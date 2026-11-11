"""평가 하네스: 기준점으로 캘리브레이션을 구역별로 채점하고, 어떤 데이터를 어떤 목적으로 썼는지 기록한다.

구역 (카메라 기준, s = HALF_SIGN[half] * x, 하프라인에서 카메라가 바라보는 쪽으로 +):
  near  s < -3        카메라 쪽 하프
  center |s| <= 3     하프라인 +-3 m
  far_mid 3 < s <= 14 먼 하프 중간
  goal   s > 14       먼 골대 구역
점(point)은 코트 좌표와의 거리와 길이(x)/폭(y) 방향 성분, 선(line)은 선까지의 수직 거리만 잰다(선을 따른 방향은 정해지지 않음).
기준점 출처(source): "tape" = 줄자로 잰 바닥 마커(G1 독립 기준), 그 밖(없으면 "video_click") = 영상에서 사람이 찍은 코트 표시.
n < 5는 참고용. 중앙값의 95% 신뢰구간은 부트스트랩(고정 시드)으로 낸다.

데이터 분리는 eval/splits.json (회차 단위). 채점할 때마다 eval/usage_log.jsonl에 커밋, 회차, 목적, 파일 해시를 남긴다.
목적이 tune인데 회차가 val/blind면 거부한다 (G4).

사용법: python eval/harness.py --match 1008 --purpose val [--cal-dir out/1008] [--labels out/1008/labels.json]
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calib_core import HALF_SIGN, match_dir  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import load, pixel_to_court  # noqa: E402
from label_points import curve_points, read_labels  # noqa: E402
from score_points import FIT_POINTS  # noqa: E402

ZONES = ("near", "center", "far_mid", "goal")
SPLITS_PATH = os.path.join(ROOT, "eval", "splits.json")
USAGE_LOG = os.path.join(ROOT, "eval", "usage_log.jsonl")
PURPOSES = ("train", "tune", "val", "blind", "report")
MIN_N = 5


def zone_of(world, half):
    s = HALF_SIGN[half] * float(world[0])
    if s < -3.0:
        return "near"
    if s <= 3.0:
        return "center"
    return "far_mid" if s <= 14.0 else "goal"


def score_rows(labels, load_cal):
    """기준점마다 한 행: cam, t, id, kind, source, zone, error(m), ex/ey(점의 길이/폭 방향 성분, m). 미확정 구간은 error None."""
    rows = []
    for lb in labels:
        cal = load_cal(lb["cam"])
        spec = spec_of(cal)
        for pt in lb["points"]:
            if pt["id"] in FIT_POINTS:  # 보정에 쓴 지점은 독립 기준이 아니다
                continue
            xy = pixel_to_court(cal, [pt["px"]], t=lb["t"])[0]
            row = {"cam": lb["cam"], "t": lb["t"], "id": pt["id"], "kind": pt["kind"],
                   "source": pt.get("source", lb.get("source", "video_click")), "error": None, "ex": None, "ey": None}
            if pt["kind"] == "point":
                truth = np.asarray(pt["world"], float)
                row["zone"] = zone_of(truth, lb["half"])
                if np.isfinite(xy).all():
                    d = xy - truth
                    row.update(error=float(np.hypot(*d)), ex=float(abs(d[0])), ey=float(abs(d[1])))
            else:
                curve = curve_points(pt["curve"], lb["half"], spec)
                near = curve[np.argmin(np.linalg.norm(curve - xy, axis=1))] if np.isfinite(xy).all() else curve[len(curve) // 2]
                row["zone"] = zone_of(near, lb["half"])
                if np.isfinite(xy).all():
                    row["error"] = float(np.linalg.norm(curve - xy, axis=1).min())
            rows.append(row)
    return rows


def median_ci(values, n_boot=2000, seed=0):
    """중앙값의 95% 부트스트랩 구간. 값이 2개 미만이면 (nan, nan)."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), (n_boot, len(v)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarize(rows, by=("cam", "zone", "kind")):
    """그룹별 n, 중앙값, 90%, 최대, 95% CI, 점의 길이/폭 방향 중앙값, 참고용 여부, 미확정 수."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    order = {z: i for i, z in enumerate(ZONES)}
    out = []
    for key in sorted(groups, key=lambda k: tuple(order.get(x, x) if isinstance(x, str) else x for x in k)):
        rs = groups[key]
        e = np.array([r["error"] for r in rs if r["error"] is not None])
        if len(e) == 0:
            out.append({**dict(zip(by, key)), "n": 0, "unresolved": len(rs)})
            continue
        lo, hi = median_ci(e)
        ex = [r["ex"] for r in rs if r["ex"] is not None]
        ey = [r["ey"] for r in rs if r["ey"] is not None]
        out.append({**dict(zip(by, key)), "n": len(e), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                    "max": float(e.max()), "ci": (lo, hi), "median_x": float(np.median(ex)) if ex else None,
                    "median_y": float(np.median(ey)) if ey else None, "reference_only": len(e) < MIN_N,
                    "unresolved": len(rs) - len(e)})
    return out


def read_splits(path=SPLITS_PATH):
    with open(path) as f:
        return json.load(f)


def role_of(match, splits):
    return next((role for role, ms in splits.items() if isinstance(ms, list) and match in ms), None)


def check_purpose(match, purpose, splits):
    """튜닝·학습에 검증/잠금 회차를 쓰려 하면 ValueError (G4)."""
    role = role_of(match, splits)
    if purpose in ("train", "tune") and role in ("val", "blind"):
        raise ValueError(f"{match}는 {role} 회차라 {purpose}에 쓸 수 없습니다 (eval/splits.json)")
    return role


def _sha(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def log_usage(match, purpose, files, extra=None, path=USAGE_LOG):
    """한 줄 JSON 기록: 시각, 커밋(작업 트리가 깨끗한지), 회차와 역할, 목적, 사용한 파일 해시."""
    rec = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "commit": _git("rev-parse", "--short", "HEAD"),
           "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")), "match": match,
           "role": role_of(match, read_splits()) if os.path.exists(SPLITS_PATH) else None, "purpose": purpose,
           "files": {os.path.relpath(p, ROOT) if os.path.isabs(p) else p: _sha(p) for p in files}, **(extra or {})}
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def gate_g1(rows):
    """G1: 줄자 기준점(source == tape)이 하나라도 채점되었는가."""
    return any(r["source"] == "tape" and r["error"] is not None for r in rows)


def fmt(s):
    if s["n"] == 0:
        return f"cam{s['cam']} {s['zone']:<8} {s['kind']:<6} n= 0 (미확정 {s['unresolved']})"
    ci = f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if np.isfinite(s["ci"][0]) else "[-]"
    xy = f" 길이 {s['median_x']:.2f} / 폭 {s['median_y']:.2f}" if s["median_x"] is not None else ""
    tag = " 참고용" if s["reference_only"] else ""
    return (f"cam{s['cam']} {s['zone']:<8} {s['kind']:<6} n={s['n']:2d} 중앙값 {s['median']:.2f} CI {ci} "
            f"90% {s['p90']:.2f} 최대 {s['max']:.2f}{xy}{tag}")


def main(argv=None):
    p = argparse.ArgumentParser(description="구역별 채점 + 사용 기록")
    p.add_argument("--match", required=True)
    p.add_argument("--purpose", required=True, choices=PURPOSES)
    p.add_argument("--cal-dir", default=None, help="채점할 캘리브레이션 폴더 (기본 out/<match>)")
    p.add_argument("--labels", default=None, help="기준점 파일 (기본 out/<match>/labels.json)")
    a = p.parse_args(argv)
    check_purpose(a.match, a.purpose, read_splits())
    cal_dir = a.cal_dir or match_dir(a.match)
    labels_path = a.labels or os.path.join(match_dir(a.match), "labels.json")
    labels = read_labels(labels_path)
    if not labels:
        raise SystemExit(f"기준점이 없습니다: {labels_path}")
    rows = score_rows(labels, lambda n: load(n, cal_dir))
    for s in summarize(rows):
        print(fmt(s))
    sources = sorted({r["source"] for r in rows})
    print(f"기준점 출처: {', '.join(sources)} / G1(줄자 기준점): {'통과' if gate_g1(rows) else '불합격'}")
    cams = sorted({lb["cam"] for lb in labels})
    log_usage(a.match, a.purpose, [labels_path] + [os.path.join(cal_dir, f"calib_cam{n}.json") for n in cams],
              {"labels_n": sum(r["error"] is not None for r in rows), "times": sorted({lb["t"] for lb in labels})})


if __name__ == "__main__":
    main()
