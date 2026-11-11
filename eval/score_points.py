"""사람이 찍은 검증 점(out/<회차>/labels.json)으로 호모그래피 정확도를 미터로 잰다.

지점(point)은 코트 좌표와의 거리, 선(line)은 해당 코트 선까지의 거리로 채점한다.
사용법: [FUTSAL_MATCH=1103] python eval/score_points.py [--out-dir out/1008] [--labels out/1008/labels.json]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court import spec_of  # noqa: E402
from court_map import load, pixel_to_court  # noqa: E402
from label_points import LABELS_PATH, curve_points, read_labels  # noqa: E402


FIT_POINTS = {"far_corner_left", "far_corner_right"}


def score(labels, load_cal):
    """라벨별 오차 행 목록. 코트를 확정 못 한 구간의 점은 error가 None이다."""
    rows = []
    for lb in labels:
        cal = load_cal(lb["cam"])
        spec = spec_of(cal)
        for pt in lb["points"]:
            if pt["id"] in FIT_POINTS:  # 보정에 쓰는 지점이라 채점에서 뺀다
                continue
            xy = pixel_to_court(cal, [pt["px"]], t=lb["t"])[0]
            if not np.isfinite(xy).all():
                err = None
            elif pt["kind"] == "point":
                err = float(np.linalg.norm(xy - pt["world"]))
            else:
                err = float(np.linalg.norm(curve_points(pt["curve"], lb["half"], spec) - xy, axis=1).min())
            rows.append({"cam": lb["cam"], "t": lb["t"], "id": pt["id"], "kind": pt["kind"], "region": pt["region"], "error": err})
    return rows


def summarize(rows):
    """(카메라, 영역, 종류)별 통계. 영역 순서: near, center, far."""
    order = {"near": 0, "center": 1, "far": 2}
    groups = {}
    for r in rows:
        if r["error"] is not None:
            groups.setdefault((r["cam"], r["region"], r["kind"]), []).append(r["error"])
    out = []
    for key in sorted(groups, key=lambda k: (k[0], order[k[1]], k[2])):
        e = np.array(groups[key])
        out.append({"cam": key[0], "region": key[1], "kind": key[2], "n": len(e), "median": float(np.median(e)),
                    "p90": float(np.percentile(e, 90)), "max": float(e.max())})
    return out


def main():
    p = argparse.ArgumentParser(description="검증 점 채점")
    p.add_argument("--out-dir", default=None, help="기본: 현재 회차 폴더 out/<FUTSAL_MATCH>")
    p.add_argument("--labels", default=LABELS_PATH)
    a = p.parse_args()
    labels = read_labels(a.labels)
    if not labels:
        raise SystemExit(f"라벨이 없습니다: {a.labels} (python eval/label_points.py --cam 1 로 먼저 찍으세요)")
    rows = score(labels, lambda n: load(n, a.out_dir))
    print(f"{'cam':>3} {'영역':<7} {'종류':<6} {'n':>3} {'중앙값':>8} {'90%':>8} {'최대':>8}")
    for s in summarize(rows):
        print(f"{s['cam']:>3} {s['region']:<7} {s['kind']:<6} {s['n']:>3} {s['median']:7.2f}m {s['p90']:7.2f}m {s['max']:7.2f}m")
    missing = sum(r["error"] is None for r in rows)
    if missing:
        print(f"코트 미확정 구간이라 채점 못 한 점: {missing}개")
    worst = sorted((r for r in rows if r["error"] is not None), key=lambda r: -r["error"])[:5]
    print("\n오차가 큰 점 (클릭 실수 확인용):")
    for r in worst:
        print(f"  cam{r['cam']} t={r['t']:g} {r['id']}: {r['error']:.2f}m")


if __name__ == "__main__":
    main()
