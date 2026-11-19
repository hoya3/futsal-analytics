"""풋살 경기 분석 리포트 (프로토타입).

python report/make_report.py <tracks.json> [--summary summary.json] [--out report_dir] [--title ...]

입력: 팀 파이프라인(python -m futsal.trackview)이 만드는 tracks.json
  {"court": {"length", "width"}, "fps", "start", "n_frames",
   "players": [{"id", "team", "name", "x": [...], "y": [...]}], "ball": {"x", "y", "cam"}?}
  좌표는 미터, 원점은 코트 모서리 (x: 0..길이, y: 0..너비), null = 안 보임.
출력: report.html (외부 리소스 없는 한 파일, 휴대폰 화면용), report.json (계산한 숫자 전부).

팀 코드는 실행 중에 import 하지 않는다. 필요한 계산은 여기서 다시 구현한다 (아래 "궤적 정리" 참고).
의존성: numpy 만.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import warnings

import numpy as np

# ---------------------------------------------------------------- 설정값 (report.json 의 "settings" 로도 나감)
SETTINGS = {
    "max_gap_s": 2.0,         # 이보다 긴 미검출 구간은 메우지 않음 (거리에서 빠짐)
    "jump_speed_ms": 10.0,    # 연속 관측 사이 이동이 이 속도 + jump_slack_m 를 넘으면 '순간이동'(ID 잘못 이음)으로 보고 끊음
    "jump_slack_m": 1.5,
    "min_segment_s": 1.0,     # 끊고 남은 조각이 이보다 짧으면 버림
    "smooth_s": 0.6,          # 이동평균 창 (7 프레임, 두 번 적용 → 삼각 가중). 팀 시뮬레이터 정답 대비 거리 오차 ±1.5%
    "smooth_ref_noise_m": 0.4,  # 흔들림 추정이 이보다 크면 창을 비례해서 넓힘
    "smooth_max_s": 3.0,
    "jump_noise_k": 6.0,      # 순간이동 기준 여유 = max(jump_slack_m, 6 x 흔들림)
    "max_speed_ms": 10.0,     # 정리 후에도 이보다 빠르면 오류로 보고 거리·속도에서 뺌
    "hi_speed_ms": 4.0,       # 고강도 달리기 기준 (약 14.4 km/h)
    "sprint_speed_ms": 5.5,   # 스프린트 기준 (팀 metrics.py 와 같음)
    "sprint_min_s": 1.0,
    "top_speed_hold_s": 0.5,  # 최고 속도 추정 = 이 시간 동안 계속 유지한 정리된 속도의 최댓값
    "top_speed_min_s": 30.0,  # 추적 시간이 이보다 짧으면 최고 속도 추정을 내지 않음
    "fragment_min_s": None,   # None → clamp(경기 시간의 10%, 30 s, 120 s)
    "gk_goal_dist_m": 6.0,    # 추적 시간의 70% 이상을 골라인 6 m 안에 있으면 골키퍼로 추정
    "gk_share": 0.7,
}
TEAM_COLORS = {"A": "#eb6834", "B": "#2a78d6"}
OTHER_COLOR = "#8a8984"
BALL_COLOR = "#3b3b38"


# ================================================================ 불러오기
def load_tracks(d: dict) -> dict:
    L, W = float(d["court"]["length"]), float(d["court"]["width"])
    fps = float(d["fps"])
    n = int(d.get("n_frames") or (len(d["players"][0]["x"]) if d["players"] else 0))
    players = []
    for p in d["players"]:
        xy = np.array([[np.nan if a is None else a, np.nan if b is None else b] for a, b in zip(p["x"], p["y"])],
                      float).reshape(-1, 2)
        if len(xy) < n:
            xy = np.vstack([xy, np.full((n - len(xy), 2), np.nan)])
        players.append({"id": p["id"], "team": str(p.get("team") if p.get("team") is not None else "?"),
                        "name": p.get("name"), "xy": xy[:n]})
    ball = None
    if d.get("ball"):
        b = d["ball"]
        ball = np.array([[np.nan if a is None else a, np.nan if c is None else c] for a, c in zip(b["x"], b["y"])],
                        float).reshape(-1, 2)
        if len(ball) < n:  # 공 추적이 먼저 끝난 경우: 선수와 같은 길이로 맞춘다 (분석이 길이 n을 가정)
            ball = np.vstack([ball, np.full((n - len(ball), 2), np.nan)])
        ball = ball[:n]
    return {"L": L, "W": W, "fps": fps, "n": n, "start": float(d.get("start", 0.0)), "players": players, "ball": ball}


# ================================================================ 궤적 정리
# 프레임마다의 위치는 수십 cm 흔들린다. 흔들림을 그대로 더하면 거리가 크게 부풀고 (특히 느린 골키퍼),
# 프레임 간 속도는 튄다. 팀 코드는 칼만+RTS 평활을 쓰는데, 여기서는 더 단순하고 검증하기 쉬운 방법을 쓴다:
#   0) 궤적마다 위치 흔들림 σ 를 2차 차분의 MAD 로 추정한다 (noise_std).
#   1) 연속 관측 사이 이동이 물리적으로 불가능하면 (jump_speed_ms x 시간 + max(jump_slack_m, 6σ)) 그 자리에서
#      궤적을 끊는다 (ID를 다른 사람에게 잘못 이어 붙인 '순간이동'). 미검출이 max_gap_s 보다 길어도 끊는다.
#   2) 끊긴 조각 안의 짧은 빈칸은 선형 보간으로 메운다. min_segment_s 보다 짧은 조각은 버린다.
#   3) 조각마다 이동평균(smooth_s, σ 가 smooth_ref_noise_m 보다 크면 비례해서 넓힘)을 두 번 적용한다.
#      끝점은 점대칭으로 늘려서 등속 직선 운동은 그대로 보존된다.
#   팀 시뮬레이터(futsal.sim tracks, 10분) 정답과 비교: 필드 선수 거리 오차 -1~+1.2 %, 골키퍼 -2~-8 %,
#   최고 속도 추정은 정답 최댓값보다 0~0.8 m/s 낮음 (골키퍼는 흔들림이 커서 ±2 m/s).
#   4) 속도 = 정리된 위치의 중앙 차분. max_speed_ms 를 넘는 값은 오류로 보고 버린다.
#   거리 = 조각 안에서 정리된 위치의 이동 합 (조각 사이, 긴 빈칸, 순간이동은 더하지 않음).
def _box(seg: np.ndarray, w: int) -> np.ndarray:
    p = min(w // 2, len(seg) - 1)
    if p < 1:
        return seg.copy()
    front = 2 * seg[0] - seg[1:p + 1][::-1]
    back = 2 * seg[-1] - seg[-p - 1:-1][::-1]
    ext = np.concatenate([front, seg, back])
    k = np.ones(2 * p + 1) / (2 * p + 1)
    return np.stack([np.convolve(ext[:, c], k, mode="valid") for c in range(seg.shape[1])], 1)


def noise_std(xy: np.ndarray) -> float:
    """위치 흔들림(m)을 연속 관측의 2차 차분에서 추정 (MAD, 두 축 합). 등속·등가속 운동에는 0."""
    sig = []
    for c in range(2):
        z = xy[:, c]
        d2 = z[2:] - 2 * z[1:-1] + z[:-2]
        d2 = d2[np.isfinite(d2)]
        if len(d2) < 5:
            return 0.0
        sig.append(1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6))
    return float(math.hypot(*sig))


def smooth_track(xy: np.ndarray, fps: float, cfg: dict = SETTINGS) -> dict:
    n = len(xy)
    s = np.full((n, 2), np.nan)
    v = np.full(n, np.nan)
    seg_id = np.full(n, -1)
    out = {"s": s, "v": v, "seg_id": seg_id, "jumps": 0, "dropped_segments": 0, "too_fast": 0, "noise_m": None,
           "smooth_s": None}
    idx = np.flatnonzero(np.isfinite(xy).all(1))
    if len(idx) < 2:
        return out
    dframes = np.diff(idx)
    step = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1)
    sigma = noise_std(xy)
    out["noise_m"] = sigma
    # 위치 흔들림이 큰 궤적(먼 쪽 골키퍼 등)은 흔들림만으로도 한 프레임에 수 m 튀므로 기준을 그만큼 넓힌다
    slack = max(cfg["jump_slack_m"], cfg["jump_noise_k"] * sigma)
    jump = step > cfg["jump_speed_ms"] * dframes / fps + slack
    long_gap = (dframes - 1) / fps > cfg["max_gap_s"] + 1e-9
    out["jumps"] = int(jump.sum())
    brk = np.flatnonzero(jump | long_gap)
    starts, ends = np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]
    # 흔들림이 기준(smooth_ref_noise_m)보다 크면 창을 비례해서 넓힌다 (최대 smooth_max_s)
    smooth_s = min(cfg["smooth_max_s"], cfg["smooth_s"] * max(1.0, sigma / cfg["smooth_ref_noise_m"]))
    out["smooth_s"] = smooth_s
    w = max(1, int(round(smooth_s * fps)) // 2 * 2 + 1)
    min_len = int(round(cfg["min_segment_s"] * fps))
    k = 0
    for st, en in zip(starts, ends):
        a, b = idx[st], idx[en]
        if b - a + 1 < max(2, min_len):
            out["dropped_segments"] += 1
            continue
        t = np.arange(a, b + 1)
        known = idx[st:en + 1]
        seg = np.stack([np.interp(t, known, xy[known, c]) for c in range(2)], 1)
        sm = _box(_box(seg, w), w)
        s[a:b + 1] = sm
        v[a:b + 1] = np.linalg.norm(np.gradient(sm, axis=0), axis=1) * fps if len(sm) > 1 else 0.0
        seg_id[a:b + 1] = k
        k += 1
    bad = v > cfg["max_speed_ms"]
    out["too_fast"] = int(bad.sum())
    v[bad] = np.nan
    return out


def _step_mask(tr: dict) -> np.ndarray:
    """i → i+1 이동을 거리에 넣을지 (같은 조각 안, 두 끝 속도 모두 정상)."""
    sid, v = tr["seg_id"], tr["v"]
    return (sid[:-1] == sid[1:]) & (sid[:-1] >= 0) & np.isfinite(v[:-1]) & np.isfinite(v[1:])


def player_metrics(xy: np.ndarray, fps: float, block_frames: int | None = None, cfg: dict = SETTINGS,
                   tr: dict | None = None) -> dict:
    tr = tr or smooth_track(xy, fps, cfg)
    s, v = tr["s"], tr["v"]
    ok = _step_mask(tr)
    steps = np.where(ok, np.linalg.norm(np.nan_to_num(np.diff(s, axis=0)), axis=1), 0.0)
    vstep = np.where(ok, (np.nan_to_num(v[:-1]) + np.nan_to_num(v[1:])) / 2, 0.0)
    covered = np.isfinite(s[:, 0])
    tracked_s = covered.sum() / fps
    seen = np.isfinite(xy).all(1)
    vv = v[np.isfinite(v)]
    dist = float(steps.sum())
    fast = np.nan_to_num(v, nan=0.0) >= cfg["sprint_speed_ms"]
    need = max(1, int(round(cfg["sprint_min_s"] * fps)))
    sprints, run = 0, 0
    for f in fast:
        run = run + 1 if f else 0
        sprints += run == need
    # 원시 최고 속도: 연속 프레임 관측 사이 이동 그대로 (흔들림·순간이동이 들어 있음 → 참고용)
    i = np.flatnonzero(seen[:-1] & seen[1:])
    raw_max = float(np.max(np.linalg.norm(xy[i + 1] - xy[i], axis=1)) * fps) if len(i) else None
    # 최고 속도 추정: top_speed_hold_s 동안 계속 유지한 속도(창 안의 최솟값)의 최댓값.
    # 한두 프레임만 튀는 값은 창의 최솟값에서 걸러진다. 순간이동은 이미 smooth_track 에서 끊겼다.
    top = None
    if len(vv) and tracked_s >= cfg["top_speed_min_s"]:
        k = max(1, int(round(cfg["top_speed_hold_s"] * fps)))
        vz = np.where(np.isfinite(v), v, -np.inf)
        if len(vz) >= k:
            win = np.lib.stride_tricks.sliding_window_view(vz, k).min(1)
            top = float(win.max()) if np.isfinite(win.max()) else None
    m = {
        "seen_s": float(seen.sum() / fps),
        "seen_share": float(seen.mean()) if len(seen) else 0.0,
        "tracked_s": float(tracked_s),
        "distance_m": dist,
        "distance_per_min": dist / (tracked_s / 60) if tracked_s > 0 else None,
        "mean_speed_ms": float(np.mean(vv)) if len(vv) else None,
        "top_speed_est_ms": top,
        "speed_p99_ms": float(np.percentile(vv, 99)) if len(vv) else None,
        "raw_max_speed_ms": raw_max,
        "hi_distance_m": float(steps[vstep >= cfg["hi_speed_ms"]].sum()),
        "sprints": int(sprints),
        "jumps": tr["jumps"],
        "noise_m": tr["noise_m"],
        "smooth_s": tr["smooth_s"],
    }
    if block_frames:
        nb = int(math.ceil(len(xy) / block_frames))
        m["distance_blocks_m"] = [float(x) for x in np.bincount(np.arange(len(steps)) // block_frames,
                                                                  weights=steps, minlength=nb)[:nb]]
    return m


def fragment_threshold_s(duration_s: float, cfg: dict = SETTINGS) -> float:
    if cfg.get("fragment_min_s") is not None:
        return float(cfg["fragment_min_s"])
    return float(min(120.0, max(30.0, 0.1 * duration_s)))


# ================================================================ 구역
THIRDS_ABS = ["왼쪽 골대 쪽 1/3", "가운데 1/3", "오른쪽 골대 쪽 1/3"]
CHANNELS_ABS = ["아래쪽 측면", "중앙", "위쪽 측면"]          # 그림 기준: y 가 위로 커짐
THIRDS_REL = ["수비 1/3", "중간 1/3", "공격 1/3"]
CHANNELS_REL = ["오른쪽 측면", "중앙", "왼쪽 측면"]          # 공격 방향(+x)을 볼 때 +y 가 왼쪽


def zone_shares(xy: np.ndarray, L: float, W: float) -> dict:
    """xy (N,2) 의 표본이 3x3 구역에 있던 비율. thirds[i]: x 방향 1/3, channels[j]: y 방향 1/3, grid[j][i]."""
    xy = xy[np.isfinite(xy).all(1)]
    if not len(xy):
        return {"n": 0, "thirds": [0, 0, 0], "channels": [0, 0, 0], "grid": [[0] * 3 for _ in range(3)]}
    ix = np.clip((xy[:, 0] / L * 3).astype(int), 0, 2)
    iy = np.clip((xy[:, 1] / W * 3).astype(int), 0, 2)
    g = np.zeros((3, 3))
    np.add.at(g, (iy, ix), 1)
    g /= len(xy)
    return {"n": int(len(xy)), "thirds": [float(a) for a in g.sum(0)], "channels": [float(a) for a in g.sum(1)],
            "grid": [[float(a) for a in row] for row in g]}


def detect_sides(cx_a: np.ndarray, cx_b: np.ndarray, fps: float, min_agree: float = 0.85) -> dict:
    """두 팀의 무게중심 x (프레임별, NaN 가능)로 진영과 후반 진영 교체를 추정한다.
    분 단위로 sign(A - B)를 보고, 한 번 바뀌는 지점 중 가장 잘 맞는 곳을 찾는다.
    반환: {"switch_frame": int|None, "a_left_first": bool|None, "agree": float}"""
    bf = int(round(60 * fps))
    nb = len(cx_a) // bf if len(cx_a) >= bf else 1
    bf = len(cx_a) // nb if len(cx_a) < bf else bf
    sgn = []
    for k in range(nb):
        a, b = cx_a[k * bf:(k + 1) * bf], cx_b[k * bf:(k + 1) * bf]
        ok = np.isfinite(a) & np.isfinite(b)
        sgn.append(np.sign(np.mean(a[ok] - b[ok])) if ok.sum() > 0.2 * bf else 0.0)
    sgn = np.array(sgn)
    known = sgn != 0
    if known.sum() < 2:
        return {"switch_frame": None, "a_left_first": None, "agree": 0.0}
    no_switch = max((sgn[known] < 0).mean(), (sgn[known] > 0).mean())
    best = (0.0, None, None)
    for k in range(3, nb - 2):                           # 양쪽 다 3분 이상
        before, after = sgn[:k][known[:k]], sgn[k:][known[k:]]
        if len(before) < 2 or len(after) < 2:
            continue
        for first in (-1, 1):
            sc = ((before == first).sum() + (after == -first).sum()) / (len(before) + len(after))
            if sc > best[0]:
                best = (sc, k, first)
    if best[1] is not None and best[0] >= min_agree and best[0] >= no_switch + 0.1:
        return {"switch_frame": int(best[1] * bf), "a_left_first": bool(best[2] < 0), "agree": float(best[0])}
    if no_switch >= 0.7:
        return {"switch_frame": None, "a_left_first": bool((sgn[known] < 0).mean() >= 0.5), "agree": float(no_switch)}
    return {"switch_frame": None, "a_left_first": None, "agree": float(no_switch)}


def to_attack_right(s: np.ndarray, orient: np.ndarray, L: float, W: float) -> np.ndarray:
    """orient == -1 (왼쪽으로 공격) 인 프레임을 180도 돌려서, 항상 오른쪽으로 공격하는 좌표로 바꾼다."""
    out = s.copy()
    m = orient < 0
    out[m] = np.array([L, W]) - s[m]
    return out


def heatmap(xy: np.ndarray, L: float, W: float, cell: float = 1.0) -> np.ndarray:
    xy = xy[np.isfinite(xy).all(1)]
    nx, ny = int(math.ceil(L / cell)), int(math.ceil(W / cell))
    h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[ny, nx], range=[[0, W], [0, L]]) if len(xy) else (
        np.zeros((ny, nx)), None, None)
    k = np.array([1, 2, 1], float)
    for _ in range(2):                                    # 살짝 흐리게 (보기 좋게, 합은 유지)
        hp = np.pad(h, 1, mode="edge")
        h = (hp[:-2, 1:-1] * k[0] + hp[1:-1, 1:-1] * k[1] + hp[2:, 1:-1] * k[2]) / 4
        hp = np.pad(h, 1, mode="edge")
        h = (hp[1:-1, :-2] * k[0] + hp[1:-1, 1:-1] * k[1] + hp[1:-1, 2:] * k[2]) / 4
    tot = h.sum()
    return h / tot if tot > 0 else h


# ================================================================ 분석
def parse_roster(spec) -> dict:
    """팀별 실제 선수 수. "A=5 B=5" 같은 목록/문자열 또는 {"A": 5} 사전 → {"A": 5, "B": 5}."""
    if spec is None:
        return {}
    if isinstance(spec, dict):
        items = list(spec.items())
    else:
        if isinstance(spec, str):
            spec = spec.replace(",", " ").split()
        items = []
        for tok in spec:
            if "=" not in tok:
                raise ValueError(f"인원 형식은 팀=숫자 (예: A=5): {tok!r}")
            k, v = tok.split("=", 1)
            items.append((k.strip(), v.strip()))
    out = {}
    for k, v in items:
        n = int(v)
        if n <= 0:
            raise ValueError(f"인원은 1 이상이어야 함: {k}={v}")
        out[str(k)] = n
    return out


def resolve_roster(cli=None, tracks: dict | None = None, summary: dict | None = None) -> tuple[dict, str | None]:
    """우선순위: CLI --players > tracks.json "roster" > summary.json "roster"."""
    for spec, src in ((cli, "명령줄 --players"), ((tracks or {}).get("roster"), "tracks.json"),
                      ((summary or {}).get("roster"), "summary.json")):
        r = parse_roster(spec)
        if r:
            return r, src
    return {}, None


def roster_checks(roster: dict, main_ids: dict) -> list[dict]:
    """팀마다 주요 ID 수와 실제 인원 비교. ID를 합치거나 고르지는 않는다 (추적 담당 몫). 인원 정보가 없으면 []."""
    out = []
    if not roster:
        return out
    for t, n in roster.items():
        k = main_ids.get(t, 0)
        if k > n:
            st, msg = "more_ids", f"{team_label(t)}: 주요 ID {k}개, 실제 선수 {n}명 → 한 선수의 ID가 여러 개로 나뉘었을 수 있음"
        elif k < n:
            st, msg = "fewer_ids", (f"{team_label(t)}: 주요 ID {k}개, 실제 선수 {n}명 → 일부 선수가 짧게만 잡혔거나 "
                                    "다른 팀으로 분류됐을 수 있음")
        else:
            st, msg = "ok", f"{team_label(t)}: 주요 ID {k}개 = 실제 선수 {n}명"
        out.append({"team": t, "roster": n, "main_ids": k, "status": st, "message": msg})
    for t, k in main_ids.items():
        if t not in roster and k:
            out.append({"team": t, "roster": None, "main_ids": k, "status": "not_in_roster",
                        "message": f"{team_label(t)}: 인원 정보에 없는 팀 (주요 ID {k}개)"})
    return out


def analyze(d: dict, summary: dict | None = None, title: str | None = None, cfg: dict = SETTINGS,
            roster: dict | None = None, roster_source: str | None = None) -> dict:
    """roster 가 None 이면 tracks.json / summary.json 의 "roster" 를 찾는다."""
    if roster is None:
        roster, roster_source = resolve_roster(None, d, summary)
    roster = roster or {}
    T = load_tracks(d)
    L, W, fps, n = T["L"], T["W"], T["fps"], T["n"]
    duration = n / fps
    block_min = 5 if duration >= 15 * 60 else 1
    bf = int(round(block_min * 60 * fps))
    nb = max(1, int(math.ceil(n / bf)))
    frag_s = fragment_threshold_s(duration, cfg)

    players, tracks = [], {}
    for p in T["players"]:
        tr = smooth_track(p["xy"], fps, cfg)
        m = player_metrics(p["xy"], fps, bf, cfg, tr)
        s = tr["s"]
        cov = np.isfinite(s[:, 0])
        gd = np.minimum(s[cov, 0], L - s[cov, 0])
        gk = bool(len(gd) and (gd <= cfg["gk_goal_dist_m"]).mean() >= cfg["gk_share"])
        players.append({"id": p["id"], "team": p["team"], "name": p["name"],
                        "label": p["name"] or f"ID {p['id']}", "fragment": m["seen_s"] < frag_s,
                        "gk_guess": gk and m["seen_s"] >= frag_s, **m,
                        "first_seen_s": float(np.argmax(np.isfinite(p["xy"][:, 0])) / fps)})
        tracks[p["id"]] = tr

    # 짧게 잡힌 ID(조각)는 팀 집계 어디에도 넣지 않는다 (데이터 품질 섹션에만 나옴)
    main_players = [p for p in players if not p["fragment"]]
    team_names = sorted({p["team"] for p in main_players}, key=lambda t: (-sum(q["tracked_s"] for q in main_players
                                                                                if q["team"] == t), t))
    main_teams = sorted([t for t in team_names if t not in ("?", "", "None")][:2])
    team_names = main_teams + [t for t in team_names if t not in main_teams]

    def team_stack(t, which="all"):
        ids = [p["id"] for p in main_players if p["team"] == t and (which == "all" or
               (which == "outfield" and not p["gk_guess"]))]
        return np.stack([tracks[i]["s"] for i in ids]) if ids else np.full((0, n, 2), np.nan)

    # ---- 진영 / 후반 교체
    sides = {"switch_frame": None, "a_left_first": None, "agree": 0.0, "teams": main_teams}
    orient = {t: np.zeros(n) for t in team_names}
    if len(main_teams) == 2 and n:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cxa = np.nanmean(team_stack(main_teams[0])[:, :, 0], axis=0)
            cxb = np.nanmean(team_stack(main_teams[1])[:, :, 0], axis=0)
        sides.update(detect_sides(cxa, cxb, fps))
        if sides["a_left_first"] is not None:
            o = np.full(n, 1.0 if sides["a_left_first"] else -1.0)      # +1: 오른쪽으로 공격
            if sides["switch_frame"] is not None:
                o[sides["switch_frame"]:] *= -1
            orient[main_teams[0]], orient[main_teams[1]] = o, -o
    sw = sides["switch_frame"]
    periods = ([("전반", 0, sw), ("후반", sw, n)] if sw is not None else [("경기 전체", 0, n)])

    # ---- 팀
    teams = {}
    for t in team_names:
        frs = [p for p in players if p["team"] == t and p["fragment"]]
        tp = mains = [p for p in main_players if p["team"] == t]
        allpos = team_stack(t)
        flat = allpos.reshape(-1, 2)
        o_rep = np.tile(orient[t], len(allpos)) if len(allpos) else np.zeros(0)
        rel = to_attack_right(flat, o_rep, L, W) if (orient[t] != 0).any() else None
        z_abs = zone_shares(flat, L, W)
        z_rel = zone_shares(rel[o_rep != 0], L, W) if rel is not None else None
        # 팀 형태: 골키퍼(추정) 뺀 필드 선수, 3명 이상 보이는 프레임만
        of = team_stack(t, "outfield")
        cnt = np.isfinite(of[:, :, 0]).sum(0) if len(of) else np.zeros(n, int)
        good = cnt >= 3
        shape = {"frames": int(good.sum())}
        if good.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                xs, ys = of[:, good, 0], of[:, good, 1]
                depth = np.nanmax(xs, 0) - np.nanmin(xs, 0)
                width = np.nanmax(ys, 0) - np.nanmin(ys, 0)
                pair = []
                for i in range(len(of)):
                    for j in range(i + 1, len(of)):
                        pair.append(np.linalg.norm(of[i, good] - of[j, good], axis=1))
                spread = np.nanmean(np.stack(pair), 0)
            fr = np.flatnonzero(good)
            blk = fr // bf
            shape.update({
                "width_m_median": float(np.median(width)), "depth_m_median": float(np.median(depth)),
                "width_m_p10_p90": [float(np.percentile(width, 10)), float(np.percentile(width, 90))],
                "depth_m_p10_p90": [float(np.percentile(depth, 10)), float(np.percentile(depth, 90))],
                "teammate_dist_m_mean": float(np.mean(spread)),
                "teammate_dist_blocks_m": [float(np.mean(spread[blk == k])) if (blk == k).any() else None
                                           for k in range(nb)],
            })
        blocks = np.sum([p["distance_blocks_m"] for p in tp], axis=0) if tp else np.zeros(nb)
        per_period = []
        for name, a, b in periods:
            per_period.append({"name": name, "distance_m": float(sum(
                np.sum(steps_between(tracks[p["id"]], a, b)) for p in tp))})
        total = float(sum(p["distance_m"] for p in tp))
        teams[t] = {
            "ids": len(tp) + len(frs), "main_ids": len(mains), "fragment_ids": len(frs),
            "distance_total_m": total,                       # 주요 ID만
            "distance_per_main_player_m": float(np.mean([p["distance_m"] for p in mains])) if mains else None,
            "roster": roster.get(t),
            "distance_per_roster_player_m": total / roster[t] if roster.get(t) else None,
            "fragment_distance_m": float(sum(p["distance_m"] for p in frs)),     # 넣었다면 더해졌을 거리 (참고)
            "distance_per_min_team": float(sum(p["distance_m"] for p in tp) / (duration / 60)) if duration else None,
            "hi_distance_m": float(sum(p["hi_distance_m"] for p in tp)),
            "sprints": int(sum(p["sprints"] for p in tp)),
            "distance_blocks_m": [float(x) for x in blocks],
            "periods": per_period,
            "zones_abs": z_abs, "zones_rel": z_rel, "direction_known": bool((orient[t] != 0).any()),
            "shape": shape,
            "heatmap_rel_frame": rel is not None,
            "_heat": heatmap(rel if rel is not None else flat, L, W, 1.0),
        }

    # ---- 선수 평균 위치 (구간별, 절대 좌표) + 선수 히트맵 (공격 방향 기준)
    for p in players:
        s = tracks[p["id"]]["s"]
        p["mean_pos"] = []
        for name, a, b in periods:
            seg = s[a:b]
            ok = np.isfinite(seg[:, 0])
            p["mean_pos"].append({"period": name, "x": float(seg[ok, 0].mean()) if ok.sum() > fps * 30 else None,
                                  "y": float(seg[ok, 1].mean()) if ok.sum() > fps * 30 else None})
        o = orient.get(p["team"], np.zeros(n))
        rel = to_attack_right(s, o, L, W)
        p["zones_abs"] = zone_shares(s, L, W)
        p["zones_rel"] = zone_shares(rel[o != 0], L, W) if (o != 0).any() else None
        p["_heat"] = heatmap(rel, L, W, 2.0) if not p["fragment"] else None

    # ---- 공
    ball = None
    if T["ball"] is not None and len(T["ball"]):
        b = T["ball"]
        seen = np.isfinite(b).all(1)
        inside = seen & (b[:, 0] >= -1) & (b[:, 0] <= L + 1) & (b[:, 1] >= -1) & (b[:, 1] <= W + 1)
        left = inside & (b[:, 0] < L / 2)
        blk = np.arange(n) // bf
        ball = {"coverage": float(seen.mean()), "off_court_share_of_seen": float((seen & ~inside).sum() / max(1, seen.sum())),
                "left_half_share": float(left.sum() / max(1, inside.sum())),
                "left_half_blocks": [float(left[blk == k].sum() / inside[blk == k].sum())
                                     if inside[blk == k].sum() > 0 else None for k in range(nb)],
                "coverage_blocks": [float(seen[blk == k].mean()) for k in range(nb)],
                "zones_abs": zone_shares(b[inside], L, W)}
        if len(main_teams) == 2 and sides["a_left_first"] is not None:
            oa = orient[main_teams[0]]
            in_a_half = inside & (((oa > 0) & (b[:, 0] < L / 2)) | ((oa < 0) & (b[:, 0] >= L / 2)))
            ball["in_half_of"] = main_teams[0]
            ball["in_team_half_share"] = float(in_a_half.sum() / max(1, inside.sum()))
            ball["in_team_half_blocks"] = [float(in_a_half[blk == k].sum() / inside[blk == k].sum())
                                           if inside[blk == k].sum() > 0 else None for k in range(nb)]

    # ---- 데이터 품질
    mains = main_players
    frag_by_team = {}
    for p in players:
        if p["fragment"]:
            f = frag_by_team.setdefault(p["team"], {"ids": 0, "distance_m": 0.0})
            f["ids"] += 1
            f["distance_m"] += p["distance_m"]
    quality = {
        "fragment_by_team": frag_by_team,
        "roster_checks": roster_checks(roster, {t: teams[t]["main_ids"] for t in team_names}),
        "ids": len(players), "main_ids": len(mains), "fragment_ids": len(players) - len(mains),
        "fragment_threshold_s": frag_s,
        "fragment_distance_share": float(sum(p["distance_m"] for p in players if p["fragment"]) /
                                         max(1e-9, sum(p["distance_m"] for p in players))),
        "seen_share_main_median": float(np.median([p["seen_share"] for p in mains])) if mains else None,
        "jumps_total": int(sum(p["jumps"] for p in players)),
        "unknown_team_ids": sum(p["team"] not in main_teams for p in players),
        "calibration_error_note": "코트 좌표 오차 약 0.15 m (가까운 쪽) ~ 0.3 m (먼 쪽)",
    }
    pipe = None
    if summary:
        keep = ("ids", "ids_over_half", "events", "rate_hz", "window_cam1_s", "offsets", "alignment", "ball")
        pipe = {k: summary[k] for k in keep if k in summary}
        sp = {q.get("id"): q for q in summary.get("players", [])}
        for p in players:
            if p["id"] in sp:
                p["pipeline_max_speed_ms"] = sp[p["id"]].get("max_speed_ms")
                p["pipeline_distance_m"] = sp[p["id"]].get("distance_m")

    return {
        "version": 1, "title": title or "풋살 경기 리포트",
        "match": {"duration_s": duration, "fps": fps, "n_frames": n, "start_s": T["start"],
                  "court": {"length": L, "width": W}, "block_min": block_min, "n_blocks": nb},
        "settings": {**cfg, "fragment_min_s": frag_s},
        "sides": sides, "periods": [{"name": a, "from_s": b / fps, "to_s": c / fps} for a, b, c in periods],
        "teams": teams, "team_order": team_names, "main_teams": main_teams,
        "roster": roster or None, "roster_source": roster_source,
        "players": players, "ball": ball, "quality": quality, "pipeline_summary": pipe,
    }


def steps_between(tr: dict, a: int, b: int) -> np.ndarray:
    s = tr["s"][a:b]
    ok = _step_mask({"seg_id": tr["seg_id"][a:b], "v": tr["v"][a:b]})
    return np.where(ok, np.linalg.norm(np.nan_to_num(np.diff(s, axis=0)), axis=1), 0.0)


def to_json(res: dict) -> dict:
    """report.json 용: 내부 배열(_heat)을 목록으로, numpy 값을 파이썬 값으로."""
    def clean(o):
        if isinstance(o, dict):
            return {k.lstrip("_"): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return [[round(float(a), 5) for a in row] for row in o] if o.ndim == 2 else [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(o) else round(float(o), 3)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    return clean(res)


# ================================================================ SVG 그리기
def team_color(t: str) -> str:
    return TEAM_COLORS.get(t, OTHER_COLOR)


def team_label(t: str) -> str:
    return f"팀 {t}" if t not in ("?", "", "None") else "팀 미상"


def _f(v, nd=0, unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:,.{nd}f}{unit}"


def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _arc(cx, cy, r, a0, a1, k=24):
    a = np.radians(np.linspace(a0, a1, k))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def court_svg(L: float, W: float, inner: str = "", attack_arrow: bool = False, label: str = "") -> str:
    """코트 그림 (x 오른쪽, y 위쪽). 중앙원 r=3, 페널티 구역 = 골대 기둥에서 6 m 사분원 두 개 + 3 m 직선."""
    def P(pts):
        return " ".join(f"{x:.2f},{W - y:.2f}" for x, y in pts)
    cy, gw = W / 2, 3.0
    lines = [f'<rect x="0" y="0" width="{L}" height="{W}" class="cl"/>',
             f'<line x1="{L / 2}" y1="0" x2="{L / 2}" y2="{W}" class="cl"/>',
             f'<circle cx="{L / 2}" cy="{W - cy}" r="3" class="cl"/>',
             f'<circle cx="{L / 2}" cy="{W - cy}" r="0.15" class="cm"/>']
    for gx, sgn in ((0.0, 1), (L, -1)):
        lo, hi = cy - gw / 2, cy + gw / 2
        if sgn > 0:
            pts = np.vstack([_arc(gx, lo, 6, -90, 0), _arc(gx, hi, 6, 0, 90)])
        else:
            pts = np.vstack([_arc(gx, lo, 6, 270, 180), _arc(gx, hi, 6, 180, 90)])
        lines.append(f'<polyline points="{P(pts)}" class="cl"/>')
        for dm in (6, 10):
            lines.append(f'<circle cx="{gx + sgn * dm}" cy="{W - cy}" r="0.15" class="cm"/>')
        gxx = gx - 1 if sgn > 0 else gx
        lines.append(f'<rect x="{gxx}" y="{W - hi}" width="1" height="{gw}" class="cg"/>')
    arrow = ""
    if attack_arrow:
        arrow = (f'<g class="arr"><line x1="{L / 2 - 4}" y1="{W + 1.6}" x2="{L / 2 + 4}" y2="{W + 1.6}"/>'
                 f'<polyline points="{L / 2 + 3},{W + 0.9} {L / 2 + 4},{W + 1.6} {L / 2 + 3},{W + 2.3}"/></g>'
                 f'<text x="{L / 2 + 4.6}" y="{W + 2.1}" class="tl">공격 방향 (추정)</text>')
    lab = f'<text x="0" y="-0.6" class="tl">{html.escape(label)}</text>' if label else ""
    h = W + (3.2 if attack_arrow else 1.2) + (1.6 if label else 0)
    top = -2.2 if label else -1.2
    return (f'<svg class="court" viewBox="-1.2 {top} {L + 2.4} {h + (1.2 if not label else 0.6)}" '
            f'role="img"><rect x="-1.2" y="{top}" width="{L + 2.4}" height="{h + 2}" class="bg"/>'
            f'<rect x="0" y="0" width="{L}" height="{W}" class="pitch"/>{inner}{"".join(lines)}{arrow}{lab}</svg>')


def heat_cells(h: np.ndarray, L: float, W: float, color: str) -> str:
    if h is None or not h.size or h.max() <= 0:
        return ""
    ny, nx = h.shape
    cw, ch = L / nx, W / ny
    mx = h.max()
    out = []
    for j in range(ny):
        for i in range(nx):
            a = (h[j, i] / mx) ** 0.6
            if a < 0.04:
                continue
            out.append(f'<rect x="{i * cw:.2f}" y="{W - (j + 1) * ch:.2f}" width="{cw + 0.02:.2f}" '
                       f'height="{ch + 0.02:.2f}" fill="{color}" fill-opacity="{0.9 * a:.2f}"/>')
    return "".join(out)


def zone_grid(z: dict, L: float, W: float) -> str:
    out = []
    for j in range(3):
        for i in range(3):
            v = z["grid"][j][i] if z and z["n"] else 0
            x, y = (i + 0.5) * L / 3, W - (j + 0.5) * W / 3
            out.append(f'<text x="{x:.2f}" y="{y + 0.6:.2f}" class="zt">{v * 100:.0f}%</text>')
    for i in (1, 2):
        out.append(f'<line x1="{i * L / 3:.2f}" y1="0" x2="{i * L / 3:.2f}" y2="{W}" class="zl"/>')
    for j in (1, 2):
        out.append(f'<line x1="0" y1="{j * W / 3:.2f}" x2="{L}" y2="{j * W / 3:.2f}" class="zl"/>')
    return "".join(out)


def line_chart(series: list[tuple[str, str, list]], labels: list[str], unit: str, marker_at: float | None = None,
               ymin0: bool = True, pct: bool = False) -> str:
    """series: (이름, 색, 값들). 하나의 y축. 점마다 <title> 툴팁."""
    Wd, Hd, ml, mr, mt, mb = 360, 170, 40, 54, 10, 26
    vals = [v for _, _, s in series for v in s if v is not None]
    if not vals:
        return "<p class='muted'>데이터 없음</p>"
    lo = 0 if ymin0 else min(vals)
    hi = max(vals) if not pct else 1.0
    if hi <= lo:
        hi = lo + 1
    if not pct:                                          # 눈금을 보기 좋은 값(1, 2, 2.5, 5 x 10^k)으로
        raw = (hi - lo) / 4
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        lo = math.floor(lo / step) * step
        hi = lo + 4 * step
        while hi < max(vals):
            hi += step
    nb = len(labels)
    X = lambda k: ml + (Wd - ml - mr) * (k + 0.5) / nb
    Y = lambda v: mt + (Hd - mt - mb) * (1 - (v - lo) / (hi - lo))
    nd = 0 if pct or (hi - lo) / 4 >= 1 or float((hi - lo) / 4).is_integer() else 1
    g = []
    for q in range(5):
        v = lo + (hi - lo) * q / 4
        g.append(f'<line x1="{ml}" x2="{Wd - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="gr"/>'
                 f'<text x="{ml - 4}" y="{Y(v) + 3:.1f}" class="ax" text-anchor="end">'
                 f'{(f"{v * 100:.0f}%" if pct else f"{v:,.{nd}f}")}</text>')
    step = max(1, nb // 8)
    for k, lb in enumerate(labels):
        if k % step == 0:
            g.append(f'<text x="{X(k):.1f}" y="{Hd - 8}" class="ax" text-anchor="middle">{lb}</text>')
    if marker_at is not None:
        xm = ml + (Wd - ml - mr) * marker_at / nb
        g.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{mt}" y2="{Hd - mb}" class="ht"/>'
                 f'<text x="{xm + 3:.1f}" y="{Hd - mb - 4}" class="ax">진영 교체</text>')
    ends = []
    for name, col, s in series:
        pts = [(X(k), Y(v)) for k, v in enumerate(s) if v is not None]
        if len(pts) > 1:
            g.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                     f'stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for k, v in enumerate(s):
            if v is None:
                continue
            txt = f"{v * 100:.0f}%" if pct else f"{v:,.0f} {unit}"
            g.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}" stroke="#fcfcfb" '
                     f'stroke-width="1.5"><title>{html.escape(name)} · {labels[k]}분: {txt}</title></circle>')
        if pts and len(series) > 1:                     # 한 줄짜리는 제목이 이름을 대신함
            ends.append([pts[-1][1] + 4, pts[-1][0] + 6, name])
    ends.sort()
    for k in range(1, len(ends)):                       # 끝 라벨이 겹치지 않게 아래로 민다
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 12)
    for y, x, name in ends:
        g.append(f'<text x="{x:.1f}" y="{y:.1f}" class="dl">{html.escape(name)}</text>')
    return f'<svg class="chart" viewBox="0 0 {Wd} {Hd}" role="img">{"".join(g)}</svg>'


def sparkline(vals: list[float], color: str, vmax: float) -> str:
    w, h = 84, 22
    if not vals or vmax <= 0:
        return ""
    n = len(vals)
    pts = " ".join(f"{(k + 0.5) * w / n:.1f},{h - 2 - (h - 4) * v / vmax:.1f}" for k, v in enumerate(vals))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><polyline points="{pts}" '
            f'fill="none" stroke="{color}" stroke-width="1.6"/></svg>')


def bar_row(label: str, shares: list[float], names: list[str], color: str) -> str:
    cells = "".join(f'<div class="seg" style="flex:{max(s, 0.001):.4f};background:{color};opacity:{0.35 + 0.65 * k / 2:.2f}" '
                    f'title="{html.escape(nm)} {s * 100:.0f}%">{s * 100:.0f}%</div>'
                    for k, (s, nm) in enumerate(zip(shares, names)))
    return f'<div class="br"><div class="bl">{html.escape(label)}</div><div class="bar">{cells}</div></div>'


# ================================================================ HTML
CSS = """
:root{--bg:#f4f4f1;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e2e1dc;--pitch:#eef3ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif}
main{max-width:860px;margin:0 auto;padding:12px 16px 40px}
h1{font-size:22px;margin:8px 0 2px}h2{font-size:18px;margin:0 0 8px}h3{font-size:15px;margin:12px 0 6px}
.sub{color:var(--ink2);font-size:13px}.muted{color:var(--muted)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.tile{border:1px solid var(--line);border-radius:10px;padding:8px 10px}.tile .k{font-size:12px;color:var(--ink2)}
.tile .v{font-size:22px;font-weight:600}.tile .u{font-size:12px;color:var(--ink2);margin-left:2px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--ink2);font-size:12px}td:first-child,th:first-child{text-align:left}
tbody tr:hover{background:#f0efec}.tag{font-size:11px;color:var(--ink2);border:1px solid var(--line);border-radius:4px;padding:0 3px;margin-left:4px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
.grid-small{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.grid-small figure{margin:0}.grid-small figcaption{font-size:12px;color:var(--ink2)}
svg.court{width:100%;height:auto;display:block}svg .bg{fill:var(--card)}svg .pitch{fill:var(--pitch)}
svg .cl{fill:none;stroke:#9a9a96;stroke-width:.12}svg .cm{fill:#9a9a96}svg .cg{fill:none;stroke:#9a9a96;stroke-width:.12}
svg .tl{font-size:1.25px;fill:var(--ink2)}svg .zt{font-size:1.8px;fill:var(--ink);text-anchor:middle;font-weight:600}
svg .zl{stroke:#b9b8b2;stroke-width:.08;stroke-dasharray:.4 .3}svg .arr line,svg .arr polyline{stroke:var(--ink2);stroke-width:.15;fill:none}
svg .pn{font-size:1.1px;fill:#fff;text-anchor:middle;font-weight:700}
svg.chart{width:100%;height:auto}svg .gr{stroke:var(--line);stroke-width:1}svg .ax{font-size:10px;fill:var(--ink2)}
svg .dl{font-size:11px;fill:var(--ink)}svg .ht{stroke:var(--muted);stroke-dasharray:3 3}
.legend{font-size:13px;color:var(--ink2);margin:4px 0}.legend span{margin-right:12px}
.br{display:flex;align-items:center;gap:6px;margin:4px 0}.bl{width:84px;font-size:12px;color:var(--ink2);flex:none}
.bar{display:flex;flex:1;gap:2px;height:22px}.seg{color:#fff;font-size:11px;display:flex;align-items:center;justify-content:center;border-radius:3px;overflow:hidden;min-width:0;text-shadow:0 0 2px rgba(0,0,0,.45)}
.note{font-size:12.5px;color:var(--ink2);background:#f6f5f1;border-left:3px solid #c3c2b7;padding:6px 10px;margin:8px 0;border-radius:0 6px 6px 0}
.warn{border-left-color:#fab219}
details summary{cursor:pointer;font-weight:600}ul{padding-left:20px;margin:6px 0}li{margin:2px 0}
"""


def render_html(res: dict, show_hi: bool = False) -> str:
    """show_hi: 고강도 거리(기준 미정)를 HTML에 보일지. report.json 에는 항상 들어 있다."""
    M = res["match"]
    L, W = M["court"]["length"], M["court"]["width"]
    T, players, Q = res["teams"], res["players"], res["quality"]
    mains = [p for p in players if not p["fragment"]]
    frags = [p for p in players if p["fragment"]]
    order = res["team_order"]
    bmin, nb = M["block_min"], M["n_blocks"]
    blabels = [f"{k * bmin}–{min((k + 1) * bmin, math.ceil(M['duration_s'] / 60))}" for k in range(nb)]
    sw = res["sides"]["switch_frame"]
    sw_block = sw / (bmin * 60 * M["fps"]) if sw is not None else None
    E = html.escape
    parts = []

    def dot(t):
        return f'<span class="dot" style="background:{team_color(t)}"></span>'

    # ---- 머리말
    dur = M["duration_s"]
    parts.append(f'<h1>{E(res["title"])}</h1><div class="sub">경기 시간 {_mmss(dur)} · 코트 {L:g}×{W:g} m · '
                 f'{M["fps"]:g} Hz 추적 · 선수 ID {Q["ids"]}개 (주요 {Q["main_ids"]} + 짧게 잡힌 {Q["fragment_ids"]})</div>')

    # ---- 개요
    tiles = [f'<div class="tile"><div class="k">경기 시간</div><div class="v">{_mmss(dur)}</div></div>']
    for t in order:
        if t not in res["main_teams"]:
            continue
        tm = T[t]
        avg = (f'선수 1명당 {_f(tm["distance_per_roster_player_m"])} m (÷ 실제 {tm["roster"]}명)' if tm.get("roster")
               else f'주요 ID 평균 {_f(tm["distance_per_main_player_m"])} m')
        tiles.append(f'<div class="tile"><div class="k">{dot(t)}{team_label(t)} 총 이동 거리</div>'
                     f'<div class="v">{_f(tm["distance_total_m"] / 1000, 2)}<span class="u">km</span></div>'
                     f'<div class="k">{avg}</div></div>')
    sides = res["sides"]
    if sw is not None:
        side_txt = (f"후반 시작 무렵 ({_mmss(sw / M['fps'])}) 두 팀이 진영을 바꾼 것으로 보입니다 "
                    f"(분 단위 일치율 {sides['agree'] * 100:.0f}%).")
    elif sides["a_left_first"] is not None:
        side_txt = "경기 내내 진영이 바뀌지 않은 것으로 보입니다."
    else:
        side_txt = "공격 방향을 추정하지 못했습니다. 구역은 코트 절대 위치로만 봅니다."
    if sides["a_left_first"] is not None and res["main_teams"]:
        a = res["main_teams"][0]
        side_txt += f" 처음에는 {team_label(a)}가 {'왼쪽' if sides['a_left_first'] else '오른쪽'} 골대를 지켰습니다 (추정)."
    parts.append(f'<section><h2>경기 개요</h2><div class="tiles">{"".join(tiles)}</div>'
                 f'<p class="note">{E(side_txt)} 방향은 팀 평균 위치(자기 골대 쪽에 더 가까움)로 추정한 값입니다.</p>')
    has_roster = any(T[t].get("roster") for t in order)
    rows = []
    for t in order:
        tm = T[t]
        sh = tm["shape"]
        rows.append(f'<tr><td>{dot(t)}{team_label(t)}</td><td>{tm["main_ids"]}</td>'
                    + (f'<td>{tm["roster"] or "–"}</td><td>{_f(tm["distance_per_roster_player_m"])}</td>' if has_roster else "")
                    + f'<td>{_f(tm["distance_total_m"])}</td><td>{_f(tm["distance_per_main_player_m"])}</td>'
                    f'<td>{_f(tm["distance_per_min_team"])}</td>'
                    + (f'<td>{_f(tm["hi_distance_m"])}</td>' if show_hi else "")
                    + f'<td>{tm["sprints"]}</td>'
                    f'<td>{_f(sh.get("width_m_median"), 1)}</td><td>{_f(sh.get("depth_m_median"), 1)}</td>'
                    f'<td>{_f(sh.get("teammate_dist_m_mean"), 1)}</td></tr>')
    roster_note = (f'“1명당”은 팀 총 거리를 실제 선수 수(앱에서 입력, 출처: {E(str(res.get("roster_source")))})로 나눈 값입니다. '
                   if has_roster else "")
    parts.append('<h3>팀 비교</h3><div class="scroll"><table><thead><tr><th>팀</th><th>주요 ID</th>'
                 + ('<th>실제 인원</th><th>1명당 m</th>' if has_roster else "")
                 + '<th>총 거리 m</th><th>주요 ID 평균 m</th><th>팀 m/분</th>'
                 + ('<th>고강도 m</th>' if show_hi else "")
                 + '<th>스프린트</th><th>폭 m</th><th>길이 m</th>'
                 f'<th>동료 간 거리 m</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                 '<p class="note">짧게 잡힌 ID(조각)는 팀 집계(거리·히트맵·구역·형태·시간대별)에서 모두 뺐습니다 '
                 '(데이터 품질 섹션에 따로 표시). ' + roster_note +
                 '“주요 ID 평균”은 주요 ID 하나당 평균입니다. 폭·길이·동료 간 거리는 골키퍼(추정)를 뺀 필드 선수가 '
                 '3명 이상 보이는 순간의 중앙값/평균입니다.</p></section>')

    # ---- 선수별 기록
    hl = [p for p in mains if not p["gk_guess"] and p["team"] in res["main_teams"]]
    hi_tiles = []
    keys = [("distance_m", "가장 많이 뛴 선수", "m", 0)] + ([("hi_distance_m", "고강도 거리 1위", "m", 0)] if show_hi else [])         + [("distance_per_min", "분당 거리 1위", "m/분", 0), ("sprints", "스프린트 1위", "회", 0)]
    for key, name, unit, nd in keys:
        if hl:
            b = max(hl, key=lambda p: p[key] or 0)
            hi_tiles.append(f'<div class="tile"><div class="k">{name}</div><div class="v" style="font-size:17px">'
                            f'{dot(b["team"])}{E(b["label"])}</div><div class="k">{_f(b[key], nd)} {unit}</div></div>')
    vmax = max([max(p["distance_blocks_m"]) for p in mains if p["distance_blocks_m"]] + [1])
    rows = []
    for t in order:
        for p in sorted([p for p in mains if p["team"] == t], key=lambda p: -p["distance_m"]):
            top = p["top_speed_est_ms"]
            rows.append(f'<tr><td>{dot(t)}{E(p["label"])}{"<span class=tag>GK?</span>" if p["gk_guess"] else ""}'
                        f'<span class="muted"> #{p["id"]}</span></td><td>{_f(p["tracked_s"] / 60, 1)}</td>'
                        f'<td>{_f(p["distance_m"])}</td><td>{_f(p["distance_per_min"])}</td>'
                        f'<td>{_f(p["mean_speed_ms"], 2)}</td>'
                        f'<td>{_f(top, 1)}{f" <span class=muted>({top * 3.6:.0f} km/h)</span>" if top else ""}</td>'
                        + (f'<td>{_f(p["hi_distance_m"])}</td>' if show_hi else "") + f'<td>{p["sprints"]}</td>'
                        f'<td>{sparkline(p["distance_blocks_m"], team_color(t), vmax)}</td></tr>')
    parts.append(f'<section><h2>선수별 기록</h2><div class="tiles">{"".join(hi_tiles)}</div>'
                 '<p class="sub">하이라이트는 골키퍼(추정)와 짧게 잡힌 ID를 뺀 필드 선수 중에서 고릅니다. 표는 옆으로 밀어서 보세요.</p>'
                 '<div class="scroll"><table><thead><tr><th>선수</th><th>추적 분</th><th>거리 m</th><th>m/분</th>'
                 '<th>평균 m/s</th><th>최고 속도 추정 m/s</th>' + ('<th>고강도 m</th>' if show_hi else "") + '<th>스프린트</th>'
                 f'<th>{bmin}분 단위 거리</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                 f'<p class="note">최고 속도는 <b>추정값</b>입니다: 정리된 궤적에서 {res["settings"]["top_speed_hold_s"]:g}초 동안 '
                 f'계속 유지한 가장 빠른 속도 (추적 {res["settings"]["top_speed_min_s"]:.0f}초 미만이면 표시 안 함). 한 프레임의 최댓값은 위치 흔들림과 '
                 'ID가 다른 사람에게 잘못 이어진 순간(순간이동) 때문에 비현실적으로 튀어서 믿지 않습니다. '
                 + (f'고강도 = {res["settings"]["hi_speed_ms"]:g} m/s 이상으로 달린 거리, ' if show_hi else "")
                 + f'스프린트 = '
                 f'{res["settings"]["sprint_speed_ms"]:g} m/s 이상을 {res["settings"]["sprint_min_s"]:g}초 이상 유지한 횟수. '
                 '“추적 분”은 그 ID가 코트에서 추적된 시간이라 실제 출전 시간보다 짧을 수 있습니다.</p></section>')

    # ---- 활동량 추이
    ser = [(team_label(t), team_color(t), T[t]["distance_blocks_m"]) for t in res["main_teams"]]
    parts.append(f'<section><h2>시간대별 활동량</h2><p class="sub">{bmin}분마다 팀 전체 이동 거리 (m)</p>'
                 f'{line_chart(ser, blabels, "m", sw_block)}'
                 f'<div class="legend">{"".join(f"<span>{dot(t)}{team_label(t)}</span>" for t in res["main_teams"])}</div>')
    if res["periods"] and len(res["periods"]) == 2:
        pr = "".join(f'<tr><td>{dot(t)}{team_label(t)}</td>' + "".join(f'<td>{_f(q["distance_m"])}</td>' for q in T[t]["periods"])
                     + '</tr>' for t in res["main_teams"])
        parts.append(f'<h3>전·후반 이동 거리 (m)</h3><table><thead><tr><th>팀</th><th>전반</th><th>후반</th></tr></thead>'
                     f'<tbody>{pr}</tbody></table>')
    parts.append('</section>')

    # ---- 팀 히트맵 + 구역
    hm = []
    for t in order:
        tm = T[t]
        if tm["main_ids"] == 0 and t not in res["main_teams"]:
            continue
        hm.append(f'<figure style="margin:0"><figcaption>{dot(t)}{team_label(t)}</figcaption>'
                  f'{court_svg(L, W, heat_cells(tm["_heat"], L, W, team_color(t)), tm["heatmap_rel_frame"])}</figure>')
    any_rel = any(T[t]["heatmap_rel_frame"] for t in order)
    parts.append(f'<section><h2>히트맵 (팀)</h2><div class="grid2">{"".join(hm)}</div><p class="note">'
                 + ("화살표가 있는 그림은 그 팀이 항상 오른쪽으로 공격하도록 돌려서 그렸습니다 (후반 진영 교체 반영, 추정). "
                    if any_rel else "코트의 실제 위치 그대로 그렸습니다. ")
                 + '진할수록 오래 머문 곳입니다. 1 m 칸, 살짝 흐리게 처리.</p></section>')

    zparts = []
    for t in res["main_teams"]:
        tm = T[t]
        zparts.append(f'<h3>{dot(t)}{team_label(t)}</h3>')
        if tm["zones_rel"]:
            zparts.append(bar_row("세로 3구역", tm["zones_rel"]["thirds"], THIRDS_REL, team_color(t)))
            zparts.append(bar_row("가로 3구역", tm["zones_rel"]["channels"], CHANNELS_REL, team_color(t)))
            zparts.append(f'<p class="sub">위: 공격 방향 기준 (수비 → 중간 → 공격 / 오른쪽 측면 → 중앙 → 왼쪽 측면)</p>')
        zparts.append(court_svg(L, W, zone_grid(tm["zones_abs"], L, W), label="코트 절대 위치 기준 (3×3 구역 비율)"))
        za = tm["zones_abs"]
        zparts.append('<div class="scroll"><table><tbody>' +
                      "".join(f'<tr><td>{n_}</td><td>{v * 100:.0f}%</td></tr>' for n_, v in zip(THIRDS_ABS, za["thirds"])) +
                      "".join(f'<tr><td>{n_}</td><td>{v * 100:.0f}%</td></tr>' for n_, v in zip(CHANNELS_ABS, za["channels"])) +
                      '</tbody></table></div>')
    parts.append(f'<section><h2>구역 점유</h2>{"".join(zparts)}<p class="note">팀 선수들이 각 구역에 머문 시간 비율 '
                 '(골키퍼·조각 포함). “왼쪽/오른쪽 골대 쪽”은 그림에서 본 코트의 실제 위치이고, 전·후반에 진영이 바뀌면 '
                 '같은 구역이라도 수비/공격 의미가 바뀝니다. 공격 방향 기준 막대는 진영 추정이 될 때만 보입니다.</p></section>')

    # ---- 평균 위치 / 팀 형태
    figs = []
    for k, per in enumerate(res["periods"]):
        dots = []
        for p in mains:
            mp = p["mean_pos"][k]
            if mp["x"] is None:
                continue
            c = team_color(p["team"])
            dots.append(f'<g><circle cx="{mp["x"]:.2f}" cy="{W - mp["y"]:.2f}" r="0.9" fill="{c}" stroke="#fcfcfb" '
                        f'stroke-width="0.2"/><text x="{mp["x"]:.2f}" y="{W - mp["y"] + 0.4:.2f}" class="pn">{p["id"]}</text>'
                        f'<title>{E(p["label"])} (#{p["id"]}) 평균 위치 x {mp["x"]:.1f} m, y {mp["y"]:.1f} m</title></g>')
        figs.append(f'<figure style="margin:0"><figcaption>{per["name"]} ({_mmss(per["from_s"])}–{_mmss(per["to_s"])})'
                    f'</figcaption>{court_svg(L, W, "".join(dots))}</figure>')
    rows = []
    for t in res["main_teams"]:
        sh = T[t]["shape"]
        if "width_m_median" not in sh:
            continue
        rows.append(f'<tr><td>{dot(t)}{team_label(t)}</td><td>{_f(sh["width_m_median"], 1)} '
                    f'<span class="muted">({_f(sh["width_m_p10_p90"][0], 1)}–{_f(sh["width_m_p10_p90"][1], 1)})</span></td>'
                    f'<td>{_f(sh["depth_m_median"], 1)} <span class="muted">({_f(sh["depth_m_p10_p90"][0], 1)}–'
                    f'{_f(sh["depth_m_p10_p90"][1], 1)})</span></td><td>{_f(sh["teammate_dist_m_mean"], 1)}</td></tr>')
    ser = [(team_label(t), team_color(t), T[t]["shape"].get("teammate_dist_blocks_m", [])) for t in res["main_teams"]]
    parts.append(f'<section><h2>평균 위치와 팀 형태</h2><div class="grid2">{"".join(figs)}</div>'
                 f'<p class="sub">원 안 숫자 = ID. 주요 ID만, 구간마다 30초 이상 추적된 경우.</p>'
                 '<div class="scroll"><table><thead><tr><th>팀</th><th>폭 m (10–90%)</th><th>길이 m (10–90%)</th>'
                 f'<th>동료 간 평균 거리 m</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                 f'<h3>동료 간 평균 거리 ({bmin}분 단위, m)</h3>{line_chart(ser, blabels, "m", sw_block, ymin0=False)}'
                 '<p class="note">폭 = 필드 선수들의 y 범위(터치라인 방향), 길이 = x 범위(골대 방향). 순간마다 재고 중앙값을 냅니다. '
                 '평균 위치는 시간 평균이라 실제 대형과 다를 수 있습니다 (자주 움직이는 선수는 가운데로 모임).</p></section>')

    # ---- 선수별 히트맵
    sm = []
    for t in order:
        for p in sorted([p for p in mains if p["team"] == t], key=lambda p: p["id"]):
            sm.append(f'<figure><figcaption>{dot(t)}{E(p["label"])} #{p["id"]} · {_f(p["tracked_s"] / 60, 0)}분</figcaption>'
                      f'{court_svg(L, W, heat_cells(p["_heat"], L, W, team_color(t)))}</figure>')
    parts.append(f'<section><h2>히트맵 (선수)</h2><div class="grid-small">{"".join(sm)}</div>'
                 '<p class="note">2 m 칸. 진영 추정이 되면 모든 그림은 그 선수가 <b>오른쪽으로 공격</b>하도록 돌려서 '
                 '그렸습니다 (후반 진영 교체 반영). 추정이 안 되면 코트 실제 위치 그대로입니다.</p></section>')

    # ---- 공
    B = res["ball"]
    if B:
        ser = [("공이 왼쪽 반에", BALL_COLOR, B["left_half_blocks"])]
        what = "공이 왼쪽 반에 있던 비율"
        extra = ""
        if "in_team_half_blocks" in B:
            a = B["in_half_of"]
            ser = [(f"{team_label(a)} 진영에 공", team_color(a), B["in_team_half_blocks"])]
            what = f"공이 {team_label(a)} 진영(반)에 있던 비율"
            extra = (f'<div class="tile"><div class="k">{team_label(a)} 진영에 공이 있던 비율</div>'
                     f'<div class="v">{B["in_team_half_share"] * 100:.0f}<span class="u">%</span></div></div>')
        parts.append(f'<section><h2>공 위치 <span class="tag">검증 전 참고값</span></h2><div class="tiles">'
                     f'<div class="tile"><div class="k">공 위치가 있는 프레임</div><div class="v">{B["coverage"] * 100:.0f}'
                     f'<span class="u">%</span></div></div><div class="tile"><div class="k">왼쪽 반에 있던 비율</div>'
                     f'<div class="v">{B["left_half_share"] * 100:.0f}<span class="u">%</span></div></div>{extra}</div>'
                     f'<h3>{bmin}분 단위 · {what}</h3>{line_chart(ser, blabels, "", sw_block, pct=True)}'
                     f'{court_svg(L, W, zone_grid(B["zones_abs"], L, W), label="공 위치 3×3 구역 비율 (코트 절대 위치)")}'
                     f'<p class="note warn">공 위치는 아직 검증하지 않았습니다. 공이 떠 있으면 바닥 좌표로 잘못 투영되고 '
                     f'(보이는 위치의 {B["off_court_share_of_seen"] * 100:.1f}%가 코트 밖), 안 보이는 시간이 많습니다. '
                     '점유율이 아니라 “공이 어느 쪽 반에 있었나”만 봅니다.</p></section>')

    # ---- 데이터 품질
    S = res["settings"]
    frows = "".join(f'<tr><td>{dot(p["team"])}{E(p["label"])} #{p["id"]}</td><td>{team_label(p["team"])}</td>'
                    f'<td>{_f(p["seen_s"], 0)}</td><td>{_f(p["distance_m"])}</td>'
                    f'<td>{_mmss(p["first_seen_s"])}</td></tr>' for p in sorted(frags, key=lambda p: -p["seen_s"]))
    pipe = res.get("pipeline_summary") or {}
    pipe_txt = ""
    if pipe:
        bits = []
        if "ids" in pipe:
            bits.append(f"파이프라인 ID {pipe['ids']}개")
        if "ids_over_half" in pipe:
            bits.append(f"경기의 절반 넘게 보인 ID {pipe['ids_over_half']}개")
        if "events" in pipe:
            bits.append(f"ID 생김/끊김 {pipe['events']}회")
        al = pipe.get("alignment") or {}
        if al.get("median_after_m") is not None:
            bits.append(f"두 카메라 위치 차이 중앙값 {al['median_after_m']} m (정렬 후)")
        pipe_txt = f'<li>summary.json: {E(", ".join(bits))}</li>' if bits else ""
    fb = Q.get("fragment_by_team") or {}
    frag_txt = ", ".join(f'{team_label(t)} {v["ids"]}개 {v["distance_m"]:,.0f} m' for t, v in sorted(fb.items())) or "없음"
    roster_txt = ""
    if Q.get("roster_checks"):
        items = "".join(f'<li>{"⚠ " if c["status"] != "ok" else ""}{E(c["message"])}</li>' for c in Q["roster_checks"])
        roster_txt = (f'<li>실제 인원과 비교 (출처: {E(str(res.get("roster_source")))}; ID를 합치거나 고르지는 않음):'
                      f'<ul>{items}</ul></li>')
    parts.append(
        f'<section><h2>데이터 품질</h2><ul>'
        f'<li>선수 ID {Q["ids"]}개 중 주요 {Q["main_ids"]}개, 짧게 잡힌 ID {Q["fragment_ids"]}개 '
        f'(추적 {Q["fragment_threshold_s"]:.0f}초 미만). 짧게 잡힌 ID는 모든 집계에서 뺐습니다 — 넣었다면 전체 거리가 '
        f'{Q["fragment_distance_share"] * 100:.1f}% ({frag_txt}) 더해졌을 것입니다.</li>{roster_txt}'
        f'<li>주요 ID가 보인 시간 비율 (중앙값): {_f((Q["seen_share_main_median"] or 0) * 100, 0)}%</li>'
        f'<li>순간이동(ID 잘못 이음 추정)으로 끊은 곳: {Q["jumps_total"]}곳 — 거리·속도에서 뺐습니다.</li>'
        f'<li>코트 좌표 오차: 약 0.15 m (가까운 쪽) ~ 0.3 m (먼 쪽). 거리는 평활로 흔들림을 줄였지만 몇 % 정도 오차가 남습니다.</li>'
        f'<li>팀 구분이 없는 ID: {Q["unknown_team_ids"]}개</li>{pipe_txt}</ul>'
        f'<details><summary>짧게 잡힌 ID {len(frags)}개</summary><p class="sub">선수 표·순위·히트맵·팀 총 거리·구역·팀 형태·'
        f'시간대별 활동량 어디에도 넣지 않았습니다. 대부분 같은 선수가 잠깐 다른 ID로 잡힌 조각입니다.</p>'
        f'<div class="scroll"><table><thead><tr><th>ID</th><th>팀</th><th>추적 초</th><th>거리 m</th><th>처음 보인 시각</th>'
        f'</tr></thead><tbody>{frows}</tbody></table></div></details>'
        f'<details><summary>계산 방법</summary><ul>'
        f'<li>궤적 정리: {S["max_gap_s"]:g}초보다 긴 미검출과 순간이동(연속 관측 사이 {S["jump_speed_ms"]:g} m/s + '
        f'max({S["jump_slack_m"]:g} m, 흔들림의 {S["jump_noise_k"]:g}배) 초과)에서 끊고, 짧은 빈칸은 직선으로 메운 뒤 '
        f'{S["smooth_s"]:g}초 이동평균을 두 번 적용 (흔들림이 큰 궤적은 창을 최대 {S["smooth_max_s"]:g}초까지 넓힘).</li>'
        f'<li>속도: 정리된 위치의 중앙 차분. {S["max_speed_ms"]:g} m/s 초과는 오류로 제외.</li>'
        f'<li>진영 추정: 1분마다 두 팀 평균 x 위치를 비교 → 순서가 한 번 뒤집히는 지점을 진영 교체로 봄.</li>'
        f'<li>골키퍼 추정: 추적 시간의 {S["gk_share"] * 100:.0f}% 이상을 골라인 {S["gk_goal_dist_m"]:g} m 안에서 보낸 ID.</li>'
        f'</ul></details></section>')
    parts.append('<p class="sub" style="text-align:center">좌표 데이터만으로 만든 리포트 · 영상·사진은 포함하지 않습니다</p>')
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{E(res["title"])}</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(parts)}</main></body></html>')


# ================================================================ CLI
def make_report(tracks_path: str, out_dir: str, summary_path: str | None = None, title: str | None = None,
                players=None, show_hi: bool = False, cfg: dict = SETTINGS) -> dict:
    """players: CLI 인원 ("A=5 B=5" 또는 목록). 없으면 tracks.json → summary.json 의 "roster" 를 찾는다."""
    with open(tracks_path, encoding="utf-8") as f:
        d = json.load(f)
    summary = None
    if summary_path:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    roster, src = resolve_roster(players, d, summary)
    res = analyze(d, summary, title, cfg, roster, src)
    res["display"] = {"high_intensity": bool(show_hi)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(render_html(res, show_hi))
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(to_json(res), f, ensure_ascii=False, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="풋살 경기 분석 리포트 (tracks.json → report.html, report.json)")
    ap.add_argument("tracks")
    ap.add_argument("--summary", help="파이프라인 summary.json (선택)")
    ap.add_argument("--out", help="출력 폴더 (기본: tracks.json 옆 report/)")
    ap.add_argument("--title")
    ap.add_argument("--players", nargs="+", metavar="팀=인원",
                    help='팀별 실제 선수 수 (예: --players A=5 B=5). 없으면 tracks.json / summary.json 의 "roster"')
    ap.add_argument("--high-intensity", action="store_true",
                    help="고강도 거리(기준 미정)를 HTML에도 표시 (report.json 에는 항상 있음)")
    ap.add_argument("--min-seen", type=float, help="이 초보다 짧게 보인 ID는 '짧게 잡힌 ID'로 분류 (기본: 경기의 10%%, 30~120초)")
    a = ap.parse_args(argv)
    cfg = dict(SETTINGS)
    if a.min_seen is not None:
        cfg["fragment_min_s"] = a.min_seen
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.tracks)), "report")
    try:
        res = make_report(a.tracks, out, a.summary, a.title, a.players, a.high_intensity, cfg)
    except ValueError as e:
        ap.error(str(e))
    Q = res["quality"]
    print(f"wrote {out}/report.html, report.json  (ID {Q['ids']}: 주요 {Q['main_ids']}, 조각 {Q['fragment_ids']})")
    for t in res["main_teams"]:
        tm = res["teams"][t]
        print(f"  {team_label(t)}: 총 {tm['distance_total_m']:.0f} m, 주요 ID 평균 {tm['distance_per_main_player_m'] or 0:.0f} m"
              + (f", 1명당 {tm['distance_per_roster_player_m']:.0f} m (÷{tm['roster']})" if tm.get("roster") else ""))
    for c in Q.get("roster_checks", []):
        print(f"  {'!' if c['status'] != 'ok' else ' '} {c['message']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
