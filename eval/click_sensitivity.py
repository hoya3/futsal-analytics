"""클릭 보정(7클릭)의 민감도: 클릭 노이즈가 코트 오차로 얼마나 커지는지, 잘못 찍은 점 하나를 잔차로 잡을 수 있는지.

실제 앵커 캘리브레이션(out/<회차>/H_cam<N>_refined.json)을 '참 카메라'로 두고, 그 카메라로 랜드마크를 투영한 픽셀을 참 클릭으로 쓴다.
1) 노이즈: 모든 클릭에 sigma px 가우스 노이즈 -> manual_result로 다시 풀고, 구역별 격자 점의 코트 오차 RMS (흰 선 ICP 전, 최악 쪽).
2) 오클릭: 클릭 하나를 bad px 옮겼을 때 가까운 하프 풀이의 점별 잔차(m) 최대가 기준을 넘는지 (검출 가능 여부).
사용법: python eval/click_sensitivity.py [--matches 1008 1103] [--sigma 1] [--bad 5] [--trials 300]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calib_core import click_landmarks, click_residuals_px, front_facing, manual_result, to_world  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import load_anchor, pixel_to_court  # noqa: E402
from harness import ZONES, zone_of  # noqa: E402
from refine import project  # noqa: E402

SIZE = (1920, 1080)


def true_clicks(cal):
    """참 카메라로 투영한 7클릭 픽셀. 사이드라인 클릭은 저장된 풀이의 선 위 위치를 쓴다."""
    lms = click_landmarks(spec_of(cal), cal.get("mirror", False))
    G_near = front_facing(np.linalg.inv(np.array(cal["H_near"])))
    G_far = front_facing(np.linalg.inv(np.array(cal["H_far"])))
    clicks = []
    for i, lm in enumerate(lms):
        if str(i) in cal["world"]["near"]:
            w, G = cal["world"]["near"][str(i)], G_near
        elif str(i) in cal["world"]["far"]:
            w, G = cal["world"]["far"][str(i)], G_far
        else:  # 이 파일에서 찍지 않은 확인 클릭: 규격 위치 (동쪽 기준 -> 실제 방향)
            w, G = to_world(lm["xy"], cal["half"]), G_near
        clicks.append(project(G, np.array([w], float))[0][0])
    return lms, np.array(clicks)


def grid(cal, step=1.0):
    """화면 안에 보이는 코트 격자 점 (월드, 픽셀, 구역)."""
    spec = spec_of(cal)
    xs = np.arange(-spec.length / 2, spec.length / 2 + 1e-9, step)
    ys = np.arange(-spec.width / 2, spec.width / 2 + 1e-9, step)
    w = np.array([(x, y) for x in xs for y in ys])
    sign = 1 if cal["half"] == "east" else -1
    near = sign * w[:, 0] <= 0
    px = np.empty_like(w)
    depth = np.empty(len(w))
    for mask, key in ((near, "H_near"), (~near, "H_far")):
        px[mask], depth[mask] = project(front_facing(np.linalg.inv(np.array(cal[key]))), w[mask])
    ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < SIZE[0]) & (px[:, 1] >= 0) & (px[:, 1] < SIZE[1])
    return w[ok], px[ok], np.array([zone_of(p, cal["half"]) for p in w[ok]])


def noise_rms(cal, sigma=1.0, trials=300, seed=0):
    """구역별 코트 오차 RMS (m)."""
    lms, clicks = true_clicks(cal)
    w, px, zones = grid(cal)
    rng = np.random.default_rng(seed)
    sq = {z: [] for z in ZONES}
    for _ in range(trials):
        res = manual_result((clicks + rng.normal(0, sigma, clicks.shape)).tolist(), lms, cal["half"], spec_of(cal))
        est = pixel_to_court({"half": cal["half"], "H_near": res["near"][0], "H_far": res["far"][0]}, px)
        d2 = np.sum((est - w) ** 2, axis=1)
        for z in ZONES:
            if (zones == z).any():
                sq[z].append(d2[zones == z])
    return {z: float(np.sqrt(np.mean(np.concatenate(v)))) if v else None for z, v in sq.items()}


def misclick_residuals(cal, bad=5.0, n_clicks=None):
    """클릭마다 bad px 옮겼을 때 재투영 잔차 최대(px, click_residuals_px). n_clicks=7이면 확인 클릭 없이."""
    lms, clicks = true_clicks(cal)
    clicks = clicks[:n_clicks] if n_clicks else clicks
    clean = click_residuals_px(manual_result(clicks.tolist(), lms, cal["half"], spec_of(cal)), clicks.tolist(), lms)
    out = {}
    for i, lm in enumerate(lms[:len(clicks)]):
        worst = 0.0
        for d in ((bad, 0), (0, bad), (-bad, 0), (0, -bad)):
            c = clicks.copy()
            c[i] += d
            res = manual_result(c.tolist(), lms, cal["half"], spec_of(cal))
            worst = max(worst, max(click_residuals_px(res, c.tolist(), lms).values()))
        out[lm["name"]] = worst
    return out, clean


def main(argv=None):
    p = argparse.ArgumentParser(description="클릭 보정 민감도")
    p.add_argument("--matches", nargs="+", default=["1008", "1103"])
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--bad", type=float, default=5.0)
    p.add_argument("--trials", type=int, default=300)
    a = p.parse_args(argv)
    from calib_core import match_dir
    for m in a.matches:
        for n in (1, 2):
            cal = load_anchor(n, match_dir(m))
            rms = noise_rms(cal, a.sigma, a.trials)
            txt = " ".join(f"{z} {v:.2f}" for z, v in rms.items() if v is not None)
            print(f"{m} cam{n}: 클릭 노이즈 sigma={a.sigma:g} px -> 코트 오차 RMS (m): {txt}")
            for n_clicks, tag in ((7, "7클릭"), (None, "9클릭")):
                res, clean = misclick_residuals(cal, a.bad, n_clicks)
                print(f"  {tag}: 클릭 하나를 {a.bad:g} px 옮겼을 때 재투영 잔차 최대 px (정상 {max(clean.values()):.2f}): "
                      + ", ".join(f"{i + 1}번 {v:.1f}" for i, v in enumerate(res.values())))


if __name__ == "__main__":
    main()
