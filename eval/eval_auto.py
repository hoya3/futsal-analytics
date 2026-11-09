"""자동 초기화를 시간별 캘리브레이션(track.py 결과)과 비교한다.

주의: 정답도 흰 선 정렬로 만들어졌으므로 개발용 잠정 지표다. 최종 판정은 손으로 찍은 검증 점과 줄자 기준점으로 한다.
사용법: python eval/eval_auto.py [--cam 1]
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_init import auto_calibrate, consensus, median_of  # noqa: E402
from calib_core import HALF_SIGN, Source  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import load  # noqa: E402

TIMES = tuple(range(200, 3200, 200))  # 15개 시점


def truth_segment(cal, t):
    for s in cal["segments"]:
        if s["t0"] <= t <= s["t1"]:
            return None if s["source"] == "unresolved" else s
    return None


def error_m(H_auto, H_true, half, spec, size=(1920, 1080)):
    """화면 안 near-half 격자의 코트 좌표 차이 (중앙값, 95%), 미터."""
    s = HALF_SIGN[half]
    xs, ys = np.meshgrid(np.linspace(-spec.length * 0.4, 0, 9), np.linspace(-spec.width / 2, spec.width / 2, 9))
    world = np.c_[xs.ravel(), ys.ravel()] * s
    G = np.linalg.inv(np.array(H_true))
    q = np.c_[world, np.ones(len(world))] @ G.T
    q = q[q[:, 2] * np.sign(G[2, 2]) > 0]
    px = q[:, :2] / q[:, 2:3]
    w_ok = (px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])
    court_true = cv2.perspectiveTransform(px[w_ok].reshape(-1, 1, 2), np.array(H_true))[:, 0]
    court_auto = cv2.perspectiveTransform(px[w_ok].reshape(-1, 1, 2), np.array(H_auto))[:, 0]
    e = np.linalg.norm(court_auto - court_true, axis=1)
    return float(np.median(e)), float(np.percentile(e, 95))


def main(args):
    cams = [int(args[args.index("--cam") + 1])] if "--cam" in args else [1, 2]
    for n in cams:
        cal = load(n)
        spec = spec_of(cal)
        src = Source(cal["video"])
        results, valid_times = [], []
        for t in TIMES:
            seg = truth_segment(cal, t) and truth_segment(cal, t + 12)
            if seg is None:
                continue
            frame = median_of(src, t)
            t0 = time.time()
            r = auto_calibrate(frame, cal["half"], spec)
            r["t_sec"], r["truth"] = float(t), seg["H_near"]
            if r["H_near"] is not None:
                r["err"] = error_m(r["H_near"], seg["H_near"], cal["half"], spec)
            results.append(r)
            valid_times.append(t)
            err = f"err med={r['err'][0]:6.2f} p95={r['err'][1]:6.2f}" if "err" in r else "err -"
            print(f"cam{n} t={t:5d}: {r['status']:14s} conf={r['confidence']:.2f} {err} m cam={r.get('camera')} ({time.time() - t0:.1f}s)", flush=True)
        good = sum(1 for r in results if r.get("err", (99, 0))[0] < 0.5)
        print(f"cam{n}: 프레임별 0.5 m 미만 {good}/{len(results)}", flush=True)
        for k in range(3):  # 겹치지 않는 시점 5개 묶음의 다수결
            group = results[k::3][:5]
            lead, members = consensus(group, spec)
            if lead is None:
                print(f"cam{n} 묶음{k}: 후보 없음", flush=True)
                continue
            status = "ok" if len(members) * 2 > len(group) else "low_confidence"
            print(f"cam{n} 묶음{k}: {status} votes={len(members)}/{len(group)} 앵커 t={group[lead]['t_sec']:.0f} "
                  f"err med={group[lead]['err'][0]:.2f} p95={group[lead]['err'][1]:.2f} m", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
