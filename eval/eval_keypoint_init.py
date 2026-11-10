"""키포인트 모델 기반 자동 초기화(keypoint_init.py)를 시간별 캘리브레이션, 사람이 찍은 검증 점과 비교한다.

1) 경기 중 여러 시점: 같은 화면 격자를 두 결과로 코트에 옮긴 차이(m), 가까운 하프와 먼 하프 따로.
   정답도 흰 선 정렬로 만든 것이라 '기존 결과를 클릭 없이 재현하는가'를 보는 지표다.
2) 검증 점 시각: 자동 결과로 사람 클릭을 채점 (score_points와 같은 방식).
사용법: python eval/eval_keypoint_init.py [--cam 1]
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_init import median_of  # noqa: E402
from calib_core import HALF_SIGN, Source, front_facing  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import homographies_at, load  # noqa: E402
from keypoint_init import keypoint_calibrate, load_model  # noqa: E402
from label_points import LABELS_PATH, read_labels  # noqa: E402
from score_points import score, summarize  # noqa: E402

TIMES = tuple(range(200, 3200, 200))  # 15개 시점


def resolved(cal, t):
    for s in cal["segments"]:
        if s["t0"] <= t <= s["t1"]:
            return None if s["source"] == "unresolved" else s
    return None


def region_error(H_auto, H_true, half, spec, region, size=(1920, 1080)):
    """정답 H로 화면에 보이는 코트 격자(한 하프)를 두 H로 코트에 옮긴 차이 (중앙값, 95%), 미터."""
    s = HALF_SIGN[half]
    L2, W2 = spec.length / 2, spec.width / 2
    xs = np.linspace(-L2 * 0.9, -0.5, 10) if region == "near" else np.linspace(0.5, L2 * 0.95, 10)
    X, Y = np.meshgrid(xs, np.linspace(-W2, W2, 9))
    world = np.c_[X.ravel(), Y.ravel()] * s
    G = front_facing(np.linalg.inv(np.array(H_true)))
    q = np.c_[world, np.ones(len(world))] @ G.T
    q = q[q[:, 2] > 0]
    px = q[:, :2] / q[:, 2:3]
    px = px[(px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])]
    if len(px) == 0:
        return None
    a = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.array(H_auto, np.float64))[:, 0]
    b = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.array(H_true, np.float64))[:, 0]
    e = np.linalg.norm(a - b, axis=1)
    return float(np.median(e)), float(np.percentile(e, 95))


def as_cal(r, half, spec):
    return {"half": half, "spec": spec.__dict__, "H_near": r["H_near"], "H_far": r["H_far"]}


def main(args):
    cams = [int(args[args.index("--cam") + 1])] if "--cam" in args else [1, 2]
    model = load_model()
    for n in cams:
        cal = load(n)
        spec = spec_of(cal)
        src = Source(cal["video"])
        near_med, far_med, fails = [], [], 0
        for t in TIMES:
            seg = resolved(cal, t) and resolved(cal, t + 12)
            if seg is None:
                continue
            frame = median_of(src, t)
            t0 = time.time()
            r = keypoint_calibrate(frame, cal["half"], spec, model)
            dt = time.time() - t0
            if r["status"] != "ok":
                fails += 1
                print(f"cam{n} t={t:5d}: 실패 (가까운 쪽 지점 {r['n_near']}개)", flush=True)
                continue
            Hn, Hf = homographies_at(seg, t + 6)  # median_of는 t ~ t+12 s 프레임
            en = region_error(r["H_near"], Hn, cal["half"], spec, "near")
            ef = region_error(r["H_far"], Hf, cal["half"], spec, "far")
            near_med.append(en[0])
            if ef:
                far_med.append(ef[0])
            far_txt = f"먼 쪽 {ef[0]:5.2f} / {ef[1]:5.2f} m" if ef else "먼 쪽 -"
            print(f"cam{n} t={t:5d}: 지점 가까운 {r['n_near']:2d} 먼 {r['n_far']:2d} | 가까운 쪽 {en[0]:5.2f} / {en[1]:5.2f} m | "
                  f"{far_txt} (중앙값 / 95%, {dt:.1f}s)", flush=True)
        print(f"cam{n}: 성공 {len(near_med)}/{len(near_med) + fails}, 시점별 중앙값의 중앙값 가까운 쪽 {np.median(near_med):.2f} m, "
              f"먼 쪽 {np.median(far_med):.2f} m, 가까운 쪽 0.2 m 미만 {sum(e < 0.2 for e in near_med)}개", flush=True)

    print("\n== 검증 점 채점: 자동 결과 vs 현재 캘리브레이션 ==")
    labels = [lb for lb in read_labels(LABELS_PATH) if lb["cam"] in cams]
    rows_auto = []
    for lb in labels:
        frame = Source(lb["video"]).frame_at(lb["t"])
        cal = load(lb["cam"])
        r = keypoint_calibrate(frame, cal["half"], spec_of(cal), model)
        if r["status"] == "ok":
            rows_auto += score([lb], lambda _: as_cal(r, cal["half"], spec_of(cal)))
    rows_cal = score(labels, load)
    for name, rows in (("자동(키포인트)", rows_auto), ("현재 캘리브레이션", rows_cal)):
        print(f"[{name}]")
        for s in summarize(rows):
            print(f"  cam{s['cam']} {s['region']:<6} {s['kind']:<5} n={s['n']:2d} 중앙값 {s['median']:.2f}m 90% {s['p90']:.2f}m 최대 {s['max']:.2f}m")


if __name__ == "__main__":
    main(sys.argv[1:])
