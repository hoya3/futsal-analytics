"""자동 인식 결과를 눈으로 확인한다. 빨강 = 자동 인식, 초록 = 기준(track.py 결과).

사용법: python eval/preview_auto.py --cam 1                 경기 전체에 고르게 뽑은 5개 시점 + 다수결 판정
        python eval/preview_auto.py --cam 2 --t 1000 2400    지정한 시점만 (동기화 영상 시각, 초)
출력: out/auto_cam<N>_t<시각>.jpg, 시점당 약 30초 소요
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_init import auto_calibrate, consensus, median_of  # noqa: E402
from calib_core import Source, draw_lines, draw_overlay, out_path  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import load  # noqa: E402
from eval_auto import error_m, truth_segment  # noqa: E402


def parse_times(args):
    if "--t" not in args:
        return None
    out = []
    for a in args[args.index("--t") + 1:]:
        if a.startswith("--"):
            break
        out.append(float(a))
    return out


def main(args):
    n = int(args[args.index("--cam") + 1]) if "--cam" in args else 1
    cal = load(n)
    spec = spec_of(cal)
    src = Source(cal["video"])
    times = parse_times(args)
    if times is None:
        times = [float(t) for t in np.linspace(src.duration * 0.15, src.duration * 0.9, 5)]

    results = []
    for t in times:
        frame = median_of(src, t)
        if frame is None:
            print(f"t={t:.0f}: 프레임을 읽을 수 없음")
            continue
        r = auto_calibrate(frame, cal["half"], spec)
        r["t_sec"] = t
        results.append(r)
        seg = truth_segment(cal, t)
        view = frame.copy()
        err = "no reference"
        if r["H_near"] is not None:
            view = draw_overlay(frame, np.array(r["H_near"]), np.array(r["H_far"]), cal["half"], spec)
        if seg is not None:
            draw_lines(view, np.array(seg["H_near"]), lambda p: np.ones(len(p), bool), color=(0, 255, 0), thickness=1, spec=spec)
            if r["H_near"] is not None:
                med, p95 = error_m(r["H_near"], seg["H_near"], cal["half"], spec)
                err = f"err med {med:.2f} m, p95 {p95:.2f} m"
        text = f"cam{n} t={t:.0f}  {r['status']}  conf={r['confidence']:.2f}  {err}"
        cv2.rectangle(view, (0, 0), (1100, 44), (0, 0, 0), -1)
        cv2.putText(view, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        path = out_path(f"auto_cam{n}_t{int(t)}.jpg")
        cv2.imwrite(path, view)
        print(f"{text.replace('  ', ' | ')} -> {path}", flush=True)

    if len(results) >= 3:
        lead, members = consensus(results, spec)
        if lead is None:
            print("다수결: 후보 없음 -> 수동 5클릭")
        else:
            passed = len(members) * 2 > len(results)
            print(f"다수결: {len(members)}/{len(results)}표, 앵커 t={results[lead]['t_sec']:.0f} -> {'자동 제안 통과' if passed else '수동 5클릭'}")


if __name__ == "__main__":
    main(sys.argv[1:])
