"""카메라별, 하프별로 실제 흰 선이 모델 선에서 몇 미터 떨어져 있는가?

모델 선의 각 샘플마다 이미지 법선 방향(+-25 px)으로 가장 강한 흰 선 응답을 찾고, 그 픽셀을 해당 하프의
호모그래피로 코트에 다시 매핑한다. 약 1 m를 넘는 오차는 안정적으로 검출되지 않는다.
사용법: python line_error.py [suffix]   (기본값 "_refined" -> out/H_cam<N>_refined.json; "" = 클릭만 사용한 결과)
"""
import json
import sys

import cv2
import numpy as np

from calib_core import front_facing, in_half, out_path
from court import spec_of
from refine import match_lines, ridge_map, sample_model

suffix = sys.argv[1] if len(sys.argv) > 1 else "_refined"
for n in (1, 2):
    d = json.load(open(out_path(f"H_cam{n}{suffix}.json")))
    model = sample_model(spec_of(d))
    ridge = cv2.GaussianBlur(ridge_map(cv2.imread(out_path(f"frame_cam{n}.jpg"))), (0, 0), 1)
    parts = []
    for region in ("near", "far"):
        H = np.array(d[f"H_{region}"])
        src, dst = match_lines(ridge, front_facing(np.linalg.inv(H)), model[in_half(model, d["half"], region)], 25)
        court = cv2.perspectiveTransform(dst.reshape(-1, 1, 2).astype(np.float32), H)[:, 0]
        parts.append(f"{region} {np.median(np.linalg.norm(court - src, axis=1)):.2f} m ({len(src)} samples)")
    print(f"cam{n}: median line error  " + ",  ".join(parts))
