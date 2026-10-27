"""How far (in meters) are the white lines from the model lines, per camera and per half?

For each model-line sample, search along the image normal (+-25 px) for the strongest white-line response and
map that pixel back to the court with the homography of that half. Errors above ~1 m are not reliably detected.
Usage: python line_error.py [suffix]   (default "_refined" -> out/H_cam<N>_refined.json; "" = clicks only)
"""
import json
import sys

import cv2
import numpy as np

from calibrate import front_facing, in_half
from refine import match_lines, ridge_map, sample_model

suffix = sys.argv[1] if len(sys.argv) > 1 else "_refined"
model = sample_model()
for n in (1, 2):
    d = json.load(open(f"out/H_cam{n}{suffix}.json"))
    ridge = cv2.GaussianBlur(ridge_map(cv2.imread(f"out/frame_cam{n}.jpg")), (0, 0), 1)
    parts = []
    for region in ("near", "far"):
        H = np.array(d[f"H_{region}"])
        src, dst = match_lines(ridge, front_facing(np.linalg.inv(H)), model[in_half(model, d["half"], region)], 25)
        court = cv2.perspectiveTransform(dst.reshape(-1, 1, 2).astype(np.float32), H)[:, 0]
        parts.append(f"{region} {np.median(np.linalg.norm(court - src, axis=1)):.2f} m ({len(src)} samples)")
    print(f"cam{n}: median line error  " + ",  ".join(parts))
