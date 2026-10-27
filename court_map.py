"""Image pixel -> court meters for one camera, using its near-half and far-half homographies.

    from court_map import load, pixel_to_court
    cam1 = load(1)
    xy = pixel_to_court(cam1, [[960, 700], [1200, 300]])   # (N, 2) court meters, origin = center spot

Court coordinates: x along the court (-20..20), y across (-10..10), meters.
"""
import json
import os

import cv2
import numpy as np

from calibrate import HALF_SIGN


def load(n, out_dir="out"):
    """Calibration of camera n (the refined one if it exists)."""
    refined = os.path.join(out_dir, f"H_cam{n}_refined.json")
    with open(refined if os.path.exists(refined) else os.path.join(out_dir, f"H_cam{n}.json")) as f:
        return json.load(f)


def pixel_to_court(cal, px):
    px = np.asarray(px, np.float32).reshape(-1, 1, 2)
    xy = cv2.perspectiveTransform(px, np.array(cal["H_near"]))[:, 0]
    far = xy[:, 0] * HALF_SIGN[cal["half"]] > 0  # beyond the halfway line: use the far-half homography
    if far.any():
        xy[far] = cv2.perspectiveTransform(px[far], np.array(cal["H_far"]))[:, 0]
    return xy
