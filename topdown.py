"""Warp camera frames onto the 40x20 m court plane (top-down) using the calibrations.

Usage: python topdown.py           -> out/topdown.jpg: each camera's near half (cam1 left, cam2 right); spots a camera
                                      cannot see (right below it) are filled from the other camera's far view
       python topdown.py --cam 1   -> out/topdown_cam1.jpg: the whole court as camera 1 maps it (near + far half)
"""
import sys

import cv2
import numpy as np

from calibrate import in_half
from court import L, W, court_lines
from court_map import load

S = 30  # pixels per meter
TO_PX = np.array([[S, 0, L / 2 * S], [0, -S, W / 2 * S], [0, 0, 1]])  # court m -> canvas px (north up)
SIZE = (int(L * S), int(W * S))
_xs = (np.arange(SIZE[0]) - L / 2 * S) / S
GRID = np.stack(np.broadcast_arrays(_xs[None, :], np.zeros((SIZE[1], 1))), -1).reshape(-1, 2)  # canvas -> court x


def paste(canvas, n, region):
    cal = load(n)
    frame = cv2.imread(f"out/frame_cam{n}.jpg")
    M = TO_PX @ np.array(cal[f"H_{region}"])
    img = cv2.warpPerspective(frame, M, SIZE)
    seen = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, SIZE) > 0  # inside the camera's picture
    mask = in_half(GRID, cal["half"], region).reshape(SIZE[1], SIZE[0]) & seen
    canvas[mask] = img[mask]
    return cal, mask


def label(canvas, text, left):
    x = 10 if left else SIZE[0] - 10 - 18 * len(text)
    cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


if __name__ == "__main__":
    canvas = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    if "--cam" in sys.argv:
        n = int(sys.argv[sys.argv.index("--cam") + 1])
        for region in ("near", "far"):
            cal, _ = paste(canvas, n, region)
        near_left = cal["half"] == "east"  # a camera facing east has its near half on the west (left)
        label(canvas, f"cam{n} near", near_left)
        label(canvas, f"cam{n} far", not near_left)
        out = f"out/topdown_cam{n}.jpg"
    else:
        filled = np.zeros((SIZE[1], SIZE[0]), bool)
        for n in (1, 2):
            cal, mask = paste(canvas, n, "near")
            filled |= mask
        for n in (1, 2):  # fill the blind spots from the other camera's far view, marked with a thin outline
            far = np.zeros_like(canvas)
            _, mask = paste(far, n, "far")
            gap = mask & ~filled
            canvas[gap] = far[gap]
            filled |= gap
            contours, _ = cv2.findContours(gap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1)
        for n in (1, 2):
            label(canvas, f"cam{n} near", load(n)["half"] == "east")
        out = "out/topdown.jpg"
    for line in court_lines():
        px = np.stack([(line[:, 0] + L / 2) * S, (W / 2 - line[:, 1]) * S], axis=1)
        cv2.polylines(canvas, [px.astype(np.int32)], False, (0, 0, 255), 1)
    cv2.imwrite(out, canvas)
    print("->", out)
