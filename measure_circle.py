"""Measure the real center circle (radius, center) from each camera: trace the white ring, map to court meters, fit a circle."""
import json
import sys

import cv2
import numpy as np

from refine import ridge_map, project, front_facing


def fit_circle(p):
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)


for n in (1, 2):
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_refined"
    d = json.load(open(f"out/H_cam{n}{suffix}.json"))
    H = np.array(d["H_near"])  # the circle straddles the halfway line; both homographies share it
    Hc2i = front_facing(np.linalg.inv(H))
    ridge = cv2.GaussianBlur(ridge_map(cv2.imread(f"out/frame_cam{n}.jpg")), (0, 0), 1.5)
    pts = []
    for th in np.radians(np.arange(0, 360, 3)):
        rs = np.linspace(1.8, 4.2, 97)  # search radii (m) around the assumed 3 m ring
        world = np.c_[rs * np.cos(th), rs * np.sin(th)]
        px, depth = project(Hc2i, world)
        if (depth <= 0).any() or (px < 5).any() or (px[:, 0] > 1914).any() or (px[:, 1] > 1074).any():
            continue
        v = cv2.remap(ridge, px[:, :1].astype(np.float32), px[:, 1:].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
        if v.max() > 0.25:
            pts.append(world[v.argmax()])
    pts = np.array(pts)
    cx, cy, r = fit_circle(pts)
    res = np.abs(np.linalg.norm(pts - [cx, cy], axis=1) - r)
    print(f"cam{n}: {len(pts)} ring points -> center=({cx:.2f},{cy:.2f}) m, radius={r:.2f} m, fit residual {res.mean():.2f} m")
