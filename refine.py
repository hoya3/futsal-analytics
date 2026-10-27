"""Snap each camera's two homographies (near half, far half) onto the white court lines.
The far one starts from the near one (calibrate.py has no far-half clicks) and maps the halfway line exactly like it.

Usage: python refine.py   -> out/H_cam<N>_refined.json + out/H_cam<N>_refined_overlay.jpg
Method (ICP): sample the court model's lines on one half, find the nearest white line along the image normal,
re-fit the homography, repeat with a shrinking search radius.
  near half: RANSAC fit (drops matches on fences/players); the hand clicks are added as heavily weighted anchors so
             the fit cannot drift away from them.
  far half:  H_far = H_near o K, where K = I + v [1 0 0] leaves every point of the halfway line (x = 0) in place, so
             the two homographies agree along the whole halfway line (no jump when a player crosses it). RANSAC only
             picks the inlier matches; v (3 numbers) is then least-squares fitted to them.
"""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate import draw_overlay, front_facing, in_half
from court import GOAL_W, PENALTY_R, court_lines

SIGMAS = (1.5, 2.5, 4.0, 6.0)
SAMPLE_SPACING = 0.15  # m between model-line samples
CIRCLE_MARGIN = 3.2  # each half also uses the whole center circle (radius 3 m)
# the far-half fit also uses the near-half lines within 8 m of the halfway line (circle, sidelines). With the halfway
# line shared, using the whole court pulls the far half toward H_near (measured: cam1 far 0.43 m, cam2 0.72 m vs
# 0.36 m, 0.61 m with 8 m)
FAR_MARGIN = 8.0
RADII = (25, 18, 12, 8, 5, 3, 3)  # px, ICP search radius per round


def ridge_map(frame):
    """Bright thin-line strength in [0, 1] (multi-scale Hessian), zero on saturated colors (grass, trees)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = np.zeros_like(gray)
    for s in SIGMAS:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        lam_min = (ixx + iyy) / 2 - np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)  # most negative curvature = bright ridge
        out = np.maximum(out, s ** 2 * np.maximum(0, -lam_min))
    out *= (hsv[..., 1] < 110)
    return np.minimum(out / np.percentile(out, 99.7), 1.0).astype(np.float32)


def sample_model(penalty_r=PENALTY_R, goal_w=GOAL_W):
    """Points every ~15 cm along all court lines."""
    pts = []
    for line in court_lines(penalty_r, goal_w):
        for a, b in zip(line[:-1], line[1:]):
            n = max(int(np.linalg.norm(b - a) / SAMPLE_SPACING), 1)
            pts.append(np.linspace(a, b, n, endpoint=False))
    return np.concatenate(pts)


def project(Hc2i, pts):
    q = np.c_[pts, np.ones(len(pts))] @ Hc2i.T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def match_lines(ridge, Hc2i, pts, radius, min_ridge=0.3):
    """For each model sample, the strongest white-line pixel along the image normal within +-radius px.
    Returns (court points, matched image points)."""
    h, w = ridge.shape
    px, depth = project(Hc2i, pts)
    offs = np.arange(-radius, radius + 1)
    src, dst = [], []
    for i in range(1, len(pts) - 1):
        t = px[i + 1] - px[i - 1]
        if depth[i] <= 0 or np.linalg.norm(t) < 1e-3 or np.linalg.norm(pts[i + 1] - pts[i - 1]) > 0.5:
            continue  # behind the camera, degenerate, or a joint between two lines
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        cand = px[i] + offs[:, None] * nrm
        if (cand < 2).any() or (cand[:, 0] > w - 3).any() or (cand[:, 1] > h - 3).any():
            continue
        v = cv2.remap(ridge, cand[:, :1].astype(np.float32), cand[:, 1:].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
        if v.max() >= min_ridge:
            src.append(pts[i])
            dst.append(cand[v.argmax()])
    return np.array(src), np.array(dst)


def icp_align(ridge, H_img2court, pts, anchor_world, anchor_img, anchor_w=5, radii=RADII):
    """Returns (H_img2court, fraction of line matches kept by RANSAC in the last round)."""
    a_src = np.repeat(np.asarray(anchor_world, float), anchor_w, axis=0)
    a_dst = np.repeat(np.asarray(anchor_img, float), anchor_w, axis=0)
    Hc2i = front_facing(np.linalg.inv(H_img2court))
    ratio = 0.0
    for r in radii:
        src, dst = match_lines(ridge, Hc2i, pts, r)
        if len(src) < 50:
            break
        H_new, inl = cv2.findHomography(np.r_[src, a_src].astype(np.float32), np.r_[dst, a_dst].astype(np.float32),
                                        cv2.RANSAC, max(2.0, r / 3))
        if H_new is None:
            break
        Hc2i, ratio = front_facing(H_new), inl[:len(src)].mean()
    H = np.linalg.inv(Hc2i)
    return H / H[2, 2], ratio


def icp_align_far(ridge, H_near, pts, radii=RADII):
    """Far half as H_near o K with K = I + v [1 0 0] (fixes the halfway line pointwise).
    Returns (H_img2court, fraction of line matches kept by RANSAC in the last round)."""
    G = front_facing(np.linalg.inv(H_near))

    def c2i(v):  # K's third column is (0, 0, 1), so (G K)[2, 2] = G[2, 2] stays positive (front facing)
        return G @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0]))

    v, ratio = np.zeros(3), 0.0
    for r in radii:
        src, dst = match_lines(ridge, c2i(v), pts, r)
        if len(src) < 50:
            break
        _, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, max(2.0, r / 3))
        if inl is None:
            break
        keep = inl[:, 0] > 0
        v = least_squares(lambda p: (project(c2i(p), src[keep])[0] - dst[keep]).ravel(), v, x_scale="jac").x
        ratio = keep.mean()
    H = np.linalg.inv(c2i(v))
    return H / H[2, 2], ratio


def refine_camera(d, frame):
    ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    model = sample_model()
    out = dict(d)
    # near half: its own lines (plus the whole center circle), anchored at the clicks
    pts = model[in_half(model, d["half"], "near", margin=CIRCLE_MARGIN)]
    idx = [int(i) for i in d["world"]["near"]]
    H_near, ratio = icp_align(ridge, np.array(d["H_near"]), pts,
                              [d["world"]["near"][str(i)] for i in idx], [d["img_pts"][i] for i in idx])
    print(f"  near half: {ratio:.0%} of line matches kept")
    # far half: its own lines are a few pixels thin and far away, so the sharp lines around the halfway line are added.
    # It inherits the refined near half's halfway line.
    pts = model[in_half(model, d["half"], "far", margin=FAR_MARGIN)]
    H_far, ratio = icp_align_far(ridge, H_near, pts)
    print(f"  far half: {ratio:.0%} of line matches kept")
    out["H_near"], out["H_far"] = H_near.tolist(), H_far.tolist()
    out["refined"] = "icp"
    return out


if __name__ == "__main__":
    for n in (1, 2):
        d = json.load(open(f"out/H_cam{n}.json"))
        if "H_near" not in d:
            raise SystemExit(f"out/H_cam{n}.json is from the old 5-point tool: re-run calibrate.py for cam{n} first")
        frame = cv2.imread(f"out/frame_cam{n}.jpg")
        print(f"cam{n}:")
        r = refine_camera(d, frame)
        json.dump(r, open(f"out/H_cam{n}_refined.json", "w"), indent=2)
        cv2.imwrite(f"out/H_cam{n}_refined_overlay.jpg",
                    draw_overlay(frame, np.array(r["H_near"]), np.array(r["H_far"]), r["half"]))
