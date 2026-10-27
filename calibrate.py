"""Mouse-only court calibration: 5 clicks per camera, all on the camera's own (near) half.

A single homography cannot fit both halves (the camera is low, so slight unevenness of the outdoor court shifts the
far half), so each camera gets
  H_near: image px -> court m, valid on the camera's own half (where it tracks players) - from the 5 clicks
  H_far:  image px -> court m, valid beyond the halfway line (fills the spots the other camera cannot see).
          Starts as H_near; refine.py fits it to the white lines while keeping every point of the halfway line
          where H_near puts it, so both agree along the whole halfway line.

Usage: python calibrate.py camera1.mp4 camera2.mp4      (images also work; "-" skips a camera, e.g. - camera2.mp4)
  Starts at t=60 s (the phones moved during the first ~30 s of the videos). Pick a frame where nobody hides the
  landmarks: a/d = -/+ 10 s, z/c = -/+ 1 s (only before the first click).
  Left/right are as seen in the picture. The minimap (bottom-left) shows the next point in red.
  "half" = which goal the camera faces: cam1 faces the east goal (x=+20), cam2 the west goal (x=-20); h flips it.
Keys: left click = place the red point, u = undo, h = flip facing (before first click), m = hide/show help,
      Enter = accept, q = quit.
Output per camera: out/H_cam<N>.json, out/frame_cam<N>.jpg, out/H_cam<N>_overlay.jpg. Run refine.py afterwards.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from court import CENTER_R, L, W, court_lines

HALF_SIGN = {"east": 1, "west": -1}

# Court coordinates for a camera facing the EAST goal: x along the court (camera side x<0), y = left in the picture.
# kind "point": an exact landmark. kind "line": any point on the right sideline (y = -W/2); its position along the
# line is solved for, so it does not need a marking. xy of a "line" click is only the starting guess.
# Near half = 4 points + 1 sideline point = exactly the 8 constraints a homography needs. (A second sideline point
# would add nothing: the halfway line's right end already lies on that sideline.)
LANDMARKS = [
    {"name": "halfway line, left end", "kind": "point", "xy": (0.0, W / 2), "halves": ("near", "far")},
    {"name": "center circle x halfway line, right crossing", "kind": "point", "xy": (0.0, -CENTER_R), "halves": ("near",)},
    {"name": "halfway line, right end", "kind": "point", "xy": (0.0, -W / 2), "halves": ("near", "far")},
    {"name": "right sideline: any point low in the picture", "kind": "line", "xy": (-14.0, -W / 2), "halves": ("near",)},
    {"name": "10 m spot (small white dot)", "kind": "point", "xy": (-10.0, 0.0), "halves": ("near",)},
]


def to_world(xy, half):
    """East-facing coordinates -> real court coordinates (a west-facing camera sees the court rotated 180 degrees)."""
    x, y = xy
    return (x, y) if half == "east" else (-x, -y)


def in_half(pts, half, region, margin=0.0):
    """Mask of court points on the camera's near half (x*sign <= margin) or far half (x*sign >= -margin)."""
    s = HALF_SIGN[half] * np.asarray(pts)[:, 0]
    return s <= margin if region == "near" else s >= -margin


def solve_half(img_pts, lms, half):
    """Homography (image -> court) from point and sideline clicks. Returns (H, solved world points, errors in m)."""
    img = np.asarray(img_pts, float)
    guess = np.array([to_world(lm["xy"], half) for lm in lms])
    H0, _ = cv2.findHomography(img, guess, 0)
    line_idx = [i for i, lm in enumerate(lms) if lm["kind"] == "line"]

    def unpack(p):
        H = np.append(p[:8], 1.0).reshape(3, 3)
        world = guess.copy()
        for k, i in enumerate(line_idx):  # slide along the sideline y = -W/2 (east-facing coordinates)
            world[i] = to_world((p[8 + k], -W / 2), half)
        return H, world

    def residuals(p):
        H, world = unpack(p)
        return (cv2.perspectiveTransform(img[None].astype(np.float64), H)[0] - world).ravel()

    p0 = np.r_[(H0 / H0[2, 2]).ravel()[:8], [lms[i]["xy"][0] for i in line_idx]]
    sol = least_squares(residuals, p0, x_scale="jac")
    H, world = unpack(sol.x)
    err = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return H, world, err


def front_facing(Hc2i):
    """A homography is defined up to scale/sign; make the center spot (0,0) lie in front of the camera."""
    return Hc2i if Hc2i[2, 2] > 0 else -Hc2i


def draw_lines(view, H_img2court, keep, color=(0, 0, 255), thickness=2):
    """Draw the court markings for which keep(points) is true, through inv(H)."""
    Hc2i = front_facing(np.linalg.inv(H_img2court))
    for line in court_lines():
        dense = np.concatenate([np.linspace(a, b, 40, endpoint=False) for a, b in zip(line[:-1], line[1:])])
        q = np.c_[dense, np.ones(len(dense))] @ Hc2i.T
        ok = keep(dense) & (q[:, 2] > 0)
        px = q[:, :2] / q[:, 2:3]
        ok &= np.isfinite(px).all(1) & (np.abs(px) < 1e5).all(1)
        for run in np.split(np.arange(len(dense)), np.where(~ok)[0]):  # draw each continuous visible run
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(view, [px[run].astype(np.int32)], False, color, thickness)


def draw_overlay(frame, H_near, H_far, half):
    view = frame.copy()
    draw_lines(view, H_near, lambda p: in_half(p, half, "near"))
    draw_lines(view, H_far, lambda p: in_half(p, half, "far"))
    return view


class Source:
    """A video (seekable) or a single image."""

    def __init__(self, path):
        self.is_video = path.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        if self.is_video:
            self.cap = cv2.VideoCapture(path)
            self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.image = cv2.imread(path)

    def frame_at(self, t):
        if not self.is_video:
            return self.image
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None


class Calibrator:
    def __init__(self, frame, half):
        self.frame = frame
        self.half = half
        self.t = 0.0
        self.show_map = True
        self.mouse = None  # last cursor position, for the magnifier
        self.clicks = []
        self.result = None  # {"near": (H, world, err), "far": (H, world, err)}

    @property
    def done(self):
        return len(self.clicks) == len(LANDMARKS)

    def _solve(self):
        if not self.done:
            return
        idx = list(range(len(LANDMARKS)))
        H, world, err = solve_half(self.clicks, LANDMARKS, self.half)
        near = (H, dict(zip(idx, world)), dict(zip(idx, err)))
        far_idx = [i for i, lm in enumerate(LANDMARKS) if "far" in lm["halves"]]  # anchors for the far-half fit
        self.result = {"near": near, "far": (H, {i: near[1][i] for i in far_idx}, {i: near[2][i] for i in far_idx})}

    def click(self, x, y):
        if not self.done:
            self.clicks.append((x, y))
            self._solve()

    def undo(self):
        if self.clicks:
            self.clicks.pop()
            self.result = None

    def flip(self):
        if not self.clicks:
            self.half = "west" if self.half == "east" else "east"

    def _minimap(self, view):
        x0, w, h = 20, 120, 240  # the whole court as the camera sees it: far goal on top, its own goal at the bottom
        y0 = view.shape[0] - h - 110
        cv2.rectangle(view, (x0 - 10, y0 - 10), (x0 + 640, y0 + h + 100), (0, 0, 0), -1)

        def to_map(x, y):
            return int(x0 + (W / 2 - y) / W * w), int(y0 + (L / 2 - x) / L * h)

        for line in court_lines():
            cv2.polylines(view, [np.array([to_map(x, y) for x, y in line], np.int32)], False, (130, 130, 130), 1)
        i_next = len(self.clicks)
        for i, lm in enumerate(LANDMARKS):
            px, py = to_map(*lm["xy"])
            color = (0, 200, 0) if i < i_next else (0, 0, 255) if i == i_next else (200, 200, 200)
            cv2.circle(view, (px, py), 8, color, -1)
            cv2.putText(view, str(i + 1), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tx = x0 + w + 25
        if self.done:
            msg = "Enter: accept   u: undo"
        else:
            msg = f"click {i_next + 1}/{len(LANDMARKS)}: {LANDMARKS[i_next]['name']}"
        cv2.putText(view, f"facing the {self.half} goal", (tx, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, msg, (tx, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if not self.done and LANDMARKS[i_next]["kind"] == "line":
            cv2.putText(view, "(anywhere ON the line - no marking needed)", (tx, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(view, f"t={self.t:.0f}s  a/d: -/+10s  z/c: -/+1s (before clicking)", (tx, y0 + h + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    def render(self):
        if self.result:
            view = draw_overlay(self.frame, self.result["near"][0], self.result["far"][0], self.half)
        else:
            view = self.frame.copy()
        for i, c in enumerate(self.clicks):
            cv2.circle(view, tuple(map(int, c)), 7, (0, 255, 0), -1)
            cv2.putText(view, str(i + 1), (int(c[0]) + 10, int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if self.show_map:
            self._minimap(view)
        self._magnifier(view)
        return view

    def _magnifier(self, view, half_size=40, zoom=5):
        """Zoomed view of the area under the cursor (opposite side of the screen), so far corners can be clicked to the pixel."""
        if self.mouse is None:
            return
        x, y = self.mouse
        h, w = self.frame.shape[:2]
        x0, y0 = min(max(x - half_size, 0), w - 2 * half_size), min(max(y - half_size, 0), h - 2 * half_size)
        crop = view[y0:y0 + 2 * half_size, x0:x0 + 2 * half_size]
        big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        cx, cy = (x - x0) * zoom + zoom // 2, (y - y0) * zoom + zoom // 2
        cv2.line(big, (cx, 0), (cx, big.shape[0]), (0, 255, 255), 1)
        cv2.line(big, (0, cy), (big.shape[1], cy), (0, 255, 255), 1)
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (255, 255, 255), 2)
        bx = 10 if x > w // 2 else w - 10 - big.shape[1]  # opposite side of the cursor, below the far court edge
        view[200:200 + big.shape[0], bx:bx + big.shape[1]] = big

    def to_json(self, video=None):
        return {
            "half": self.half, "video": video, "t_sec": self.t,
            "names": [lm["name"] for lm in LANDMARKS],
            "img_pts": [list(map(float, c)) for c in self.clicks],
            "world": {region: {str(i): list(map(float, p)) for i, p in self.result[region][1].items()} for region in ("near", "far")},
            "H_near": self.result["near"][0].tolist(),
            "H_far": self.result["far"][0].tolist(),
        }

    def save(self, out_json, video=None):
        cv2.imwrite(out_json.replace("H_", "frame_").replace(".json", ".jpg"), self.frame)
        with open(out_json, "w") as f:
            json.dump(self.to_json(video), f, indent=2)
        cv2.imwrite(out_json.replace(".json", "_overlay.jpg"), self.render())


def run_camera(n, path, half, t=60.0):  # both phones were still being adjusted during the first ~30 s
    src = Source(path)
    cal = Calibrator(src.frame_at(t), half)
    cal.t = t
    win = f"cam{n} - {os.path.basename(path)}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def on_mouse(ev, x, y, *_):
        cal.mouse = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            cal.click(x, y)

    cv2.setMouseCallback(win, on_mouse)
    steps = {ord("a"): -10, ord("d"): 10, ord("z"): -1, ord("c"): 1}
    while True:
        cv2.imshow(win, cal.render())
        key = cv2.waitKey(30) & 0xFF
        if key in steps and src.is_video and not cal.clicks:
            new_t = min(max(cal.t + steps[key], 0), src.duration - 1)
            frame = src.frame_at(new_t)
            if frame is not None:
                cal.frame, cal.t = frame, new_t
        elif key == ord("u"):
            cal.undo()
        elif key == ord("h"):
            cal.flip()
        elif key == ord("m"):
            cal.show_map = not cal.show_map
        elif key == ord("q"):
            cv2.destroyWindow(win)
            return False
        elif key == 13 and cal.result:
            os.makedirs("out", exist_ok=True)
            cal.save(f"out/H_cam{n}.json", video=path)
            print(f"cam{n}: facing {cal.half}, t={cal.t:.0f}s -> out/H_cam{n}.json")
            cv2.destroyWindow(win)
            return True


if __name__ == "__main__":
    for n, (path, half) in enumerate(zip(sys.argv[1:3], ("east", "west")), start=1):
        if path == "-":  # keep this camera's existing calibration
            continue
        if not run_camera(n, path, half):
            break
