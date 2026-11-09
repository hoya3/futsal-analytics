"""화면(OpenCV 창) 없이 도는 캘리브레이션 공용 로직. GUI(calibrate.py)와 서버 처리가 함께 쓴다."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from court import DEFAULT_SPEC, court_lines

HALF_SIGN = {"east": 1, "west": -1}


def click_landmarks(spec=DEFAULT_SPEC):
    """5클릭 랜드마크.

    동쪽 골대를 바라보는 카메라 기준 코트 좌표: x = 코트 길이 방향(카메라 쪽이 x<0), y = 화면의 왼쪽.
    kind "point": 정확한 랜드마크. kind "line": 오른쪽 사이드라인(y = -W/2) 위의 아무 점; 선을 따른 위치는
    풀이로 구하므로 표시가 없어도 된다. "line" 클릭의 xy는 초기 추정값일 뿐이다.
    가까운 하프 = 점 4개 + 사이드라인 점 1개 = 호모그래피에 필요한 제약 8개와 정확히 같다. (사이드라인 점을
    하나 더 추가해도 소용없다: 하프라인 오른쪽 끝이 이미 그 사이드라인 위에 있다.)
    """
    L2, W2 = spec.length / 2, spec.width / 2
    return [
        {"name": "halfway line, left end", "kind": "point", "xy": (0.0, W2), "halves": ("near", "far")},
        {"name": "center circle x halfway line, right crossing", "kind": "point", "xy": (0.0, -spec.center_r), "halves": ("near",)},
        {"name": "halfway line, right end", "kind": "point", "xy": (0.0, -W2), "halves": ("near", "far")},
        {"name": "right sideline: any point low in the picture", "kind": "line", "xy": (-0.7 * L2, -W2), "halves": ("near",)},
        {"name": "10 m spot (small white dot)", "kind": "point", "xy": (-(L2 - 10.0), 0.0), "halves": ("near",)},
    ]


LANDMARKS = click_landmarks()


def to_world(xy, half):
    """동쪽 기준 좌표 -> 실제 코트 좌표 (서쪽을 보는 카메라에게는 코트가 180도 돌아가 보인다)."""
    x, y = xy
    return (x, y) if half == "east" else (-x, -y)


def in_half(pts, half, region, margin=0.0):
    """코트 점이 카메라의 가까운 하프(x*sign <= margin) 또는 먼 하프(x*sign >= -margin)에 있는지의 마스크."""
    s = HALF_SIGN[half] * np.asarray(pts)[:, 0]
    return s <= margin if region == "near" else s >= -margin


def solve_half(img_pts, lms, half, spec=DEFAULT_SPEC):
    """점 클릭과 사이드라인 클릭으로 호모그래피(이미지 -> 코트)를 구한다. (H, 풀이된 월드 좌표, 오차(m))를 반환."""
    img = np.asarray(img_pts, float)
    guess = np.array([to_world(lm["xy"], half) for lm in lms])
    H0, _ = cv2.findHomography(img, guess, 0)
    line_idx = [i for i, lm in enumerate(lms) if lm["kind"] == "line"]

    def unpack(p):
        H = np.append(p[:8], 1.0).reshape(3, 3)
        world = guess.copy()
        for k, i in enumerate(line_idx):  # 사이드라인 y = -W/2를 따라 이동 (동쪽 기준 좌표)
            world[i] = to_world((p[8 + k], -spec.width / 2), half)
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
    """호모그래피는 스케일/부호가 정해지지 않으므로, 센터 스폿 (0,0)이 카메라 앞에 오도록 맞춘다."""
    return Hc2i if Hc2i[2, 2] > 0 else -Hc2i


def draw_lines(view, H_img2court, keep, color=(0, 0, 255), thickness=2, spec=DEFAULT_SPEC):
    """keep(points)가 참인 코트 라인을 inv(H)로 투영해 그린다."""
    Hc2i = front_facing(np.linalg.inv(H_img2court))
    for line in court_lines(spec):
        dense = np.concatenate([np.linspace(a, b, 40, endpoint=False) for a, b in zip(line[:-1], line[1:])])
        q = np.c_[dense, np.ones(len(dense))] @ Hc2i.T
        ok = keep(dense) & (q[:, 2] > 0)
        px = q[:, :2] / q[:, 2:3]
        ok &= np.isfinite(px).all(1) & (np.abs(px) < 1e5).all(1)
        for run in np.split(np.arange(len(dense)), np.where(~ok)[0]):  # 연속으로 보이는 구간마다 그린다
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(view, [px[run].astype(np.int32)], False, color, thickness)


def draw_overlay(frame, H_near, H_far, half, spec=DEFAULT_SPEC):
    view = frame.copy()
    draw_lines(view, H_near, lambda p: in_half(p, half, "near"), spec=spec)
    draw_lines(view, H_far, lambda p: in_half(p, half, "far"), spec=spec)
    return view


class Source:
    """영상(탐색 가능) 또는 단일 이미지."""

    def __init__(self, path):
        self.is_video = path.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        if self.is_video:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
            self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.image = cv2.imread(path)

    def frame_at(self, t):
        if not self.is_video:
            return self.image
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None
