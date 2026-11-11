"""2차원 코트 지도를 만든다.

기본은 코트 치수로 그린 합성 지도다. 카메라 합성에서 생기는 경계와 어긋남이 없어 히트맵과 선수 위치의 배경으로 쓴다.
--photo는 실제 영상을 펼쳐 합성한 확인용 이미지다: 경기 전체에서 고르게 뽑은 프레임의 픽셀별 중앙값으로 움직이는 선수를 지우고,
코트의 각 지점을 더 선명하게 보는 카메라 쪽으로 섞는다.

사용법: python topdown.py                             -> out/topdown.jpg
        python topdown.py --photo [--frames 11] [--lines]  -> out/topdown_photo.jpg (--lines: 모델 코트 선을 빨강으로 겹침)
"""
import sys

import cv2
import numpy as np

from calib_core import Source, front_facing, in_half, out_path
from court import court_lines, landmarks, spec_of
from court_map import homographies_at, load

S = 30  # 미터당 픽셀
MARGIN = 1.0  # 코트 둘레 여백 (m): 테두리 선과 골대가 잘리지 않게
GRASS = (60, 125, 55)  # 합성 지도 배경 (BGR)
LINE = (245, 245, 245)
LINE_PX = 3  # 코트 선 굵기 (약 10 cm)
GOAL_DEPTH = 0.8  # m
SHARPNESS = 4  # 클수록 더 선명한 카메라 쪽으로 치우친다
SEAM_BLUR = 15.0  # 카메라 경계를 섞는 폭 (px)
EDGE_FADE = 120.0  # 카메라 시야 끝에서 비중을 줄여 가는 폭 (px)


class Canvas:
    """코트 치수에 맞는 탑다운 캔버스의 좌표 변환."""

    def __init__(self, spec):
        self.spec = spec
        L, W = spec.length, spec.width
        cx, cy = (L / 2 + MARGIN) * S, (W / 2 + MARGIN) * S
        self.to_px = np.array([[S, 0, cx], [0, -S, cy], [0, 0, 1]])  # 코트 m -> 캔버스 px (북쪽이 위)
        self.size = (int((L + 2 * MARGIN) * S), int((W + 2 * MARGIN) * S))
        gx, gy = np.meshgrid((np.arange(self.size[0]) - cx) / S, (cy - np.arange(self.size[1])) / S)
        self.grid = np.c_[gx.ravel(), gy.ravel()]  # 캔버스 픽셀의 코트 좌표

    def px(self, pts):
        """코트 좌표 (N, 2) -> 캔버스 정수 픽셀."""
        pts = np.asarray(pts, float)
        return np.rint(pts @ self.to_px[:2, :2].T + self.to_px[:2, 2]).astype(np.int32)


def render_court(spec):
    """합성 코트 지도. (이미지, Canvas)를 돌려준다."""
    geo = Canvas(spec)
    w, h = geo.size
    canvas = np.full((h, w, 3), GRASS, np.uint8)
    for line in court_lines(spec):
        cv2.polylines(canvas, [geo.px(line)], False, LINE, LINE_PX, cv2.LINE_AA)
    for name, xy in landmarks(spec).items():
        if name == "center" or name.startswith(("pen_mark", "pen2_mark")):
            cv2.circle(canvas, tuple(int(v) for v in geo.px([xy])[0]), 4, LINE, -1, cv2.LINE_AA)
    L2, g = spec.length / 2, spec.goal_w / 2
    for sx in (1, -1):
        goal = np.array([[sx * L2, g], [sx * (L2 + GOAL_DEPTH), g], [sx * (L2 + GOAL_DEPTH), -g], [sx * L2, -g]])
        cv2.polylines(canvas, [geo.px(goal)], False, LINE, 2, cv2.LINE_AA)
    return canvas, geo


def _usable(cal, t):
    for seg in cal["segments"]:
        if seg["t0"] <= t <= seg["t1"]:
            return None if seg["source"] == "unresolved" else seg
    return None


def warp_frame(frame, seg, half, geo, t):
    """시각 t의 프레임 한 장을 코트로 펼친다. 카메라 쪽 절반은 H_near, 나머지는 H_far. 보이지 않는 곳은 검정."""
    w, h = geo.size
    out = np.zeros((h, w, 3), np.uint8)
    for region, H in zip(("near", "far"), homographies_at(seg, t)):
        img = cv2.warpPerspective(frame, geo.to_px @ H, geo.size)
        mask = in_half(geo.grid, half, region).reshape(h, w)
        out[mask] = img[mask]
    return out


def median_court(cal, source, times, geo):
    """(코트로 펼친 중앙값 이미지, 대표 구간)을 돌려준다. 프레임 밖이었던 값은 중앙값에서 뺀다."""
    stack, rep = [], None
    for t in times:
        seg = _usable(cal, float(t))
        frame = source.frame_at(float(t)) if seg is not None else None
        if frame is not None:
            stack.append(warp_frame(frame, seg, cal["half"], geo, float(t)))
            rep = rep or seg
    if not stack:
        raise ValueError("코트를 펼칠 수 있는 프레임이 없습니다")
    data = np.stack(stack).astype(np.float32)
    data[data.sum(-1) == 0] = np.nan
    return np.nan_to_num(np.nanmedian(data, axis=0)).astype(np.uint8), rep


def sharpness(seg, half, geo):
    """코트 1 m가 화면에서 차지하는 픽셀 수의 최솟값(깊이 방향). 클수록 그 카메라가 이 지점을 선명하게 본다."""
    w, h = geo.size
    near = in_half(geo.grid, half, "near")
    out = np.zeros(len(geo.grid), np.float32)
    for region, sel in (("near", near), ("far", ~near)):
        G = front_facing(np.linalg.inv(np.array(seg[f"H_{region}"])))
        p = geo.grid[sel]
        q = np.c_[p, np.ones(len(p))] @ G.T
        u, v, d = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2], q[:, 2]
        a = (G[0, :2][None] - u[:, None] * G[2, :2][None]) / d[:, None]
        b = (G[1, :2][None] - v[:, None] * G[2, :2][None]) / d[:, None]
        out[sel] = np.linalg.svd(np.stack([a, b], 1), compute_uv=False)[:, 1] * (d > 0)
    return out.reshape(h, w)


def build(cals, open_source=Source, frames=11):
    """두 카메라의 시간별 캘리브레이션 {1: cal, 2: cal}로 깨끗한 코트 맵을 만든다."""
    geo = Canvas(spec_of(cals[1]))
    sources = {n: open_source(c["video"]) for n, c in cals.items()}
    span = min(s.duration for s in sources.values())
    times = np.linspace(span * 0.05, span * 0.95, frames)
    imgs, weights = {}, {}
    for n, cal in cals.items():
        imgs[n], rep = median_court(cal, sources[n], times, geo)
        seen = imgs[n].any(2).astype(np.uint8)
        edge = np.minimum(cv2.distanceTransform(seen, cv2.DIST_L2, 3) / EDGE_FADE, 1.0)  # 시야 끝에서 서서히 비중을 줄인다
        weights[n] = sharpness(rep, cal["half"], geo) * edge

    both = (weights[1] > 0) & (weights[2] > 0)
    if both.any():  # 두 카메라의 색과 밝기를 가운데로 맞춘다
        gain = np.sqrt(np.median(imgs[1][both], axis=0) / np.maximum(np.median(imgs[2][both], axis=0), 1.0))
        imgs[1] = np.clip(imgs[1] / np.maximum(gain, 1e-3), 0, 255).astype(np.uint8)
        imgs[2] = np.clip(imgs[2] * gain, 0, 255).astype(np.uint8)

    w1, w2 = weights[1] ** SHARPNESS, weights[2] ** SHARPNESS
    alpha = np.divide(w1, w1 + w2, out=np.zeros_like(w1), where=(w1 + w2) > 0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), SEAM_BLUR)  # 경계가 선으로 보이지 않게 섞는다
    alpha[weights[2] == 0] = 1.0  # 한 카메라만 보는 곳은 그 영상 그대로
    alpha[weights[1] == 0] = 0.0
    out = alpha[..., None] * imgs[1] + (1 - alpha[..., None]) * imgs[2]
    return out.astype(np.uint8), geo


def draw_model_lines(canvas, geo):
    for line in court_lines(geo.spec):
        cv2.polylines(canvas, [geo.px(line)], False, (0, 0, 255), 1)


def main(args):
    cals = {n: load(n) for n in (1, 2)}
    if "--photo" not in args:
        cv2.imwrite(out_path("topdown.jpg"), render_court(spec_of(cals[1]))[0])
        print("-> out/topdown.jpg")
        return
    for n, cal in cals.items():
        if "segments" not in cal:
            raise SystemExit(f"cam{n}: 시간별 캘리브레이션이 없습니다. 먼저 track.py를 실행하세요.")
    frames = int(args[args.index("--frames") + 1]) if "--frames" in args else 11
    canvas, geo = build(cals, frames=frames)
    if "--lines" in args:
        draw_model_lines(canvas, geo)
    cv2.imwrite(out_path("topdown_photo.jpg"), canvas)
    print("-> out/topdown_photo.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
