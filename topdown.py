"""캘리브레이션을 이용해 카메라 프레임을 코트 평면(탑다운)으로 워핑한다.

사용법: python topdown.py           -> out/topdown.jpg: 각 카메라의 가까운 하프 (cam1 왼쪽, cam2 오른쪽); 카메라가
                                       못 보는 곳(바로 아래)은 다른 카메라의 먼 하프 영상으로 채운다
        python topdown.py --cam 1   -> out/topdown_cam1.jpg: 카메라 1이 매핑한 코트 전체 (가까운 + 먼 하프)
"""
import sys

import cv2
import numpy as np

from calib_core import in_half
from court import court_lines, spec_of
from court_map import load

S = 30  # 미터당 픽셀


class Canvas:
    """코트 치수에 맞는 탑다운 캔버스의 좌표 변환."""

    def __init__(self, spec):
        self.spec = spec
        L, W = spec.length, spec.width
        self.to_px = np.array([[S, 0, L / 2 * S], [0, -S, W / 2 * S], [0, 0, 1]])  # 코트 m -> 캔버스 px (북쪽이 위)
        self.size = (int(L * S), int(W * S))
        xs = (np.arange(self.size[0]) - L / 2 * S) / S
        self.grid = np.stack(np.broadcast_arrays(xs[None, :], np.zeros((self.size[1], 1))), -1).reshape(-1, 2)  # 캔버스 -> 코트 x


def paste(canvas, n, region, geo):
    cal = load(n)
    frame = cv2.imread(f"out/frame_cam{n}.jpg")
    M = geo.to_px @ np.array(cal[f"H_{region}"])
    img = cv2.warpPerspective(frame, M, geo.size)
    seen = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, geo.size) > 0  # 카메라 화면 안쪽
    mask = in_half(geo.grid, cal["half"], region).reshape(geo.size[1], geo.size[0]) & seen
    canvas[mask] = img[mask]
    return cal, mask


def label(canvas, text, left):
    x = 10 if left else canvas.shape[1] - 10 - 18 * len(text)
    cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


if __name__ == "__main__":
    geo = Canvas(spec_of(load(1)))
    w, h = geo.size
    canvas = np.zeros((h, w, 3), np.uint8)
    if "--cam" in sys.argv:
        n = int(sys.argv[sys.argv.index("--cam") + 1])
        for region in ("near", "far"):
            cal, _ = paste(canvas, n, region, geo)
        near_left = cal["half"] == "east"  # 동쪽을 보는 카메라는 가까운 하프가 서쪽(왼쪽)
        label(canvas, f"cam{n} near", near_left)
        label(canvas, f"cam{n} far", not near_left)
        out = f"out/topdown_cam{n}.jpg"
    else:
        filled = np.zeros((h, w), bool)
        for n in (1, 2):
            cal, mask = paste(canvas, n, "near", geo)
            filled |= mask
        for n in (1, 2):  # 사각지대를 다른 카메라의 먼 하프 영상으로 채우고 얇은 윤곽선으로 표시
            far = np.zeros_like(canvas)
            _, mask = paste(far, n, "far", geo)
            gap = mask & ~filled
            canvas[gap] = far[gap]
            filled |= gap
            contours, _ = cv2.findContours(gap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1)
        for n in (1, 2):
            label(canvas, f"cam{n} near", load(n)["half"] == "east")
        out = "out/topdown.jpg"
    for line in court_lines(geo.spec):
        px = np.stack([(line[:, 0] + geo.spec.length / 2) * S, (geo.spec.width / 2 - line[:, 1]) * S], axis=1)
        cv2.polylines(canvas, [px.astype(np.int32)], False, (0, 0, 255), 1)
    cv2.imwrite(out, canvas)
    print("->", out)
