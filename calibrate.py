"""마우스만으로 하는 코트 캘리브레이션: 카메라당 5번 클릭, 모두 카메라 쪽(가까운) 하프에서.

호모그래피 하나로는 두 하프를 다 맞출 수 없으므로(카메라가 낮아서 야외 코트의 약간의 굴곡에도 먼 하프가
어긋남) 카메라마다 두 개를 둔다
  H_near: 이미지 px -> 코트 m, 카메라 쪽 하프(선수를 추적하는 영역)에서 유효 - 5번의 클릭으로 구함
  H_far:  이미지 px -> 코트 m, 하프라인 너머에서 유효(다른 카메라가 못 보는 곳을 채움).
          처음엔 H_near와 같고, refine.py가 하프라인 위의 모든 점은 H_near가 놓은 자리에 고정한 채
          흰 선에 맞추므로, 두 호모그래피가 하프라인 전체에서 일치한다.

사용법: python calibrate.py camera1.mp4 camera2.mp4      (이미지도 가능; "-"는 해당 카메라 건너뜀, 예: - camera2.mp4)
  t=60 s에서 시작(영상 처음 ~30 s 동안 폰이 움직였음). 랜드마크를 아무도 가리지 않는 프레임을 고른다:
  a/d = -/+ 10 s, z/c = -/+ 1 s (첫 클릭 전에만).
  좌/우는 화면에 보이는 기준. 미니맵(왼쪽 아래)에 다음 점이 빨간색으로 표시된다.
  "half" = 카메라가 바라보는 골대: cam1은 동쪽 골대(x=+20), cam2는 서쪽 골대(x=-20); h로 뒤집는다.
키: 왼쪽 클릭 = 빨간 점 찍기, u = 되돌리기, h = 방향 뒤집기(첫 클릭 전), m = 도움말 숨기기/보이기,
    Enter = 확정, q = 종료.
카메라별 출력: out/H_cam<N>.json, out/frame_cam<N>.jpg, out/H_cam<N>_overlay.jpg. 그다음 refine.py를 실행한다.
"""
import json
import os
import sys
from dataclasses import asdict

import cv2
import numpy as np

from calib_core import Source, click_landmarks, draw_overlay, solve_half
from court import DEFAULT_SPEC, CourtSpec, court_lines


class Calibrator:
    def __init__(self, frame, half, spec=DEFAULT_SPEC):
        self.frame = frame
        self.half = half
        self.spec = spec
        self.lms = click_landmarks(spec)
        self.t = 0.0
        self.show_map = True
        self.mouse = None  # 돋보기용 마지막 커서 위치
        self.clicks = []
        self.result = None  # {"near": (H, world, err), "far": (H, world, err)}

    @property
    def done(self):
        return len(self.clicks) == len(self.lms)

    def _solve(self):
        if not self.done:
            return
        idx = list(range(len(self.lms)))
        H, world, err = solve_half(self.clicks, self.lms, self.half, self.spec)
        near = (H, dict(zip(idx, world)), dict(zip(idx, err)))
        far_idx = [i for i, lm in enumerate(self.lms) if "far" in lm["halves"]]  # 먼 하프 피팅의 앵커
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
        x0, w, h = 20, 120, 240  # 카메라가 보는 방향의 코트 전체: 먼 골대가 위, 카메라 쪽 골대가 아래
        L, W = self.spec.length, self.spec.width
        y0 = view.shape[0] - h - 110
        cv2.rectangle(view, (x0 - 10, y0 - 10), (x0 + 640, y0 + h + 100), (0, 0, 0), -1)

        def to_map(x, y):
            return int(x0 + (W / 2 - y) / W * w), int(y0 + (L / 2 - x) / L * h)

        for line in court_lines(self.spec):
            cv2.polylines(view, [np.array([to_map(x, y) for x, y in line], np.int32)], False, (130, 130, 130), 1)
        i_next = len(self.clicks)
        for i, lm in enumerate(self.lms):
            px, py = to_map(*lm["xy"])
            color = (0, 200, 0) if i < i_next else (0, 0, 255) if i == i_next else (200, 200, 200)
            cv2.circle(view, (px, py), 8, color, -1)
            cv2.putText(view, str(i + 1), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tx = x0 + w + 25
        if self.done:
            msg = "Enter: accept   u: undo"
        else:
            msg = f"click {i_next + 1}/{len(self.lms)}: {self.lms[i_next]['name']}"
        cv2.putText(view, f"facing the {self.half} goal", (tx, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, msg, (tx, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if not self.done and self.lms[i_next]["kind"] == "line":
            cv2.putText(view, "(anywhere ON the line - no marking needed)", (tx, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(view, f"t={self.t:.0f}s  a/d: -/+10s  z/c: -/+1s (before clicking)", (tx, y0 + h + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    def render(self):
        if self.result:
            view = draw_overlay(self.frame, self.result["near"][0], self.result["far"][0], self.half, self.spec)
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
        """커서 아래 영역을 확대해 화면 반대편에 보여준다. 먼 코너도 픽셀 단위로 클릭할 수 있게."""
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
        bx = 10 if x > w // 2 else w - 10 - big.shape[1]  # 커서 반대편, 먼 코트 끝선 아래
        view[200:200 + big.shape[0], bx:bx + big.shape[1]] = big

    def to_json(self, video=None):
        return {
            "half": self.half, "video": video, "t_sec": self.t, "spec": asdict(self.spec),
            "names": [lm["name"] for lm in self.lms],
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


def run_camera(n, path, half, t=60.0, spec=DEFAULT_SPEC):  # 처음 ~30 s 동안은 두 폰 모두 아직 조정 중이었음
    src = Source(path)
    cal = Calibrator(src.frame_at(t), half, spec)
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
    args = sys.argv[1:]
    spec = DEFAULT_SPEC
    if "--court" in args:  # 예: --court 38x18 (길이x폭, 미터)
        i = args.index("--court")
        spec = CourtSpec(*map(float, args[i + 1].lower().split("x")))
        del args[i:i + 2]
    for n, (path, half) in enumerate(zip(args[:2], ("east", "west")), start=1):
        if path == "-":  # 이 카메라의 기존 캘리브레이션 유지
            continue
        if not run_camera(n, path, half, spec=spec):
            break
