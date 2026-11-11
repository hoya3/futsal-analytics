"""합성 2차원 코트 지도를 만들고, 필요하면 시간별 호모그래피를 카메라 화면에서 확인한다.

사용법: python view_calib.py                       합성 2차원 코트 지도  -> out/topdown.jpg
        python view_calib.py --t 1000              (확인용) 그 시각의 두 카메라 + 코트 선  -> out/view_t1000.jpg
        python view_calib.py --start 5 --end 30    (확인용) 클립 저장  -> out/view_5_30.mp4
        python view_calib.py --play [--start 5]    (확인용) 실시간 재생 (space 일시정지, a/d -/+10 s, z/c -/+1 s, q 종료)
"""
import argparse
import time

import cv2
import numpy as np

from calib_core import draw_overlay, out_path
from court import spec_of
from court_map import homographies_at, load
from topdown import main as build_clean_map

PANEL = (960, 540)


class Reader:
    """시각으로 프레임을 읽는다. 조금 앞으로 가는 요청은 탐색 없이 이어 읽어 재생을 빠르게 한다."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
        self.pos = None
        self.frame = None

    def at(self, t):
        if self.pos is None or not 0 <= t - self.pos <= 1.0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, self.frame = self.cap.read()
            self.pos = t if ok else None
            return self.frame if ok else None
        ok, moved = True, False
        while ok and self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 < t:
            ok, moved = self.cap.grab(), True
        if ok and moved:
            ok, self.frame = self.cap.retrieve()
        self.pos = t if ok else None
        return self.frame if ok else None


def segment_at(cal, t):
    for seg in cal["segments"]:
        if seg["t0"] <= t <= seg["t1"]:
            return seg
    return None


class View:
    def __init__(self):
        self.cals = {n: load(n) for n in (1, 2)}
        for n, cal in self.cals.items():
            if "segments" not in cal:
                raise SystemExit(f"cam{n}: 시간별 캘리브레이션이 없습니다. 먼저 track.py를 실행하세요.")
        self.readers = {n: Reader(cal["video"]) for n, cal in self.cals.items()}
        self.spec = spec_of(self.cals[1])
        self.duration = min(cal["segments"][-1]["t1"] for cal in self.cals.values())

    def _panel(self, n, t, frame, seg):
        cal = self.cals[n]
        if frame is None:
            view = np.zeros((1080, 1920, 3), np.uint8)
            text = f"cam{n} t={t:.1f}  no frame"
        elif seg is None or seg["source"] == "unresolved":
            view = (frame * 0.45).astype(np.uint8)
            text = f"cam{n} t={t:.1f}  UNRESOLVED (no court)"
        else:
            view = draw_overlay(frame, *homographies_at(seg, t), cal["half"], self.spec)
            shift = seg.get("shift")
            text = f"cam{n} t={t:.1f}  {seg['source']}  line score {seg['score']:.2f}" + (f"  shift ({shift[0]:+.0f},{shift[1]:+.0f}) px" if shift else "")
        view = cv2.resize(view, PANEL, interpolation=cv2.INTER_AREA)
        cv2.rectangle(view, (0, 0), (PANEL[0], 34), (0, 0, 0), -1)
        cv2.putText(view, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return view

    def compose(self, t):
        frames = {n: self.readers[n].at(t) for n in (1, 2)}
        return np.hstack([self._panel(n, t, frames[n], segment_at(self.cals[n], t)) for n in (1, 2)])


def save_clip(view, start, end, step, path):
    first = view.compose(start)
    h, w = first.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 1 / step, (w, h))
    for t in np.arange(start, end, step):
        out.write(view.compose(float(t)))
    out.release()


def play(view, start):
    win = "calibration over time (space: pause, a/d: -/+10s, z/c: -/+1s, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    t, paused, last = start, False, time.time()
    jumps = {ord("a"): -10, ord("d"): 10, ord("z"): -1, ord("c"): 1}
    while t < view.duration:
        cv2.imshow(win, view.compose(t))
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if not paused:
            t += now - last  # 처리에 걸린 시간만큼 건너뛰어 실제 속도를 유지한다
        last = now
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key in jumps:
            t = max(t + jumps[key], 0.0)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="합성 2차원 코트 지도 / 시간별 호모그래피 확인 (t는 동기화 영상 시각, 초)")
    p.add_argument("--t", type=float, help="이 시각의 사진 한 장")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, help="클립 끝 시각")
    p.add_argument("--step", type=float, default=0.1, help="클립의 시간 간격 (s)")
    p.add_argument("--play", action="store_true")
    a = p.parse_args()
    if not a.play and a.end is None and a.t is None:
        build_clean_map([])
        return
    view = View()
    if a.play:
        play(view, a.start)
    elif a.end is not None:
        path = out_path(f"view_{a.start:g}_{a.end:g}.mp4")
        save_clip(view, a.start, a.end, a.step, path)
        print("->", path)
    else:
        path = out_path(f"view_t{a.t:g}.jpg")
        cv2.imwrite(path, view.compose(a.t))
        print("->", path)


if __name__ == "__main__":
    main()
