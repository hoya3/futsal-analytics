"""검증 점 라벨링: 보정에 쓰지 않은 코트 지점을 사람이 찍어 독립적인 정확도 기준을 만든다.

사용법: python eval/label_points.py --cam 1 [--times 300 1100 2000]    (t = 동기화 영상 시각, 초)
키: 왼쪽 클릭 = 찍기, s = 안 보이는 지점 건너뛰기, u = 되돌리기, Enter = 이 프레임 저장 후 다음, q = 종료
    a/d = -/+ 10 s, z/c = -/+ 1 s (첫 클릭 전에만)
  미니맵의 빨간 점이 지금 찍을 지점이다. 넓은 선은 폭의 가운데를 찍는다. 돋보기로 픽셀 단위까지 맞춘다.
  확실하지 않은 지점은 s로 건너뛴다. 잘못 찍은 점 하나가 평가를 망친다.
결과: out/<회차>/labels.json (프레임마다 저장되므로 중간에 끊어도 이어서 할 수 있다). 채점: python eval/score_points.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_init import median_of  # noqa: E402
from calib_core import Source, out_path, to_world  # noqa: E402
from calibrate import Calibrator  # noqa: E402
from court import DEFAULT_SPEC, court_lines, spec_of  # noqa: E402
from court_map import load  # noqa: E402

LABELS_PATH = out_path("labels.json")  # 회차별 (out/<FUTSAL_MATCH>/labels.json)
DEFAULT_TIMES = (300.0, 1100.0, 2000.0)


def validation_points(spec=DEFAULT_SPEC):
    """동쪽을 바라보는 카메라 기준 좌표의 검증 지점. 7클릭(하프라인 양 끝, 서클 오른쪽 교차, 가까운 10m 지점, 먼 쪽 코너)과 겹치지 않는다.
    kind "point": 정확한 지점. kind "line": 해당 선 위의 아무 점 (넓은 선은 가운데) - 선까지의 거리로 채점한다."""
    L2, W2, r, g, p = spec.length / 2, spec.width / 2, spec.center_r, spec.goal_w / 2, spec.penalty_r
    arc = np.radians(45)
    return [
        {"id": "circle_left", "name": "center circle x halfway line, LEFT crossing", "kind": "point", "xy": (0.0, r), "region": "center"},
        {"id": "far_10m", "name": "FAR 10 m spot (small white dot)", "kind": "point", "xy": (L2 - 10.0, 0.0), "region": "far"},
        {"id": "far_arc_left", "name": "FAR penalty arc x goal line, left", "kind": "point", "xy": (L2, g + p), "region": "far"},
        {"id": "far_arc_right", "name": "FAR penalty arc x goal line, right", "kind": "point", "xy": (L2, -(g + p)), "region": "far"},
        {"id": "far_post_left", "name": "FAR goal post base, left", "kind": "point", "xy": (L2, g), "region": "far"},
        {"id": "far_post_right", "name": "FAR goal post base, right", "kind": "point", "xy": (L2, -g), "region": "far"},
        {"id": "right_sideline", "name": "right sideline, low in the picture", "kind": "line", "curve": "right_sideline",
         "xy": (-0.8 * L2, -W2), "region": "near"},
        {"id": "near_arc", "name": "NEAR penalty arc", "kind": "line", "curve": "near_arc",
         "xy": (-L2 + p * np.cos(arc), g + p * np.sin(arc)), "region": "near"},
        {"id": "circle_near", "name": "center circle, side nearest the camera", "kind": "line", "curve": "circle", "xy": (-r, 0.0), "region": "center"},
        {"id": "circle_far", "name": "center circle, side farthest from the camera", "kind": "line", "curve": "circle", "xy": (r, 0.0), "region": "center"},
    ]


def curve_points(name, half, spec=DEFAULT_SPEC):
    """검증 선의 실제 코트 좌표 (촘촘한 점열)."""
    lines = court_lines(spec)
    L2, W2 = spec.length / 2, spec.width / 2
    if name == "right_sideline":
        polys = [np.array([[-L2, -W2], [L2, -W2]])]
    elif name == "near_arc":
        polys = lines[6:9]  # 동쪽 기준에서 카메라 쪽(x<0) 골대의 호와 앞쪽 직선
    elif name == "circle":
        polys = [lines[2]]
    else:
        raise ValueError(name)
    dense = np.concatenate([np.linspace(a, b, max(int(np.linalg.norm(b - a) / 0.01), 2)) for poly in polys for a, b in zip(poly[:-1], poly[1:])])
    return dense if half == "east" else -dense


def make_label(cam, t, video, half, points, clicks):
    """찍은 결과를 저장 형식으로. 건너뛴 지점(None)은 뺀다."""
    out = []
    for lm, px in zip(points, clicks):
        if px is None:
            continue
        item = {"id": lm["id"], "kind": lm["kind"], "region": lm["region"], "px": [float(px[0]), float(px[1])]}
        if lm["kind"] == "point":
            item["world"] = [float(v) for v in to_world(lm["xy"], half)]
        else:
            item["curve"] = lm["curve"]
        out.append(item)
    return {"cam": cam, "t": float(t), "video": video, "half": half, "points": out}


def read_labels(path=LABELS_PATH):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["labels"]


def write_labels(labels, path=LABELS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"labels": labels}, f, indent=1)
    os.replace(tmp, path)


class Labeler(Calibrator):
    """5클릭 창의 미니맵과 돋보기를 재사용한다. 호모그래피는 풀지 않는다."""

    def __init__(self, frame, half, spec):
        super().__init__(frame, half, spec)
        self.lms = [dict(lm, halves=()) for lm in validation_points(spec)]

    def _solve(self):
        pass

    def skip(self):
        if not self.done:
            self.clicks.append(None)

    def render(self):
        view = self.frame.copy()
        for i, c in enumerate(self.clicks):
            if c is not None:
                cv2.drawMarker(view, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(view, str(i + 1), (int(c[0]) + 10, int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.show_map:
            self._minimap(view)
        self._magnifier(view)
        return view


def label_frame(n, cal, source, t, spec):
    """프레임 한 장을 라벨링한다. 저장하면 라벨 dict, q면 None."""
    tool = Labeler(median_of(source, t), cal["half"], spec)
    tool.t = t
    win = f"validation points cam{n}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def on_mouse(ev, x, y, *_):
        tool.mouse = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            tool.click(x, y)

    cv2.setMouseCallback(win, on_mouse)
    steps = {ord("a"): -10, ord("d"): 10, ord("z"): -1, ord("c"): 1}
    while True:
        cv2.imshow(win, tool.render())
        key = cv2.waitKey(30) & 0xFF
        if key in steps and not tool.clicks:
            new_t = min(max(tool.t + steps[key], 0.0), source.duration - 15.0)
            frame = median_of(source, new_t)
            if frame is not None:
                tool.frame, tool.t = frame, new_t
        elif key == ord("s"):
            tool.skip()
        elif key == ord("u"):
            tool.undo()
        elif key == ord("m"):
            tool.show_map = not tool.show_map
        elif key == ord("q"):
            cv2.destroyWindow(win)
            return None
        elif key == 13 and tool.done:
            cv2.destroyWindow(win)
            return make_label(n, tool.t, cal["video"], cal["half"], tool.lms, tool.clicks)


def main():
    p = argparse.ArgumentParser(description="검증 점 라벨링")
    p.add_argument("--cam", type=int, required=True, choices=(1, 2))
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    a = p.parse_args()
    cal = load(a.cam)
    spec = spec_of(cal)
    source = Source(cal["video"])
    labels = read_labels()
    done = {(lb["cam"], lb["t"]) for lb in labels}
    for t in a.times:
        if (a.cam, t) in done:
            print(f"t={t:g}: 이미 라벨링됨, 건너뜀")
            continue
        label = label_frame(a.cam, cal, source, t, spec)
        if label is None:
            break
        labels.append(label)
        write_labels(labels)
        print(f"t={label['t']:g}: {len(label['points'])}개 저장 -> {LABELS_PATH}", flush=True)


if __name__ == "__main__":
    main()
