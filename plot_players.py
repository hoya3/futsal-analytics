"""팀원의 선수 검출 결과를 탑다운 코트 위에 표시한다.

입력 CSV (프레임마다 검출된 선수당 한 행), 해당 카메라 영상에서 선수 '발'의 픽셀 좌표:
    cam,t,id,x,y
    1,61.20,7,812.5,640.0
    2,61.20,3,1102.0,415.5
  cam = 1 또는 2, t = 동기화 영상(*_sync.mp4)의 시간(초), id = 트랙 id, x/y = 발 픽셀 (1920x1080).

사용법: python plot_players.py detections.csv [--t 61.2] [--bg out/topdown.jpg]
  --t: 이 시각에서 0.1 s 이내의 행만 사용(한 순간); 기본값 = 모든 행(이동 경로).
out/players.jpg (코트 위 점)와 out/players_court.csv (같은 행 + 코트 미터 X_m, Y_m)를 저장한다.
코트를 확정하지 못한 구간의 행은 X_m, Y_m이 비고 그림에서 빠진다.
"""
import csv
import math
import sys

import cv2
import numpy as np

from calib_core import out_path
from court import spec_of
from court_map import load, pixel_to_court
from topdown import Canvas

COLORS = [(66, 135, 245), (60, 200, 60), (245, 90, 60), (200, 60, 200), (0, 200, 220), (240, 240, 60), (255, 255, 255)]


def to_canvas(xy, geo):
    return tuple(int(v) for v in geo.px([xy])[0])


def main(path, t=None, bg=out_path("topdown.jpg")):
    rows = list(csv.DictReader(open(path)))
    if t is not None:
        rows = [r for r in rows if abs(float(r["t"]) - t) < 0.1]
    cals = {n: load(n) for n in (1, 2)}
    geo = Canvas(spec_of(cals[1]))
    for n in (1, 2):
        sel = [r for r in rows if int(r["cam"]) == n]
        if sel:
            xy = pixel_to_court(cals[n], [[float(r["x"]), float(r["y"])] for r in sel], t=[float(r["t"]) for r in sel])
            for r, (X, Y) in zip(sel, xy):
                known = math.isfinite(X) and math.isfinite(Y)
                r["X_m"], r["Y_m"] = (round(float(X), 2), round(float(Y), 2)) if known else ("", "")

    canvas = cv2.imread(bg)
    canvas = (canvas * 0.6).astype(np.uint8)  # 점이 잘 보이도록 배경을 어둡게
    for r in sorted(rows, key=lambda r: float(r["t"])):
        if r["X_m"] == "":
            continue
        c = to_canvas((r["X_m"], r["Y_m"]), geo)
        color = COLORS[int(r["id"]) % len(COLORS)]
        if t is None:
            cv2.circle(canvas, c, 3, color, -1)
        else:
            cv2.circle(canvas, c, 9, color, -1)
            cv2.circle(canvas, c, 9, (0, 0, 0), 1)
            cv2.putText(canvas, f"{r['id']}", (c[0] + 11, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path("players.jpg"), canvas)
    with open(out_path("players_court.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} detections -> out/players.jpg, out/players_court.csv")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[a.index("--t") + 1]) if "--t" in a else None, a[a.index("--bg") + 1] if "--bg" in a else out_path("topdown.jpg"))
