"""Plot teammates' player detections on the top-down court.

Input CSV (one row per detected player per frame), pixel coordinates of the player's FEET in that camera's video:
    cam,t,id,x,y
    1,61.20,7,812.5,640.0
    2,61.20,3,1102.0,415.5
  cam = 1 or 2, t = seconds in the video, id = track id, x/y = foot pixel (1920x1080).

Usage: python plot_players.py detections.csv [--t 61.2] [--bg out/topdown.jpg]
  --t: only rows within 0.1 s of this time (one moment); default = all rows (paths).
Writes out/players.jpg (dots on the court) and out/players_court.csv (same rows + X_m, Y_m court meters).
"""
import csv
import sys

import cv2
import numpy as np

from court import L, W
from court_map import load, pixel_to_court
from topdown import S

COLORS = [(66, 135, 245), (60, 200, 60), (245, 90, 60), (200, 60, 200), (0, 200, 220), (240, 240, 60), (255, 255, 255)]


def to_canvas(xy):
    return int((xy[0] + L / 2) * S), int((W / 2 - xy[1]) * S)


def main(path, t=None, bg="out/topdown.jpg"):
    rows = list(csv.DictReader(open(path)))
    if t is not None:
        rows = [r for r in rows if abs(float(r["t"]) - t) < 0.1]
    cals = {n: load(n) for n in (1, 2)}
    for n in (1, 2):
        sel = [r for r in rows if int(r["cam"]) == n]
        if sel:
            xy = pixel_to_court(cals[n], [[float(r["x"]), float(r["y"])] for r in sel])
            for r, (X, Y) in zip(sel, xy):
                r["X_m"], r["Y_m"] = round(float(X), 2), round(float(Y), 2)

    canvas = cv2.imread(bg)
    canvas = (canvas * 0.6).astype(np.uint8)  # dim the background so the dots stand out
    for r in sorted(rows, key=lambda r: float(r["t"])):
        c = to_canvas((r["X_m"], r["Y_m"]))
        color = COLORS[int(r["id"]) % len(COLORS)]
        if t is None:
            cv2.circle(canvas, c, 3, color, -1)
        else:
            cv2.circle(canvas, c, 9, color, -1)
            cv2.circle(canvas, c, 9, (0, 0, 0), 1)
            cv2.putText(canvas, f"{r['id']}", (c[0] + 11, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite("out/players.jpg", canvas)
    with open("out/players_court.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} detections -> out/players.jpg, out/players_court.csv")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[a.index("--t") + 1]) if "--t" in a else None, a[a.index("--bg") + 1] if "--bg" in a else "out/topdown.jpg")
