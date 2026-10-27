"""Official futsal court model (40 x 20 m). Origin = center spot, x = long axis, y = short axis (meters)."""
import numpy as np

L, W = 40.0, 20.0
CENTER_R = 3.0
PENALTY_R = 6.0
GOAL_W = 3.0

LANDMARKS = {
    "center": (0, 0),
    "hw_n": (0, W / 2), "hw_s": (0, -W / 2),
    "cc_n": (0, CENTER_R), "cc_s": (0, -CENTER_R),
    "cc_e": (CENTER_R, 0), "cc_w": (-CENTER_R, 0),
    "corner_ne": (L / 2, W / 2), "corner_se": (L / 2, -W / 2),
    "corner_nw": (-L / 2, W / 2), "corner_sw": (-L / 2, -W / 2),
    "post_ne": (L / 2, GOAL_W / 2), "post_se": (L / 2, -GOAL_W / 2),
    "post_nw": (-L / 2, GOAL_W / 2), "post_sw": (-L / 2, -GOAL_W / 2),
}
# penalty-area corners: arc meets the goal line, and arc meets the straight front line
for sx, tag in ((1, "e"), (-1, "w")):
    for sy, side in ((1, "n"), (-1, "s")):
        LANDMARKS[f"pa_goal_{side}{tag}"] = (sx * L / 2, sy * (GOAL_W / 2 + PENALTY_R))
        LANDMARKS[f"pa_front_{side}{tag}"] = (sx * (L / 2 - PENALTY_R), sy * GOAL_W / 2)
    LANDMARKS[f"pen_mark_{tag}"] = (sx * (L / 2 - 6.0), 0)
    LANDMARKS[f"pen2_mark_{tag}"] = (sx * (L / 2 - 10.0), 0)


def _arc(cx, cy, r, a0, a1, n=40):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def court_lines(penalty_r=PENALTY_R, goal_w=GOAL_W):
    """Polylines (Nx2 arrays, meters) of all court markings."""
    lines = [
        np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2]]),
        np.array([[0, -W / 2], [0, W / 2]]),
        _arc(0, 0, CENTER_R, 0, 360, 80),
    ]
    g = goal_w / 2
    for sx in (1, -1):
        gx = sx * L / 2
        # quarter arcs centered on each post, running from the goal line to the front line
        if sx == 1:
            top = _arc(gx, g, penalty_r, 90, 180)
            bot = _arc(gx, -g, penalty_r, 180, 270)
        else:
            top = _arc(gx, g, penalty_r, 0, 90)
            bot = _arc(gx, -g, penalty_r, 270, 360)
        lines += [top, bot, np.array([[gx - sx * penalty_r, -g], [gx - sx * penalty_r, g]])]
    return lines
