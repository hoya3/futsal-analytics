"""풋살 코트 모델. 원점 = 센터 스폿, x = 긴 축, y = 짧은 축 (미터). 기본값은 공식 규격 40 x 20 m.

구장마다 크기가 다르므로 치수는 CourtSpec으로 넘긴다. 라인 구조(하프라인, 센터 서클, 페널티 호)는 표준을 가정한다.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    length: float = 40.0
    width: float = 20.0
    center_r: float = 3.0
    penalty_r: float = 6.0
    goal_w: float = 3.0


DEFAULT_SPEC = CourtSpec()
L, W = DEFAULT_SPEC.length, DEFAULT_SPEC.width
CENTER_R = DEFAULT_SPEC.center_r
PENALTY_R = DEFAULT_SPEC.penalty_r
GOAL_W = DEFAULT_SPEC.goal_w


def spec_of(cal):
    """캘리브레이션 dict의 코트 치수. spec이 없는 옛 파일은 기본 규격."""
    return CourtSpec(**cal["spec"]) if cal.get("spec") else DEFAULT_SPEC


def landmarks(spec=DEFAULT_SPEC):
    """이름 -> 코트 좌표 (미터)."""
    L2, W2, r, g, p = spec.length / 2, spec.width / 2, spec.center_r, spec.goal_w / 2, spec.penalty_r
    pts = {
        "center": (0, 0),
        "hw_n": (0, W2), "hw_s": (0, -W2),
        "cc_n": (0, r), "cc_s": (0, -r),
        "cc_e": (r, 0), "cc_w": (-r, 0),
        "corner_ne": (L2, W2), "corner_se": (L2, -W2),
        "corner_nw": (-L2, W2), "corner_sw": (-L2, -W2),
        "post_ne": (L2, g), "post_se": (L2, -g),
        "post_nw": (-L2, g), "post_sw": (-L2, -g),
    }
    # 페널티 에어리어 꼭짓점: 호와 골라인이 만나는 점, 호와 앞쪽 직선이 만나는 점
    for sx, tag in ((1, "e"), (-1, "w")):
        for sy, side in ((1, "n"), (-1, "s")):
            pts[f"pa_goal_{side}{tag}"] = (sx * L2, sy * (g + p))
            pts[f"pa_front_{side}{tag}"] = (sx * (L2 - p), sy * g)
        pts[f"pen_mark_{tag}"] = (sx * (L2 - 6.0), 0)
        pts[f"pen2_mark_{tag}"] = (sx * (L2 - 10.0), 0)
    return pts


LANDMARKS = landmarks()


def _arc(cx, cy, r, a0, a1, n=40):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def court_lines(spec=DEFAULT_SPEC):
    """모든 코트 라인의 폴리라인 (Nx2 배열, 미터)."""
    L2, W2 = spec.length / 2, spec.width / 2
    lines = [
        np.array([[-L2, -W2], [L2, -W2], [L2, W2], [-L2, W2], [-L2, -W2]]),
        np.array([[0, -W2], [0, W2]]),
        _arc(0, 0, spec.center_r, 0, 360, 80),
    ]
    g, p = spec.goal_w / 2, spec.penalty_r
    for sx in (1, -1):
        gx = sx * L2
        # 각 골포스트를 중심으로 골라인에서 앞쪽 직선까지 이어지는 1/4 원호
        if sx == 1:
            top = _arc(gx, g, p, 90, 180)
            bot = _arc(gx, -g, p, 180, 270)
        else:
            top = _arc(gx, g, p, 0, 90)
            bot = _arc(gx, -g, p, 270, 360)
        lines += [top, bot, np.array([[gx - sx * p, -g], [gx - sx * p, g]])]
    return lines
