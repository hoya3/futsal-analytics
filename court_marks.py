"""코트 표시 정의: 라인 분할(segmentation)의 클래스와 표시 좌표, 키포인트 이름과 좌표.
학습 데이터(train/make_seg_dataset.py, train/make_dataset.py)와 추론(seg_init.py, keypoint_init.py)이 같은 정의를 쓴다.
여기 두는 이유: 추론 모듈이 train/ 폴더를 sys.path에 넣지 않고 바로 불러오게.

클래스 (동쪽을 바라보는 카메라 기준, make_dataset.keypoints와 같은 이름 규칙: near = 카메라 쪽 x<0, far = x>0,
left = y>0, right = y<0. 좌우 반전 배치(1103)는 canonical 뒤의 이름):
  0 background, 1 halfway_line, 2 center_circle, 3 sideline_left, 4 sideline_right, 5 far_goal_line,
  6 near_goal_line, 7 far_penalty_area, 8 near_penalty_area, 9 spots, 255 = 무시(ignore).
"""
import math

import numpy as np

from court import DEFAULT_SPEC, _arc

NAMES = ["background", "halfway_line", "center_circle", "sideline_left", "sideline_right", "far_goal_line",
         "near_goal_line", "far_penalty_area", "near_penalty_area", "spots"]
CLS = {n: i for i, n in enumerate(NAMES)}
IGNORE = 255
FLIP_CLASS = {CLS["sideline_left"]: CLS["sideline_right"], CLS["sideline_right"]: CLS["sideline_left"]}
STEP_M = 0.25  # 선을 이 길이 이하 조각으로 나눠 투영한다 (원근 + 하프 전환)


def markings(spec=DEFAULT_SPEC):
    """(이름, 클래스, 모양, 좌표). 동쪽을 바라보는 카메라 기준 코트 좌표(m). 모양 "line" = 폴리라인, "spot" = 원판 중심.
    클래스 순서대로 나열한다 (= 칠하는 순서)."""
    L2, W2, r, g, p = spec.length / 2, spec.width / 2, spec.center_r, spec.goal_w / 2, spec.penalty_r
    far_pa = np.vstack([_arc(L2, g, p, 90, 180), _arc(L2, -g, p, 180, 270)])  # (L2, g+p) -> 앞쪽 직선 -> (L2, -g-p)
    near_pa = np.vstack([_arc(-L2, -g, p, 270, 360), _arc(-L2, g, p, 0, 90)])
    return [
        ("halfway_line", 1, "line", np.array([[0.0, -W2], [0.0, W2]])),
        ("center_circle", 2, "line", _arc(0, 0, r, 0, 360, 160)),
        ("sideline_left", 3, "line", np.array([[-L2, W2], [L2, W2]])),
        ("sideline_right", 4, "line", np.array([[-L2, -W2], [L2, -W2]])),
        ("far_goal_line", 5, "line", np.array([[L2, -W2], [L2, W2]])),
        ("near_goal_line", 6, "line", np.array([[-L2, -W2], [-L2, W2]])),
        ("far_penalty_area", 7, "line", far_pa),
        ("near_penalty_area", 8, "line", near_pa),
        ("center_spot", 9, "spot", np.array([0.0, 0.0])),
        ("far_6m_spot", 9, "spot", np.array([L2 - 6.0, 0.0])),
        ("near_6m_spot", 9, "spot", np.array([-(L2 - 6.0), 0.0])),
        ("far_10m_spot", 9, "spot", np.array([L2 - 10.0, 0.0])),
        ("near_10m_spot", 9, "spot", np.array([-(L2 - 10.0), 0.0])),
    ]


def keypoints(spec=DEFAULT_SPEC):
    """(이름, 동쪽을 바라보는 카메라 기준 좌표). 사람이 볼 수 있는 코트 표시만 쓴다. 순서 = YOLO pose 라벨 순서."""
    L2, W2, r, g, p = spec.length / 2, spec.width / 2, spec.center_r, spec.goal_w / 2, spec.penalty_r
    return [
        ("halfway_left", (0.0, W2)), ("halfway_right", (0.0, -W2)),
        ("circle_halfway_left", (0.0, r)), ("circle_halfway_right", (0.0, -r)),
        ("near_10m", (-(L2 - 10.0), 0.0)), ("far_10m", (L2 - 10.0, 0.0)),
        ("far_corner_left", (L2, W2)), ("far_corner_right", (L2, -W2)),
        ("far_arc_left", (L2, g + p)), ("far_arc_right", (L2, -(g + p))),
        ("far_post_left", (L2, g)), ("far_post_right", (L2, -g)),
        ("near_penalty_front", (-(L2 - p), 0.0)),
    ]


# 좌우 반전 증강에서 서로 바뀌는 키포인트의 인덱스 (keypoints 순서)
FLIP_IDX = [1, 0, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10, 12]


def densify(line, step=STEP_M):
    """폴리라인을 step 이하 조각으로. 조각 수를 짝수로 해 대칭인 선(사이드라인)은 x=0에 꼭짓점이 오게 한다."""
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        k = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        k += k % 2
        out.append(a + (b - a) * np.linspace(0, 1, k + 1)[1:, None])
    return np.vstack(out)
