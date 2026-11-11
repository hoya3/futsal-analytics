"""화면(OpenCV 창) 없이 도는 캘리브레이션 공용 로직. GUI(calibrate.py)와 서버 처리가 함께 쓴다."""
import os
from dataclasses import asdict

import cv2
import numpy as np
from scipy.optimize import least_squares

from court import DEFAULT_SPEC, court_lines

HALF_SIGN = {"east": 1, "west": -1}

# 촬영 회차(경기)마다 카메라 배치가 달라 결과를 out/<회차>/에 따로 둔다. 회차 이름은 촬영 날짜(MMDD).
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MATCH = "1008"


def match_dir(match=None):
    """회차의 결과 폴더. 회차는 인자 > 환경변수 FUTSAL_MATCH > DEFAULT_MATCH 순으로 정한다."""
    return os.path.join(ROOT, "out", match or os.environ.get("FUTSAL_MATCH") or DEFAULT_MATCH)


def out_path(name, match=None):
    return os.path.join(match_dir(match), name)


def click_landmarks(spec=DEFAULT_SPEC, mirror=False):
    """7클릭 랜드마크. 앞의 5점은 카메라 쪽 절반(H_near), 뒤의 2점(먼 쪽 코너)은 H_far에만 쓴다.

    동쪽 골대를 바라보는 카메라 기준 코트 좌표: x = 코트 길이 방향(카메라 쪽이 x<0), y = 화면의 왼쪽.
    kind "point": 정확한 랜드마크. kind "line": 오른쪽 사이드라인(y = -W/2) 위의 아무 점; 선을 따른 위치는
    풀이로 구하므로 표시가 없어도 된다. "line" 클릭의 xy는 초기 추정값일 뿐이다.
    가까운 하프 = 점 4개 + 사이드라인 점 1개 = 호모그래피에 필요한 제약 8개와 정확히 같다. (사이드라인 점을
    하나 더 추가해도 소용없다: 하프라인 오른쪽 끝이 이미 그 사이드라인 위에 있다.)
    mirror=True: 보이는 사이드라인이 화면 왼쪽인 배치(1103 촬영). 이름의 좌우와 y 부호를 바꾼다.

    8, 9번은 선택 확인 클릭(optional, 기본은 7클릭 후 바로 확정): 가까운 하프에 여분 제약 3개를 더해, 그것 없이는 잔차가
    0이라 틀려도 드러나지 않는 10 m 지점 오클릭을 재투영 잔차로 잡는다(eval/click_sensitivity.py). 화면에 없으면 건너뛴다(None).
    기존 7클릭 파일과 순서가 같도록 뒤에 붙인다.
    """
    L2, W2 = spec.length / 2, spec.width / 2
    lms = [
        {"name": "halfway line, left end", "kind": "point", "xy": (0.0, W2), "halves": ("near", "far")},
        {"name": "center circle x halfway line, right crossing", "kind": "point", "xy": (0.0, -spec.center_r), "halves": ("near",)},
        {"name": "halfway line, right end", "kind": "point", "xy": (0.0, -W2), "halves": ("near", "far")},
        {"name": "right sideline: any point low in the picture", "kind": "line", "xy": (-0.7 * L2, -W2), "halves": ("near",)},
        {"name": "10 m spot (small white dot)", "kind": "point", "xy": (-(L2 - 10.0), 0.0), "halves": ("near",)},
        {"name": "FAR left corner", "kind": "point", "xy": (L2, W2), "halves": ("far",)},
        {"name": "FAR right corner", "kind": "point", "xy": (L2, -W2), "halves": ("far",)},
        {"name": "CHECK penalty arc front (middle of the near arc)", "kind": "point", "xy": (-(L2 - spec.penalty_r), 0.0),
         "halves": ("near",), "optional": True},
        {"name": "CHECK right sideline: a second point, far from click 4", "kind": "line", "xy": (-0.9 * L2, -W2),
         "halves": ("near",), "optional": True},
    ]
    if not mirror:
        return lms
    # 카메라가 반대쪽 모서리에 있어 보이는 사이드라인이 화면 왼쪽인 경우: 같은 순서로 찍되 좌우를 바꾼다 (코트 y 반전)
    swap = {"left": "right", "right": "left"}
    return [{**lm, "name": " ".join(swap.get(w.rstrip(","), w.rstrip(",")) + ("," if w.endswith(",") else "") for w in lm["name"].split()),
             "xy": (lm["xy"][0], -lm["xy"][1])} for lm in lms]


LANDMARKS = click_landmarks()
REQUIRED_CLICKS = sum(not lm.get("optional") for lm in LANDMARKS)  # 7
# 클릭 재투영 잔차가 이보다 크면 다시 찍기 권고. 1103(학습 회차) 측정: 정상 클릭(sigma 1 px)의 잔차 최대는 중앙값 2.3 px,
# 95% 4.9 px라 3 px면 오경보 31%. 5 px 오클릭은 하프라인 점 4.5~11.6 px, 10 m 지점(확인 클릭 있을 때) 6~9.5 px.
MISCLICK_PX = 5.0


def to_world(xy, half):
    """동쪽 기준 좌표 -> 실제 코트 좌표 (서쪽을 보는 카메라에게는 코트가 180도 돌아가 보인다)."""
    x, y = xy
    return (x, y) if half == "east" else (-x, -y)


def in_half(pts, half, region, margin=0.0):
    """코트 점이 카메라의 가까운 하프(x*sign <= margin) 또는 먼 하프(x*sign >= -margin)에 있는지의 마스크."""
    s = HALF_SIGN[half] * np.asarray(pts)[:, 0]
    return s <= margin if region == "near" else s >= -margin


def solve_half(img_pts, lms, half, spec=DEFAULT_SPEC):
    """점 클릭과 사이드라인 클릭으로 호모그래피(이미지 -> 코트)를 구한다. (H, 풀이된 월드 좌표, 오차(m))를 반환."""
    img = np.asarray(img_pts, float)
    guess = np.array([to_world(lm["xy"], half) for lm in lms])
    H0, _ = cv2.findHomography(img, guess, 0)
    line_idx = [i for i, lm in enumerate(lms) if lm["kind"] == "line"]

    def unpack(p):
        H = np.append(p[:8], 1.0).reshape(3, 3)
        world = guess.copy()
        for k, i in enumerate(line_idx):  # 사이드라인(y = lm의 y, 보통 -W/2)을 따라 이동 (동쪽 기준 좌표)
            world[i] = to_world((p[8 + k], lms[i]["xy"][1]), half)
        return H, world

    def residuals(p):
        H, world = unpack(p)
        return (cv2.perspectiveTransform(img[None].astype(np.float64), H)[0] - world).ravel()

    p0 = np.r_[(H0 / H0[2, 2]).ravel()[:8], [lms[i]["xy"][0] for i in line_idx]]
    sol = least_squares(residuals, p0, x_scale="jac")
    H, world = unpack(sol.x)
    err = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return H, world, err


def _padded(clicks, lms):
    """건너뛴 선택 클릭은 None. 클릭이 lms보다 짧으면(기존 7클릭) 나머지를 None으로 채운다."""
    out = list(clicks) + [None] * (len(lms) - len(clicks))
    missing = [lm["name"] for c, lm in zip(out, lms) if c is None and not lm.get("optional")]
    if missing:
        raise ValueError(f"필수 클릭이 빠졌습니다: {missing}")
    return out


def manual_result(clicks, lms, half, spec=DEFAULT_SPEC):
    """클릭의 호모그래피. {"near": (H, 월드 좌표, 오차), "far": ...}.
    near는 카메라 쪽 점만으로 풀고, far는 하프라인을 near와 공유한 채 먼 쪽 코너에 맞춘다. 건너뛴 선택 클릭(None)은 뺀다."""
    clicks = _padded(clicks, lms)
    near_idx = [i for i, lm in enumerate(lms) if "near" in lm["halves"] and clicks[i] is not None]
    H, world, err = solve_half([clicks[i] for i in near_idx], [lms[i] for i in near_idx], half, spec)
    near = (H, dict(zip(near_idx, world)), dict(zip(near_idx, err)))
    far_world, far_err = {}, {}
    beyond = [i for i, lm in enumerate(lms) if "near" not in lm["halves"] and clicks[i] is not None]  # 하프라인 너머의 점
    for i, lm in enumerate(lms):
        if "far" in lm["halves"] and clicks[i] is not None:
            far_world[i] = near[1][i] if i in near[1] else np.array(to_world(lm["xy"], half), float)
            far_err[i] = near[2].get(i, 0.0)
    H_far = far_from_anchors(H, [far_world[i] for i in beyond], [clicks[i] for i in beyond])
    return {"near": near, "far": (H_far, far_world, far_err)}


def manual_json(result, clicks, lms, half, spec=DEFAULT_SPEC, video=None, t=0.0, mirror=False):
    """7클릭 결과의 저장 형식 (out/<회차>/H_cam<N>.json, refine.py의 입력)."""
    return {
        "half": half, "video": video, "t_sec": t, "spec": asdict(spec), "mirror": mirror,
        "names": [lm["name"] for lm in lms],
        "img_pts": [None if c is None else list(map(float, c)) for c in _padded(clicks, lms)],
        "world": {region: {str(i): list(map(float, p)) for i, p in result[region][1].items()} for region in ("near", "far")},
        "H_near": result["near"][0].tolist(),
        "H_far": result["far"][0].tolist(),
    }


def far_from_anchors(H_near, anchor_world, anchor_img):
    """H_far = H_near o K, K = I + v [1 0 0] (하프라인 전체를 고정). v를 먼 쪽 기준점에 맞춘다. 기준점이 없으면 H_near."""
    if len(anchor_world) == 0:
        return H_near
    world, img = np.asarray(anchor_world, float), np.asarray(anchor_img, float)
    G = front_facing(np.linalg.inv(H_near))
    homog = np.c_[world, np.ones(len(world))]

    def residual(v):
        q = homog @ (G @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0]))).T
        return (q[:, :2] / q[:, 2:3] - img).ravel()

    v = least_squares(residual, np.zeros(3), x_scale=[1e-3, 1e-3, 1e-4]).x
    H = np.linalg.inv(G @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0])))
    return H / H[2, 2]


def front_facing(Hc2i):
    """호모그래피는 스케일/부호가 정해지지 않으므로, 센터 스폿 (0,0)이 카메라 앞에 오도록 맞춘다."""
    return Hc2i if Hc2i[2, 2] > 0 else -Hc2i


def draw_lines(view, H_img2court, keep, color=(0, 0, 255), thickness=2, spec=DEFAULT_SPEC):
    """keep(points)가 참인 코트 라인을 inv(H)로 투영해 그린다."""
    Hc2i = front_facing(np.linalg.inv(H_img2court))
    for line in court_lines(spec):
        dense = np.concatenate([np.linspace(a, b, 40, endpoint=False) for a, b in zip(line[:-1], line[1:])])
        q = np.c_[dense, np.ones(len(dense))] @ Hc2i.T
        ok = keep(dense) & (q[:, 2] > 0)
        px = q[:, :2] / q[:, 2:3]
        ok &= np.isfinite(px).all(1) & (np.abs(px) < 1e5).all(1)
        for run in np.split(np.arange(len(dense)), np.where(~ok)[0]):  # 연속으로 보이는 구간마다 그린다
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(view, [px[run].astype(np.int32)], False, color, thickness)


def draw_overlay(frame, H_near, H_far, half, spec=DEFAULT_SPEC):
    view = frame.copy()
    draw_lines(view, H_near, lambda p: in_half(p, half, "near"), spec=spec)
    draw_lines(view, H_far, lambda p: in_half(p, half, "far"), spec=spec)
    return view


class Source:
    """영상(탐색 가능) 또는 단일 이미지."""

    def __init__(self, path):
        self.is_video = path.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        if self.is_video:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
            self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.image = cv2.imread(path)

    def frame_at(self, t):
        if not self.is_video:
            return self.image
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None


def click_residuals_px(result, clicks, lms):
    """클릭마다 풀이 행렬로 다시 투영한 위치와의 거리 (px). 선 클릭은 풀이가 선 위로 옮긴 위치 기준(선에 수직인 차이)."""
    clicks = _padded(clicks, lms)
    out = {}
    for region in ("near", "far"):
        G = front_facing(np.linalg.inv(np.asarray(result[region][0], float)))
        for i, w in result[region][1].items():
            if i in out or clicks[i] is None or (region == "far" and i in result["near"][1]):
                continue
            q = G @ np.array([w[0], w[1], 1.0])
            out[i] = float(np.hypot(q[0] / q[2] - clicks[i][0], q[1] / q[2] - clicks[i][1]))
    return out
