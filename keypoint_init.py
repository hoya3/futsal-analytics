"""클릭 없이 코트 키포인트 모델(train/best.pt)로 한 프레임의 코트를 찾는다.

모델이 찾은 지점으로 가까운 하프 호모그래피를 바로 풀고, 그 지점을 앵커로 둔 채 흰 선 ICP(refine.py)로 다듬는다.
먼 하프는 하프라인을 공유하는 H_far = H_near o K 구조에 먼 쪽 지점을 앵커로 넣어 맞춘다.
지점 이름은 카메라에서 본 기준(make_dataset.keypoints, 동쪽 기준 좌표)이고, 실제 방향은 half로 정한다.
"""
import os
import sys

import cv2
import numpy as np

from calib_core import in_half, to_world
from court import DEFAULT_SPEC
from refine import CIRCLE_PAD, FAR_MARGIN, icp_align, icp_align_far, ridge_map, sample_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))
from make_dataset import keypoints  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train", "best.pt")
CONF_MIN = 0.5  # 이보다 낮은 키포인트는 화면에 없다고 본다
MIN_POINTS = 6  # 필요한 지점 수: 하프라인 4점은 한 직선 위라 제약이 5개뿐이므로 그 밖의 점이 2개 이상 있어야 풀린다
INIT_FAR_X = 10.0  # 초기 행렬에 쓰는 먼 쪽 지점의 하프라인으로부터 거리 상한 (m): 먼 10 m 지점까지
# 키포인트는 화면 위쪽(하프라인, 10 m 지점)에 몰려 있어 화면 아래쪽 기울기가 거의 정해지지 않는다. 앵커를 강하게 걸어
# 위쪽을 고정하고, 넓은 반경으로 아래쪽 선(가까운 사이드라인, 페널티 호)을 잡는다.
# 측정 (15개 시점): 앵커 5 / 반경 8 px부터면 cam1 가까운 사이드라인이 화면 아래에서 227 px 벗어남 -> 이 설정에서 17 px
KP_RADII = (60, 40, 25, 18, 12, 8, 5, 3, 3)
ANCHOR_W = 100  # ICP에서 키포인트 하나의 비중 (흰 선 매칭 몇 개만큼)

_model = None


def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(path)
    return _model


def detect(frame, model=None, imgsz=960):
    """{이름: (x, y)} 원본 픽셀. 신뢰도 CONF_MIN 미만은 뺀다."""
    r = (model or load_model()).predict(frame, imgsz=imgsz, conf=0.05, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return {}
    k = int(r.boxes.conf.argmax())
    xy, conf = r.keypoints.xy[k].cpu().numpy(), r.keypoints.conf[k].cpu().numpy()
    return {name: xy[i] for i, (name, _) in enumerate(keypoints()) if conf[i] >= CONF_MIN}


def keypoint_calibrate(frame, half, spec=DEFAULT_SPEC, model=None):
    """{status, H_near, H_far, n_near, n_far}. status: ok / failed(지점 부족)."""
    found = detect(frame, model)
    names = dict(keypoints(spec))
    near = [(to_world(names[k], half), p) for k, p in found.items() if names[k][0] <= 0]
    far = [(to_world(names[k], half), p) for k, p in found.items() if names[k][0] > 0]
    if len(near) + len(far) < MIN_POINTS or len({names[k][0] for k in found}) < 2:
        return {"status": "failed", "H_near": None, "H_far": None, "n_near": len(near), "n_far": len(far)}
    nw, ni = np.array([w for w, _ in near], float).reshape(-1, 2), np.array([p for _, p in near], float).reshape(-1, 2)
    # 초기값은 평면 하나: 가까운 쪽 지점만으로는 하프라인 4점이 한 직선 위라 제약이 7개뿐이라 먼 쪽 지점을 더한다.
    # 바닥이 하프라인에서 꺾여 있으므로 하프라인에서 INIT_FAR_X 안의 먼 쪽 지점만 쓴다 (먼 골라인 지점은 가까운 쪽을 비튼다)
    init = [(w, p) for w, p in far if abs(w[0]) <= INIT_FAR_X]
    aw = np.r_[nw, np.array([w for w, _ in init], float).reshape(-1, 2)]
    ai = np.r_[ni, np.array([p for _, p in init], float).reshape(-1, 2)]
    if len(aw) < MIN_POINTS:
        return {"status": "failed", "H_near": None, "H_far": None, "n_near": len(near), "n_far": len(far)}
    H0, _ = cv2.findHomography(ai, aw, 0)
    if H0 is None:
        return {"status": "failed", "H_near": None, "H_far": None, "n_near": len(near), "n_far": len(far)}

    ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    model_pts = sample_model(spec)
    near_pts = model_pts[in_half(model_pts, half, "near", margin=spec.center_r + CIRCLE_PAD)]
    H_near, _ = icp_align(ridge, H0, near_pts, nw, ni, anchor_w=ANCHOR_W, radii=KP_RADII)

    fw = np.array([w for w, _ in far], float).reshape(-1, 2)
    fi = np.array([p for _, p in far], float).reshape(-1, 2)
    far_pts = model_pts[in_half(model_pts, half, "far", margin=FAR_MARGIN)]
    H_far, _ = icp_align_far(ridge, H_near, far_pts, radii=KP_RADII, anchor_world=fw, anchor_img=fi)
    return {"status": "ok", "H_near": H_near.tolist(), "H_far": H_far.tolist(), "n_near": len(near), "n_far": len(far)}
