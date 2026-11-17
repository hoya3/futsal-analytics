"""클릭 없이 코트 라인 세그멘테이션 모델(court_seg.py, train/seg_best.ts)의 클래스 지도로 한 프레임의 코트를 찾는다.

keypoint_init.keypoint_calibrate와 같은 결과 dict를 내므로 keypoint_init.decide(다수결 + 흰 선 일치율 판정)와
eval/eval_autocal.py가 그대로 쓴다.

방법
  1. 클래스 지도: 확률의 argmax, 확률이 CONF_MIN 미만이면 배경. 클래스 이름은 동쪽을 바라보는 카메라 기준 좌표
     (near = 카메라 쪽 x<0, far = x>0, left = y>0, right = y<0)이고 정규(1008) 배치 기준이다. 좌우 반전 배치(mirror,
     1103)는 court_seg.predict(mirror=True)가 픽셀 위치만 되돌리고 이름은 바꾸지 않으므로 sideline_left/right를 바꿔 읽는다.
  2. 초기 행렬: 하프라인(x=0), 두 사이드라인(y=+-W/2), 먼/가까운 골라인(x=+-L/2)에 RANSAC + 최소제곱 직선을 맞춰
     교점으로 코트 점((0, +-W/2), (L/2, +-W/2) 등)을 얻는다. 센터 서클은 타원을 맞춰 하프라인과의 교점 (0, +-r),
     그리고 (먼 골라인이 없을 때) 하프라인 위의 1차원 사영으로 구한 센터 스폿과 x 방향 소실점을 잇는 직선과의 교점
     (+-r, 0)을 더한다 (어느 쪽이 +r인지는 두 경우를 다 풀어 클래스 잔차로 고른다). 이 점들로 DLT.
  3. 클래스별 ICP (ClassICP): 클래스 픽셀을 현재 행렬로 투영한 같은 클래스 코트 표시의 가장 가까운 샘플과 짝짓고,
     점-선 거리(투영 접선에 수직)를 soft_l1 최소제곱으로 줄여 행렬을 다시 푼다 (사이드라인 픽셀은 사이드라인에만,
     서클 픽셀은 서클에만). 초기값 후보(골라인 교점을 넣은/뺀 것 x 서클 (+-r, 0) 배정)마다 x 방향 축척을 바꾼
     시작점(X_SCALES)을 두고, 잔차(짝 없는 픽셀은 상한값)가 작은 것만 끝까지 맞춰 고른다. 하프라인 + 사이드라인은 x 축척을
     정하지 못해, 가까운 표시를 카메라 뒤로 보낸 엉뚱한 답(1008 cam2 + 잡음에서 가까운 쪽 12.6 m)이 생기기 때문이다.
     그다음 가까운 하프만 다시 맞춘다.
     점-점 ICP + 거리 자르기는 굵은 선 안에서 아무 곳에나 멈추고 선을 따라 미끄러지지 못해 0.5 m 넘게 틀렸다
     (1008 cam1 정답 마스크, 측정).
     먼 하프는 하프라인을 공유하는 H_near o K (refine.icp_align_far와 같은 3자유도)를 먼 쪽 클래스 픽셀에 맞춘다.
  4. 흰 선 ICP (refine): 흰 선 지도를 세그멘테이션 선 픽셀 둘레(SEG_DILATE px)로 제한해 펜스·다른 종목 선을 버리고,
     가까운 하프는 icp_align(클래스 ICP 결과의 지점을 앵커로), 먼 하프는 icp_align_far(먼 코너/골대 지점을 앵커로).
  5. 결과 행렬로 투영한 표시가 같은 클래스 픽셀에 떨어지는 비율(seg_score)이 SEG_SCORE_MIN 미만이면 실패.
     confidence = 가까운 쪽 흰 선 일치율 (keypoint_init와 같은 정의라 ACCEPT_SCORE가 그대로 적용된다).
     진단: 찾은 선과 inlier 수, 클래스 픽셀 수, 클래스 ICP 잔차, 먼 쪽 흰 선 일치율, 세그멘테이션 일치율.
"""
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from auto_init import ROT180
from calib_core import HALF_SIGN, front_facing, in_half
from court import DEFAULT_SPEC
from refine import (CIRCLE_PAD, FAR_ANCHOR_W, FAR_MARGIN, RADII, icp_align, icp_align_far, icp_align_far_free,
                    project, ridge_map, sample_model)
from track import line_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))
from make_seg_dataset import CLS, FLIP_CLASS, NAMES, densify, markings  # noqa: E402

CONF_MIN = 0.5  # 이보다 낮은 확률의 픽셀은 배경으로 본다
MIN_PIXELS = 40  # 클래스를 쓰는 최소 픽셀 수 (1080p)
MAX_PIXELS = 1500  # 클래스 ICP에 쓰는 클래스별 최대 픽셀 수 (고르게 뽑는다)
LINE_THR = 6.0  # RANSAC 직선 inlier 거리 (px). 이후 굵기에 맞춰 넓힌다
MODEL_STEP = 0.05  # 클래스 ICP의 코트 표시 샘플 간격 (m)
CLASS_ICP_GATES = (80.0, 50.0, 30.0, 20.0, 20.0, 20.0, 20.0, 20.0)  # 라운드별 짝 허용 거리 (px): 이보다 먼 픽셀은 그 라운드에서 뺀다
X_SCALES = (0.5, 0.7, 1.0, 1.4, 2.0)  # 초기값 후보마다 시험하는 x 방향 축척
QUICK_KEEP = 8  # 시작점 중 그대로의 잔차가 작은 몇 개를 짧게 맞춰 볼지
START_KEEP = 3  # 끝까지 맞춰 볼 시작점 수
CLASS_F_SCALE = 3.0  # 클래스 ICP soft_l1 척도 (px)
SEG_SCORE_MIN = 0.4  # 결과 행렬로 투영한 표시 중 같은 클래스 픽셀에 떨어지는 비율 하한. 정답 마스크(잡음 포함)의 맞는 답 0.52 이상,
# 엉뚱한 x 축척 답은 0~0.6 (겹치지만 아래쪽을 거른다). 미만이면 실패로 낸다
SEG_DILATE = 7  # 흰 선 지도를 남길 세그멘테이션 선 둘레 (px)
SEG_ANCHOR_W = 20  # 가까운 쪽 앵커 하나의 비중 (흰 선 매칭 몇 개만큼). 키포인트(100)보다 약하게: 클래스 ICP도 몇 px 틀릴 수 있다
SEG_FAR_ANCHOR_W = FAR_ANCHOR_W
SEG_RADII = RADII  # 클래스 ICP로 이미 몇 px 안에 들어오므로 refine 기본 반경
FAR_CLASSES = (CLS["far_goal_line"], CLS["far_penalty_area"])  # 먼 하프만의 표시: 있어야 먼 쪽 앵커를 만든다
LINE_OF = {CLS["halfway_line"]: ("x", 0.0), CLS["sideline_left"]: ("y", 1.0), CLS["sideline_right"]: ("y", -1.0),
           CLS["far_goal_line"]: ("x", 1.0), CLS["near_goal_line"]: ("x", -1.0)}  # 직선 클래스 -> (축, L/2 또는 W/2 배수)


# ---- 클래스 지도


def seg_labels(seg_probs, conf_min=CONF_MIN):
    """세그멘테이션 출력 -> (labels uint8 (H, W), conf float32 (H, W)). seg_probs는 (C, H, W) 확률 또는 (labels, conf).
    확률이 conf_min 미만인 픽셀은 배경(0)."""
    if isinstance(seg_probs, (tuple, list)):
        labels, conf = np.asarray(seg_probs[0]), np.asarray(seg_probs[1], np.float32)
    else:
        p = np.asarray(seg_probs)
        labels, conf = p.argmax(axis=0).astype(np.uint8), p.max(axis=0).astype(np.float32)
    labels = np.where(conf >= conf_min, labels, 0).astype(np.uint8)
    labels[labels >= len(NAMES)] = 0  # 무시(255) 등은 배경
    return labels, conf


def true_classes(labels, mirror):
    """모델 클래스 -> 동쪽 기준 실제 코트 의미. mirror면 sideline_left/right를 바꾼다 (court_seg.predict(mirror=True)는
    픽셀만 되돌리고 이름은 정규 배치 그대로라, 원래 화면의 sideline_left는 코트의 y<0 사이드라인이다)."""
    if not mirror:
        return labels
    lut = np.arange(256, dtype=np.uint8)
    for a, b in FLIP_CLASS.items():
        lut[a] = b
    return lut[labels]


def class_pixels(labels, rng=None, max_n=None):
    """{클래스: (N, 2) 픽셀 중심 (x, y)}. MIN_PIXELS 미만인 클래스는 뺀다. max_n이면 고르게 뽑는다."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for c in range(1, len(NAMES)):
        ys, xs = np.nonzero(labels == c)
        if len(xs) < MIN_PIXELS:
            continue
        pts = np.c_[xs, ys].astype(np.float64)
        if max_n and len(pts) > max_n:
            pts = pts[np.sort(rng.choice(len(pts), max_n, replace=False))]
        out[c] = pts
    return out


# ---- 이미지 직선과 타원


def fit_line(pts, rng, iters=300, thr=LINE_THR):
    """RANSAC 뒤 inlier에 전최소제곱(TLS). 굵은 선은 잔차의 퍼짐(MAD)에 맞춰 inlier 폭을 넓힌다.
    {line: (a, b, c) (a^2 + b^2 = 1), n: inlier 수, rms: px} 또는 None."""
    if len(pts) < MIN_PIXELS:
        return None
    sub = pts if len(pts) <= 3000 else pts[rng.choice(len(pts), 3000, replace=False)]
    ij = rng.integers(0, len(sub), (iters, 2))
    d = sub[ij[:, 1]] - sub[ij[:, 0]]
    length = np.linalg.norm(d, axis=1)
    ok = length > 15
    if not ok.any():
        return None
    nrm = np.c_[-d[ok, 1], d[ok, 0]] / length[ok, None]
    c = -(nrm * sub[ij[ok, 0]]).sum(1)
    counts = (np.abs(sub @ nrm.T + c) < thr).sum(0)
    k = int(counts.argmax())
    line = np.r_[nrm[k], c[k]]
    t = thr
    for _ in range(4):
        r = pts @ line[:2] + line[2]
        inl = np.abs(r) < t
        if inl.sum() < MIN_PIXELS // 2:
            return None
        q = pts[inl]
        m = q.mean(0)
        w, v = np.linalg.eigh(np.cov((q - m).T))
        n = v[:, 0]
        line = np.r_[n, -n @ m]
        r = pts @ line[:2] + line[2]
        mad = np.median(np.abs(r[np.abs(r) < t]))
        t = float(np.clip(3.0 * 1.4826 * mad, 3.0, 25.0))
    inl = np.abs(pts @ line[:2] + line[2]) < t
    rr = pts[inl] @ line[:2] + line[2]
    return {"line": line, "n": int(inl.sum()), "rms": float(np.sqrt(np.mean(rr ** 2))), "thr": t}


def intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9 * max(1.0, np.abs(p[:2]).max()):
        return None
    return p[:2] / p[2]


def ellipse_conic(e):
    """cv2.fitEllipse 결과 ((cx, cy), (w, h), 각도) -> 원뿔곡선 행렬 C (p^T C p = 0)."""
    (cx, cy), (w, h), ang = e
    a, b = max(w / 2, 1e-3), max(h / 2, 1e-3)
    t = np.radians(ang)
    M = np.array([[np.cos(t), np.sin(t), 0.0], [-np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]) @ \
        np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    return M.T @ np.diag([1 / a ** 2, 1 / b ** 2, -1.0]) @ M


def fit_ellipse(pts):
    """센터 서클 픽셀에 타원. 샘프슨 거리로 벗어난 픽셀을 두 번 걸러 다시 맞춘다. (C, inlier 수) 또는 None."""
    if len(pts) < MIN_PIXELS:
        return None
    q = pts
    C = None
    for _ in range(3):
        if len(q) < 20:
            return None
        e = cv2.fitEllipseDirect(q.astype(np.float32))
        if not np.all(np.isfinite(e[1])) or min(e[1]) < 2:
            return None
        C = ellipse_conic(e)
        P = np.c_[pts, np.ones(len(pts))]
        f = np.einsum("ij,jk,ik->i", P, C, P)
        g = 2 * (P @ C)[:, :2]
        dist = np.abs(f) / np.maximum(np.linalg.norm(g, axis=1), 1e-9)
        t = max(3.0, 3.0 * 1.4826 * np.median(dist))
        q = pts[dist < t]
    return C, len(q)


def line_conic(line, C):
    """직선과 원뿔곡선의 교점 (0~2개)."""
    a, b, c = line
    p0 = -c * np.array([a, b]) / (a * a + b * b)
    d = np.array([-b, a])
    P0, D = np.r_[p0, 1.0], np.r_[d, 0.0]
    A, B, K = D @ C @ D, 2 * D @ C @ P0, P0 @ C @ P0
    disc = B * B - 4 * A * K
    if abs(A) < 1e-15 or disc < 0:
        return []
    s = np.sqrt(disc)
    return [p0 + d * (-B + sg * s) / (2 * A) for sg in (1, -1)]


# ---- 코트 표시 (동쪽 기준 좌표)


def class_models(spec=DEFAULT_SPEC, step=MODEL_STEP):
    """{클래스: ((M, 2) 코트 표시 위의 촘촘한 점, (M, 2) 단위 접선)} (동쪽 기준). 스폿의 접선은 0 (점-점 거리)."""
    out = {}
    for _, cls, kind, geom in markings(spec):
        if kind == "spot":
            pts, tan = geom[None], np.zeros((1, 2))
        else:
            pts = densify(geom, step)
            tan = np.gradient(pts, axis=0)
            tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
        if cls in out:
            pts, tan = np.vstack([out[cls][0], pts]), np.vstack([out[cls][1], tan])
        out[cls] = (pts, tan)
    return out


def _court_points(spec):
    """초기값에 쓰는 직선 교점의 코트 좌표 (동쪽 기준). 키 = (직선 클래스, 직선 클래스)."""
    L2, W2 = spec.length / 2, spec.width / 2
    h, sl, sr, fg, ng = (CLS[n] for n in ("halfway_line", "sideline_left", "sideline_right", "far_goal_line", "near_goal_line"))
    return {(h, sl): (0.0, W2), (h, sr): (0.0, -W2), (fg, sl): (L2, W2), (fg, sr): (L2, -W2),
            (ng, sl): (-L2, W2), (ng, sr): (-L2, -W2)}


def init_candidates(pix, spec, size, rng):
    """직선/타원에서 얻은 (코트 점, 이미지 점) 목록 후보들과 진단 정보. 후보마다 DLT로 G0(코트 -> 이미지)를 만든다."""
    lines = {c: fit_line(pix[c], rng) for c in LINE_OF if c in pix}
    lines = {c: v for c, v in lines.items() if v is not None}
    info = {"lines": {NAMES[c]: v["n"] for c, v in lines.items()}}
    w, h = size
    big = max(w, h) * 3
    corr = []
    for (a, b), xy in _court_points(spec).items():
        if a in lines and b in lines:
            p = intersect(lines[a]["line"], lines[b]["line"])
            if p is not None and abs(p[0]) < big and abs(p[1]) < big:
                corr.append((xy, p))
    hw = CLS["halfway_line"]
    extra = []  # 두 가지 (+r, 0) 배정
    ell = fit_ellipse(pix[CLS["center_circle"]]) if CLS["center_circle"] in pix else None
    info["circle"] = ell[1] if ell else 0
    if ell is not None and hw in lines:
        C = ell[0]
        lh = lines[hw]["line"]
        cross = line_conic(lh, C)
        if len(cross) == 2:
            d = np.array([-lh[1], lh[0]])
            ends = {xy[1]: p for xy, p in corr if xy[0] == 0.0}
            if spec.width / 2 in ends and -spec.width / 2 in ends:  # 하프라인 방향: -W/2 끝 -> +W/2 끝
                d = ends[spec.width / 2] - ends[-spec.width / 2]
            elif d[0] > 0:  # 사이드라인이 없으면 화면 왼쪽이 y>0 (두 배치 모두)
                d = -d
            cross.sort(key=lambda p: p @ d)  # 앞쪽 = -r
            corr += [((0.0, -spec.center_r), cross[0]), ((0.0, spec.center_r), cross[1])]
            # 하프라인 위 점들(y 알려짐)로 1차원 사영 y = (a t + b) / (c t + 1) -> 센터 스폿(y=0)의 이미지
            on = [(xy[1], p) for xy, p in corr if xy[0] == 0.0]
            if len(on) >= 3:
                o = on[0][1]
                u = d / np.linalg.norm(d)
                t = np.array([(p - o) @ u for _, p in on]) / 1000.0
                y = np.array([v for v, _ in on])
                A = np.c_[t, np.ones_like(t), -y * t]
                (a1, b1, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
                if abs(a1) > 1e-9:
                    centre = o + u * (-b1 / a1) * 1000.0
                    sl, sr = CLS["sideline_left"], CLS["sideline_right"]
                    if sl in lines and sr in lines:
                        V = np.cross(lines[sl]["line"], lines[sr]["line"])  # x 방향 소실점
                    else:
                        V = np.linalg.solve(C, lh)  # 하프라인의 극점 = x 방향 무한원점의 이미지
                    l0 = np.cross(np.r_[centre, 1.0], V)
                    if np.abs(l0[:2]).max() > 1e-12:
                        l0 = l0 / np.linalg.norm(l0[:2])
                        e2 = line_conic(l0, C)
                        if len(e2) == 2:
                            r = spec.center_r
                            extra = [[((r, 0.0), e2[0]), ((-r, 0.0), e2[1])], [((r, 0.0), e2[1]), ((-r, 0.0), e2[0])]]
    info["points"] = len(corr)
    # 골라인 교점(코너)은 골라인 클래스가 틀리면(엉뚱한 선 조각) 초기값을 망치므로 넣은 것과 뺀 것을 모두 후보로 둔다
    has = {x for x in (spec.length / 2, -spec.length / 2) if any(c[0][0] == x for c in corr)}
    drops = [set()] + [{x} for x in sorted(has)] + ([has] if len(has) == 2 else [])
    cands = []
    for drop in drops:
        base = [c for c in corr if c[0][0] not in drop]
        for ext in ([[]] + extra) if extra else [[]]:
            cc = base + ext
            world = np.array([c[0] for c in cc], float).reshape(-1, 2)
            img = np.array([c[1] for c in cc], float).reshape(-1, 2)
            off = world[np.abs(world[:, 0]) > 1e-6]
            if len(cc) < 4 or len(off) < 2 or len(np.unique(world, axis=0)) < 4:
                continue
            if len(np.unique(off, axis=0)) < 2:
                continue
            G0, _ = cv2.findHomography(world, img, 0)
            if G0 is not None and np.all(np.isfinite(G0)):
                cands.append(front_facing(G0 / np.abs(G0).max()))
    return cands, info


# ---- 클래스별 ICP


class ClassICP:
    """클래스 픽셀을 같은 클래스 코트 표시에 맞춰 G(코트 -> 이미지, 동쪽 기준)를 구한다.
    짝: 픽셀마다 G로 투영한 같은 클래스 표시 샘플 중 화면에서 가장 가까운 것. 잔차: 그 샘플의 투영 접선에 수직인 거리
    (점-선 ICP: 선을 따라 미끄러지는 방향은 벌하지 않아 점-점 ICP처럼 엉뚱한 곳에 멈추지 않는다), 스폿은 점-점 거리.
    짝을 고정한 채 G의 8개 값을 soft_l1 최소제곱으로 풀고 짝을 다시 찾는 것을 반복한다. 굵은 선은 픽셀이 중심선 양쪽에
    고르게 있어 중심에 맞는다."""

    def __init__(self, pix, models):
        self.pix = {c: p for c, p in pix.items() if c in models}
        self.models = models

    def pairs(self, G, region, spec, gate=np.inf):
        """(코트 점, 접선, 이미지 점) 짝. region: "all" / "near" / "far" (짝이 된 코트 점의 x로 나눈다). gate: 최대 거리 px."""
        margin_near = spec.center_r + CIRCLE_PAD
        src, tan, dst = [], [], []
        for c, p in self.pix.items():
            m, t = self.models[c]
            if region == "near":
                if c in FAR_CLASSES:
                    continue
                keep = np.ones(len(m), bool) if c in (CLS["halfway_line"], CLS["center_circle"]) else m[:, 0] <= margin_near
            elif region == "far":
                if c in (CLS["near_goal_line"], CLS["near_penalty_area"], CLS["halfway_line"], CLS["center_circle"]):
                    continue
                keep = m[:, 0] > 0.5
            else:
                keep = np.ones(len(m), bool)
            px, depth = project(G, m)
            keep &= (depth > 0) & np.isfinite(px).all(1) & (np.abs(px) < 1e5).all(1)
            if keep.sum() == 0:
                continue
            d, idx = cKDTree(px[keep]).query(p)
            ok = d < gate
            if ok.sum() < 5:
                continue
            src.append(m[keep][idx[ok]])
            tan.append(t[keep][idx[ok]])
            dst.append(p[ok])
        if not src:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
        return np.vstack(src), np.vstack(tan), np.vstack(dst)

    @staticmethod
    def residuals(G, src, tan, dst):
        """점-선 거리 (px). 접선이 0인 짝(스폿)은 점-점 거리의 두 성분."""
        pm, _ = project(G, src)
        line = np.linalg.norm(tan, axis=1) > 0
        pt, _ = project(G, src[line] + 0.05 * tan[line])
        d = pt - pm[line]
        n = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        r_line = ((dst[line] - pm[line]) * n).sum(1)
        return np.r_[r_line, (dst[~line] - pm[~line]).ravel()]

    def fit(self, G, region, spec, gates=CLASS_ICP_GATES):
        """(G, 마지막 라운드의 |잔차| 중앙값 px, 짝 수)."""
        G = G / G[2, 2]
        res, n = np.inf, 0
        for gate in gates:
            src, tan, dst = self.pairs(G, region, spec, gate)
            if len(src) < 20:
                break

            def fun(q):
                return self.residuals(np.append(q, 1.0).reshape(3, 3), src, tan, dst)

            q = least_squares(fun, G.ravel()[:8], loss="soft_l1", f_scale=CLASS_F_SCALE, x_scale="jac", max_nfev=50).x
            G_new = np.append(q, 1.0).reshape(3, 3)
            if not np.all(np.isfinite(G_new)):
                break
            step = np.abs(project(G_new, src)[0] - project(G, src)[0]).max()
            G = G_new
            r = np.abs(fun(q))
            res, n = float(np.median(r)), len(src)
            if step < 0.05:
                break
        return front_facing(G), res, n

    def cost(self, G, spec, cap=15.0):
        """모든 클래스 픽셀의 점-선 거리(cap에서 자름) 평균 px: 초기값 후보 고르기. 짝이 없는 픽셀(그 클래스 표시가
        카메라 뒤로 간 경우 등)은 cap으로 센다: 표시 하나를 카메라 뒤로 보내 잔차를 피하는 답을 막는다."""
        total = sum(len(p) for p in self.pix.values())
        src, tan, dst = self.pairs(G, "all", spec)
        if total == 0 or len(src) == 0:
            return np.inf
        r = np.abs(self.residuals(G, src, tan, dst))
        line = np.linalg.norm(tan, axis=1) > 0
        per_px = np.r_[np.minimum(r[:line.sum()], cap),
                       np.minimum(np.hypot(*r[line.sum():].reshape(-1, 2).T), cap) if (~line).any() else []]
        return float((per_px.sum() + cap * (total - len(dst))) / total)

    def fit_far(self, G_near, spec, gates=CLASS_ICP_GATES):
        """먼 하프 G_far = G_near K(v), K = I + v [1 0 0] (하프라인 고정)를 먼 쪽 픽셀에 맞춘다. (G_far, 잔차 px, 짝 수)."""
        def c2i(v):
            return G_near @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0]))

        v, res, n = np.zeros(3), np.inf, 0
        for gate in gates:
            src, tan, dst = self.pairs(c2i(v), "far", spec, gate)
            if len(src) < 20:
                break
            v_new = least_squares(lambda p: self.residuals(c2i(p), src, tan, dst), v, loss="soft_l1",
                                  f_scale=CLASS_F_SCALE, x_scale=[1e-3, 1e-3, 1e-4], max_nfev=50).x
            step = np.abs(project(c2i(v_new), src)[0] - project(c2i(v), src)[0]).max()
            v = v_new
            r = np.abs(self.residuals(c2i(v), src, tan, dst))
            res, n = float(np.median(r)), len(src)
            if step < 0.05:
                break
        return c2i(v), res, n


# ---- 앵커 (클래스 ICP 결과에서 뽑은 지점)


def near_anchor_points(classes, spec):
    """가까운 쪽 앵커로 쓸 코트 점 (동쪽 기준): 실제로 보인 클래스 위의 지점만."""
    L2, W2, r, g, p = spec.length / 2, spec.width / 2, spec.center_r, spec.goal_w / 2, spec.penalty_r
    pts = []
    h = CLS["halfway_line"] in classes
    if h and CLS["sideline_left"] in classes:
        pts.append((0.0, W2))
    if h and CLS["sideline_right"] in classes:
        pts.append((0.0, -W2))
    if CLS["center_circle"] in classes:
        pts += [(0.0, r), (0.0, -r), (-r, 0.0)]
    if CLS["near_penalty_area"] in classes:
        pts += [(-(L2 - p), g), (-(L2 - p), -g)]
    if CLS["near_goal_line"] in classes:
        if CLS["sideline_left"] in classes:
            pts.append((-L2, W2))
        if CLS["sideline_right"] in classes:
            pts.append((-L2, -W2))
    return np.array(pts, float).reshape(-1, 2)


def far_anchor_points(classes, spec):
    """먼 쪽 앵커 (동쪽 기준): 먼 골라인이 보이면 코너와 골대, 먼 페널티 에어리어가 보이면 그 앞쪽 끝."""
    L2, W2, g, p = spec.length / 2, spec.width / 2, spec.goal_w / 2, spec.penalty_r
    pts = []
    if CLS["far_goal_line"] in classes:
        pts += [(L2, g), (L2, -g)]
        if CLS["sideline_left"] in classes:
            pts.append((L2, W2))
        if CLS["sideline_right"] in classes:
            pts.append((L2, -W2))
    if CLS["far_penalty_area"] in classes:
        pts += [(L2 - p, g), (L2 - p, -g)]
    return np.array(pts, float).reshape(-1, 2)


def _seen_spots(pix, G, spec, gate, min_px=3):
    """먼 하프의 스폿(6 m, 10 m) 중 G로 투영한 자리 gate px 안에 스폿 픽셀이 min_px개 이상 있는 것 (동쪽 기준)."""
    L2 = spec.length / 2
    far = np.array([[L2 - 6.0, 0.0], [L2 - 10.0, 0.0]])
    if pix is None:
        return np.empty((0, 2))
    px, depth = project(G, far)
    ok = [depth[i] > 0 and (np.linalg.norm(pix - px[i], axis=1) < gate).sum() >= min_px for i in range(len(far))]
    return far[np.array(ok, bool)]


def _visible(G, pts, size, pad=5):
    px, depth = project(G, pts)
    w, h = size
    ok = (depth > 0) & (px[:, 0] >= pad) & (px[:, 0] < w - pad) & (px[:, 1] >= pad) & (px[:, 1] < h - pad)
    return pts[ok], px[ok]


def seg_line_mask(labels, dilate=SEG_DILATE):
    """세그멘테이션 선 픽셀(배경 아닌 모든 클래스)을 dilate px 넓힌 마스크 (float32 0/1)."""
    m = (labels > 0).astype(np.uint8)
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return m.astype(np.float32)


def seg_score(labels, G, spec, models, size):
    """클래스 ICP 행렬로 투영한 코트 표시 샘플 중 같은 클래스 픽셀(3 px 안)에 떨어지는 비율 (진단용)."""
    hit, tot = 0, 0
    for c, (m, _) in models.items():
        if c == CLS["spots"]:
            continue
        mask = cv2.dilate((labels == c).astype(np.uint8), np.ones((7, 7), np.uint8))
        _, px = _visible(G, m[::4], size)
        if len(px):
            hit += int(mask[px[:, 1].astype(int), px[:, 0].astype(int)].sum())
            tot += len(px)
    return hit / tot if tot else 0.0


def _failed(reason, **extra):
    return {"status": "failed", "reason": reason, "H_near": None, "H_far": None, "n_near": 0, "n_far": 0,
            "confidence": 0.0, **extra}


def seg_calibrate(frame, half, spec=DEFAULT_SPEC, seg_probs=None, model=None, mirror=False, anchor_w=None,
                  far_anchor_w=None, radii=None, far_model="shared", dilate=SEG_DILATE, seed=0):
    """세그멘테이션으로 한 프레임의 코트를 찾는다. keypoint_init.keypoint_calibrate와 같은 dict:
    {status, H_near, H_far, n_near, n_far, confidence, _G, _size, flipped} + 진단(reason, lines, classes, ...).
    seg_probs: (C, H, W) 확률 또는 (labels, conf). None이면 court_seg.predict(frame, model, mirror)로 구한다.
    mirror: 좌우 반전 배치(calibration의 mirror). 모델 클래스의 좌우 사이드라인 의미를 바꾼다.
    anchor_w / far_anchor_w / radii: 가까운 앵커 비중, 먼 앵커 비중, ICP 반경 (None이면 SEG_* 기본값)."""
    anchor_w = SEG_ANCHOR_W if anchor_w is None else int(round(anchor_w))
    far_anchor_w = SEG_FAR_ANCHOR_W if far_anchor_w is None else float(far_anchor_w)
    radii = tuple(SEG_RADII if radii is None else radii)
    size = (frame.shape[1], frame.shape[0])
    if seg_probs is None:
        import court_seg
        seg_probs = court_seg.predict(frame, model, mirror=mirror)
    labels, _ = seg_labels(seg_probs)
    if labels.shape != frame.shape[:2]:
        labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    labels = true_classes(labels, mirror)
    rng = np.random.default_rng(seed)
    pix_all = class_pixels(labels)
    classes = {NAMES[c]: len(p) for c, p in pix_all.items()}
    diag = {"classes": classes, "flipped": False}
    if CLS["halfway_line"] not in pix_all:
        return _failed("하프라인 픽셀 없음", **diag)
    cands, info = init_candidates(pix_all, spec, size, rng)
    diag.update(info)
    if not cands:
        return _failed("초기 행렬을 만들 지점 부족 (하프라인 + 사이드라인 + 먼 골라인 또는 센터 서클 필요)", **diag)

    models = class_models(spec)
    pix = {c: (p if len(p) <= MAX_PIXELS else p[np.sort(rng.choice(len(p), MAX_PIXELS, replace=False))])
           for c, p in pix_all.items()}
    cicp = ClassICP(pix, models)
    # 하프라인 + 사이드라인 + 서클은 x 방향 축척을 거의 못 정하므로(가까운 쪽 표시를 카메라 뒤로 보내는 답이 생긴다)
    # 후보마다 x 축척을 바꾼 시작점도 짧게 맞춰 보고, 잔차가 작은 START_KEEP개만 끝까지 맞춘다
    starts = [G0 @ np.diag([s, 1.0, 1.0]) for G0 in cands for s in X_SCALES]
    raw = sorted(range(len(starts)), key=lambda k: cicp.cost(starts[k], spec))[:QUICK_KEEP]
    quick = sorted(((cicp.cost(cicp.fit(starts[k], "all", spec, gates=CLASS_ICP_GATES[:2])[0], spec), k)
                    for k in raw), key=lambda f: f[0])
    fits = []
    for _, k in quick[:START_KEEP]:
        G1, _, _ = cicp.fit(starts[k], "all", spec)
        G2, _, _ = cicp.fit(G1, "near", spec)
        fits.append((cicp.cost(G2, spec), G1))
    cost, G_all = min(fits, key=lambda f: f[0])
    if not np.isfinite(cost):
        return _failed("클래스 ICP 실패", **diag)
    G_near_e, res_near, n_pairs = cicp.fit(G_all, "near", spec)
    diag.update({"class_cost_px": round(cost, 2), "class_res_px": round(res_near, 2), "class_pairs": n_pairs})
    if n_pairs < 20:
        return _failed("가까운 하프 클래스 짝 부족", **diag)

    R = np.diag([HALF_SIGN[half], HALF_SIGN[half], 1.0])  # 동쪽 기준 <-> 실제 코트 (자기 역행렬)
    present = set(pix)
    ridge_full = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    ridge = ridge_full * seg_line_mask(labels, dilate)
    model_pts = sample_model(spec)
    near_pts = model_pts[in_half(model_pts, half, "near", margin=spec.center_r + CIRCLE_PAD)]
    aw_e, ai = _visible(G_near_e, near_anchor_points(present, spec), size)
    H0 = np.linalg.inv(G_near_e @ R)
    H_near, near_ratio = icp_align(ridge, H0 / H0[2, 2], near_pts, aw_e @ R[:2, :2], ai, anchor_w=anchor_w, radii=radii)

    far_pts = model_pts[in_half(model_pts, half, "far", margin=FAR_MARGIN)]
    fw_w, fi = np.empty((0, 2)), np.empty((0, 2))
    far_res = None
    G_near_fit_e = front_facing(np.linalg.inv(H_near)) @ R  # 흰 선으로 다듬은 가까운 하프에서 먼 쪽을 맞춘다
    spots = _seen_spots(pix.get(CLS["spots"]), G_near_fit_e, spec, gate=25.0)
    if present & set(FAR_CLASSES) or len(spots):
        G_far_e, far_res, _ = cicp.fit_far(G_near_fit_e, spec)
        spots = _seen_spots(pix.get(CLS["spots"]), G_far_e, spec, gate=6.0)
        fw_e, fi = _visible(G_far_e, np.r_[far_anchor_points(present, spec), spots], size)
        fw_w = fw_e @ R[:2, :2]
    H_far, far_ratio = icp_align_far(ridge, H_near, far_pts, radii=radii, anchor_world=fw_w, anchor_img=fi,
                                     anchor_w=far_anchor_w)
    if far_model == "free":
        from refine import FREE_FAR_MARGIN
        free_pts = far_pts[in_half(far_pts, half, "far", margin=FREE_FAR_MARGIN)]
        H_far, far_ratio = icp_align_far_free(ridge, H_near, free_pts, H_init=H_far, radii=radii, anchor_world=fw_w,
                                              anchor_img=fi, anchor_w=far_anchor_w, spec=spec)

    G_near = front_facing(np.linalg.inv(H_near))
    G_east = front_facing(np.linalg.inv((np.eye(3) if half == "east" else ROT180) @ H_near))
    far_only = model_pts[in_half(model_pts, half, "far")]
    agree = seg_score(labels, G_east, spec, models, size)
    if agree < SEG_SCORE_MIN:
        return _failed(f"세그멘테이션과 불일치 (클래스 일치율 {agree:.2f} < {SEG_SCORE_MIN})", seg_score=round(agree, 3), **diag)
    return {"status": "ok", "H_near": H_near.tolist(), "H_far": H_far.tolist(), "n_near": len(ai), "n_far": len(fi),
            "confidence": round(line_score(ridge_full, G_near, near_pts), 3),
            "far_score": round(line_score(ridge_full, front_facing(np.linalg.inv(H_far)), far_only), 3),
            "seg_score": round(agree, 3),
            "near_ratio": round(float(near_ratio), 3), "far_ratio": round(float(far_ratio), 3),
            "far_class_res_px": None if far_res is None else round(far_res, 2),
            "_G": G_east, "_size": size, **diag}


# ---- 영상: 여러 시점 + 다수결 (calib_service.find_court가 쓴다)

SEG_FAR_MODEL = "free"  # 영상 자동 보정의 먼 하프 모델. eval_autocal(실제 모델, 1008/1103 4대): 통과 결과의 먼 쪽 오차가
# shared보다 같거나 작다 (1008 cam2 0.47 -> 0.32 m, 1103 cam2 0.11 -> 0.06 m), 자동 통과 100%와 틀린 통과 0회는 그대로


def _plain(x):
    """numpy 값이 섞인 진단 dict -> JSON으로 쓸 수 있는 값 (세션 파일에 저장한다)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def seg_calibrate_video(source, half, spec=DEFAULT_SPEC, times=None, n_windows=5, model=None, mirror=None,
                        far_model=SEG_FAR_MODEL, log=lambda *_: None):
    """여러 시점(선수를 지운 중앙값 프레임)에서 seg_calibrate를 돌리고 keypoint_init.decide로 판정한다.
    keypoint_init.keypoint_calibrate_video와 같은 형식에 mirror, far_model을 더해 돌려준다 (method "segmentation").
    mirror None: 좌우 배치를 모르면 처음 풀리는 시점에서 두 배치를 다 풀어 클래스 일치율이 높은 쪽으로 정하고 이후 시점은
    그 배치만 쓴다. 틀린 배치는 클래스 일치율이 낮아 실패한다 (실제 모델, 4대: 맞는 배치 0.78 이상, 틀린 배치 0.34 이하)."""
    import court_seg
    from auto_init import median_of
    from keypoint_init import decide

    model = model or court_seg.load_seg_model()
    if times is None:
        times = np.linspace(source.duration * 0.15, source.duration * 0.9, n_windows)
    results = []
    for t in times:
        frame = median_of(source, float(t))
        if frame is None:
            r = {"status": "failed", "reason": "프레임을 읽을 수 없음", "H_near": None, "confidence": 0.0}
        elif mirror is None:
            tries = [dict(seg_calibrate(frame, half, spec, model=model, mirror=m, far_model=far_model), mirror=m)
                     for m in (False, True)]
            r = max(tries, key=lambda x: (x["status"] == "ok", x.get("seg_score") or 0.0))
            if r["status"] == "ok":
                mirror = r["mirror"]
                log(f"  좌우 배치: {'반전(mirror)' if mirror else '기본'} "
                    f"(클래스 일치율 {tries[0].get('seg_score')} / 반전 {tries[1].get('seg_score')})")
        else:
            r = dict(seg_calibrate(frame, half, spec, model=model, mirror=mirror, far_model=far_model), mirror=mirror)
        r.setdefault("confidence", 0.0)
        r["t_sec"] = float(t)
        log(f"  t={float(t):8.1f}: {r['status']} 일치율={r['confidence']:.2f} 클래스 일치율={r.get('seg_score')}"
            + (f" ({r['reason']})" if r.get("reason") else ""))
        results.append(r)
    lead, members, status = decide(results, spec)
    public = [_plain({k: v for k, v in r.items() if not k.startswith("_") and not k.startswith("H_")}) for r in results]
    base = {"method": "segmentation", "windows": len(results), "details": public, "far_model": far_model,
            "mirror": bool(mirror) if mirror is not None else None}
    if lead is None:
        return {**base, "status": "failed", "votes": 0, "H_near": None, "H_far": None, "confidence": 0.0}
    best = _plain({k: v for k, v in results[lead].items() if not k.startswith("_")})
    return {**best, **base, "status": status, "votes": len(members), "mirror": bool(results[lead]["mirror"])}
