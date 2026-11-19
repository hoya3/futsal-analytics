"""코트 라인 분할(segmentation) 학습 데이터를 시간별 호모그래피에서 자동으로 만든다.

make_dataset.py(키포인트)와 같은 프레임 선택(--step, 안정 구간만, 밀림 반영)과 같은 좌우 정규화(canonical)를 쓰고,
정답은 점 대신 코트 표시 전체를 클래스별로 칠한 마스크다. 각 표시는 실제 칠한 폭(기본 0.08 m)의 얇은 코트 다각형을
호모그래피로 투영해 칠하므로 먼 선은 가늘고 가까운 선은 두껍다. 먼 선이 사라지지 않도록 중심선을 최소 2 px(1280 폭)로 덧칠한다.

클래스 (동쪽을 바라보는 카메라 기준, make_dataset.keypoints와 같은 이름 규칙: near = 카메라 쪽 x<0, far = x>0,
left = y>0, right = y<0. 좌우 반전 배치(1103)는 canonical 뒤의 이름):
  0 background, 1 halfway_line, 2 center_circle, 3 sideline_left, 4 sideline_right, 5 far_goal_line,
  6 near_goal_line, 7 far_penalty_area, 8 near_penalty_area, 9 spots, 255 = 무시(ignore).
칠하는 순서 = 클래스 번호 순서. 겹치는 곳은 뒤의 클래스가 덮는다(서클 > 하프라인, 사이드라인 > 하프라인 끝,
골라인 > 사이드라인 코너, 페널티 에어리어 > 골라인, 스폿 > 모든 선).
무시(255): make_dataset.UNRELIABLE의 지점이 속한 표시(1008 cam2의 먼 골라인/페널티 에어리어/먼 6 m 스폿)는 클래스 대신 255.
far_model "shared"에서 H_near와 H_far가 하프라인 위치에서 어긋나면(반올림 이상) 두 투영의 차이 부분을 255로 둔다.
far_model "free"는 court_map.pixel_to_court처럼 하프라인 너머 FAR_BLEND_M까지 두 투영을 섞어 그리므로 어긋남이 없다.
코트 밖(펜스 위 등)은 표시가 없으므로 그냥 배경(0)이다.

분할: 기본은 회차 단위(eval/splits.json의 train 회차 -> train, val 회차 -> val). --split time은 make_dataset처럼
시간 블록으로 나누며, 이때 모든 회차가 학습에 들어가므로 val/잠금 회차는 거부한다.

사용법: python train/make_seg_dataset.py [--matches 1103 1008] [--step 3] [--out train/seg_dataset] [--limit 0] [--workers 4]
                                         [--split match|time] [--no-zip] [--preview 6]
출력: train/seg_dataset/{images,masks}/{train,val}, classes.json, train/court_seg_dataset.zip, train/seg_preview/*.jpg
"""
import argparse
import json
import math
import multiprocessing
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import make_dataset as md  # noqa: E402
from calib_core import HALF_SIGN, Source, front_facing, match_dir  # noqa: E402
from court import DEFAULT_SPEC, _arc, spec_of  # noqa: E402
from court_map import blend_width, homographies_at, load  # noqa: E402
from harness import _git, check_purpose, log_usage, read_splits, rel_to_root  # noqa: E402

NAMES = ["background", "halfway_line", "center_circle", "sideline_left", "sideline_right", "far_goal_line",
         "near_goal_line", "far_penalty_area", "near_penalty_area", "spots"]
CLS = {n: i for i, n in enumerate(NAMES)}
IGNORE = 255
FLIP_CLASS = {CLS["sideline_left"]: CLS["sideline_right"], CLS["sideline_right"]: CLS["sideline_left"]}
LINE_WIDTH_M = 0.08  # 칠한 선 폭 (m)
SPOT_RADIUS_M = 0.1
MIN_WIDTH_PX = 2  # 저장 해상도(1280 폭)에서 최소 선 굵기
STEP_M = 0.25  # 선을 이 길이 이하 조각으로 나눠 투영한다 (원근 + 하프 전환)
HALFWAY_TOL_PX = 1.0  # shared 모델의 하프라인에서 H_near/H_far 투영 차이가 이보다 크면 차이 부분을 무시

# 키포인트 데이터셋에서 뺀 지점(make_dataset.UNRELIABLE) -> 그 지점이 놓인 표시. 두 데이터셋이 같은 근거로 같은 곳을 뺀다.
KEYPOINT_MARKINGS = {
    "far_corner_left": {"far_goal_line"}, "far_corner_right": {"far_goal_line"},
    "far_arc_left": {"far_goal_line", "far_penalty_area"}, "far_arc_right": {"far_goal_line", "far_penalty_area"},
    "far_post_left": {"far_goal_line", "far_penalty_area", "far_6m_spot"},
    "far_post_right": {"far_goal_line", "far_penalty_area", "far_6m_spot"},
}


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


def unreliable_markings(match, cam):
    names = set()
    for kp in md.UNRELIABLE.get(match, {}).get(cam, set()):
        names |= KEYPOINT_MARKINGS.get(kp, set())
    return names


def densify(line, step=STEP_M):
    """폴리라인을 step 이하 조각으로. 조각 수를 짝수로 해 대칭인 선(사이드라인)은 x=0에 꼭짓점이 오게 한다."""
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        k = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        k += k % 2
        out.append(a + (b - a) * np.linspace(0, 1, k + 1)[1:, None])
    return np.vstack(out)


def _clip(poly, a, b, c):
    """볼록 다각형을 a*x + b*y + c >= 0 쪽으로 자른다 (Sutherland-Hodgman)."""
    if len(poly) == 0:
        return poly
    d = poly @ np.array([a, b]) + c
    out = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if d[i] >= 0:
            out.append(poly[i])
        if (d[i] >= 0) != (d[j] >= 0):
            out.append(poly[i] + (poly[j] - poly[i]) * (d[i] / (d[i] - d[j])))
    return np.array(out).reshape(-1, 2)


class Projector:
    """동쪽 기준 코트 좌표 -> 저장 해상도 픽셀. 카메라 쪽 하프(x<=0)는 H_near, 너머는 H_far.
    blend > 0 (far_model "free")이면 pixel_to_court처럼 하프라인 너머 blend m까지 두 투영을 거리에 비례해 섞는다."""

    def __init__(self, H_near, H_far, half, scale, blend=0.0):
        s = scale  # 원본 픽셀 중심 u -> 저장 해상도 (u + 0.5) * s - 0.5
        S = np.array([[s[0], 0, 0.5 * s[0] - 0.5], [0, s[1], 0.5 * s[1] - 0.5], [0, 0, 1.0]])
        R = np.diag([HALF_SIGN[half], HALF_SIGN[half], 1.0])  # 동쪽 기준 -> 실제 코트 (to_world)
        self.Gn = S @ front_facing(np.linalg.inv(np.asarray(H_near, float))) @ R
        self.Gf = S @ front_facing(np.linalg.inv(np.asarray(H_far, float))) @ R
        self.blend = blend
        self.size = None

    @staticmethod
    def _apply(G, pts):
        q = pts @ G[:, :2].T + G[:, 2]
        return q[..., :2] / q[..., 2:3], q[..., 2]

    def _mats(self, poly, force):
        """다각형에 쓸 행렬 목록: [G] 또는 섞을 때 [Gn, Gf]."""
        if force:
            return [self.Gn if force == "near" else self.Gf]
        if self.blend > 0:
            return [self.Gn, self.Gf] if poly[:, 0].max() > 0 else [self.Gn]
        return [self.Gn if poly[:, 0].mean() <= 0 else self.Gf]

    def _project(self, poly, Gs):
        if len(Gs) == 1:
            return self._apply(Gs[0], poly)[0]
        w = np.clip(poly[..., 0] / self.blend, 0.0, 1.0)[..., None]
        return (1 - w) * self._apply(self.Gn, poly)[0] + w * self._apply(self.Gf, poly)[0]

    def _min_depth(self, G):
        return 1e-3 * abs(G[2, 2])  # 센터 스폿 깊이의 1/1000보다 가까우면(사실상 카메라 뒤) 잘라낸다

    def polygon(self, poly, force=None):
        """코트 다각형(볼록) -> 픽셀 다각형. 카메라 뒤와 아주 먼 화면 밖은 잘라낸다. force = "near"/"far"면 그 행렬만."""
        Gs = self._mats(poly, force)
        for G in Gs:  # 깊이(투영의 세 번째 성분)는 코트 좌표의 일차식: 앞쪽만 남긴다
            poly = _clip(poly, G[2, 0], G[2, 1], G[2, 2] - self._min_depth(G))
        if len(poly) < 2:
            return None
        px = self._project(poly, Gs)
        w_, h_ = self.size
        for a, b, c in ((1, 0, 2 * w_), (-1, 0, 3 * w_), (0, 1, 2 * h_), (0, -1, 3 * h_)):
            px = _clip(px, a, b, c)
        return px if len(px) >= 2 else None

    def polygons(self, polys, force=None):
        """다각형 묶음 (K, V, 2). 대부분은 한 번에 투영하고, 카메라 뒤나 화면에서 아주 먼 것만 polygon()으로 잘라낸다."""
        out = [None] * len(polys)
        groups = {}
        for k, poly in enumerate(polys):
            groups.setdefault(tuple(id(G) for G in self._mats(poly, force)), []).append(k)
        w_, h_ = self.size
        for key, idx in groups.items():
            Gs = [self.Gn if i == id(self.Gn) else self.Gf for i in key]
            batch = polys[idx]
            ok = np.ones(len(idx), bool)
            for G in Gs:
                ok &= (self._apply(G, batch)[1] > self._min_depth(G)).all(1)
            px = self._project(batch, Gs)
            ok &= ((px[..., 0] > -2 * w_) & (px[..., 0] < 3 * w_) & (px[..., 1] > -2 * h_) & (px[..., 1] < 3 * h_)).all(1)
            for j, k in enumerate(idx):
                out[k] = px[j] if ok[j] else self.polygon(polys[k], force)
        return out


def _fill(mask, polys, value):
    for px in polys:
        if px is not None and len(px) >= 3:
            cv2.fillConvexPoly(mask, np.rint(px * 16).astype(np.int32), int(value), cv2.LINE_8, shift=4)


def draw_marking(mask, proj, kind, geom, value, width_m=LINE_WIDTH_M, force=None):
    """한 표시를 칠한다: 실제 폭의 코트 다각형 + 최소 굵기(MIN_WIDTH_PX)의 중심선."""
    if kind == "spot":
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        px = proj.polygon(geom + SPOT_RADIUS_M * np.c_[np.cos(ang), np.sin(ang)], force)
        if px is not None:
            _fill(mask, [px], value)
            c = np.rint(px.mean(0)).astype(int)
            cv2.circle(mask, (int(c[0]), int(c[1])), MIN_WIDTH_PX // 2, int(value), -1)
        return
    pts = densify(geom)
    a, b = pts[:-1], pts[1:]
    d = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    n = np.c_[-d[:, 1], d[:, 0]]
    h = width_m / 2
    a2, b2 = a - d * h, b + d * h  # 끝을 반 폭만큼 늘려 꺾이는 곳(코너)에 틈이 없게 한다
    quads = np.stack([a2 + n * h, b2 + n * h, b2 - n * h, a2 - n * h], axis=1)
    _fill(mask, proj.polygons(quads, force), value)
    lines = [np.rint(px * 16).astype(np.int32) for px in proj.polygons(np.stack([a, b], axis=1), force) if px is not None]
    if lines:
        cv2.polylines(mask, lines, False, int(value), MIN_WIDTH_PX, cv2.LINE_8, shift=4)


def halfway_gap_px(proj, spec=DEFAULT_SPEC):
    """하프라인 위 점들을 H_near와 H_far로 투영했을 때 화면 안에서의 최대 차이 (px, 저장 해상도)."""
    ys = np.linspace(-spec.width / 2, spec.width / 2, 41)
    line = np.c_[np.zeros_like(ys), ys]
    (pn, dn), (pf, df) = proj._apply(proj.Gn, line), proj._apply(proj.Gf, line)
    w_, h_ = proj.size
    inside = (dn > 0) & (df > 0) & (pn[:, 0] >= 0) & (pn[:, 0] < w_) & (pn[:, 1] >= 0) & (pn[:, 1] < h_)
    return float(np.linalg.norm(pn - pf, axis=1)[inside].max()) if inside.any() else 0.0


def render_mask(H_near, H_far, half, cam=1, match="", spec=DEFAULT_SPEC, size=(1280, 720), src_size=(1920, 1080),
                blend=0.0, width_m=LINE_WIDTH_M):
    """동쪽 기준 클래스의 마스크 (size[1], size[0]) uint8. 이미지 -> 코트 호모그래피(원본 해상도 src_size 기준)를 받는다.
    좌우 반전 배치는 이 결과를 canonical_mask로 뒤집는다."""
    proj = Projector(H_near, H_far, half, (size[0] / src_size[0], size[1] / src_size[1]), blend)
    proj.size = size
    mask = np.zeros((size[1], size[0]), np.uint8)
    skip = unreliable_markings(match, cam)
    for name, cls, kind, geom in markings(spec):
        value = IGNORE if name in skip else cls
        if name == "halfway_line" and blend <= 0 and halfway_gap_px(proj, spec) > HALFWAY_TOL_PX:
            # 두 하프의 투영이 하프라인에서 어긋나면 어느 쪽이 맞는지 모른다: 겹치는 부분만 클래스, 나머지는 무시
            near, far = np.zeros_like(mask), np.zeros_like(mask)
            draw_marking(near, proj, kind, geom, 1, width_m, force="near")
            draw_marking(far, proj, kind, geom, 1, width_m, force="far")
            mask[(near > 0) != (far > 0)] = IGNORE
            mask[(near > 0) & (far > 0)] = value
            continue
        draw_marking(mask, proj, kind, geom, value, width_m)
    return mask


def canonical_mask(mask):
    """좌우 반전 배치의 마스크를 1008 배치로: 화면 좌우 반전 + sideline_left/right 교환 (make_dataset.canonical과 같은 규칙)."""
    out = np.ascontiguousarray(mask[:, ::-1])
    lut = np.arange(256, dtype=np.uint8)
    for a, b in FLIP_CLASS.items():
        lut[a] = b
    return lut[out]


PALETTE = np.array([(0, 0, 0), (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255),
                    (255, 0, 0), (0, 255, 255), (128, 0, 128), (255, 255, 255)], np.uint8)  # BGR


def overlay(image, mask):
    """미리보기: 클래스 색을 반투명으로, 무시 영역은 회색."""
    view = image.copy()
    color = np.zeros_like(image)
    lab = mask < len(NAMES)
    color[lab] = PALETTE[mask[lab]]
    color[mask == IGNORE] = (128, 128, 128)
    on = mask > 0
    view[on] = (0.35 * view[on] + 0.65 * color[on]).astype(np.uint8)
    y = 18
    for i, n in enumerate(NAMES[1:], 1):
        cv2.putText(view, n, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tuple(int(v) for v in PALETTE[i]), 1)
        y += 16
    cv2.putText(view, "ignore", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
    return view


def camera_frames(job):
    """한 카메라의 프레임과 마스크를 저장한다. 다른 프로세스에서 돌 수 있도록 결과(장수, 클래스별 픽셀 수)를 돌려준다.
    영상 탐색이 프레임당 수 초 걸려도 키포인트 데이터셋(train/dataset) 이미지를 재사용하지 않는다: 그 이미지는 다른 PC(WSL)의
    디코더로 뽑아 같은 프레임인데도 색이 달라(픽셀 평균 차 18) 회차마다 색이 다른 데이터가 된다."""
    m, n, out = job["match"], job["cam"], job["out"]
    cal = load(n, match_dir(m))
    if "segments" not in cal:  # main이 미리 확인한다. SystemExit은 Pool 작업자가 잡지 못해 전체가 멈추므로 RuntimeError
        raise RuntimeError(f"{m} cam{n}: 시간별 캘리브레이션이 없습니다 (FUTSAL_MATCH={m} python track.py {n})")
    spec, mirror, blend = spec_of(cal), bool(cal.get("mirror")), blend_width(cal)
    source = Source(cal["video"])
    fw, fh = int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (md.IMAGE_WIDTH, md.IMAGE_WIDTH * fh // fw)
    times = np.arange(job["step"] / 2, source.duration, job["step"])
    if job["limit"]:
        times = times[np.linspace(0, len(times) - 1, job["limit"]).astype(int)]
    k = job["n_preview"]
    shown = sorted(set(((np.arange(k) + 0.5) * len(times) / max(k, 1)).astype(int)))
    res = {"match": m, "cam": n, "mirror": mirror, "frames": 0, "counts": {"train": 0, "val": 0},
           "pixels": {s: np.zeros(256, np.int64) for s in ("train", "val")}}
    for i, t in enumerate(times):
        t = float(t)
        seg = md.usable_segment(cal, t)
        if seg is None:
            continue
        H_near, H_far = homographies_at(seg, t)
        mask = render_mask(H_near, H_far, cal["half"], n, m, spec, size, (fw, fh), blend)
        if mirror:
            mask = canonical_mask(mask)
        if not ((mask > 0) & (mask != IGNORE)).any():
            continue
        stem = f"{m}_cam{n}_{t:07.1f}"
        s = job["split"] or md.split_of(t)
        dst = os.path.join(out, "images", s, stem + ".jpg")
        frame = source.frame_at(t)
        if frame is None:
            continue
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(out, "masks", s, stem + ".png"), mask)
        res["pixels"][s] += np.bincount(mask.ravel(), minlength=256)
        if shown and i >= shown[0]:  # 고르게 고른 시각 (그 시각이 빠졌으면 다음 프레임)
            shown = [j for j in shown if j > i]
            cv2.imwrite(os.path.join(job["preview"], stem + ".jpg"), overlay(small, mask), [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["frames"] += 1
        res["counts"][s] += 1
        if res["frames"] % 100 == 0:
            print(f"{m} cam{n}: {i + 1} / {len(times)}", flush=True)
    print(f"{m} cam{n}: {res['frames']}장", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description="코트 라인 분할 데이터셋 생성")
    p.add_argument("--step", type=float, default=3.0, help="샘플 간격 (s)")
    p.add_argument("--out", default=os.path.join(ROOT, "train", "seg_dataset"))
    p.add_argument("--limit", type=int, default=0, help="카메라당 최대 프레임 수 (시험용)")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--preview", type=int, default=0, help="미리보기 장수 (train/seg_preview/)")
    p.add_argument("--split", choices=("match", "time"), default="match",
                   help="match: splits.json의 train 회차 -> train, val 회차 -> val. time: 시간 블록 (학습 회차만 허용)")
    p.add_argument("--matches", nargs="+", default=None, help="회차 (기본: eval/splits.json의 train + val)")
    p.add_argument("--workers", type=int, default=4, help="카메라별 병렬 프로세스 수")
    a = p.parse_args()
    splits = read_splits()
    a.matches = a.matches or list(splits["train"]) + list(splits["val"])
    split_of_match, purpose = {}, {}
    for m in a.matches:
        if a.split == "time":  # 시간 블록: 모든 회차가 학습에 들어간다
            check_purpose(m, "train", splits)
            purpose[m] = "train"
        else:  # 회차 단위: val 회차는 검증에만 쓴다 (잠금 회차는 train 목적이라 거부된다)
            role = "val" if m in splits["val"] else "train"
            check_purpose(m, role, splits)
            purpose[m] = split_of_match[m] = role
    md.require_timelines(a.matches)  # 작업자 프로세스 안에서 실패하면 Pool이 멈추므로, 시작 전에 확인
    md.clear_dataset_dir(a.out, ("images", "masks", "classes.json"))
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", s))
        os.makedirs(os.path.join(a.out, "masks", s))
    preview = os.path.join(ROOT, "train", "seg_preview")
    if a.preview:
        shutil.rmtree(preview, ignore_errors=True)
        os.makedirs(preview)
    cams = [(m, n) for m in a.matches for n in (1, 2)]
    per_cam_preview = math.ceil(a.preview / len(cams)) if a.preview else 0

    jobs = [dict(match=m, cam=n, step=a.step, limit=a.limit, out=a.out, preview=preview, n_preview=per_cam_preview,
                 split=split_of_match.get(m)) for m, n in cams]
    counts = {"train": 0, "val": 0}
    frames = {}
    pixels = {s: np.zeros(256, np.int64) for s in ("train", "val")}
    meta = {m: {"split": split_of_match.get(m, "time"), "mirror": None, "frames": {}} for m in a.matches}
    if a.workers > 1:
        with multiprocessing.Pool(min(a.workers, len(jobs))) as pool:
            results = list(pool.imap_unordered(camera_frames, jobs))
    else:
        results = [camera_frames(job) for job in jobs]
    for r in sorted(results, key=lambda r: (a.matches.index(r["match"]), r["cam"])):
        m = r["match"]
        meta[m]["mirror"] = r["mirror"]
        meta[m]["frames"][f"cam{r['cam']}"] = r["frames"]
        frames[m] = frames.get(m, 0) + r["frames"]
        for s in ("train", "val"):
            counts[s] += r["counts"][s]
            pixels[s] += r["pixels"][s]

    fractions = {}
    for s, px in pixels.items():
        total = max(int(px.sum()), 1)
        fractions[s] = {n: round(100.0 * int(px[i]) / total, 4) for i, n in enumerate(NAMES)}
        fractions[s]["ignore"] = round(100.0 * int(px[IGNORE]) / total, 4)
    info = {
        "names": NAMES, "ignore_index": IGNORE, "image_size": [md.IMAGE_WIDTH, md.IMAGE_WIDTH * 9 // 16],
        "line_width_m": LINE_WIDTH_M, "spot_radius_m": SPOT_RADIUS_M, "min_width_px": MIN_WIDTH_PX,
        "canonical": "1008 layout (visible near sideline on the right); mirrored matches flipped",
        "matches": meta,
        "split": ("by match (eval/splits.json): " + ", ".join(f"{m} -> {v['split']}" for m, v in meta.items())
                  if a.split == "match" else f"time blocks of {md.VAL_BLOCK:.0f} s, 1 in 5 -> val"),
        "draw_order": "class index ascending; later classes overwrite earlier ones",
        "step_s": a.step, "pixel_percent": fractions,
        "commit": _git("rev-parse", "--short", "HEAD"),
    }
    with open(os.path.join(a.out, "classes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    for m in a.matches if not a.limit else []:  # 시험 실행(--limit)은 기록하지 않는다
        log_usage(m, purpose[m], [os.path.join(match_dir(m), f"calib_cam{n}.json") for n in (1, 2)],
                  {"dataset": rel_to_root(os.path.abspath(a.out)), "kind": "segmentation", "frames": frames.get(m, 0),
                   "step": a.step})
    print(f"학습 {counts['train']}장, 검증 {counts['val']}장, 회차별 {frames} -> {a.out}", flush=True)
    print(json.dumps(fractions, ensure_ascii=False), flush=True)
    if not a.no_zip:
        path = shutil.make_archive(os.path.join(ROOT, "train", "court_seg_dataset"), "zip", a.out)
        print(f"-> {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
