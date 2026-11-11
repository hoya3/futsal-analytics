"""앵커 프레임의 캘리브레이션을 경기 전체로 전파해 시간별 호모그래피를 만든다.

삼각대의 폰은 경기 중 조금씩 밀린다(측정: cam1은 약 10 px, 코트 위 0.65~1.5 m). 그래서 일정 간격으로 프레임을 뽑아
앵커 대비 영상 평행이동을 흰 선으로 추정하고, 이동이 같은 샘플을 구간으로 묶은 뒤 구간마다 호모그래피를 다시 맞춘다.
선이 맞지 않는 구간(촬영 시작 직후의 조정, 폰을 크게 건드린 경우)은 unresolved로 남겨 수동 재지정 대상으로 넘긴다.
구간을 나누지 않는 tol_px 안의 느린 밀림은 구간마다 서브픽셀로 따로 재서 drift로 저장한다(court_map이 시각에 맞춰 반영).

사용법: python track.py 1 2 [--step 5]
        python track.py 1 2 --drift-only   (기존 out/calib_cam<N>.json의 구간은 두고 밀림만 다시 잰다)
  입력: out/H_cam<N>_refined.json (없으면 out/H_cam<N>.json), 그 안의 video 경로
  출력: out/calib_cam<N>.json, 구간별 확인용 out/track_cam<N>_seg<K>.jpg
"""
import json
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from calib_core import Source, draw_overlay, front_facing, in_half, out_path
from court import spec_of
from court_map import load_anchor
from refine import CIRCLE_PAD, FAR_MARGIN, far_anchors, icp_align, icp_align_far, project, ridge_map, sample_model

HIT_THRESHOLD = 0.25
TRACK_RADII = (8, 5, 3, 3)
FAR_RADII = (12, 8, 5, 3, 2, 2)


@dataclass(frozen=True)
class TrackConfig:
    step: float = 5.0  # 샘플 간격 (s)
    radius: int = 40  # 직전 샘플 주변 평행이동 탐색 반경 (px)
    wide_radius: int = 120  # 놓쳤을 때 넓혀서 찾는 반경 (px)
    tol_px: float = 3.0  # 같은 구간으로 보는 이동 차이 (px)
    rel_score: float = 0.7  # 유효 샘플의 일치율 하한 (앵커 일치율 대비)
    min_score: float = 0.4  # 유효 샘플의 일치율 절대 하한
    median_frames: int = 41  # 구간 재추정에 쓰는 프레임 수: 고정된 카메라는 많이 합칠수록 먼 쪽 선이 또렷해진다
    boundary_eps: float = 0.5  # 구간 경계 시각의 허용 오차 (s)
    merge_gap: int = 2  # 양옆 이동이 같으면 메우는 무효 샘플 수
    # tol_px 안의 느린 밀림은 구간을 나누지 않는다. 먼 쪽은 1 px가 약 0.3 m라 구간 안에서 따로 잰다
    # (측정: 두 카메라 모두 경기 중 약 3 px 밀림, 반영 시 cam1 먼 쪽 검증 오차 중앙값 0.61 -> 0.44 m)
    drift_step: float = 30.0  # 구간 안 밀림 측정 간격 (s)
    drift_frames: int = 5  # 측정 시각마다 합치는 프레임 수 (선수를 지운다)
    drift_span: float = 2.0  # 그 프레임들을 뽑는 범위 (+- s)
    # 배경 특징점(펜스, 건물, 차)으로 카메라 이동을 먼저 재고 흰 선은 그 주변에서만 맞춘다. 흰 선만 보면 사이드라인을
    # 따라 미끄러진 위치도 비슷한 점수를 받는다 (1103 cam1: 실제로는 고정인데 (-23,-31), (-60,-82)로 잘못 추적,
    # 야간 조명 구간은 미끄러진 쪽 점수가 더 높음)
    use_features: bool = True
    feature_radius: int = 4  # 특징점 이동 주변 흰 선 탐색 반경 (px)
    feature_min_inliers: int = 20  # 같은 이동을 가리키는 매칭 수. 우연한 매칭은 한 점에 모이지 않는다


@dataclass
class Sample:
    t: float
    dx: float
    dy: float
    score: float
    ok: bool = False
    by: str = "lines"  # lines: 흰 선으로 확인 / features: 선은 흐리지만 배경 특징점으로 확인된 이동


FEATURE_SCALE = 0.5  # 특징점은 절반 해상도에서 찾는다
_CLAHE = cv2.createCLAHE(3.0, (8, 8))  # 밝기 평준화: 야간 조명이 켜져도 낮 프레임과 맞는다 (1103 cam1, 2845 s)


def describe(frame):
    """배경 특징점 (좌표는 원본 px, 기술자). 프레임이 없으면 None."""
    if frame is None:
        return None
    gray = _CLAHE.apply(cv2.cvtColor(cv2.resize(frame, None, fx=FEATURE_SCALE, fy=FEATURE_SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    kp, desc = cv2.SIFT_create(3000).detectAndCompute(gray, None)
    if desc is None or len(kp) < 10:
        return None
    return np.float32([k.pt for k in kp]) / FEATURE_SCALE, desc


def background_shift(a, b, min_inliers=20):
    """특징점 a -> b의 카메라 평행이동 (dx, dy) px. 배율이나 회전이 섞였거나 근거가 약하면 None."""
    if a is None or b is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(a[1], b[1], k=2)
    good = [m for m in pairs if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
    if len(good) < min_inliers:
        return None
    pa = a[0][[m[0].queryIdx for m in good]]
    pb = b[0][[m[0].trainIdx for m in good]]
    A, inl = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=3.0)
    if A is None or int(inl.sum()) < min_inliers or inl.mean() < 0.6:
        return None
    scale, rot = np.hypot(A[0, 0], A[1, 0]), np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    if abs(scale - 1) > 0.01 or abs(rot) > 0.3:  # 평행이동 모델로 다룰 수 없다: 흰 선 탐색에 맡긴다
        return None
    d = np.median(pb[inl.ravel() > 0] - pa[inl.ravel() > 0], axis=0)
    return float(d[0]), float(d[1])


def translation(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def to_ridge(frame):
    return cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)


def visible_px(Hc2i, pts, shape):
    px, depth = project(Hc2i, pts)
    h, w = shape
    return px[(depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)]


def line_score(ridge, Hc2i, pts):
    """화면 안에 들어오는 모델 샘플 중 흰 선 위에 떨어지는 비율. 샘플이 너무 적으면 0."""
    px = visible_px(Hc2i, pts, ridge.shape)
    if len(px) < 50:
        return 0.0
    return float((ridge[px[:, 1].astype(int), px[:, 0].astype(int)] > HIT_THRESHOLD).mean())


def search_shift(field, px, center, radius, step):
    """field 값의 평균이 최대인 평행이동을 center 주변 격자에서 찾는다. (dx, dy, 평균값)을 반환."""
    h, w = field.shape
    axis = np.arange(-radius, radius + 1, step)
    off = np.stack(np.meshgrid(axis + center[0], axis + center[1]), -1).reshape(-1, 2)
    q = np.rint(px[None] + off[:, None]).astype(int)
    inside = (q[..., 0] >= 0) & (q[..., 0] < w) & (q[..., 1] >= 0) & (q[..., 1] < h)
    vals = field[q[..., 1].clip(0, h - 1), q[..., 0].clip(0, w - 1)] * inside
    count = inside.sum(1)
    mean = vals.sum(1) / np.maximum(count, 1)
    mean[count < 0.6 * px.shape[0]] = -1  # 대부분 화면 밖으로 나가는 이동은 제외
    best = int(mean.argmax())
    return float(off[best, 0]), float(off[best, 1]), float(mean[best])


def estimate_shift(ridge, G, pts, center, radius):
    """앵커 호모그래피 G 대비 평행이동 추정. 넓은 흰 선은 능선이 둘로 갈라지므로 흐린 맵에서 봉우리를 찾는다."""
    field = cv2.GaussianBlur(ridge, (0, 0), 2.5)
    px = visible_px(translation(*center) @ G, pts, ridge.shape) - center
    if len(px) < 50:
        return (*center, 0.0)
    dx, dy = center
    if radius > 4:  # 넓은 반경은 4 px 격자로 먼저 찾는다
        dx, dy, _ = search_shift(field, px, center, radius, 4)
    dx, dy, _ = search_shift(field, px, (dx, dy), min(radius, 4), 1)
    return dx, dy, line_score(ridge, translation(dx, dy) @ G, pts)


def subpixel_shift(field, px, radius=4.0):
    """field 값의 평균이 최대인 평행이동 (dx, dy)를 1 -> 0.25 -> 0.05 px 격자로 좁혀 가며 찾는다. (dx, dy, 평균값)."""
    best, val = np.zeros(2), 0.0
    for rad, st in ((radius, 1.0), (1.0, 0.25), (0.25, 0.05)):
        axis = np.arange(-rad, rad + 1e-9, st)
        off = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2) + best
        q = (px[None] + off[:, None]).astype(np.float32)
        mean = cv2.remap(field, q[..., 0], q[..., 1], cv2.INTER_LINEAR, borderValue=0).mean(1)
        k = int(mean.argmax())
        best, val = off[k], float(mean[k])
    return float(best[0]), float(best[1]), val


def segment_lines_px(seg, half, model, shape, margin=8):
    """구간 행렬로 투영한 코트 선 픽셀. 가까운 하프는 H_near, 먼 하프는 H_far로 (하프라인에서 이어진다)."""
    h, w = shape
    out = []
    for region in ("near", "far"):
        px, depth = project(front_facing(np.linalg.inv(np.array(seg[f"H_{region}"]))), model[in_half(model, half, region)])
        keep = (depth > 0) & (px[:, 0] > margin) & (px[:, 1] > margin) & (px[:, 0] < w - margin) & (px[:, 1] < h - margin)
        out.append(px[keep])
    return np.concatenate(out)


def measure_drift(source, result, cfg=TrackConfig(), log=lambda *_: None):
    """확정된 구간마다 구간 행렬 대비 서브픽셀 평행이동을 drift_step 간격으로 재서 seg["drift"] = [[t, dx, dy], ...]에 넣는다.
    가까운 쪽과 먼 쪽에 같은 이동을 주므로 하프라인 연결은 유지된다. 선이 잘 안 보인 시각은 버리고, 3점 이동 중앙값으로 튐을 누른다."""
    model = sample_model(spec_of(result))
    for seg in result["segments"]:
        seg.pop("drift", None)
        if seg["source"] == "unresolved":
            continue
        times = np.arange(seg["t0"] + cfg.drift_step / 2, seg["t1"], cfg.drift_step)
        if len(times) == 0:
            times = np.array([(seg["t0"] + seg["t1"]) / 2])
        rows = []
        for t in times:
            frames = [f for f in (source.frame_at(float(x)) for x in np.clip(np.linspace(t - cfg.drift_span, t + cfg.drift_span, cfg.drift_frames), seg["t0"], seg["t1"])) if f is not None]
            if not frames:
                continue
            ridge = to_ridge(np.median(np.stack(frames), axis=0).astype(np.uint8))
            px = segment_lines_px(seg, result["half"], model, ridge.shape)
            if len(px) < 100:
                continue
            dx, dy, _ = subpixel_shift(cv2.GaussianBlur(ridge, (0, 0), 1.5), px)
            hit = float((ridge[np.rint(px[:, 1] + dy).astype(int).clip(0, ridge.shape[0] - 1),
                               np.rint(px[:, 0] + dx).astype(int).clip(0, ridge.shape[1] - 1)] > HIT_THRESHOLD).mean())
            if hit >= cfg.min_score:
                rows.append([float(t), dx, dy])
        if rows:
            d = np.array(rows)
            if len(d) >= 3:
                for c in (1, 2):
                    d[1:-1, c] = np.median(np.stack([d[:-2, c], d[1:-1, c], d[2:, c]]), axis=0)
            seg["drift"] = d.round(3).tolist()
            log(f"  [{seg['t0']:8.1f} ~ {seg['t1']:8.1f}] 밀림 {len(d)}점: dx {d[:, 1].min():+.1f}..{d[:, 1].max():+.1f}, "
                f"dy {d[:, 2].min():+.1f}..{d[:, 2].max():+.1f} px")
    return result


def sample_at(source, t, G, pts, center, cfg, floor, frame=None, radius=None, widen=True):
    frame = source.frame_at(t) if frame is None else frame
    if frame is None:
        return Sample(t, *center, 0.0)
    ridge = to_ridge(frame)
    dx, dy, score = estimate_shift(ridge, G, pts, center, radius or cfg.radius)
    if widen and score < floor:  # 폰을 크게 건드린 경우: 넓게 다시 찾는다
        wide = estimate_shift(ridge, G, pts, center, cfg.wide_radius)
        if wide[2] > score:
            dx, dy, score = wide
    return Sample(t, dx, dy, score, score >= floor)


def walk(source, times, G, pts, start, cfg, floor, log, start_frame=None):
    """times 순서대로 샘플링한다. 탐색 중심은 마지막 유효 샘플의 이동.
    배경 특징점으로 마지막 유효 샘플 대비 카메라 이동을 잴 수 있으면 그 자리 주변(feature_radius)에서만 흰 선을 맞추고,
    흰 선이 흐려도(야간 조명) 그 이동을 받아들인다. 잴 수 없으면 흰 선만으로 찾는다."""
    center, out = start, []
    ref = describe(start_frame) if cfg.use_features and start_frame is not None else None
    for t in times:
        frame = source.frame_at(t)
        cur = describe(frame) if cfg.use_features else None
        moved = background_shift(ref, cur, cfg.feature_min_inliers)
        if moved is not None:
            guess = (center[0] + moved[0], center[1] + moved[1])
            s = sample_at(source, t, G, pts, guess, cfg, floor, frame=frame, radius=cfg.feature_radius, widen=False)
            if not s.ok:
                s = Sample(t, *guess, s.score, True, "features")
        else:
            s = sample_at(source, t, G, pts, center, cfg, floor, frame=frame)
        if s.ok:
            center = (s.dx, s.dy)
            ref = cur if cur is not None else ref
        out.append(s)
        tag = "ok" if s.ok and s.by == "lines" else "ok(배경)" if s.ok else "LOST"
        log(f"  t={t:8.1f}  shift=({s.dx:+.0f},{s.dy:+.0f})  score={s.score:.2f}  {tag}")
    return out


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ref(run):
    return float(np.median(run["dx"])), float(np.median(run["dy"]))


def _new_run(sample):
    run = {"ok": sample.ok, "samples": [sample], "dx": [], "dy": []}
    if sample.ok:
        run["dx"].append(sample.dx)
        run["dy"].append(sample.dy)
    return run


def group_samples(samples, tol, merge_gap=2):
    """시간순 샘플을 이동이 같은 연속 구간(ok)과 선을 놓친 구간(not ok)으로 묶는다.
    혼자 튄 샘플은 구간을 나누지 않고, 양옆 이동이 같은 짧은 무효 구간은 메운다."""
    runs = []
    for i, s in enumerate(samples):
        cur = runs[-1] if runs else None
        if cur is None or cur["ok"] != s.ok:
            runs.append(_new_run(s))
        elif not s.ok:
            cur["samples"].append(s)
        elif _dist((s.dx, s.dy), _ref(cur)) <= tol:
            cur["samples"].append(s)
            cur["dx"].append(s.dx)
            cur["dy"].append(s.dy)
        else:
            nxt = samples[i + 1] if i + 1 < len(samples) else None
            if nxt is not None and nxt.ok and _dist((nxt.dx, nxt.dy), _ref(cur)) <= tol:
                cur["samples"].append(s)  # 혼자 튄 샘플: 구간에는 넣되 이동 추정에서는 뺀다
            else:
                runs.append(_new_run(s))

    merged = []
    for run in runs:
        if (len(merged) >= 2 and merged[-1]["ok"] is False and run["ok"] and merged[-2]["ok"]
                and len(merged[-1]["samples"]) <= merge_gap and _dist(_ref(merged[-2]), _ref(run)) <= tol):
            gap = merged.pop()
            prev = merged[-1]
            prev["samples"] += gap["samples"] + run["samples"]
            prev["dx"] += run["dx"]
            prev["dy"] += run["dy"]
        else:
            merged.append(run)
    return merged


def locate_boundary(source, G, pts, lo, hi, a, b, eps):
    """이동 a였던 시각 lo와 이동 b가 된 시각 hi 사이에서 바뀐 시점을 이분 탐색한다."""
    Ga, Gb = translation(*a) @ G, translation(*b) @ G
    while hi - lo > eps:
        mid = (lo + hi) / 2
        frame = source.frame_at(mid)
        if frame is None:
            break
        ridge = to_ridge(frame)
        if line_score(ridge, Ga, pts) >= line_score(ridge, Gb, pts):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def median_frame(source, times, n):
    times = [times[i] for i in np.linspace(0, len(times) - 1, min(n, len(times))).astype(int)]
    frames = [f for f in (source.frame_at(t) for t in times) if f is not None]
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def refit(frame, Gs_near, Gs_far, near_pts, far_pts, far_world=None, far_img=None):
    """평행이동한 호모그래피를 시작점으로 흰 선에 다시 맞춘다. 나빠지면 평행이동 결과를 유지한다.
    이미지 -> 코트 (H_near, H_far, 일치율)을 반환."""
    ridge = to_ridge(frame)
    px, depth = project(Gs_near, near_pts)
    h, w = ridge.shape
    vis = np.where((depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h))[0]
    pick = vis[np.linspace(0, len(vis) - 1, 8).astype(int)]  # 시작점에서 멀어지지 않게 묶는 약한 앵커

    H_near = np.linalg.inv(Gs_near)
    H_far = np.linalg.inv(Gs_far)
    score0 = line_score(ridge, Gs_near, near_pts)
    H_new, _ = icp_align(ridge, H_near, near_pts, near_pts[pick], px[pick], anchor_w=3, radii=TRACK_RADII)
    score1 = line_score(ridge, front_facing(np.linalg.inv(H_new)), near_pts)
    if score1 < score0:
        return H_near / H_near[2, 2], H_far / H_far[2, 2], score0

    far_new, _ = icp_align_far(ridge, H_new, far_pts, radii=FAR_RADII, anchor_world=far_world, anchor_img=far_img)
    keep_far = line_score(ridge, front_facing(np.linalg.inv(far_new)), far_pts) >= line_score(ridge, Gs_far, far_pts) - 0.05
    return H_new, far_new if keep_far else H_far / H_far[2, 2], score1


def track_camera(source, cal, cfg=TrackConfig(), log=lambda *_: None):
    """앵커 캘리브레이션 cal을 영상 전체로 전파해 version 2 캘리브레이션 dict를 돌려준다."""
    spec = spec_of(cal)
    half, t_a = cal["half"], float(cal["t_sec"])
    model = sample_model(spec)
    near_pts = model[in_half(model, half, "near", margin=spec.center_r + CIRCLE_PAD)]
    far_pts = model[in_half(model, half, "far", margin=FAR_MARGIN)]
    G_near = front_facing(np.linalg.inv(np.array(cal["H_near"])))
    G_far = front_facing(np.linalg.inv(np.array(cal["H_far"])))
    far_world, far_click = far_anchors(cal) if "world" in cal and cal.get("img_pts") else (None, None)  # 카메라가 밀린 구간에서도 먼 쪽 기준을 유지
    if far_world is not None and len(far_world) == 0:
        far_world, far_click = None, None

    frame = source.frame_at(t_a)
    if frame is None:
        raise ValueError(f"앵커 시각 t={t_a}의 프레임을 읽을 수 없습니다")
    ridge = to_ridge(frame)
    dx0, dy0, _ = estimate_shift(ridge, G_near, near_pts, (0.0, 0.0), 8)
    score_a = line_score(ridge, G_near, near_pts)
    floor = max(cfg.min_score, cfg.rel_score * score_a)
    log(f"anchor t={t_a:.1f}: score={score_a:.2f}, 유효 하한={floor:.2f}")
    if score_a < cfg.min_score:
        raise ValueError(f"앵커 호모그래피가 흰 선과 맞지 않습니다 (일치율 {score_a:.2f})")

    start = Sample(t_a, dx0, dy0, score_a, True)
    forward = np.arange(t_a + cfg.step, source.duration, cfg.step)
    backward = np.arange(t_a - cfg.step, -1e-6, -cfg.step)
    samples = walk(source, backward, G_near, near_pts, (dx0, dy0), cfg, floor, log, frame)[::-1]
    samples += [start] + walk(source, forward, G_near, near_pts, (dx0, dy0), cfg, floor, log, frame)
    # 앵커 자체의 측정 편향을 빼서, 앵커 구간의 이동이 0이 되게 한다
    for s in samples:
        s.dx -= dx0
        s.dy -= dy0

    runs = group_samples(samples, cfg.tol_px, cfg.merge_gap)
    bounds = [0.0]
    for a, b in zip(runs[:-1], runs[1:]):
        lo, hi = a["samples"][-1].t, b["samples"][0].t
        if a["ok"] and b["ok"]:
            ra, rb = _ref(a), _ref(b)
            bounds.append(locate_boundary(source, G_near, near_pts, lo, hi, (ra[0] + dx0, ra[1] + dy0),
                                          (rb[0] + dx0, rb[1] + dy0), cfg.boundary_eps))
        else:  # 선을 놓친 쪽이 애매한 시간을 떠안는다
            bounds.append(lo if a["ok"] else hi)
    bounds.append(float(source.duration))

    segments = []
    for k, run in enumerate(runs):
        seg = {"t0": bounds[k], "t1": bounds[k + 1]}
        if not run["ok"]:
            segments.append({**seg, "source": "unresolved", "H_near": None, "H_far": None, "score": 0.0})
            continue
        dx, dy = _ref(run)
        times = [s.t for s in run["samples"]]
        if seg["t0"] <= t_a <= seg["t1"] and _dist((dx, dy), (0, 0)) <= cfg.tol_px:  # 사용자가 승인한 앵커는 그대로 쓴다
            H_far = None
            if far_world is not None:
                H_far = refit_far(source, times, np.array(cal["H_near"]), G_far, far_pts, far_world, far_click, cfg)
            segments.append({**seg, "source": cal.get("source", "manual"), "H_near": cal["H_near"],
                             "H_far": cal["H_far"] if H_far is None else H_far.tolist(), "score": score_a, "shift": [0.0, 0.0]})
            continue
        T = translation(dx, dy)
        far_img = None if far_world is None else far_click + [dx, dy]
        H_near, H_far, score = fit_segment(source, times, T @ G_near, T @ G_far, near_pts, far_pts, cfg, far_world, far_img)
        segments.append({**seg, "source": "tracked", "H_near": H_near.tolist(), "H_far": H_far.tolist(),
                         "score": score, "shift": [dx, dy]})

    return measure_drift(source, {**cal, "version": 2, "spec": asdict(spec), "segments": segments}, cfg, log)


def refit_far(source, times, H_near, G_far, far_pts, far_world, far_img, cfg):
    """사용자가 승인한 앵커 구간은 H_near를 그대로 두고, 여러 프레임을 합친 영상으로 먼 쪽만 정밀하게 다시 맞춘다."""
    frame = median_frame(source, times, cfg.median_frames)
    if frame is None:
        return None
    ridge = to_ridge(frame)
    new, _ = icp_align_far(ridge, H_near, far_pts, radii=FAR_RADII, anchor_world=far_world, anchor_img=far_img)
    return new if line_score(ridge, front_facing(np.linalg.inv(new)), far_pts) >= line_score(ridge, G_far, far_pts) - 0.05 else None


def fit_segment(source, times, Gs_near, Gs_far, near_pts, far_pts, cfg, far_world=None, far_img=None):
    frame = median_frame(source, times, cfg.median_frames)
    if frame is None:
        H_near, H_far = np.linalg.inv(Gs_near), np.linalg.inv(Gs_far)
        return H_near / H_near[2, 2], H_far / H_far[2, 2], 0.0
    return refit(frame, Gs_near, Gs_far, near_pts, far_pts, far_world, far_img)


def save_previews(source, result, n):
    """구간마다 중간 프레임에 코트 선을 그려 저장한다."""
    spec = spec_of(result)
    for k, seg in enumerate(result["segments"]):
        if seg["source"] == "unresolved":
            continue
        frame = source.frame_at((seg["t0"] + seg["t1"]) / 2)
        if frame is not None:
            view = draw_overlay(frame, np.array(seg["H_near"]), np.array(seg["H_far"]), result["half"], spec)
            cv2.imwrite(out_path(f"track_cam{n}_seg{k:02d}.jpg"), view)


def main(args):
    drift_only = "--drift-only" in args
    args = [a for a in args if a != "--drift-only"]
    step = float(args[args.index("--step") + 1]) if "--step" in args else TrackConfig.step
    cams = [int(a) for a in args[: args.index("--step")]] if "--step" in args else [int(a) for a in args]
    for n in cams or (1, 2):
        if drift_only:
            with open(out_path(f"calib_cam{n}.json")) as f:
                result = json.load(f)
            print(f"cam{n}: 밀림만 다시 잰다 ({result['video']})", flush=True)
            measure_drift(Source(result["video"]), result, TrackConfig(), log=lambda m: print(m, flush=True))
            with open(out_path(f"calib_cam{n}.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"  -> out/calib_cam{n}.json", flush=True)
            continue
        cal = load_anchor(n)
        source = Source(cal["video"])
        print(f"cam{n}: {cal['video']} ({source.duration:.0f} s)", flush=True)
        result = track_camera(source, cal, TrackConfig(step=step), log=lambda m: print(m, flush=True))
        with open(out_path(f"calib_cam{n}.json"), "w") as f:
            json.dump(result, f, indent=2)
        save_previews(source, result, n)
        for seg in result["segments"]:
            print(f"  [{seg['t0']:8.1f} ~ {seg['t1']:8.1f}] {seg['source']:10s} score={seg['score']:.2f} shift={seg.get('shift')}")
        print(f"  -> out/calib_cam{n}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
