"""클릭 없이 한 프레임에서 코트를 찾는다: 카메라 자세 사전지식 + 합성 코트 대조.

코너 삼각대라는 촬영 규칙 덕분에 카메라 자세의 범위가 좁다. 가능한 자세(위치, 높이, 초점거리, 시선, 기울기) 격자를 전부
흰 선의 거리 맵과 대조해 후보를 고르고, 지역 탐색과 ICP(refine.py)로 정밀화한 뒤 양방향 점수로 최종 판정한다.

코트 대칭: 코트는 180도 회전에 대칭이라 선만으로는 어느 쪽 골대인지 알 수 없다. 카메라 쪽 골대를 x<0으로 두는 '동쪽 기준'
좌표에서 풀고, 실제 방향은 half(세션 정보)로 정한다. 좌우 반전은 실제 카메라 자세만 탐색하므로 나오지 않는다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from calib_core import front_facing, in_half
from court import DEFAULT_SPEC, court_lines
from refine import CIRCLE_PAD, FAR_MARGIN, icp_align, icp_align_far, project, ridge_map, sample_model

ROT180 = np.diag([-1.0, -1.0, 1.0])  # 동쪽 기준 좌표 <-> 서쪽을 보는 카메라의 실제 코트 좌표
AUTO_RADII = (12, 8, 5, 3, 3)  # 지역 탐색이 이미 10 px 수준까지 맞춘 뒤이므로 좁게 시작


@dataclass(frozen=True)
class AutoConfig:
    behind: tuple = (-3.0, -1.0, 1.0)  # 골라인 기준 카메라 x 오프셋 (m, 음수 = 골라인 뒤)
    beside: tuple = (-1.0, 1.0, 3.0, 5.0)  # 사이드라인 기준 카메라 y 오프셋 (m, 양수 = 코트 밖)
    heights: tuple = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # m
    focals: tuple = (1300.0, 1500.0, 1700.0, 1900.0, 2100.0)  # px, 1080p 기준
    pan_step: float = 1.5  # 도, 화면에서 약 40 px
    pan_max: float = 45.0
    pan_inward: float = 10.0  # 코너에서 코트 바깥쪽으로 돌아간 시선의 허용치 (도)
    tilts: tuple = tuple(float(t) for t in range(2, 17))  # 수평 아래 각도 (도)
    rolls: tuple = (-2.0, 0.0, 2.0)  # 도
    tau_coarse: float = 70.0  # 격자 단계의 거리 맵 절단 거리 (px): 정답 칸과 어긋나도 점수가 나도록 넓게
    tau_refine: tuple = (70.0, 45.0, 30.0, 20.0, 14.0, 10.0)  # 지역 탐색 라운드별 절단 거리
    line_threshold: float = 0.45  # 흰 선 픽셀 판정 (ridge)
    min_component: int = 400  # 이보다 작은 선 조각은 잔디 잡음으로 버림 (px)
    grass_pad: int = 15  # 잔디 윗경계 위로 허용하는 여유 (px): 먼 골라인이 경계에 붙어 있음
    top_coarse: int = 400  # 지역 탐색으로 넘기는 후보 수
    refine_draws: int = 200
    top_icp: int = 16  # ICP까지 하는 후보 수
    min_visible: int = 120  # 화면에 보여야 하는 모델 샘플 수 (간격 약 0.45 m)
    accept_score: float = 0.55  # 자동 통과 점수 (평가셋으로 조정)
    accept_margin: float = 1.15  # 2위 후보 대비 배수


def grass_top(frame, pad):
    """열마다 잔디가 시작되는 행 - pad. 펜스와 나무의 직선이 코트 선으로 잡히지 않게 그 위를 잘라낸다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 화면 아래쪽은 거의 잔디이므로 거기서 색을 잰다 (맑은 날의 초록과 흐린 날의 회녹색이 크게 다르다)
    h0, s0, v0 = np.median(hsv[frame.shape[0] * 2 // 3:].reshape(-1, 3), axis=0)
    grass = cv2.inRange(hsv, (max(h0 - 14, 0), s0 * 0.45, v0 * 0.4), (min(h0 + 14, 179), 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grass, connectivity=8)
    if n <= 1:
        return np.zeros(frame.shape[1], int)
    main = labels == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    top = np.where(main.any(0), main.argmax(0), frame.shape[0])
    return np.maximum(cv2.blur(top.astype(np.float32)[None], (101, 1))[0].astype(int) - pad, 0)


def line_pixels(frame, cfg):
    """(ridge, 코트 선 후보 픽셀 마스크)."""
    ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    mask = (ridge > cfg.line_threshold).astype(np.uint8)
    rows = np.arange(frame.shape[0])[:, None]
    mask[rows < grass_top(frame, cfg.grass_pad)[None]] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= cfg.min_component
    keep[0] = False
    return ridge, keep[labels]


def distance_field(lines, tau):
    """흰 선 위에서 0, tau px 이상 떨어지면 1인 거리 맵."""
    dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(dist / tau, 1.0).astype(np.float32)


def homographies(p, size):
    """자세 파라미터 (N, 7) [Cx, Cy, h, pan, tilt, f, roll] (각도는 도) -> 코트(동쪽 기준) -> 이미지 호모그래피 (N, 3, 3).
    pan은 +x(먼 골대 방향)에서 +y로 돌아간 각도, tilt는 수평 아래로 숙인 각도."""
    w, h = size
    C = p[:, :3]
    pan, tilt = np.radians(p[:, 3]), np.radians(p[:, 4])
    z = np.c_[np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), -np.sin(tilt)]
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    a = np.radians(p[:, 6])[:, None]
    x, y = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    R = np.stack([x, y, z], axis=1)
    t = -(R @ C[..., None])[..., 0]
    K = np.zeros((len(p), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = p[:, 5]
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2, h / 2, 1.0
    return K @ np.concatenate([R[:, :, :2], t[..., None]], axis=2)


def pose_grid(spec, cfg):
    """코너 양쪽의 자세 격자. 각 코너에서 시선은 코트 안쪽을 향한다."""
    L2, W2 = spec.length / 2, spec.width / 2
    pans = np.arange(-cfg.pan_max, cfg.pan_max + 1e-6, cfg.pan_step)
    out = []
    for side in (-1, 1):  # side=-1: 화면 오른쪽 사이드라인(y<0) 쪽 코너, 시선은 +y로
        ys = [side * (W2 + b) for b in cfg.beside]
        side_pans = pans[pans * -side >= -cfg.pan_inward]
        axes = np.meshgrid([-L2 + b for b in cfg.behind], ys, cfg.heights, side_pans, cfg.tilts, cfg.focals, cfg.rolls,
                           indexing="ij")
        out.append(np.stack([a.ravel() for a in axes], axis=1))
    return np.concatenate(out)


def score_poses(field, G, pts, min_visible, chunk=4000):
    """포즈별 점수: 화면 안 모델 샘플의 (적중 - 0.5 빗나감). 화면에 너무 적게 보이면 -무한대."""
    h, w = field.shape
    homog = np.c_[pts, np.ones(len(pts))].T
    out = np.empty(len(G))
    for i in range(0, len(G), chunk):
        q = G[i:i + chunk] @ homog
        depth = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = q[:, 0] / depth, q[:, 1] / depth
        vis = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = field[np.nan_to_num(v).clip(0, h - 1).astype(int), np.nan_to_num(u).clip(0, w - 1).astype(int)]
        n = vis.sum(1)
        miss = (d * vis).sum(1)
        s = (n - miss) - 0.5 * miss
        s[n < min_visible] = -np.inf
        out[i:i + chunk] = s
    return out


def diverse_top(poses, scores, k):
    """비슷한 포즈가 몰리지 않게 위치/시선 칸마다 최고 후보만 남겨 상위 k개를 고른다."""
    order = np.argsort(-scores)
    seen, keep = set(), []
    for i in order:
        if not np.isfinite(scores[i]):
            break
        key = tuple(np.round(poses[i, :5] / [2, 2, 1, 3, 2]).astype(int))
        if key not in seen:
            seen.add(key)
            keep.append(i)
            if len(keep) == k:
                break
    return np.array(keep, int)


def pose_bounds(spec, cfg):
    """지역 탐색이 벗어나면 안 되는 물리 범위. 격자 범위에서 한 칸 정도만 더 허용한다."""
    L2, W2 = spec.length / 2, spec.width / 2
    side = W2 + max(cfg.beside) + 2.0
    lo = [-L2 + min(cfg.behind) - 2.0, -side, min(cfg.heights) - 0.3, -cfg.pan_max - 10.0, 1.0, min(cfg.focals) - 200.0, -4.0]
    hi = [-L2 + max(cfg.behind) + 2.0, side, max(cfg.heights) + 0.5, cfg.pan_max + 10.0, 20.0, max(cfg.focals) + 300.0, 4.0]
    return np.array(lo), np.array(hi)


def local_refine(lines, poses, pts, cfg, size, rng, spec=DEFAULT_SPEC):
    """거리 맵을 점점 날카롭게 하며 후보마다 파라미터를 흔들어 더 나은 자세로 옮긴다."""
    lo, hi = pose_bounds(spec, cfg)
    sigma = np.array([1.0, 1.0, 0.4, 1.5, 0.8, 120.0, 1.0])
    best = poses.copy()
    for tau in cfg.tau_refine:
        field = distance_field(lines, tau)
        best_s = score_poses(field, homographies(best, size), pts, cfg.min_visible)
        draws = np.repeat(best, cfg.refine_draws, axis=0) + rng.normal(size=(len(best) * cfg.refine_draws, 7)) * sigma
        draws = np.clip(draws, lo, hi)  # 높이가 0에 가까워지면 선이 수평선으로 뭉쳐 점수만 오르는 가짜 해가 생긴다
        s = score_poses(field, homographies(draws, size), pts, cfg.min_visible).reshape(len(best), -1)
        j = s.argmax(1)
        idx = np.arange(len(best))
        better = s[idx, j] > best_s
        best[better] = draws.reshape(len(best), cfg.refine_draws, -1)[idx, j][better]
        sigma = sigma * 0.7
    return best, score_poses(field, homographies(best, size), pts, cfg.min_visible)


def hit_rate(ridge, G, pts, thr=0.25):
    px, depth = project(G, pts)
    h, w = ridge.shape
    ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if ok.sum() < 50:
        return 0.0
    return float((ridge[px[ok, 1].astype(int), px[ok, 0].astype(int)] > thr).mean())


def explained(lines, G, spec):
    """코트 안쪽으로 투영되는 흰 선 픽셀 중 모델 선 근처에 있는 비율."""
    h, w = lines.shape
    L2, W2 = spec.length / 2, spec.width / 2
    near_cut = -L2
    corners = np.array([[near_cut, -W2], [L2, -W2], [L2, W2], [near_cut, W2]])
    px, depth = project(G, corners)
    if (depth <= 0).any():  # 코트 앞쪽이 카메라 뒤로 넘어가면 카메라 앞까지만 자른다
        cam_x = -L2 * 0.6
        corners[[0, 3], 0] = cam_x
        px, depth = project(G, corners)
    if (depth <= 0).any() or (np.abs(px) > 2e4).any():
        return 0.0
    region, drawn = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, px.astype(np.int32), 1)
    for line in court_lines(spec):
        dense = np.concatenate([np.linspace(a, b, 30) for a, b in zip(line[:-1], line[1:])])
        p, d = project(G, dense)
        ok = (d > 0) & (np.abs(p) < 1e4).all(1)
        if ok.sum() > 1:
            cv2.polylines(drawn, [p[ok].astype(np.int32)], False, 1, 17)
    inside = lines & (region > 0)
    return float((inside & (drawn > 0)).sum()) / max(int(inside.sum()), 1)


def camera_from(G, size):
    """코트 -> 이미지 호모그래피의 물리 해석: (초점거리, 두 축의 스케일 비, 카메라 위치). 해석 불가면 None."""
    T = np.array([[1, 0, -size[0] / 2], [0, 1, -size[1] / 2], [0, 0, 1.0]])
    A = T @ G
    h1, h2 = A[:, 0], A[:, 1]
    den = h1[2] * h2[2]
    if abs(den) < 1e-14:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
    if f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ h1, Kinv @ h2, Kinv @ A[:, 2]
    ratio = float(np.linalg.norm(r1) / np.linalg.norm(r2))
    s = 1 / np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
    R = np.c_[r1 * s, r2 * s, np.cross(r1 * s, r2 * s)]
    return f, ratio, -R.T @ (t * s)


def plausible(G, size, spec):
    cam = camera_from(G, size)
    if cam is None:
        return False
    f, ratio, C = cam
    return 800 < f < 3200 and 0.8 < ratio < 1.25 and 1.3 <= abs(C[2]) < 7 and C[0] < -spec.length / 4


def same_solution(Ga, Gb, spec, size, tol=1.0):
    """화면 안 near-half 격자에서 두 해의 코트 좌표가 tol m 안으로 같은가."""
    xs, ys = np.meshgrid(np.linspace(-spec.length * 0.4, 0, 6), np.linspace(-spec.width / 2, spec.width / 2, 5))
    world = np.c_[xs.ravel(), ys.ravel()]
    px, depth = project(Ga, world)
    ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])
    if ok.sum() < 3:
        return False
    back = cv2.perspectiveTransform(px[ok].reshape(-1, 1, 2), np.linalg.inv(Gb))[:, 0]
    return float(np.median(np.linalg.norm(back - world[ok], axis=1))) < tol


def auto_calibrate(frame, half, spec=DEFAULT_SPEC, cfg=AutoConfig(), seed=0):
    """프레임 하나에서 코트를 찾는다. {status, confidence, margin, H_near, H_far, camera, half, spec용 값}을 돌려준다.
    status: ok(자동 통과) / low_confidence(사용자 확인 또는 수동) / failed(후보 없음)."""
    size = frame.shape[1], frame.shape[0]
    scale = 1080.0 / frame.shape[0]
    work = cv2.resize(frame, (int(round(frame.shape[1] * scale)), 1080)) if scale != 1.0 else frame
    size_w = work.shape[1], work.shape[0]

    ridge, lines = line_pixels(work, cfg)
    model = sample_model(spec)
    coarse_pts = model[::3]
    grid = pose_grid(spec, cfg)
    scores = score_poses(distance_field(lines, cfg.tau_coarse), homographies(grid, size_w), coarse_pts, cfg.min_visible)
    keep = diverse_top(grid, scores, cfg.top_coarse)
    if len(keep) == 0:
        return {"status": "failed", "confidence": 0.0, "margin": 0.0, "H_near": None, "H_far": None}

    rng = np.random.default_rng(seed)
    poses, refined = local_refine(lines, grid[keep], coarse_pts, cfg, size_w, rng, spec)
    # 탐색 범위 경계에 붙은 자세는 버린다: 낮은 높이에서 선이 수평선으로 뭉쳐 점수만 높아지는 가짜 해가 여기 몰린다
    lo, hi = pose_bounds(spec, cfg)
    refined[(poses[:, 2] <= lo[2] + 0.1) | (poses[:, 5] <= lo[5] + 20) | (poses[:, 5] >= hi[5] - 20)] = -np.inf
    order = np.argsort(-refined)[: cfg.top_icp]
    order = order[np.isfinite(refined[order])]

    near_pts = model[in_half(model, "east", "near", margin=spec.center_r + CIRCLE_PAD)]
    far_pts = model[in_half(model, "east", "far", margin=FAR_MARGIN)]
    empty = np.empty((0, 2))
    cands = []
    for G0 in homographies(poses[order], size_w):
        G0 = front_facing(G0)
        H0 = np.linalg.inv(G0)
        try:
            H, _ = icp_align(ridge, H0, near_pts, empty, empty, radii=AUTO_RADII)
            G = front_facing(np.linalg.inv(H))
        except np.linalg.LinAlgError:  # ICP가 퇴화한 경우
            H, G = H0 / H0[2, 2], G0
        if not (plausible(G, size_w, spec) and hit_rate(ridge, G, near_pts) >= hit_rate(ridge, G0, near_pts)):
            H, G = H0 / H0[2, 2], G0  # ICP가 비물리적인 해로 끌고 가면 자세 탐색 결과를 유지
        if not plausible(G, size_w, spec):
            continue
        hit = hit_rate(ridge, G, model)
        cands.append({"score": hit * explained(lines, G, spec), "hit": hit, "G": G, "H": H})
    if not cands:
        return {"status": "failed", "confidence": 0.0, "margin": 0.0, "H_near": None, "H_far": None}
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]
    rival = next((c["score"] for c in cands[1:] if not same_solution(best["G"], c["G"], spec, size_w)), 0.0)
    margin = best["score"] / max(rival, 1e-6)

    H_far, _ = icp_align_far(ridge, best["H"], far_pts, radii=AUTO_RADII)
    S = np.eye(3) if half == "east" else ROT180
    to_px = np.diag([scale, scale, 1.0])  # 원본 픽셀 -> 작업 해상도
    H_near_out, H_far_out = S @ best["H"] @ to_px, S @ H_far @ to_px
    H_near_out, H_far_out = H_near_out / H_near_out[2, 2], H_far_out / H_far_out[2, 2]
    status = "ok" if best["score"] >= cfg.accept_score and margin >= cfg.accept_margin else "low_confidence"
    f, ratio, C = camera_from(best["G"], size_w)
    return {"status": status, "_G": best["G"], "_size": size_w, "confidence": round(best["score"], 3), "margin": round(float(min(margin, 99.0)), 2),
            "hit": round(best["hit"], 3), "H_near": H_near_out.tolist(), "H_far": H_far_out.tolist(), "half": half,
            "camera": {"f": round(f, 1), "scale_ratio": round(ratio, 3), "position": [round(float(c), 2) for c in C]}}


def median_of(source, t, n=5, dt=3.0):
    """t 주변 프레임의 시간 중앙값 (움직이는 선수를 지운다). 프레임을 못 읽으면 None."""
    frames = [f for f in (source.frame_at(max(t + k * dt, 0)) for k in range(n)) if f is not None]
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def consensus(results, spec, tol=2.0):
    """시점별 결과 중 서로 같은 답을 낸 가장 큰 무리를 찾는다. (대표 인덱스, 무리 인덱스 목록)을 돌려준다.
    카메라가 경기 중 밀리므로 tol은 그만큼(약 1.5 m) 여유를 둔다. 틀린 답은 시점마다 흩어지지만 맞는 답은 모인다."""
    valid = [i for i, r in enumerate(results) if r.get("H_near") is not None]
    groups = {i: [j for j in valid if same_solution(results[i]["_G"], results[j]["_G"], spec, results[i]["_size"], tol)]
              for i in valid}
    if not groups:
        return None, []
    lead = max(groups, key=lambda i: (len(groups[i]), results[i]["confidence"]))
    members = groups[lead]
    return max(members, key=lambda i: results[i]["confidence"]), members


def auto_calibrate_video(source, half, spec=DEFAULT_SPEC, times=None, cfg=AutoConfig(), n_windows=5, log=lambda *_: None):
    """영상의 여러 시점에서 코트를 찾아 다수결로 앵커를 정한다. 과반이 같은 답을 내야 status가 ok다."""
    if times is None:
        times = np.linspace(source.duration * 0.15, source.duration * 0.9, n_windows)
    results = []
    for t in times:
        frame = median_of(source, float(t))
        r = auto_calibrate(frame, half, spec, cfg) if frame is not None else {"status": "failed", "confidence": 0.0, "H_near": None}
        r["t_sec"] = float(t)
        log(f"  t={float(t):8.1f}: {r['status']} conf={r['confidence']:.2f} {r.get('camera')}")
        results.append(r)
    lead, members = consensus(results, spec)
    public = [{k: v for k, v in r.items() if not k.startswith("_") and not k.startswith("H_")} for r in results]
    if lead is None:
        return {"status": "failed", "votes": 0, "windows": len(results), "H_near": None, "H_far": None, "details": public}
    best = results[lead]
    status = "ok" if len(members) * 2 > len(results) else "low_confidence"
    return {**{k: v for k, v in best.items() if not k.startswith("_")}, "status": status, "votes": len(members),
            "windows": len(results), "details": public}
