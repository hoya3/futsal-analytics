"""각 카메라의 두 호모그래피(가까운 하프, 먼 하프)를 코트의 흰 선에 맞춘다.
먼 하프는 가까운 하프에서 시작하며(calibrate.py에는 먼 하프 클릭이 없음), 하프라인은 가까운 하프와 똑같이 매핑한다.

사용법: python refine.py   -> out/H_cam<N>_refined.json + out/H_cam<N>_refined_overlay.jpg
방법 (ICP): 한 하프의 코트 모델 선을 샘플링하고, 이미지 법선 방향으로 가장 가까운 흰 선을 찾아
호모그래피를 다시 피팅한다. 탐색 반경을 줄여가며 반복.
  가까운 하프: RANSAC 피팅(펜스/선수 위의 매칭은 버림); 손으로 찍은 클릭을 큰 가중치의 앵커로 추가해
             피팅이 클릭에서 멀어지지 않게 한다.
  먼 하프:   (기본 far_model "shared") H_far = H_near o K, 여기서 K = I + v [1 0 0]은 하프라인(x = 0) 위의 모든 점을 그대로 두므로
             두 호모그래피가 하프라인 전체에서 일치한다(선수가 하프라인을 넘어도 튀지 않음). RANSAC은
             inlier 매칭을 고르는 데만 쓰고, v (3개 값)는 그 매칭에 최소제곱으로 피팅한다.
             far_model "free"(캘리브레이션 json의 키)면 그 결과에서 8자유도로 더 맞추고, 하프라인은 부드러운 제약으로만
             잇는다(court_map이 하프라인 근처를 섞는다). 아래 FAR_MODELS 설명 참고.
"""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from calib_core import HALF_SIGN, draw_overlay, front_facing, in_half, out_path
from court import DEFAULT_SPEC, court_lines, spec_of

SIGMAS = (1.5, 2.5, 4.0, 6.0)
SAMPLE_SPACING = 0.15  # 모델 선 샘플 간격 (m)
CIRCLE_PAD = 0.2  # 각 하프는 센터 서클 전체도 사용 (반지름 + 여유, m)
# 먼 하프 피팅은 하프라인에서 8 m 이내의 가까운 하프 선(서클, 사이드라인)도 사용한다. 하프라인을 공유하는
# 상태에서 코트 전체를 쓰면 먼 하프가 H_near 쪽으로 끌려간다 (측정값: cam1 먼 하프 0.43 m, cam2 0.72 m,
# 8 m일 때는 0.36 m, 0.61 m)
FAR_MARGIN = 8.0
RADII = (25, 18, 12, 8, 5, 3, 3)  # px, 라운드별 ICP 탐색 반경
FAR_ANCHOR_W = 30.0  # 먼 쪽 기준점 하나가 흰 선 매칭 몇 개만큼의 비중을 갖는가
# 먼 하프 모델. "shared" = H_near o K (3자유도, 하프라인을 정확히 공유, 기본값).
# "free" = 8자유도 호모그래피, 하프라인은 HALFWAY_LAM 가중치의 부드러운 제약으로만 잇는다. 먼 하프 바닥이 평면이
# 아니어서 3자유도로 먼 쪽을 다 맞추지 못하는 카메라용. 하프라인에서 H_near와 정확히 이어지지 않으므로 court_map이
# 하프라인 너머 FAR_BLEND_M까지 두 결과를 섞는다. 1008 cam2 (track.py --refit-far, 검증 라벨, 중앙값 / 90%):
#   shared            센터 점 0.12 / 0.23 m, 먼 쪽 0.45 / 1.59 m
#   free (lam 3, 섞기) 센터 점 0.12 / 0.24 m, 먼 쪽 0.37 / 0.66 m
# lam 0.3은 구간 피팅이 흰 선 일치율 검사에서 떨어져 적용되지 않았고, 30은 먼 쪽이 0.50 m로 나빠졌다.
# 1008 cam1에는 효과가 없다(오차가 골라인을 따른 방향이라 흰 선이 보지 못함).
FAR_MODELS = ("shared", "free")
HALFWAY_LAM = 3.0  # 하프라인 위 21점의 H_near-H_far 픽셀 차이에 곱하는 가중치
# "free"가 쓰는 가까운 하프 선의 범위 (하프라인에서 m). 합성 꺾임에서는 3.2 m가 나았지만(0.11 vs 0.14 m)
# 1008 cam2 실제 데이터에서는 8 m와 3.2 m가 같았다.
FREE_FAR_MARGIN = 8.0


def ridge_map(frame):
    """밝고 가는 선의 강도 [0, 1] (멀티스케일 헤시안), 채도가 높은 색(잔디, 나무)에서는 0."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = np.zeros_like(gray)
    for s in SIGMAS:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        lam_min = (ixx + iyy) / 2 - np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)  # 가장 음수인 곡률 = 밝은 능선
        out = np.maximum(out, s ** 2 * np.maximum(0, -lam_min))
    out *= (hsv[..., 1] < 110)
    return np.minimum(out / max(np.percentile(out, 99.7), 1e-6), 1.0).astype(np.float32)


def sample_model(spec=DEFAULT_SPEC):
    """모든 코트 선을 따라 ~15 cm마다 찍은 점."""
    pts = []
    for line in court_lines(spec):
        for a, b in zip(line[:-1], line[1:]):
            n = max(int(np.linalg.norm(b - a) / SAMPLE_SPACING), 1)
            pts.append(np.linspace(a, b, n, endpoint=False))
    return np.concatenate(pts)


def project(Hc2i, pts):
    q = np.c_[pts, np.ones(len(pts))] @ Hc2i.T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def subpixel_offset(v, k):
    """봉우리 k 주변 세 값에 포물선을 맞춰 봉우리의 소수점 위치를 구한다. 먼 쪽은 1 px가 약 0.3 m라 소수점이 중요하다."""
    if not 0 < k < len(v) - 1:
        return 0.0
    curve = v[k - 1] - 2 * v[k] + v[k + 1]
    return float(np.clip(0.5 * (v[k - 1] - v[k + 1]) / curve, -0.5, 0.5)) if curve < -1e-6 else 0.0


def match_lines(ridge, Hc2i, pts, radius, min_ridge=0.3):
    """모델 샘플마다 이미지 법선 방향 +-radius px 안에서 가장 강한 흰 선 위치(소수점)를 찾는다.
    (코트 점, 매칭된 이미지 점)을 반환."""
    h, w = ridge.shape
    px, depth = project(Hc2i, pts)
    offs = np.arange(-radius, radius + 1)
    src, dst = [], []
    for i in range(1, len(pts) - 1):
        t = px[i + 1] - px[i - 1]
        if depth[i] <= 0 or np.linalg.norm(t) < 1e-3 or np.linalg.norm(pts[i + 1] - pts[i - 1]) > 0.5:
            continue  # 카메라 뒤, 퇴화된 경우, 또는 두 선의 이음매
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        cand = px[i] + offs[:, None] * nrm
        if (cand < 2).any() or (cand[:, 0] > w - 3).any() or (cand[:, 1] > h - 3).any():
            continue
        v = cv2.remap(ridge, cand[:, :1].astype(np.float32), cand[:, 1:].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
        k = int(v.argmax())
        if v[k] >= min_ridge:
            src.append(pts[i])
            dst.append(cand[k] + subpixel_offset(v, k) * nrm)
    return np.array(src), np.array(dst)


def icp_align(ridge, H_img2court, pts, anchor_world, anchor_img, anchor_w=5, radii=RADII):
    """(H_img2court, 마지막 라운드에서 RANSAC이 남긴 선 매칭 비율)을 반환."""
    a_src = np.repeat(np.asarray(anchor_world, float), anchor_w, axis=0)
    a_dst = np.repeat(np.asarray(anchor_img, float), anchor_w, axis=0)
    Hc2i = front_facing(np.linalg.inv(H_img2court))
    ratio = 0.0
    for r in radii:
        src, dst = match_lines(ridge, Hc2i, pts, r)
        if len(src) < 50:
            break
        H_new, inl = cv2.findHomography(np.r_[src, a_src].astype(np.float32), np.r_[dst, a_dst].astype(np.float32),
                                        cv2.RANSAC, max(2.0, r / 3))
        if H_new is None:
            break
        Hc2i, ratio = front_facing(H_new), inl[:len(src)].mean()
    H = np.linalg.inv(Hc2i)
    return H / H[2, 2], ratio


def icp_align_far(ridge, H_near, pts, radii=RADII, anchor_world=None, anchor_img=None, anchor_w=FAR_ANCHOR_W):
    """먼 하프를 H_near o K, K = I + v [1 0 0]로 구한다 (하프라인의 모든 점을 고정).
    (H_img2court, 마지막 라운드에서 RANSAC이 남긴 선 매칭 비율)을 반환."""
    G = front_facing(np.linalg.inv(H_near))

    def c2i(v):  # K의 세 번째 열이 (0, 0, 1)이므로 (G K)[2, 2] = G[2, 2]는 양수로 유지된다 (카메라 앞)
        return G @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0]))

    has_anchor = anchor_world is not None and len(anchor_world) > 0
    if has_anchor:
        aw, ai = np.asarray(anchor_world, float), np.asarray(anchor_img, float)

    def anchor_residual(p):  # 먼 쪽 기준점: 흰 선 매칭이 먼 골대 쪽을 잘못 끌고 가지 않게 묶는다
        return np.sqrt(anchor_w) * (project(c2i(p), aw)[0] - ai).ravel() if has_anchor else np.empty(0)

    v, ratio = np.zeros(3), 0.0
    if has_anchor:
        v = least_squares(anchor_residual, v, x_scale=[1e-3, 1e-3, 1e-4]).x
    for r in radii:
        src, dst = match_lines(ridge, c2i(v), pts, r)
        if len(src) < 50:
            break
        _, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, max(2.0, r / 3))
        if inl is None:
            break
        keep = inl[:, 0] > 0
        v = least_squares(lambda p: np.r_[(project(c2i(p), src[keep])[0] - dst[keep]).ravel(), anchor_residual(p)], v, x_scale="jac").x
        ratio = keep.mean()
    H = np.linalg.inv(c2i(v))
    return H / H[2, 2], ratio


def _halfway(H_near, spec=DEFAULT_SPEC, n=21):
    """하프라인 위 코트 점과 H_near로 투영한 픽셀."""
    hw = np.c_[np.zeros(n), np.linspace(-spec.width / 2, spec.width / 2, n)]
    return hw, project(front_facing(np.linalg.inv(H_near)), hw)[0]


def icp_align_far_free(ridge, H_near, pts, H_init=None, radii=RADII, anchor_world=None, anchor_img=None,
                       anchor_w=FAR_ANCHOR_W, lam=None, spec=DEFAULT_SPEC):
    """먼 하프를 8자유도 호모그래피로 흰 선에 맞춘다. 하프라인은 H_near와 픽셀 차이에 lam(기본 HALFWAY_LAM) 가중치를 준
    부드러운 제약. H_init(보통 icp_align_far 결과)에서 시작한다. (H_img2court, 마지막 라운드의 RANSAC 선 매칭 비율)을 반환."""
    lam = HALFWAY_LAM if lam is None else lam
    hw, hw_img = _halfway(H_near, spec)
    G0 = front_facing(np.linalg.inv(H_near if H_init is None else H_init))
    p = (G0 / G0[2, 2]).ravel()[:8]  # G0[2, 2] > 0 이므로 나눠도 카메라 앞/뒤 부호가 유지된다
    has_anchor = anchor_world is not None and len(anchor_world) > 0
    if has_anchor:
        aw, ai = np.asarray(anchor_world, float), np.asarray(anchor_img, float)

    def G_of(q):
        return np.append(q, 1.0).reshape(3, 3)

    def soft(q):
        G = G_of(q)
        r = [np.sqrt(lam) * (project(G, hw)[0] - hw_img).ravel()]
        if has_anchor:
            r.append(np.sqrt(anchor_w) * (project(G, aw)[0] - ai).ravel())
        return np.concatenate(r)

    ratio = 0.0
    for r in radii:
        src, dst = match_lines(ridge, G_of(p), pts, r)
        if len(src) < 50:
            break
        _, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, max(2.0, r / 3))
        if inl is None:
            break
        keep = inl[:, 0] > 0
        p = least_squares(lambda q: np.r_[(project(G_of(q), src[keep])[0] - dst[keep]).ravel(), soft(q)], p,
                          x_scale="jac").x
        ratio = keep.mean()
    H = np.linalg.inv(G_of(p))
    return H / H[2, 2], ratio


def align_far(ridge, H_near, pts, model="shared", radii=RADII, anchor_world=None, anchor_img=None, spec=DEFAULT_SPEC,
              half=None):
    """먼 하프 피팅. model "shared"는 icp_align_far, "free"는 그 결과에서 시작하는 icp_align_far_free.
    half를 주면 "free"는 pts 중 하프라인에서 FREE_FAR_MARGIN 안쪽까지의 선만 쓴다."""
    if model not in FAR_MODELS:
        raise ValueError(f"far_model은 {FAR_MODELS} 중 하나여야 합니다: {model!r}")
    H, ratio = icp_align_far(ridge, H_near, pts, radii=radii, anchor_world=anchor_world, anchor_img=anchor_img)
    if model == "free":
        free_pts = pts if half is None else pts[in_half(pts, half, "far", margin=FREE_FAR_MARGIN)]
        H, ratio = icp_align_far_free(ridge, H_near, free_pts, H_init=H, radii=radii, anchor_world=anchor_world,
                                      anchor_img=anchor_img, spec=spec)
    return H, ratio


def halfway_gap_m(H_near, H_far, spec=DEFAULT_SPEC):
    """하프라인 위에서 H_near와 H_far가 내는 코트 좌표의 최대 차이 (m). "shared"는 0, "free"는 작아야 한다."""
    _, px = _halfway(H_near, spec)
    a = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.asarray(H_near, float))[:, 0]
    b = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.asarray(H_far, float))[:, 0]
    return float(np.linalg.norm(a - b, axis=1).max())


def far_anchors(d):
    """캘리브레이션 dict의 먼 쪽 기준점 (월드, 픽셀). 5클릭 시절 파일이나 자동 후보에는 없다."""
    far = d.get("world", {}).get("far", {})
    keep = [i for i, p in far.items() if HALF_SIGN[d["half"]] * p[0] > 0.01]
    return np.array([far[i] for i in keep], float).reshape(-1, 2), np.array([d["img_pts"][int(i)] for i in keep], float).reshape(-1, 2)


def refine_camera(d, frame, log=print):
    ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    spec = spec_of(d)
    model = sample_model(spec)
    out = dict(d)
    # 가까운 하프: 자기 하프의 선(+ 센터 서클 전체)을 사용, 클릭을 앵커로 고정
    pts = model[in_half(model, d["half"], "near", margin=spec.center_r + CIRCLE_PAD)]
    idx = [int(i) for i in d["world"]["near"]]
    H_near, ratio = icp_align(ridge, np.array(d["H_near"]), pts,
                              [d["world"]["near"][str(i)] for i in idx], [d["img_pts"][i] for i in idx])
    log(f"  near half: {ratio:.0%} of line matches kept")
    # 먼 하프: 자기 하프의 선은 몇 픽셀 두께로 가늘고 멀어서, 하프라인 주변의 선명한 선을 추가로 쓴다.
    # 하프라인은 보정된 가까운 하프의 것을 그대로 물려받는다.
    pts = model[in_half(model, d["half"], "far", margin=FAR_MARGIN)]
    far_world, far_img = far_anchors(d)
    model_name = d.get("far_model", "shared")
    H_far, ratio = align_far(ridge, H_near, pts, model_name, anchor_world=far_world, anchor_img=far_img, spec=spec,
                             half=d["half"])
    log(f"  far half ({model_name}): {ratio:.0%} of line matches kept, halfway gap {halfway_gap_m(H_near, H_far, spec):.2f} m")
    out["H_near"], out["H_far"] = H_near.tolist(), H_far.tolist()
    out["refined"] = "icp"
    return out


if __name__ == "__main__":
    for n in (1, 2):
        with open(out_path(f"H_cam{n}.json"), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("source") == "auto":  # 자동 후보는 클릭 앵커가 없고 이미 흰 선에 맞춰져 있다
            print(f"cam{n}: 자동 인식 결과라 건너뜀")
            continue
        if "H_near" not in d:
            raise SystemExit(f"{out_path(f'H_cam{n}.json')} is from the old 5-point tool: re-run calibrate.py for cam{n} first")
        frame = cv2.imread(out_path(f"frame_cam{n}.jpg"))
        print(f"cam{n}:")
        r = refine_camera(d, frame)
        with open(out_path(f"H_cam{n}_refined.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        cv2.imwrite(out_path(f"H_cam{n}_refined_overlay.jpg"),
                    draw_overlay(frame, np.array(r["H_near"]), np.array(r["H_far"]), r["half"], spec_of(r)))
