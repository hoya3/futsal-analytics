"""클릭 없는 자동 보정의 더 엄격한 자동 통과 판정(게이트) 분석.

지금 판정(keypoint_init.decide)은 '과반이 같은 답 + 대표의 가까운 쪽 흰 선 일치율 >= 0.6'이다. 앵커를 약하게 하면
(가까운 10 / 먼 30) 1103은 다 맞지만 처음 보는 1008 cam1에서 서로 일치하는 틀린 답(먼 쪽 ~5 m)이 일치율 0.58~0.6으로
통과한다. 여기서는 시점별 결과를 행렬까지 다시 풀어 후보 특징을 모두 계산하고, eval_autocal.simulate와 같은 세션에서
특징 문턱을 더한 판정의 통과/틀린 통과를 센다.

단계
  solve    설정(앵커 가까운/먼)마다 시점 풀을 다시 풀고 특징 + 정답 오차를 work/gate/windows_<설정>.json에,
           코트 평면으로 펼친 흰 선 지도(카메라 사이 비교용)를 work/gate/rect_<설정>_<회차>_cam<n>.npz에 저장
  analyze  저장한 시점 결과로 세션을 다시 만들어(eval_autocal과 같은 난수) 게이트를 평가하고 표를 출력

사용법:
  python eval/gate_analysis.py solve --setting 10,30 --match 1008 1103
  python eval/gate_analysis.py analyze --settings 100,30 10,30 3,10
"""
import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

import eval_autocal as ea  # noqa: E402
import keypoint_init as ki  # noqa: E402
from auto_init import ROT180  # noqa: E402
from calib_core import HALF_SIGN, Source, front_facing, match_dir  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import load  # noqa: E402
from refine import match_lines, project, ridge_map  # noqa: E402
from track import HIT_THRESHOLD  # noqa: E402

OUT = os.path.join(os.path.dirname(ROOT), "work", "gate")  # D:\02_Projects\futsal\work\gate
SPACING = 0.15  # 선 샘플 간격 (m), refine.SAMPLE_SPACING과 같게
RECT_STEP = 0.1  # 코트 평면 지도 격자 (m)
RECT_X, RECT_Y = 21.0, 11.0  # 지도 범위: |x| <= 21, |y| <= 11 (코트 40 x 20 + 여유)
RESID_RADIUS = 10  # 선 무리별 잔차를 잴 때 법선 방향 탐색 반경 (px)


# ----------------------------------------------------------------------------------------------------------------
# 특징
# ----------------------------------------------------------------------------------------------------------------

def _seg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / SPACING), 2)
    return np.linspace(a, b, n)


def _arc(cx, cy, r, a0, a1):
    n = max(int(np.radians(abs(a1 - a0)) * r / SPACING), 4)
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def line_groups(spec):
    """카메라 기준(동쪽 기준, 가까운 하프 = x < 0) 코트 선 무리: {이름: (N, 2)}. 가까운/먼은 이름 끝 _near/_far.
    사이드라인은 y 부호(pos/neg)로 나눈다. 카메라는 한쪽 사이드라인 밖에 서므로 둘 중 하나는 화면 아래로 잘린다."""
    L2, W2, g, p, r = spec.length / 2, spec.width / 2, spec.goal_w / 2, spec.penalty_r, spec.center_r
    out = {"halfway": _seg((0, -W2), (0, W2))}
    circle = _arc(0, 0, r, 0, 360)
    out["circle_near"], out["circle_far"] = circle[circle[:, 0] <= 0], circle[circle[:, 0] > 0]
    for tag, sx in (("near", -1), ("far", 1)):
        gx = sx * L2
        out[f"goal_{tag}"] = _seg((gx, -W2), (gx, W2))
        out[f"side_pos_{tag}"] = _seg((0, W2), (gx, W2))
        out[f"side_neg_{tag}"] = _seg((0, -W2), (gx, -W2))
        if sx == 1:
            arcs = [_arc(gx, g, p, 90, 180), _arc(gx, -g, p, 180, 270)]
        else:
            arcs = [_arc(gx, g, p, 0, 90), _arc(gx, -g, p, 270, 360)]
        out[f"pen_{tag}"] = np.concatenate(arcs + [_seg((gx - sx * p, -g), (gx - sx * p, g))])
    return out


def to_real(pts, half):
    return pts if half == "east" else -pts


def c2i(H):
    """이미지 -> 코트 H의 역(코트 -> 이미지), 카메라 앞 부호."""
    return front_facing(np.linalg.inv(np.asarray(H, float)))


def east_G(H, half):
    """동쪽 기준 코트 -> 이미지 (keypoint_init이 _G로 쓰는 꼴)."""
    S = np.eye(3) if half == "east" else ROT180
    return front_facing(np.linalg.inv(S @ np.asarray(H, float)))


def group_stats(ridge, G, pts_real, H):
    """한 무리의 (화면 안 샘플 수, 흰 선 일치율, 매칭 잔차 중앙값 px, 같은 잔차의 코트 m)."""
    h, w = ridge.shape
    px, depth = project(G, pts_real)
    vis = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    n = int(vis.sum())
    if n < 10:
        return n, None, None, None
    q = px[vis]
    hit = float((ridge[q[:, 1].astype(int), q[:, 0].astype(int)] > HIT_THRESHOLD).mean())
    src, dst = match_lines(ridge, G, pts_real, RESID_RADIUS)
    if len(src) < 5:
        return n, hit, None, None
    sp = project(G, src)[0]
    rpx = float(np.median(np.linalg.norm(dst - sp, axis=1)))
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.asarray(H, float))[:, 0]
    rm = float(np.median(np.linalg.norm(back - src, axis=1)))
    return n, hit, rpx, rm


def camera_pose(G, size):
    """코트 -> 이미지 G(동쪽 기준)를 주점 = 화면 중앙, 정사각 화소로 분해. 해석이 안 되면 None.
    f: 직교 조건의 초점거리, f_eq: 두 축 길이가 같다는 조건의 초점거리 (맞는 해면 둘이 같다), ratio: 두 축 스케일 비,
    C: 카메라 위치 (동쪽 기준 코트 m, z = 바닥 위 높이), tilt: 광축이 바닥 아래로 숙인 각, roll: 화면 가로축의 기울기."""
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
    d_eq = h1[2] ** 2 - h2[2] ** 2
    f2_eq = (h2[0] ** 2 + h2[1] ** 2 - h1[0] ** 2 - h1[1] ** 2) / d_eq if abs(d_eq) > 1e-18 else -1
    f_eq = float(np.sqrt(f2_eq)) if f2_eq > 0 else float("nan")
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ h1, Kinv @ h2, Kinv @ A[:, 2]
    ratio = float(np.linalg.norm(r1) / np.linalg.norm(r2))
    s = 1 / np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
    r1, r2, t = r1 * s, r2 * s, t * s
    R = np.c_[r1, r2, np.cross(r1, r2)]  # 코트 -> 카메라
    C = -R.T @ t
    if C[2] < 0:  # 코트 z 축 부호: 카메라가 바닥 위(+z)에 오도록 뒤집는다
        R = R @ np.diag([1, 1, -1.0])
        C = C * np.array([1, 1, -1.0])
    axis = R[2, :]  # 광축의 코트 좌표 방향
    xcam = R[0, :]  # 화면 가로축의 코트 좌표 방향
    return {"f": f, "f_eq": f_eq, "ratio": ratio, "cx": float(C[0]), "cy": float(C[1]), "height": float(C[2]),
            "tilt": float(np.degrees(np.arcsin(np.clip(-axis[2], -1, 1)))),
            "roll": float(np.degrees(np.arcsin(np.clip(xcam[2], -1, 1)))),
            "hfov": float(np.degrees(2 * np.arctan(size[0] / 2 / f)))}


def pose_feats(prefix, cam, spec):
    if cam is None:
        return {f"{prefix}_ok": 0}
    L2, W2 = spec.length / 2, spec.width / 2
    out = {f"{prefix}_ok": 1, **{f"{prefix}_{k}": v for k, v in cam.items()}}
    # 코트 밖으로 나간 거리 (양수 = 밖): 가까운 골라인 뒤 / 사이드라인 밖 중 큰 쪽
    out[f"{prefix}_behind_goal"] = -L2 - cam["cx"]
    out[f"{prefix}_outside_side"] = abs(cam["cy"]) - W2
    out[f"{prefix}_outside"] = max(out[f"{prefix}_behind_goal"], out[f"{prefix}_outside_side"])
    out[f"{prefix}_logratio"] = abs(np.log(cam["ratio"]))
    out[f"{prefix}_fdis"] = abs(np.log(cam["f_eq"] / cam["f"])) if np.isfinite(cam["f_eq"]) else 9.0
    return out


def features(ridge, H_near, H_far, half, spec, size):
    """한 결과(H_near, H_far, 이미지 -> 실제 코트)의 후보 특징 dict."""
    Gn, Gf = c2i(H_near), c2i(H_far)
    groups = line_groups(spec)
    f = {}
    tot_n = tot_hit = 0
    by_region = {"near": [0, 0.0], "far": [0, 0.0]}
    for name, pts in groups.items():
        region = "far" if name.endswith("_far") else "near"
        G, H = (Gf, H_far) if region == "far" else (Gn, H_near)
        n, hit, rpx, rm = group_stats(ridge, G, to_real(pts, half), H)
        f[f"n_{name}"], f[f"hit_{name}"], f[f"rpx_{name}"], f[f"rm_{name}"] = n, hit, rpx, rm
        f[f"vis_{name}"] = n / len(pts)
        if hit is not None:
            tot_n += n
            tot_hit += hit * n
            by_region[region][0] += n
            by_region[region][1] += hit * n
    f["vis_all"] = tot_n / sum(len(p) for p in groups.values())
    f["hit_all"] = tot_hit / tot_n if tot_n else 0.0
    for region, (n, h) in by_region.items():
        f[f"hit_{region}"] = h / n if n else 0.0
    # 가장 나쁜 무리 (화면에 충분히 보이는 무리 중): 틀린 해는 몇 무리만 맞추고 나머지는 빗나가는 경우가 많다
    seen = [k for k in groups if f[f"n_{k}"] >= 40 and f[f"hit_{k}"] is not None]
    f["hit_min_group"] = min((f[f"hit_{k}"] for k in seen), default=0.0)
    f["hit_min_group_far"] = min((f[f"hit_{k}"] for k in seen if k.endswith("_far")), default=None)
    # 물리 해석: 가까운 쪽과 먼 쪽 행렬이 각각 암시하는 카메라. 바닥이 평면이면 같은 카메라여야 한다
    pn = camera_pose(east_G(H_near, half), size)
    pf = camera_pose(east_G(H_far, half), size)
    f.update(pose_feats("cam", pn, spec))
    f.update(pose_feats("camfar", pf, spec))
    if pn is not None and pf is not None:
        f["camfar_dpos"] = float(np.linalg.norm([pn["cx"] - pf["cx"], pn["cy"] - pf["cy"], pn["height"] - pf["height"]]))
        f["camfar_df"] = abs(np.log(pf["f"] / pn["f"]))
    else:
        f["camfar_dpos"], f["camfar_df"] = 99.0, 9.0
    return f


def rectify(ridge, H_near, H_far, half):
    """흰 선 지도를 실제 코트 평면 격자(RECT_STEP m)로 펼친다. 화면 밖·카메라 뒤는 NaN. (2*RECT_Y/step+1, 2*RECT_X/step+1)."""
    xs = np.arange(-RECT_X, RECT_X + 1e-9, RECT_STEP)
    ys = np.arange(-RECT_Y, RECT_Y + 1e-9, RECT_STEP)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    far = P[:, 0] * HALF_SIGN[half] > 0
    px = np.empty((len(P), 2))
    ok = np.zeros(len(P), bool)
    for sel, H in ((~far, H_near), (far, H_far)):
        q, d = project(c2i(H), P[sel])
        px[sel] = q
        ok[sel] = d > 0
    h, w = ridge.shape
    ok &= (px[:, 0] >= 0) & (px[:, 0] < w - 1) & (px[:, 1] >= 0) & (px[:, 1] < h - 1)
    v = np.full(len(P), np.nan, np.float32)
    if ok.any():  # 쌍선형 보간 (cv2.remap은 32767개가 넘는 점을 한 번에 받지 못한다)
        q = px[ok]
        x0, y0 = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
        fx, fy = q[:, 0] - x0, q[:, 1] - y0
        v[ok] = (ridge[y0, x0] * (1 - fx) * (1 - fy) + ridge[y0, x0 + 1] * fx * (1 - fy)
                 + ridge[y0 + 1, x0] * (1 - fx) * fy + ridge[y0 + 1, x0 + 1] * fx * fy)
    return v.reshape(X.shape).astype(np.float16)


# ----------------------------------------------------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------------------------------------------------

def setting_tag(aw, faw):
    return f"aw{aw:g}_faw{faw:g}"


def solve_camera(match, n, model, log):
    out_dir = match_dir(match)
    cal = load(n, out_dir)
    spec = spec_of(cal)
    source = Source(cal["video"])
    cache = os.path.join(out_dir, "cache_autocal")
    if ea.DETECT_CACHE is not None:
        ea.DETECT_CACHE.open(cache)
    times = ea.pool_times(cal, source.duration, 15)
    rows, rect_auto, rect_ref = [], [], []
    for t in times:
        t0 = time.time()
        frame = ea.cached_median(source, cal["video"], t, cache)
        r = ea.solve_window(frame, cal, t, spec, "model", model, 0.0, None) if frame is not None else {"status": "failed"}
        r.setdefault("confidence", 0.0)
        near, far = ea.window_errors(r, cal, t, spec)
        Hn_ref, Hf_ref = ea.reference(cal, t + ea.WINDOW / 2)
        ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)  # keypoint_init과 같은 흐림
        size = (frame.shape[1], frame.shape[0])
        row = {"t": t, "status": r["status"], "confidence": r["confidence"], "n_near": r.get("n_near"),
               "n_far": r.get("n_far"), "flipped": bool(r.get("flipped", False)), "near_m": near, "far_m": far,
               "wrong": r["status"] == "ok" and ea.is_wrong(near, far), "size": list(size),
               "H_near_ref": np.asarray(Hn_ref).tolist(), "H_far_ref": np.asarray(Hf_ref).tolist(),
               "ref_feats": features(ridge, Hn_ref, Hf_ref, cal["half"], spec, size)}
        rect_ref.append(rectify(ridge, Hn_ref, Hf_ref, cal["half"]))
        if r["status"] == "ok":
            row["H_near"], row["H_far"] = np.asarray(r["H_near"]).tolist(), np.asarray(r["H_far"]).tolist()
            row["feats"] = features(ridge, r["H_near"], r["H_far"], cal["half"], spec, size)
            rect_auto.append(rectify(ridge, r["H_near"], r["H_far"], cal["half"]))
        else:
            rect_auto.append(np.full_like(rect_ref[-1], np.nan))
        rows.append(row)
        ft = row.get("feats", {})
        log(f"  {match} cam{n} t={t:7.1f} {r['status']:7s} 일치 {r['confidence']:.2f} near {near if near is not None else -1:5.2f} "
            f"far {far if far is not None else -1:5.2f} 먼쪽일치 {ft.get('hit_far', -1):.2f} 높이 {ft.get('cam_height', -1):.1f} "
            f"카메라차 {ft.get('camfar_dpos', -1):.1f} ({time.time() - t0:.1f}s)")
    if ea.DETECT_CACHE is not None:
        ea.DETECT_CACHE.flush()
    return {"match": match, "cam": n, "half": cal["half"], "spec": spec.__dict__, "windows": rows}, \
        np.stack(rect_auto), np.stack(rect_ref)


def cmd_solve(a):
    os.makedirs(OUT, exist_ok=True)
    aw, faw = (float(x) for x in a.setting.split(","))
    tag = setting_tag(aw, faw)
    log_path = os.path.join(OUT, f"solve_{tag}.log")
    logf = open(log_path, "a", encoding="utf-8", newline="\n")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    model = ki.load_model()
    ea.DETECT_CACHE = ea.DetectCache()
    path = os.path.join(OUT, f"windows_{tag}.json")
    data = {"setting": {"anchor_w": aw, "far_anchor_w": faw}, "cams": []}
    if os.path.exists(path):  # 다른 회차를 이미 푼 파일이면 이어 붙인다
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    with ea.variant(aw, faw, None):
        log(f"설정 {tag}: {ea.effective_variant()}")
        for match in a.match:
            data["cams"] = [c for c in data["cams"] if c["match"] != match]
            for n in a.cams:
                res, ra, rr = solve_camera(match, n, model, log)
                data["cams"].append(res)
                np.savez_compressed(os.path.join(OUT, f"rect_{tag}_{match}_cam{n}.npz"), auto=ra, ref=rr)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, default=float)
            log(f"-> {path}")
    log(f"검출 캐시: 적중 {ea.DETECT_CACHE.hits}, 새로 검출 {ea.DETECT_CACHE.misses}")


# ----------------------------------------------------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------------------------------------------------

def blur_nan(m, sigma_cells):
    """NaN을 비운 칸으로 보고 정규화한 가우시안 흐림."""
    v = np.nan_to_num(m.astype(np.float32))
    w = np.isfinite(m).astype(np.float32)
    k = (0, 0)
    num, den = cv2.GaussianBlur(v, k, sigma_cells), cv2.GaussianBlur(w, k, sigma_cells)
    out = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
    return out.astype(np.float32)


def ncc(a, b, mask):
    if mask.sum() < 200:
        return None
    x, y = a[mask], b[mask]
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else None


def cross_feats(ra, rb, band_x=8.0, max_shift=4.0, step=0.2):
    """두 카메라가 같은 시각에 펼친 흰 선 지도의 일치. 둘 다 맞으면 같은 바닥 무늬가 같은 자리에 온다.
    ncc0: 그대로 겹친 상관, shift_m: 상관이 최대인 평행이동 크기 (하프라인 +-band_x m 띠, 0.2 m 격자)."""
    sig = 0.15 / RECT_STEP
    A, B = blur_nan(ra, sig), blur_nan(rb, sig)
    ny, nx = A.shape
    xs = np.arange(nx) * RECT_STEP - RECT_X
    band = np.abs(xs) <= band_x
    both = np.isfinite(A) & np.isfinite(B)
    out = {"x_overlap": float(both[:, band].mean()), "x_ncc0_band": ncc(A[:, band], B[:, band], both[:, band]),
           "x_ncc0_all": ncc(A, B, both)}
    # 평행이동 탐색: 0.2 m 격자로 줄여서
    s = int(round(step / RECT_STEP))
    A2, B2 = A[::s, ::s], B[::s, ::s]
    band2 = band[::s]
    m = int(round(max_shift / step))
    best = (-2.0, 0, 0)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            Bs = np.full_like(B2, np.nan)
            ys_a = slice(max(dy, 0), B2.shape[0] + min(dy, 0))
            ys_b = slice(max(-dy, 0), B2.shape[0] + min(-dy, 0))
            xs_a = slice(max(dx, 0), B2.shape[1] + min(dx, 0))
            xs_b = slice(max(-dx, 0), B2.shape[1] + min(-dx, 0))
            Bs[ys_a, xs_a] = B2[ys_b, xs_b]
            mk = np.isfinite(A2) & np.isfinite(Bs)
            mk[:, ~band2] = False
            c = ncc(A2, Bs, mk)
            if c is not None and c > best[0]:
                best = (c, dx, dy)
    out["x_ncc_best"] = best[0]
    out["x_shift_m"] = float(np.hypot(best[1], best[2]) * step)
    return out


def grid_diff(Ga, Gb, size, region, spec):
    """동쪽 기준 G 두 개가 화면 안 격자(가까운 하프 또는 먼 하프)에서 내는 코트 좌표 차이 중앙값 (m). same_solution과 같은 방식."""
    if region == "near":
        xs = np.linspace(-spec.length * 0.4, 0, 6)
    else:
        xs = np.linspace(0, spec.length * 0.4, 6)
    X, Y = np.meshgrid(xs, np.linspace(-spec.width / 2, spec.width / 2, 5))
    world = np.c_[X.ravel(), Y.ravel()]
    px, depth = project(Ga, world)
    ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])
    if ok.sum() < 3:
        return 99.0
    back = cv2.perspectiveTransform(px[ok].reshape(-1, 1, 2), np.linalg.inv(Gb))[:, 0]
    return float(np.median(np.linalg.norm(back - world[ok], axis=1)))


def load_setting(tag):
    with open(os.path.join(OUT, f"windows_{tag}.json"), encoding="utf-8") as f:
        data = json.load(f)
    cams = {}
    for c in data["cams"]:
        rect = np.load(os.path.join(OUT, f"rect_{tag}_{c['match']}_cam{c['cam']}.npz"))
        c["rect_auto"], c["rect_ref"] = rect["auto"], rect["ref"]
        cams[(c["match"], c["cam"])] = c
    return cams


def prepare(cams):
    """시점마다 keypoint_init.decide가 쓰는 꼴의 결과, 시점 사이 먼 쪽 차이, 카메라 사이 특징을 붙인다."""
    for (match, n), c in cams.items():
        spec = spec_of({"spec": c["spec"]})
        c["specobj"] = spec
        res = []
        for w in c["windows"]:
            if w["status"] == "ok":
                res.append({"status": "ok", "H_near": w["H_near"], "confidence": w["confidence"],
                            "_G": east_G(w["H_near"], c["half"]), "_Gf": east_G(w["H_far"], c["half"]),
                            "_size": tuple(w["size"])})
            else:
                res.append({"status": w["status"], "H_near": None, "confidence": w["confidence"]})
        c["results"] = res
        k = len(res)
        c["dnear"], c["dfar"] = np.full((k, k), np.nan), np.full((k, k), np.nan)
        for i, j in itertools.product(range(k), range(k)):
            if res[i]["H_near"] is not None and res[j]["H_near"] is not None:
                c["dnear"][i, j] = grid_diff(res[i]["_G"], res[j]["_G"], res[i]["_size"], "near", spec)
                c["dfar"][i, j] = grid_diff(res[i]["_Gf"], res[j]["_Gf"], res[i]["_size"], "far", spec)
    for (match, n), c in cams.items():
        other = cams.get((match, 3 - n))
        for i, w in enumerate(c["windows"]):
            w["cross"], w["cross_ref"], w["cross_vref"] = None, None, None
            if other is None:
                continue
            j = int(np.argmin([abs(o["t"] - w["t"]) for o in other["windows"]]))
            if abs(other["windows"][j]["t"] - w["t"]) > 60:
                continue
            ra = c["rect_auto"][i].astype(np.float32)
            if w["status"] == "ok" and other["windows"][j]["status"] == "ok":
                w["cross"] = {**cross_feats(ra, other["rect_auto"][j].astype(np.float32)), "other_wrong": other["windows"][j]["wrong"]}
            if w["status"] == "ok":
                # 다른 카메라가 맞았다고 가정한 경우 (그 카메라의 정답 행렬): 1008은 두 카메라가 모두 틀려서,
                # 이 값이 '맞는 상대와 비교하면 틀린 해를 잡아내는가'를 따로 보여 준다
                w["cross_vref"] = cross_feats(ra, other["rect_ref"][j].astype(np.float32))
            # 참고용: 두 카메라 모두 정답 행렬일 때의 값 (맞는 해가 받는 점수의 기준)
            w["cross_ref"] = cross_feats(c["rect_ref"][i].astype(np.float32), other["rect_ref"][j].astype(np.float32))


def simulate(cams, runs=300, k=5, seed=0):
    """eval_autocal.simulate와 같은 난수 순서(회차마다 seed, cam1 다음 cam2)로 세션을 만든다."""
    sessions = []
    for match in sorted({m for m, _ in cams}):
        rng = np.random.default_rng(seed)
        for n in (1, 2):
            c = cams.get((match, n))
            if c is None:
                continue
            res = c["results"]
            for _ in range(runs):
                pick = sorted(rng.choice(list(range(len(res))), size=k, replace=False).tolist())
                lead, members, status = ki.decide([res[i] for i in pick], c["specobj"])
                if lead is None:
                    sessions.append({"match": match, "cam": n, "status": status, "lead": None})
                    continue
                li = pick[lead]
                mem = [pick[m] for m in members]
                others = [m for m in mem if m != li]
                w = c["windows"][li]
                s = {"match": match, "cam": n, "status": status, "lead": li, "votes": len(mem),
                     "wrong": bool(ea.is_wrong(w["near_m"], w["far_m"])),
                     "spread_near": float(np.nanmax(c["dnear"][li, others])) if others else 0.0,
                     "spread_far": float(np.nanmax(c["dfar"][li, others])) if others else 0.0,
                     "spread_far_med": float(np.nanmedian(c["dfar"][li, others])) if others else 0.0,
                     "min_member_conf": float(min(res[m]["confidence"] for m in mem)),
                     "min_member_hitfar": float(min(c["windows"][m]["feats"]["hit_far"] for m in mem))}
                sessions.append(s)
    return sessions


def window_value(w, key):
    """시점 w의 특징 값 (없으면 None). x_: 두 카메라 자동 결과 비교, xr_: 다른 카메라 정답과 비교(진단용)."""
    if key == "confidence":
        return w["confidence"]
    if key.startswith("x_"):
        return (w.get("cross") or {}).get(key)
    if key.startswith("xr_"):
        return (w.get("cross_vref") or {}).get("x_" + key[3:])
    f = w.get("feats") or {}
    if key == "cam_absroll":
        return abs(f["cam_roll"]) if "cam_roll" in f else None
    return f.get(key)


# 특징: (이름, 방향) 방향 +1이면 값이 클수록 좋음 (문턱 이상 통과), -1이면 작을수록 좋음
CANDIDATES = [
    ("confidence", +1), ("hit_near", +1), ("hit_far", +1), ("hit_all", +1), ("hit_min_group", +1),
    ("hit_min_group_far", +1), ("hit_halfway", +1), ("hit_circle_near", +1), ("hit_circle_far", +1),
    ("hit_goal_far", +1), ("hit_pen_far", +1), ("hit_pen_near", +1), ("hit_side_pos_far", +1), ("hit_side_neg_far", +1),
    ("hit_side_pos_near", +1), ("hit_side_neg_near", +1),
    ("rpx_halfway", -1), ("rpx_circle_near", -1), ("rpx_circle_far", -1), ("rm_circle_near", -1), ("rm_circle_far", -1),
    ("rpx_goal_far", -1), ("vis_all", +1),
    ("cam_height", +1), ("cam_height", -1), ("cam_outside", +1), ("cam_logratio", -1), ("cam_fdis", -1),
    ("cam_tilt", +1), ("cam_tilt", -1), ("cam_absroll", -1), ("cam_hfov", +1), ("cam_hfov", -1),
    ("camfar_height", +1), ("camfar_logratio", -1), ("camfar_dpos", -1), ("camfar_df", -1),
    ("x_ncc0_band", +1), ("x_ncc0_all", +1), ("x_ncc_best", +1), ("x_shift_m", -1),
    ("spread_near", -1), ("spread_far", -1), ("spread_far_med", -1), ("min_member_conf", +1), ("min_member_hitfar", +1),
]
DIAG = [("xr_ncc0_band", +1), ("xr_shift_m", -1)]  # 진단용 (배포 때는 다른 카메라 정답이 없다)
SESSION_KEYS = {"spread_near", "spread_far", "spread_far_med", "min_member_conf", "min_member_hitfar", "votes"}


class Table:
    """통과(status ok)한 세션마다 대표 시점의 특징 값 배열. 문턱 평가를 벡터로 한다."""

    def __init__(self, cams_by_tag, sessions_by_tag):
        rows = []
        self.cells = {}
        for tag, sessions in sessions_by_tag.items():
            cams = cams_by_tag[tag]
            for s in sessions:
                key = (tag, s["match"], s["cam"])
                self.cells.setdefault(key, 0)
                self.cells[key] += 1
                if s["status"] != "ok":
                    continue
                w = cams[(s["match"], s["cam"])]["windows"][s["lead"]]
                v = {k: s[k] for k in SESSION_KEYS}
                for k, _ in CANDIDATES + DIAG:
                    if k not in SESSION_KEYS:
                        v[k] = window_value(w, k)
                rows.append({"cell": key, "wrong": s["wrong"], "near": w["near_m"], "far": w["far_m"], "v": v})
        self.rows = rows
        self.cell = [r["cell"] for r in rows]
        self.match = np.array([r["cell"][1] for r in rows])
        self.wrong = np.array([r["wrong"] for r in rows], bool)
        self.near = np.array([np.nan if r["near"] is None else r["near"] for r in rows], float)
        self.far = np.array([np.nan if r["far"] is None else r["far"] for r in rows], float)
        self.vals = {k: np.array([np.nan if r["v"][k] is None else r["v"][k] for r in rows], float)
                     for k in rows[0]["v"]}

    def mask(self, rule):
        m = np.ones(len(self.rows), bool)
        for k, d, t in rule:
            v = self.vals[k]
            with np.errstate(invalid="ignore"):
                m &= (v >= t) if d > 0 else (v <= t)  # NaN(계산 불가)은 탈락
        return m

    def count(self, rule, sel):
        m = self.mask(rule) & sel
        return int(m.sum()), int((m & self.wrong).sum())


def tune_single(tb, sel, k, d):
    """sel 세션에서 틀린 통과가 0이 되는 가장 느슨한 문턱. 틀린 통과가 원래 없으면 None."""
    v, bad = tb.vals[k][sel], tb.wrong[sel]
    vb = v[bad]
    if not len(vb):
        return None
    vb = vb[np.isfinite(vb)]
    if not len(vb):
        return -np.inf if d > 0 else np.inf  # 틀린 것은 모두 계산 불가 -> 계산만 되면 통과
    return float(vb.max() + 1e-6) if d > 0 else float(vb.min() - 1e-6)


def tune_pair(tb, sel, f1, f2, n_q=30):
    """두 특징 AND: sel에서 틀린 통과 0, 맞는 통과 최대인 문턱 쌍 (t1, t2, 통과 수)."""
    v1, v2, bad = tb.vals[f1[0]][sel], tb.vals[f2[0]][sel], tb.wrong[sel]
    if not bad.any():
        return None

    def cands(v, d):
        v = v[np.isfinite(v)]
        if not len(v):
            return np.array([])
        q = np.unique(np.quantile(v, np.linspace(0, 1, n_q)))
        return np.r_[q, q.max() + 1e-6] if d > 0 else np.r_[q.min() - 1e-6, q]

    c1, c2 = cands(v1, f1[1]), cands(v2, f2[1])
    if not len(c1) or not len(c2):
        return None
    with np.errstate(invalid="ignore"):
        p1 = (v1[None] >= c1[:, None]) if f1[1] > 0 else (v1[None] <= c1[:, None])
        p2 = (v2[None] >= c2[:, None]) if f2[1] > 0 else (v2[None] <= c2[:, None])
    p1f, p2f = p1.astype(np.int32), p2.astype(np.int32)
    acc = p1f @ p2f.T
    wr = (p1f * bad) @ p2f.T
    acc[wr > 0] = -1
    i, j = np.unravel_index(int(acc.argmax()), acc.shape)
    if acc[i, j] < 0:
        return None
    return float(c1[i]), float(c2[j]), int(acc[i, j])


def sym(d):
    return "≥" if d > 0 else "≤"


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


# 물리 해석으로 정한 문턱 (데이터로 고르지 않음): 폰 + 삼각대 높이, 코트 밖, 수평, 정사각 화소, 폰 화각
PHYSICS_RULE = [("cam_height", +1, 1.0), ("cam_height", -1, 5.0), ("cam_outside", +1, 0.0), ("cam_absroll", -1, 5.0),
                ("cam_logratio", -1, float(np.log(1.15))), ("cam_hfov", +1, 45.0), ("cam_hfov", -1, 90.0)]


def cell_table(tb, rule, out, title):
    out(f"\n### {title}")
    out("| 설정 | 회차 cam | 세션 | 기존 통과 | 기존 틀린 통과 | 게이트 통과 | 게이트 틀린 통과 | 게이트 통과 near/far 중앙 (m) |")
    out("|---|---|---|---|---|---|---|---|")
    m = tb.mask(rule)
    cells = np.array([f"{c[0]}|{c[1]}|{c[2]}" for c in tb.cell])
    for key in sorted(tb.cells, key=lambda c: (c[1], c[2], c[0])):
        sel = cells == f"{key[0]}|{key[1]}|{key[2]}"
        ns = tb.cells[key]
        a0, w0 = int(sel.sum()), int((sel & tb.wrong).sum())
        a1, w1 = int((sel & m).sum()), int((sel & m & tb.wrong).sum())
        err = f"{np.nanmedian(tb.near[sel & m]):.2f} / {np.nanmedian(tb.far[sel & m]):.2f}" if a1 else "-"
        out(f"| {key[0]} | {key[1]} cam{key[2]} | {ns} | {a0} ({a0 / ns:.0%}) | {w0} | {a1} ({a1 / ns:.0%}) | {w1} | {err} |")


def parse_rule(text):
    """'hit_far>=0.45,cam_height<=6' -> [(key, dir, thr)]."""
    rule = []
    for part in text.split(","):
        if ">=" in part:
            k, v = part.split(">=")
            rule.append((k.strip(), +1, float(v)))
        else:
            k, v = part.split("<=")
            rule.append((k.strip(), -1, float(v)))
    return rule


def cmd_analyze(a):
    tags = [setting_tag(*(float(x) for x in s.split(","))) for s in a.settings]
    cams_by_tag, sessions_by_tag = {}, {}
    lines = []

    def out(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    for tag in tags:
        cams = load_setting(tag)
        t0 = time.time()
        prepare(cams)
        cams_by_tag[tag] = cams
        sessions_by_tag[tag] = simulate(cams)
        print(f"{tag}: 준비 {time.time() - t0:.0f}s", flush=True)
    tb = Table(cams_by_tag, sessions_by_tag)
    sel = {m: tb.match == m for m in ("1008", "1103")}
    sel["all"] = np.ones(len(tb.rows), bool)
    out(f"설정: {', '.join(tags)}. 세션 = eval_autocal과 같은 난수(회차마다 seed 0, 300회, 시점 5개). "
        f"틀림 = 가까운 쪽 > {ea.WRONG_NEAR_M} m 또는 먼 쪽 > {ea.WRONG_FAR_M} m.")

    # 1) 시점별 특징 분포
    win_rows = []
    keys = [k for k, _ in CANDIDATES + DIAG if k not in SESSION_KEYS]
    keys = list(dict.fromkeys(keys))
    for tag in tags:
        for (match, n), c in sorted(cams_by_tag[tag].items()):
            for w in c["windows"]:
                if w["status"] == "ok":
                    win_rows.append({"tag": tag, "match": match, "cam": n, "t": w["t"], "wrong": w["wrong"],
                                     "near_m": w["near_m"], "far_m": w["far_m"],
                                     "other_wrong": (w["cross"] or {}).get("other_wrong"),
                                     **{k: window_value(w, k) for k in keys}})
    with open(os.path.join(OUT, "window_features.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(win_rows, f, ensure_ascii=False, indent=0, default=float)
    out("\n## 시점별 특징 분포 (status ok 시점, 모든 설정; 최소 / 중앙 / 최대)")
    groups = [("1103 맞음", "1103", False), ("1103 틀림", "1103", True), ("1008 맞음", "1008", False), ("1008 틀림", "1008", True)]
    out("| 특징 | " + " | ".join(g[0] for g in groups) + " |")
    out("|---|" + "---|" * len(groups))
    for k in keys:
        cells = []
        for _, match, wrong in groups:
            v = np.array([r[k] for r in win_rows if r["match"] == match and r["wrong"] == wrong and r[k] is not None], float)
            v = v[np.isfinite(v)]
            cells.append(f"{v.min():.2f} / {np.median(v):.2f} / {v.max():.2f} (n={len(v)})" if len(v) else "-")
        out(f"| {k} | " + " | ".join(cells) + " |")

    out("\n## 정답 행렬의 물리 해석 (카메라별, 시점 범위)")
    for (match, n), c in sorted(cams_by_tag[tags[0]].items()):
        rf = [w["ref_feats"] for w in c["windows"]]
        xr = [w["cross_ref"] for w in c["windows"] if w["cross_ref"]]
        out(f"- {match} cam{n}: 높이 {min(f['cam_height'] for f in rf):.2f}~{max(f['cam_height'] for f in rf):.2f} m, "
            f"위치 x {np.median([f['cam_cx'] for f in rf]):.1f} y {np.median([f['cam_cy'] for f in rf]):.1f} (동쪽 기준), "
            f"코트 밖 {min(f['cam_outside'] for f in rf):.1f} m, 숙임 {np.median([f['cam_tilt'] for f in rf]):.1f}°, "
            f"|roll| 최대 {max(abs(f['cam_roll']) for f in rf):.1f}°, 화각 {np.median([f['cam_hfov'] for f in rf]):.0f}°, "
            f"|log 비| 최대 {max(f['cam_logratio'] for f in rf):.3f}, 먼쪽 카메라차 {np.median([f['camfar_dpos'] for f in rf]):.2f} m, "
            f"일치(가까운/먼) {np.median([f['hit_near'] for f in rf]):.2f}/{np.median([f['hit_far'] for f in rf]):.2f}, "
            f"카메라 사이(정답끼리) ncc0 {fmt(float(np.median([x['x_ncc0_band'] for x in xr])))} "
            f"이동 {fmt(float(np.median([x['x_shift_m'] for x in xr])))} m")

    # 2) 단일 특징 교차 확인
    out("\n## 단일 특징: 한 회차로 문턱을 고르고 다른 회차에서 확인 (통과 / 틀린 통과, 기존 판정 위에 AND)")
    out("문턱 = 조율 회차에서 틀린 통과가 0이 되는 가장 느슨한 값 (여유 없음).")
    out("| 특징 | 방향 | 1103 조율 문턱 | 1103 | → 1008 | 1008 조율 문턱 | 1008 | → 1103 | 전체 조율 문턱 | 전체 |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    b = {m: tb.count([], sel[m]) for m in sel}
    out(f"| (기존 판정) | | | {b['1103'][0]}/{b['1103'][1]} | {b['1008'][0]}/{b['1008'][1]} | | {b['1008'][0]}/{b['1008'][1]} | "
        f"{b['1103'][0]}/{b['1103'][1]} | | {b['all'][0]}/{b['all'][1]} |")
    for k, d in CANDIDATES + DIAG:
        row = [k, sym(d)]
        for tm, cm in (("1103", "1008"), ("1008", "1103")):
            thr = tune_single(tb, sel[tm], k, d)
            if thr is None:
                row += ["-", "-", "-"]
                continue
            rt, rc = tb.count([(k, d, thr)], sel[tm]), tb.count([(k, d, thr)], sel[cm])
            row += [f"{thr:.3f}", f"{rt[0]}/{rt[1]}", f"{rc[0]}/{rc[1]}"]
        thr = tune_single(tb, sel["all"], k, d)
        ra = tb.count([(k, d, thr)], sel["all"]) if thr is not None else None
        row += [fmt(thr, 3), f"{ra[0]}/{ra[1]}" if ra else "-"]
        out("| " + " | ".join(row) + " |")

    # 3) 두 특징 AND
    pool = [c for c in CANDIDATES if c[0] != "confidence"]
    for tm, cm in (("1103", "1008"), ("1008", "1103"), ("all", None)):
        rows = []
        for f1, f2 in itertools.combinations(pool, 2):
            if f1[0] == f2[0]:
                continue
            best = tune_pair(tb, sel[tm], f1, f2)
            if best is None:
                continue
            rule = [(f1[0], f1[1], best[0]), (f2[0], f2[1], best[1])]
            rc = tb.count(rule, sel[cm]) if cm else (0, 0)
            rows.append((f1, f2, best, rc))
        rows.sort(key=lambda r: (r[3][1], -(r[2][2] + r[3][0])))
        title = f"{tm}으로 조율, {cm}에서 확인" if cm else "두 회차 모두로 조율 (확인 없음, 낙관적)"
        out(f"\n## 두 특징 AND - {title} (상위 20)")
        out("| 규칙 | 조율 통과 | 확인 통과/틀림 |")
        out("|---|---|---|")
        for f1, f2, best, rc in rows[:20]:
            out(f"| {f1[0]} {sym(f1[1])} {best[0]:.3f} AND {f2[0]} {sym(f2[1])} {best[1]:.3f} | {best[2]} | "
                f"{(str(rc[0]) + '/' + str(rc[1])) if cm else '-'} |")

    # 4) 물리 문턱
    out("\n## 물리 문턱만 (데이터로 고르지 않음: 높이 1~5 m, 코트 밖, |roll| ≤ 5°, 두 축 비 ≤ 1.15, 화각 45~90°)")
    for m in ("1103", "1008"):
        out(f"- {m}: 통과/틀린 통과 {tb.count(PHYSICS_RULE, sel[m])} (기존 {b[m]})")

    for text in a.rule or []:
        cell_table(tb, parse_rule(text), out, f"규칙 {text}")
    cell_table(tb, [], out, "기존 판정 (확인용: eval_autocal 결과와 같아야 한다)")

    name = a.out or "analysis.md"
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {os.path.join(OUT, name)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("solve")
    p.add_argument("--setting", required=True, help="가까운,먼 앵커 비중 예: 10,30")
    p.add_argument("--match", nargs="+", default=["1008", "1103"])
    p.add_argument("--cams", type=int, nargs="+", default=[1, 2])
    p = sub.add_parser("analyze")
    p.add_argument("--settings", nargs="+", default=["100,30", "10,30", "3,10"])
    p.add_argument("--rule", nargs="*", default=None, help="설정 x 회차 x 카메라 표를 낼 규칙, 예: hit_far>=0.45,cam_height<=6")
    p.add_argument("--out", default=None, help="결과 Markdown 이름 (work/gate/ 안, 기본 analysis.md)")
    a = ap.parse_args()
    if a.cmd == "solve":
        cmd_solve(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()
