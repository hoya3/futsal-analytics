"""클릭 없는 자동 보정(키포인트 모델)의 기준선: 자동 통과율, 통과한 결과의 정확도, 틀렸는데 통과한 비율.

실제 흐름(keypoint_calibrate_video)은 영상에서 5개 시점을 골라 시점마다 코트를 찾고, 과반이 같은 답을 내고 흰 선
일치율이 충분하면 자동 후보(ok)로 낸다. 경기 영상이 몇 개뿐이라, 카메라마다 시점 --pool개를 미리 풀어 두고
그중 5개를 무작위로 고르는 '세션'을 --runs번 만들어 같은 판정(keypoint_init.decide)을 내린다.

정답은 시간별 캘리브레이션(court_map.load, 그 시각의 구간 행렬과 밀림)이다. 오차는 그 정답이 화면에 보여주는
코트 격자(한 하프)를 두 행렬로 코트에 옮긴 차이의 중앙값(m, eval_keypoint_init.region_error). 검증 점이 있는 회차는
라벨 시각마다 자동 결과를 따로 풀어 사람이 찍은 점으로도 채점한다.

모드
  model   train/best.pt (YOLO pose) - 실제 성능
  oracle  키포인트를 정답 위치(+ 가우시안 잡음 --noise px)로 주는 이상적 검출기 - 뒤 단계(ICP, 투표, 판정)만의 한계.
          두 모드의 차이가 모델이 개선할 수 있는 몫이다.
  seg     train/seg_best.ts (라인 세그멘테이션, court_seg.py) + seg_init.seg_calibrate - 실제 성능. 모델 파일이 없으면 건너뛴다.
          클래스 지도(argmax + 최대 확률)는 out/<회차>/cache_autocal/seg_<모델>_<수정시각>/에 압축 npz로 저장해 다시 쓴다
          (--no-seg-cache).
  segoracle  세그멘테이션 확률 = 정답 마스크(make_seg_dataset.render_mask, 시점 창 가운데 시각)의 one-hot. mirror 배치는
          모델 출력처럼 sideline_left/right를 바꿔 넣는다. --seg-noise로 불완전한 모델을 흉내 낸다
          (예: "grow=2,drop=0.2,fp=5": 클래스 영역 2 px 넓힘(음수면 깎음), 48 px 타일 20% 지움, 엉뚱한 선 조각 5개).

사용법:
  python eval/eval_autocal.py --mode oracle            (모델 없이 지금 돌릴 수 있다)
  python eval/eval_autocal.py --mode model oracle --match 1008 1103 --pool 15 --runs 300
  python eval/eval_autocal.py --mode model --match 1008 --anchor-w 10 --far-anchor-w 30 --radii 100,60,40,25,18,12,8,5,3,3
  (앵커 비중·반경 실험. 결과 파일 이름에 설정이 붙는다: out/autocal_model_1008_aw10_faw30_r100-....json)
  python eval/eval_autocal.py --mode segoracle seg --match 1008 1103 --pool 15 --runs 300
  (seg 모드에서 --anchor-w / --far-anchor-w / --radii는 seg_init의 SEG_ANCHOR_W / SEG_FAR_ANCHOR_W / SEG_RADII를 바꾼다)
model 모드의 키포인트 검출은 out/<회차>/cache_autocal/detect_<모델>_<수정시각>.json에 저장해 다시 쓴다 (--no-detect-cache).
결과: 표 출력 + out/autocal_<모드>_<회차>[_설정].json. 시점 중앙값 프레임은 out/<회차>/cache_autocal/에 저장해 다시 쓴다.
1008은 학습에 쓰지 않은 검증 회차(eval/splits.json), 1103은 학습 회차라 낙관적이다.
"""
import argparse
import contextlib
import functools
import hashlib
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

import keypoint_init as ki  # noqa: E402
import refine  # noqa: E402
import seg_init as si  # noqa: E402
from auto_init import median_of  # noqa: E402
from calib_core import HALF_SIGN, Source, front_facing, match_dir, to_world  # noqa: E402
from court import spec_of  # noqa: E402
from court_map import blend_width, homographies_at, load  # noqa: E402
from eval_keypoint_init import region_error  # noqa: E402
from label_points import read_labels  # noqa: E402
from make_dataset import keypoints  # noqa: E402
from score_points import score, summarize  # noqa: E402

WINDOW = 12.0  # median_of(t)가 쓰는 범위: t ~ t + 12 s
WRONG_NEAR_M = 0.3  # 이보다 틀린 자동 결과는 '틀림' (가까운 쪽 격자 오차 중앙값)
WRONG_FAR_M = 1.0  # 먼 쪽 격자 오차 중앙값
DETECT_CACHE = None  # model 모드의 DetectCache (main에서 만든다)
SEG_MODES = ("seg", "segoracle")
SEG_MODEL = None  # seg 모드의 court_seg.SegModel (main에서 읽는다)
SEG_CACHE = None  # seg 모드의 SegCache
SEG_NOISE = {}  # segoracle 잡음 설정 (parse_seg_noise)
SEG_OPTS = {}  # seg_calibrate에 넘길 anchor_w / far_anchor_w / radii (None이면 seg_init 기본값)


def segment_at(cal, t):
    for s in cal["segments"]:
        if s["t0"] <= t <= s["t1"]:
            return None if s["source"] == "unresolved" else s
    return None


def pool_times(cal, duration, n):
    """영상 15~90% 구간에 고르게, 정답 구간이 확정된 시점만."""
    out = []
    for t in np.linspace(duration * 0.15, duration * 0.9, n * 3):
        if segment_at(cal, t) is not None and segment_at(cal, t + WINDOW) is not None:
            out.append(float(round(t, 1)))
    idx = np.linspace(0, len(out) - 1, min(n, len(out))).astype(int) if out else []
    return [out[i] for i in idx]


def cached_median(source, video, t, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{os.path.basename(video)}_{t:g}.png")
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return img
    img = median_of(source, t)
    if img is not None:
        cv2.imwrite(path, img)
    return img


def reference(cal, t):
    """시각 t(중앙값 창의 가운데)의 정답 (H_near, H_far), 이미지 -> 코트."""
    seg = segment_at(cal, t)
    return homographies_at(seg, t) if seg is not None else (None, None)


def oracle_detector(cal, t, spec, noise_px, rng, size):
    """정답 행렬로 키포인트를 화면에 투영한 이상적 검출기. 화면 밖·카메라 뒤 지점은 '안 보임'."""
    H_near, H_far = reference(cal, t)
    half = cal["half"]
    G_near, G_far = front_facing(np.linalg.inv(H_near)), front_facing(np.linalg.inv(H_far))
    w, h = size
    found = {}
    for name, xy in keypoints(spec):
        world = np.array(to_world(xy, half), float)
        G = G_far if world[0] * HALF_SIGN[half] > 0 else G_near
        q = G @ np.array([world[0], world[1], 1.0])
        if q[2] <= 0:
            continue
        p = q[:2] / q[2] + rng.normal(0.0, noise_px, 2)
        if 2 <= p[0] < w - 2 and 2 <= p[1] < h - 2:
            found[name] = p
    return found


@contextlib.contextmanager
def detector(fn):
    """keypoint_init.detect를 잠시 바꾼다 (oracle)."""
    original = ki.detect
    ki.detect = fn
    try:
        yield
    finally:
        ki.detect = original


@contextlib.contextmanager
def variant(anchor_w=None, far_anchor_w=None, radii=None):
    """앵커 비중·ICP 반경을 이번 실행 동안만 바꾼다 (None이면 keypoint_init 기본값 그대로). 끝나면 되돌린다.
    anchor_w: 가까운 쪽 키포인트 하나가 흰 선 매칭 몇 개만큼인가 (icp_align이 행을 반복하므로 정수). 0이면 앵커 행 없음.
    far_anchor_w: 먼 쪽 앵커 비중. _calibrate_once가 icp_align_far를 anchor_w 없이 부르므로(기본값은 정의 시점에 묶임)
    ki.icp_align_far를 잠시 감싼다. 0이면 키포인트로 초기값만 맞추고 ICP에는 앵커를 넣지 않는다."""
    saved = (ki.ANCHOR_W, ki.icp_align_far, ki.KP_RADII)
    original_far = ki.icp_align_far
    if anchor_w is not None:
        ki.ANCHOR_W = int(round(anchor_w))  # np.repeat는 정수 반복 수만 받는다
    if far_anchor_w is not None:
        if far_anchor_w > 0:
            ki.icp_align_far = functools.partial(original_far, anchor_w=far_anchor_w)
        else:
            def far_init_only(ridge, H_near, pts, radii=refine.RADII, anchor_world=None, anchor_img=None):
                # 1) 반경 없이 부르면 앵커만으로 v를 맞춘 초기 행렬이 나온다 (ICP 라운드 없음)
                H0, _ = original_far(ridge, H_near, pts, radii=(), anchor_world=anchor_world, anchor_img=anchor_img)
                # 2) 그 행렬을 기준으로 앵커 없이 ICP. G K(v0) K(v)도 같은 하프라인 고정 꼴이라 탐색 공간이 같다
                return original_far(ridge, H0, pts, radii=radii)
            ki.icp_align_far = far_init_only
    if radii is not None:
        ki.KP_RADII = tuple(radii)
    try:
        yield
    finally:
        ki.ANCHOR_W, ki.icp_align_far, ki.KP_RADII = saved


class DetectCache:
    """keypoint_init.detect 결과를 프레임 내용(md5)별로 메모리와 디스크(JSON)에 둔다. 같은 시점을 여러 설정으로
    돌릴 때 YOLO를 다시 돌리지 않으려는 것. keypoint_calibrate는 원본과 좌우 반전 프레임 둘 다 detect하는데,
    반전 프레임은 내용이 달라 키도 따로 생긴다. 디스크 파일 이름에 모델 파일 수정 시각을 넣어 모델이 바뀌면 새로 쓴다."""

    def __init__(self, model_path=ki.MODEL_PATH):
        self.tag = f"{os.path.splitext(os.path.basename(model_path))[0]}_{int(os.path.getmtime(model_path))}"
        self.path, self.data, self.dirty, self.hits, self.misses = None, {}, False, 0, 0
        self.original = ki.detect

    def open(self, cache_dir):
        """회차의 cache_autocal 폴더에 있는 캐시 파일로 바꾼다."""
        self.flush()
        self.path = os.path.join(cache_dir, f"detect_{self.tag}.json")
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)

    def flush(self):
        if self.path and self.dirty:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.path)
            self.dirty = False

    def __call__(self, frame, model=None, imgsz=960):
        key = f"{hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()}_{frame.shape[1]}x{frame.shape[0]}_{imgsz}"
        if key not in self.data:
            self.misses += 1
            found = self.original(frame, model, imgsz=imgsz)
            self.data[key] = {k: [float(v[0]), float(v[1])] for k, v in found.items()}
            self.dirty = True
        else:
            self.hits += 1
        return {k: np.array(v, np.float32) for k, v in self.data[key].items()}


class SegCache:
    """세그멘테이션 클래스 지도 (argmax uint8 + 최대 확률 float16)를 프레임 내용(md5)과 mirror별로 메모리와 디스크(npz)에
    둔다. 폴더 이름에 모델 파일 이름과 수정 시각을 넣어 모델이 바뀌면 새로 쓴다."""

    def __init__(self, model_path):
        self.tag = f"{os.path.splitext(os.path.basename(model_path))[0]}_{int(os.path.getmtime(model_path))}"
        self.dir, self.mem, self.hits, self.misses = None, {}, 0, 0

    def open(self, cache_dir):
        self.dir = os.path.join(cache_dir, f"seg_{self.tag}")
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}

    def __call__(self, frame, model, mirror):
        key = f"{hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()}_m{int(bool(mirror))}"
        if key in self.mem:
            self.hits += 1
            return self.mem[key]
        path = os.path.join(self.dir, key + ".npz") if self.dir else None
        if path and os.path.exists(path):
            with np.load(path) as z:
                out = (z["labels"], z["conf"].astype(np.float32))
            self.hits += 1
        else:
            out = seg_predict(frame, model, mirror)
            self.misses += 1
            if path:
                tmp = path[:-4] + ".tmp.npz"
                np.savez_compressed(tmp, labels=out[0], conf=out[1].astype(np.float16))
                os.replace(tmp, path)
        self.mem[key] = out
        return out


def seg_predict(frame, model, mirror):
    """court_seg로 (labels, conf) 원본 해상도."""
    import court_seg
    prob = court_seg.predict(frame, model, mirror=mirror)
    return prob.argmax(axis=0).astype(np.uint8), prob.max(axis=0).astype(np.float32)


def parse_seg_noise(text):
    """"grow=2,drop=0.2,fp=5" -> dict. 빈 문자열이면 {}."""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split("=")
            if k.strip() not in ("grow", "drop", "fp"):
                raise ValueError(f"--seg-noise 항목은 grow / drop / fp 중 하나: {k!r}")
            out[k.strip()] = float(v)
    return out


def seg_noise_suffix(noise):
    return "".join(f"_{k}{v:g}" for k, v in sorted(noise.items()))


def perturb_labels(labels, noise, rng):
    """불완전한 모델 흉내: grow(px, 클래스 영역 넓힘/음수면 깎음), drop(48 px 타일을 지우는 비율), fp(엉뚱한 선 조각 수)."""
    out = labels.copy()
    grow = int(round(noise.get("grow", 0)))
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1, 2 * abs(grow) + 1))
        res = np.zeros_like(out) if grow < 0 else out.copy()
        for c in range(1, len(si.NAMES)):
            m = (labels == c).astype(np.uint8)
            if not m.any():
                continue
            if grow > 0:
                res[(cv2.dilate(m, k) > 0) & (res == 0)] = c
            else:
                res[cv2.erode(m, k) > 0] = c
        out = res
    drop = noise.get("drop", 0.0)
    if drop > 0:
        t = 48
        h, w = out.shape
        gone = rng.random(((h + t - 1) // t, (w + t - 1) // t)) < drop
        out[np.kron(gone, np.ones((t, t), bool))[:h, :w]] = 0
    h, w = out.shape
    for _ in range(int(noise.get("fp", 0))):
        p = rng.uniform([0, h * 0.1], [w, h])
        ang, length = rng.uniform(0, np.pi), rng.uniform(80, 400)
        q = p + length * np.array([np.cos(ang), np.sin(ang)])
        cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), int(rng.integers(1, 9)), int(rng.integers(3, 11)))
    return out


def segoracle_labels(cal, t, spec, size, match="", cam=1, noise=None, rng=None):
    """정답 마스크(시각 t의 정답 행렬)를 모델 출력처럼: one-hot (labels, conf). mirror면 sideline_left/right 교환.
    정답 구간이 없으면 None."""
    H_near, H_far = reference(cal, t)
    if H_near is None:
        return None
    mask = si_render_mask(H_near, H_far, cal["half"], cam, match, spec, size, blend_width(cal))
    mask[mask >= len(si.NAMES)] = 0  # 무시(255) -> 배경 (모델은 그곳을 배운 적이 없다)
    if cal.get("mirror"):
        mask = si.true_classes(mask, True)  # 같은 교환표 (자기 역)
    if noise:
        mask = perturb_labels(mask, noise, rng if rng is not None else np.random.default_rng(0))
    return mask, np.ones(mask.shape, np.float32)


def si_render_mask(H_near, H_far, half, cam, match, spec, size, blend):
    import make_seg_dataset as sd
    return sd.render_mask(H_near, H_far, half, cam, match, spec, size, size, blend)


def seg_variant():
    """seg 모드의 실제 설정 (report가 읽는 키 이름은 effective_variant와 같다)."""
    return {"anchor_w": si.SEG_ANCHOR_W if SEG_OPTS.get("anchor_w") is None else int(round(SEG_OPTS["anchor_w"])),
            "far_anchor_w": si.SEG_FAR_ANCHOR_W if SEG_OPTS.get("far_anchor_w") is None else SEG_OPTS["far_anchor_w"],
            "radii": [int(r) for r in (si.SEG_RADII if SEG_OPTS.get("radii") is None else SEG_OPTS["radii"])],
            "dilate": si.SEG_DILATE, "seg_noise": dict(SEG_NOISE), "far_model": SEG_OPTS.get("far_model", "shared")}


def effective_variant():
    """지금 keypoint_init에 걸려 있는 실제 설정 (보고서와 JSON에 남긴다)."""
    far = ki.icp_align_far
    if isinstance(far, functools.partial):
        far_w = far.keywords["anchor_w"]
    elif getattr(far, "__name__", "") == "far_init_only":
        far_w = 0.0
    else:
        far_w = refine.FAR_ANCHOR_W
    return {"anchor_w": ki.ANCHOR_W, "far_anchor_w": far_w, "radii": [int(r) for r in ki.KP_RADII]}


def variant_suffix(a):
    """출력 파일 이름에 붙일 설정 표시. 아무것도 바꾸지 않으면 빈 문자열(예전 이름 그대로)."""
    parts = []
    if a.anchor_w is not None:
        parts.append(f"aw{a.anchor_w:g}")
    if a.far_anchor_w is not None:
        parts.append(f"faw{a.far_anchor_w:g}")
    if a.radii is not None:
        parts.append("r" + "-".join(f"{r:g}" for r in a.radii))
    if getattr(a, "far_model", None) is not None:
        parts.append(f"far{a.far_model}")
    return "".join("_" + p for p in parts)


def solve_window(frame, cal, t, spec, mode, model, noise_px, rng, match=None, cam=None):
    """한 시점의 자동 결과 (keypoint_calibrate와 같은 dict). oracle은 좌우 반전 재시도를 하지 않는다.
    seg / segoracle은 seg_init.seg_calibrate (mirror = 캘리브레이션의 mirror). match / cam은 segoracle 정답 마스크의
    UNRELIABLE 구역(1008 cam2 먼 골대)에 쓴다 (None이면 cal의 _match / _cam)."""
    if mode in SEG_MODES:
        mirror = bool(cal.get("mirror"))
        size = (frame.shape[1], frame.shape[0])
        if mode == "seg":
            seg = SEG_CACHE(frame, SEG_MODEL, mirror) if SEG_CACHE is not None else seg_predict(frame, SEG_MODEL, mirror)
        else:
            seg = segoracle_labels(cal, t + WINDOW / 2, spec, size, match or cal.get("_match", ""),
                                   cam or cal.get("_cam", 1), SEG_NOISE, rng)
            if seg is None:
                return {"status": "failed", "reason": "정답 구간 없음", "H_near": None, "H_far": None, "confidence": 0.0}
        return si.seg_calibrate(frame, cal["half"], spec, seg_probs=seg, mirror=mirror, **SEG_OPTS)
    if mode == "model":
        if DETECT_CACHE is not None:
            with detector(DETECT_CACHE):
                return ki.keypoint_calibrate(frame, cal["half"], spec, model)
        return ki.keypoint_calibrate(frame, cal["half"], spec, model)
    size = (frame.shape[1], frame.shape[0])
    with detector(lambda *_a, **_k: oracle_detector(cal, t + WINDOW / 2, spec, noise_px, rng, size)):
        return ki.keypoint_calibrate(frame, cal["half"], spec, model=None, try_flip=False)


def window_errors(r, cal, t, spec):
    """자동 결과 r의 정답 대비 격자 오차 (가까운 쪽, 먼 쪽 중앙값 m). 실패면 None."""
    if r.get("status") != "ok":
        return None, None
    H_near, H_far = reference(cal, t + WINDOW / 2)
    en = region_error(r["H_near"], H_near, cal["half"], spec, "near")
    ef = region_error(r["H_far"], H_far, cal["half"], spec, "far")
    return (en[0] if en else None), (ef[0] if ef else None)


def is_wrong(near, far):
    return (near is None or near > WRONG_NEAR_M) or (far is not None and far > WRONG_FAR_M)


def simulate(results, errors, spec, runs, k, rng):
    """시점 풀에서 k개를 고르는 세션을 runs번: decide와 같은 판정, 대표 결과의 오차."""
    out = []
    valid = list(range(len(results)))
    if len(valid) < k:
        return out
    for _ in range(runs):
        pick = sorted(rng.choice(valid, size=k, replace=False).tolist())
        lead, members, status = ki.decide([results[i] for i in pick], spec)
        near, far = errors[pick[lead]] if lead is not None else (None, None)
        out.append({"status": status, "votes": len(members), "near": near, "far": far,
                    "wrong": lead is not None and is_wrong(near, far)})
    return out


def pct(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else float("nan")


def run_camera(match, n, mode, args, model, rng):
    out_dir = match_dir(match)
    cal = load(n, out_dir)
    spec = spec_of(cal)
    source = Source(cal["video"])
    cache = os.path.join(out_dir, "cache_autocal")
    if mode == "model" and DETECT_CACHE is not None:
        DETECT_CACHE.open(cache)
    if mode == "seg" and SEG_CACHE is not None:
        SEG_CACHE.open(cache)
    times = pool_times(cal, source.duration, args.pool)
    results, errors, windows = [], [], []
    for t in times:
        frame = cached_median(source, cal["video"], t, cache)
        t0 = time.time()
        r = solve_window(frame, cal, t, spec, mode, model, args.noise, rng, match, n) if frame is not None else {"status": "failed", "H_near": None}
        r.setdefault("confidence", 0.0)
        near, far = window_errors(r, cal, t, spec)
        results.append(r)
        errors.append((near, far))
        windows.append({"t": t, "status": r["status"], "confidence": r["confidence"], "n_near": r.get("n_near"),
                        "n_far": r.get("n_far"), "near_m": near, "far_m": far, "flipped": r.get("flipped", False)})
        if mode in SEG_MODES:
            windows[-1].update({k: r.get(k) for k in ("reason", "far_score", "seg_score", "lines", "classes",
                                                       "class_res_px", "far_class_res_px")})
        print(f"  {mode} {match} cam{n} t={t:7.1f}: {r['status']:7s} 일치율 {r['confidence']:.2f} "
              f"가까운 쪽 {near if near is not None else float('nan'):5.2f} m 먼 쪽 {far if far is not None else float('nan'):5.2f} m "
              f"({time.time() - t0:.1f}s)" + (f" {r.get('reason')}" if r.get("reason") else ""), flush=True)
    sessions = simulate(results, errors, spec, args.runs, args.windows, rng)

    label_rows, label_rows_cal = [], []
    labels = [lb for lb in read_labels(os.path.join(out_dir, "labels.json")) if lb["cam"] == n]
    for lb in labels:
        frame = cached_median(source, cal["video"], lb["t"], cache)
        r = solve_window(frame, cal, lb["t"], spec, mode, model, args.noise, rng, match, n) if frame is not None else {"status": "failed"}
        if r.get("status") == "ok":
            auto = {"half": cal["half"], "spec": spec.__dict__, "H_near": r["H_near"], "H_far": r["H_far"]}
            if mode in SEG_MODES and SEG_OPTS.get("far_model") == "free":
                auto["far_model"] = "free"  # court_map이 하프라인 근처를 섞는다 (직접 캘리브레이션과 같게)
            label_rows += score([lb], lambda _: auto)
    if labels:
        label_rows_cal = score(labels, lambda _: cal)
    if mode == "model" and DETECT_CACHE is not None:
        DETECT_CACHE.flush()
    return {"match": match, "cam": n, "mode": mode, "noise_px": args.noise if mode == "oracle" else None,
            "variant": seg_variant() if mode in SEG_MODES else effective_variant(),
            "windows_per_session": args.windows,
            "windows": windows, "sessions": sessions, "labels_auto": summarize(label_rows),
            "labels_calibration": summarize(label_rows_cal), "n_labels": len(labels)}


def report(res):
    w, s = res["windows"], res["sessions"]
    ok_w = [x for x in w if x["status"] == "ok"]
    v = res.get("variant") or {}
    head = (f"{res['mode']:<6} {res['match']} cam{res['cam']} [앵커 가까운 {v.get('anchor_w')} / 먼 {v.get('far_anchor_w')}, "
            f"반경 {','.join(str(r) for r in v.get('radii', []))}]")
    print(f"\n== {head}")
    print(f"  시점별: 성공 {len(ok_w)}/{len(w)}, 가까운 쪽 중앙값 {pct([x['near_m'] for x in ok_w], 50):.2f} m "
          f"(90% {pct([x['near_m'] for x in ok_w], 90):.2f}), 먼 쪽 중앙값 {pct([x['far_m'] for x in ok_w], 50):.2f} m "
          f"(90% {pct([x['far_m'] for x in ok_w], 90):.2f})")
    if res["mode"] in SEG_MODES:
        reasons = {}
        for x in w:
            if x.get("reason"):
                reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        print(f"  세그멘테이션: 일치율 중앙값 {pct([x['confidence'] for x in ok_w], 50):.2f} (최소 "
              f"{min([x['confidence'] for x in ok_w], default=float('nan')):.2f}), 먼 쪽 일치율 "
              f"{pct([x.get('far_score') for x in ok_w], 50):.2f}, 클래스 일치율 {pct([x.get('seg_score') for x in ok_w], 50):.2f}"
              + (f", 실패 이유 {reasons}" if reasons else ""))
    if s:
        acc = [x for x in s if x["status"] == "ok"]
        wrong_acc = [x for x in acc if x["wrong"]]
        print(f"  세션 {len(s)}회 (시점 {res['windows_per_session']}개씩): 자동 통과 {len(acc) / len(s):.0%}, "
              f"통과했는데 틀림 {len(wrong_acc)}회 ({len(wrong_acc) / max(len(acc), 1):.0%} of 통과), "
              f"수동으로 넘어감 {1 - len(acc) / len(s):.0%}")
        if acc:
            print(f"  통과한 결과: 가까운 쪽 {pct([x['near'] for x in acc], 50):.2f} / {pct([x['near'] for x in acc], 90):.2f} m, "
                  f"먼 쪽 {pct([x['far'] for x in acc], 50):.2f} / {pct([x['far'] for x in acc], 90):.2f} m (중앙값 / 90%)")
    if res["n_labels"]:
        for name, rows in (("자동", res["labels_auto"]), ("현재 캘리브레이션", res["labels_calibration"])):
            cells = ", ".join(f"{r['region']} {r['kind']} {r['median']:.2f}/{r['p90']:.2f}" for r in rows)
            print(f"  검증 점 [{name}]: {cells or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", nargs="+", default=["model"], choices=("model", "oracle") + SEG_MODES)
    ap.add_argument("--match", nargs="+", default=["1008"])
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--pool", type=int, default=15, help="카메라마다 미리 푸는 시점 수")
    ap.add_argument("--windows", type=int, default=5, help="세션 하나가 쓰는 시점 수 (keypoint_calibrate_video와 같게)")
    ap.add_argument("--runs", type=int, default=300, help="시점 부분집합 세션 수")
    ap.add_argument("--noise", type=float, default=1.5, help="oracle 키포인트 잡음 (px, 표준편차)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-w", type=float, default=None,
                    help=f"가까운 쪽 키포인트 앵커 비중 (기본 keypoint_init.ANCHOR_W={ki.ANCHOR_W}, 0이면 앵커 없음, 정수로 반올림)")
    ap.add_argument("--far-anchor-w", type=float, default=None,
                    help=f"먼 쪽 앵커 비중 (기본 refine.FAR_ANCHOR_W={refine.FAR_ANCHOR_W:g}, 0이면 초기값에만 쓰고 ICP에는 없음)")
    ap.add_argument("--radii", type=lambda s: [int(x) for x in s.split(",")], default=None,
                    help=f"ICP 탐색 반경 px, 쉼표 목록 (기본 keypoint_init.KP_RADII={','.join(map(str, ki.KP_RADII))})")
    ap.add_argument("--no-detect-cache", action="store_true", help="키포인트 검출 캐시(cache_autocal/detect_*.json)를 쓰지 않는다")
    ap.add_argument("--seg-noise", default="", help='segoracle 잡음, 예 "grow=2,drop=0.2,fp=5" (grow px, drop 타일 비율, fp 조각 수)')
    ap.add_argument("--no-seg-cache", action="store_true", help="세그멘테이션 캐시(cache_autocal/seg_*/)를 쓰지 않는다")
    ap.add_argument("--far-model", choices=refine.FAR_MODELS, default=None,
                    help="seg / segoracle 먼 하프 모델 (기본 seg_init.seg_calibrate의 shared, free = 8자유도)")
    a = ap.parse_args()
    global DETECT_CACHE, SEG_MODEL, SEG_CACHE, SEG_NOISE, SEG_OPTS
    SEG_NOISE = parse_seg_noise(a.seg_noise)
    SEG_OPTS = {"anchor_w": a.anchor_w, "far_anchor_w": a.far_anchor_w, "radii": a.radii}
    if a.far_model is not None:
        SEG_OPTS["far_model"] = a.far_model
    modes = list(a.mode)
    if "seg" in modes:
        import court_seg
        try:
            SEG_MODEL = court_seg.load_seg_model()
        except FileNotFoundError as e:
            print(f"seg 모드를 건너뜁니다: {e}", flush=True)
            modes.remove("seg")
        else:
            print(f"세그멘테이션 모델: {SEG_MODEL}", flush=True)
            if not a.no_seg_cache:
                SEG_CACHE = SegCache(SEG_MODEL.path)
    model = ki.load_model() if "model" in modes else None
    if model is not None and not a.no_detect_cache:
        DETECT_CACHE = DetectCache()
    suffix = variant_suffix(a)
    with variant(a.anchor_w, a.far_anchor_w, a.radii):
        print(f"설정: {effective_variant()}", flush=True)
        if set(modes) & set(SEG_MODES):
            print(f"세그멘테이션 설정: {seg_variant()}", flush=True)
        for mode in modes:
            for match in a.match:
                rng = np.random.default_rng(a.seed)
                results = [run_camera(match, n, mode, a, model, rng) for n in a.cams]
                for r in results:
                    report(r)
                extra = seg_noise_suffix(SEG_NOISE) if mode == "segoracle" else ""
                path = os.path.join(ROOT, "out", f"autocal_{mode}_{match}{suffix}{extra}.json")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump({"settings": vars(a), "variant": seg_variant() if mode in SEG_MODES else effective_variant(),
                               "wrong_near_m": WRONG_NEAR_M, "wrong_far_m": WRONG_FAR_M, "results": results}, f,
                              ensure_ascii=False, indent=1, default=_json_default)
                print(f"-> {path}")
    if DETECT_CACHE is not None:
        print(f"검출 캐시: 적중 {DETECT_CACHE.hits}, 새로 검출 {DETECT_CACHE.misses}")
    if SEG_CACHE is not None:
        print(f"세그멘테이션 캐시: 적중 {SEG_CACHE.hits}, 새로 예측 {SEG_CACHE.misses}")


def _json_default(o):
    """numpy 수 -> float/int (seg 모드 진단의 정수도 들어 있다)."""
    if isinstance(o, np.integer):
        return int(o)
    return float(o)


if __name__ == "__main__":
    main()
