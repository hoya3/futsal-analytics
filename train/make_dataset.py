"""코트 키포인트 학습 데이터를 시간별 호모그래피에서 자동으로 만든다 (YOLO pose 형식).

카메라가 고정된 구간의 프레임을 뽑고, 코트 지점을 그 시각의 호모그래피로 화면에 투영해 정답을 만든다.
지점 이름은 '카메라에서 본' 기준이다(near/far, 왼쪽/오른쪽). 그래서 서로 마주 보는 두 카메라가 같은 의미를 갖는다.

회차(out/<회차>/)마다 카메라 배치가 달라(1103은 1008의 좌우 반전 + 더 높은 위치) 여러 회차를 섞어 만든다.

사용법: python train/make_dataset.py [--matches 1008 1103] [--step 3] [--out train/dataset] [--limit 0]
출력: train/dataset/{images,labels}/{train,val}, data.yaml, train/court_pose_dataset.zip, train/preview/*.jpg
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from calib_core import Source, front_facing, match_dir, to_world  # noqa: E402
from court import DEFAULT_SPEC, spec_of  # noqa: E402
from court_map import homographies_at, load  # noqa: E402

IMAGE_WIDTH = 1280  # 저장 해상도
EDGE_MARGIN = 5  # 화면 끝에서 이만큼(px, 1080p) 안쪽만 보이는 점으로 친다
MIN_KEYPOINTS = 4
BOUNDARY_GUARD = 1.0  # 카메라가 움직인 구간 경계에서 제외하는 시간 (s)
VAL_BLOCK = 120.0  # 시간 블록(s) 5개 중 1개를 검증용으로: 이웃 프레임이 학습과 검증에 나뉘지 않게
UNRELIABLE = {"1008": {2: {"far_corner_left", "far_corner_right", "far_arc_left", "far_arc_right", "far_post_left", "far_post_right"}}}
# 회차 -> 카메라 -> 라벨에서 뺄 지점. 1008 검증 점 채점에서 cam2의 먼 골라인은 1.7~3.4 m 어긋났다. 틀린 정답을 가르치지 않도록 뺀다.
MATCHES = ("1008", "1103")


def keypoints(spec=DEFAULT_SPEC):
    """(이름, 동쪽을 바라보는 카메라 기준 좌표). 사람이 볼 수 있는 코트 표시만 쓴다."""
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


# 좌우 반전 증강에서 서로 바뀌는 지점의 인덱스
FLIP_IDX = [1, 0, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10, 12]


def project_keypoints(seg, half, cam, spec=DEFAULT_SPEC, size=(1920, 1080), t=None, match="1008"):
    """구간의 호모그래피로 키포인트를 화면에 투영한다. (N, 3) [x, y, v] (px), v = 2 보임 / 0 라벨 없음.
    t를 주면 구간 안의 느린 밀림(seg["drift"])을 반영한다. 먼 쪽은 1 px가 약 0.3 m라 정답에 밀림이 섞이면 안 된다."""
    skip = UNRELIABLE.get(match, {}).get(cam, set())
    H_near, H_far = homographies_at(seg, t) if t is not None else (seg["H_near"], seg["H_far"])
    out = np.zeros((len(keypoints(spec)), 3))
    for i, (name, xy) in enumerate(keypoints(spec)):
        G = front_facing(np.linalg.inv(np.array(H_near if xy[0] <= 0 else H_far)))
        q = G @ np.r_[to_world(xy, half), 1.0]
        if q[2] <= 0 or name in skip:
            continue
        x, y = q[0] / q[2], q[1] / q[2]
        if EDGE_MARGIN <= x < size[0] - EDGE_MARGIN and EDGE_MARGIN <= y < size[1] - EDGE_MARGIN:
            out[i] = (x, y, 2)
    return out


def yolo_line(kp, size=(1920, 1080), pad=0.03):
    """YOLO pose 라벨 한 줄. 보이는 키포인트가 너무 적으면 None."""
    vis = kp[:, 2] > 0
    if vis.sum() < MIN_KEYPOINTS:
        return None
    w, h = size
    x0, y0 = kp[vis, :2].min(0) / [w, h] - pad
    x1, y1 = kp[vis, :2].max(0) / [w, h] + pad
    x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)
    vals = [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    for x, y, v in kp:
        vals += [x / w, y / h, int(v)] if v > 0 else [0.0, 0.0, 0]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def usable_segment(cal, t):
    """카메라가 안정된 구간이면 그 구간을, 아니면 None."""
    for seg in cal["segments"]:
        if seg["t0"] <= t <= seg["t1"]:
            ok = seg["source"] != "unresolved" and t - seg["t0"] >= BOUNDARY_GUARD and seg["t1"] - t >= BOUNDARY_GUARD
            return seg if ok else None
    return None


def split_of(t):
    return "val" if int(t // VAL_BLOCK) % 5 == 0 else "train"


def data_yaml(root="/content/court_pose"):
    names = [n for n, _ in keypoints()]
    return (f"path: {root}\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [{len(names)}, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: court\n"
            f"# keypoints: {', '.join(names)}\n")


def draw_preview(frame, kp, names):
    view = frame.copy()
    for (x, y, v), name in zip(kp, names):
        if v > 0:
            cv2.circle(view, (int(x), int(y)), 7, (0, 0, 255), 2)
            cv2.putText(view, name, (int(x) + 9, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return view


def main():
    p = argparse.ArgumentParser(description="코트 키포인트 데이터셋 생성")
    p.add_argument("--step", type=float, default=3.0, help="샘플 간격 (s)")
    p.add_argument("--out", default=os.path.join(ROOT, "train", "dataset"))
    p.add_argument("--limit", type=int, default=0, help="카메라당 최대 프레임 수 (시험용)")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--matches", nargs="+", default=list(MATCHES), help="섞을 촬영 회차 (out/<회차>/calib_cam<N>.json)")
    a = p.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", s))
        os.makedirs(os.path.join(a.out, "labels", s))
    preview = os.path.join(ROOT, "train", "preview")
    os.makedirs(preview, exist_ok=True)

    counts = {"train": 0, "val": 0}
    per_match = {}
    for m, n in [(m, n) for m in a.matches for n in (1, 2)]:
        cal = load(n, match_dir(m))
        spec = spec_of(cal)
        if "segments" not in cal:
            raise SystemExit(f"{m} cam{n}: 시간별 캘리브레이션이 없습니다 (FUTSAL_MATCH={m} python track.py {n})")
        names = [k for k, _ in keypoints(spec)]
        source = Source(cal["video"])
        times = np.arange(a.step / 2, source.duration, a.step)
        if a.limit:
            times = times[np.linspace(0, len(times) - 1, a.limit).astype(int)]
        done = 0
        for t in times:
            t = float(t)
            seg = usable_segment(cal, t)
            if seg is None:
                continue
            kp = project_keypoints(seg, cal["half"], n, spec, t=t, match=m)
            line = yolo_line(kp)
            frame = source.frame_at(t) if line else None
            if frame is None:
                continue
            stem, s = f"{m}_cam{n}_{t:07.1f}", split_of(t)
            small = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_WIDTH * frame.shape[0] // frame.shape[1]), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.out, "images", s, stem + ".jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(os.path.join(a.out, "labels", s, stem + ".txt"), "w") as f:
                f.write(line + "\n")
            if done % max(len(times) // 3, 1) == 0:
                cv2.imwrite(os.path.join(preview, stem + ".jpg"), draw_preview(frame, kp, names))
            done += 1
            counts[s] += 1
            per_match[m] = per_match.get(m, 0) + 1
            if done % 100 == 0:
                print(f"{m} cam{n}: {done} / {len(times)}", flush=True)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(data_yaml())
    print(f"학습 {counts['train']}장, 검증 {counts['val']}장, 회차별 {per_match} -> {a.out}", flush=True)
    if not a.no_zip:
        path = shutil.make_archive(os.path.join(ROOT, "train", "court_pose_dataset"), "zip", a.out)
        print(f"-> {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
