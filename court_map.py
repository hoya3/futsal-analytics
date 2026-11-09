"""한 카메라의 이미지 픽셀 -> 코트 미터 변환. 가까운 하프와 먼 하프의 호모그래피를 사용한다.

    from court_map import load, pixel_to_court
    cam1 = load(1)
    xy = pixel_to_court(cam1, [[960, 700], [1200, 300]], t=61.2)   # (N, 2) 코트 미터, 원점 = 센터 스폿

코트 좌표: x = 길이 방향, y = 폭 방향, 미터 (기본 규격이면 -20..20, -10..10).

track.py를 돌린 캘리브레이션(version 2)은 카메라가 움직인 구간마다 호모그래피가 다르므로 t가 필수다. load()의 t는
동기화 영상(*_sync.mp4)의 시각(초)이다. t는 스칼라 또는 점마다 하나씩이며, 코트를 확정하지 못한 구간(unresolved)의 점은 NaN을 돌려준다.
"""
import json
import os

import cv2
import numpy as np

from calib_core import HALF_SIGN


def load_anchor(n, out_dir="out"):
    """카메라 n의 단일 프레임 캘리브레이션 (보정본이 있으면 그것을 사용)."""
    refined = os.path.join(out_dir, f"H_cam{n}_refined.json")
    with open(refined if os.path.exists(refined) else os.path.join(out_dir, f"H_cam{n}.json")) as f:
        return json.load(f)


def load_source(n, out_dir="out"):
    """카메라 n의 캘리브레이션, 원본 영상 시각 기준. 시간별 결과(calib_cam<n>.json)가 있으면 그것을 우선 사용."""
    tracked = os.path.join(out_dir, f"calib_cam{n}.json")
    if os.path.exists(tracked):
        with open(tracked) as f:
            return json.load(f)
    return load_anchor(n, out_dir)


def load(n, out_dir="out"):
    """카메라 n의 캘리브레이션, 동기화 영상(out/sync.json) 시각 기준. 팀 파이프라인은 이 함수를 쓴다.
    sync.json이 없거나 시간별 결과가 아니면 load_source와 같다."""
    cal = load_source(n, out_dir)
    sync_path = os.path.join(out_dir, "sync.json")
    if "segments" not in cal or not os.path.exists(sync_path):
        return cal
    with open(sync_path) as f:
        sync = json.load(f)
    name = sync["cameras"][str(n)]
    entry = sync["videos"][name]
    return to_sync_time(cal, os.path.join(os.path.dirname(cal["video"]), name), entry["source_time_of_output_0_s"], entry["duration_s"])


def to_sync_time(cal, video_path, source_time_of_output_0, duration):
    """원본 영상 시각의 시간별 캘리브레이션을 잘라낸 영상(sync.json)의 시각으로 옮긴다. 원본 dict는 바꾸지 않는다.
    잘라낸 영상이 재인코딩 없는 복사라 화면은 같으므로 호모그래피는 그대로 쓰고 시각만 옮긴다."""
    off = source_time_of_output_0
    segments = []
    for seg in cal["segments"]:
        t0, t1 = max(seg["t0"] - off, 0.0), min(seg["t1"] - off, duration)
        if t1 > t0:
            segments.append({**seg, "t0": t0, "t1": t1})
    return {**cal, "video": video_path, "t_sec": cal["t_sec"] - off, "segments": segments}


def _apply(H_near, H_far, half, px):
    xy = cv2.perspectiveTransform(px, np.array(H_near, np.float64))[:, 0]
    far = xy[:, 0] * HALF_SIGN[half] > 0  # 하프라인 너머: 먼 하프 호모그래피 사용
    if far.any():
        xy[far] = cv2.perspectiveTransform(px[far], np.array(H_far, np.float64))[:, 0]
    return xy


def pixel_to_court(cal, px, t=None):
    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    segments = cal.get("segments")
    if segments is None:
        return _apply(cal["H_near"], cal["H_far"], cal["half"], px)
    if t is None:
        raise ValueError("시간별 캘리브레이션에는 t(영상 시각, 초)가 필요합니다")
    times = np.broadcast_to(np.asarray(t, np.float64), len(px))
    which = np.searchsorted([s["t0"] for s in segments], times, side="right") - 1
    out = np.full((len(px), 2), np.nan)
    for i in np.unique(which[(which >= 0)]):
        seg = segments[i]
        sel = (which == i) & (times <= seg["t1"])
        if seg["source"] != "unresolved" and sel.any():
            out[sel] = _apply(seg["H_near"], seg["H_far"], cal["half"], px[sel])
    return out
