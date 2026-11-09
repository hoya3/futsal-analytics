"""합성 코트 영상으로 코트 치수, 픽셀 -> 코트 변환, 경기 전체 추적을 검증한다."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calib_core import click_landmarks, front_facing, solve_half  # noqa: E402
from court import CourtSpec, court_lines, landmarks  # noqa: E402
from court_map import load, load_source, pixel_to_court, to_sync_time  # noqa: E402
from refine import project  # noqa: E402
from track import Sample, TrackConfig, group_samples, track_camera, translation  # noqa: E402

# 실제 cam1 캘리브레이션 값 (이미지 -> 코트, 동쪽을 바라봄)
H_NEAR = np.array([
    [-0.010180008591786995, 0.1760407766849706, -44.98493034090651],
    [0.01916148202257321, 0.09856178768674712, -46.95904931565724],
    [6.233799427706753e-06, -0.007983995938345938, 1.0],
])
G = front_facing(np.linalg.inv(H_NEAR))


def render(Hc2i, spec=CourtSpec(), size=(1920, 1080), seed=0):
    """잔디색 배경에 코트 선을 그린 프레임."""
    rng = np.random.default_rng(seed)
    w, h = size
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = (60, 130, 50)
    frame = np.clip(frame.astype(np.int16) + rng.integers(-12, 12, frame.shape), 0, 255).astype(np.uint8)
    for line in court_lines(spec):
        dense = np.concatenate([np.linspace(a, b, 60, endpoint=False) for a, b in zip(line[:-1], line[1:])])
        px, depth = project(Hc2i, dense)
        keep = (depth > 0) & (np.abs(px) < 1e4).all(1)
        for run in np.split(np.arange(len(dense)), np.where(~keep)[0]):
            run = run[keep[run]]
            if len(run) > 1:
                cv2.polylines(frame, [px[run].astype(np.int32)], False, (235, 235, 235), 5, cv2.LINE_AA)
    return frame


class FakeSource:
    """시각 -> 이동량 함수로 카메라 이동을 흉내 내는 영상."""

    def __init__(self, duration, shift_at, spec=CourtSpec()):
        self.duration = duration
        self.shift_at = shift_at
        self.spec = spec

    def frame_at(self, t):
        if t >= self.duration:
            return None
        shift = self.shift_at(t)
        if shift is None:  # 선이 보이지 않는 구간
            return np.full((1080, 1920, 3), (60, 130, 50), np.uint8)
        return render(translation(*shift) @ G, self.spec, seed=int(t * 10))


def anchor(t=20.0):
    return {"half": "east", "t_sec": t, "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist()}


CFG = TrackConfig(step=5.0, median_frames=3)


def test_spec_scales_court_lines_and_landmarks():
    spec = CourtSpec(length=38, width=18)
    assert landmarks(spec)["corner_ne"] == (19, 9)
    assert landmarks(spec)["pen2_mark_e"] == (9, 0)
    outline = court_lines(spec)[0]
    assert outline[:, 0].max() == 19 and outline[:, 1].max() == 9
    assert click_landmarks(spec)[4]["xy"] == (-9.0, 0.0)


def test_default_court_is_unchanged():
    lms = click_landmarks()
    assert [lm["xy"] for lm in lms] == [(0.0, 10.0), (0.0, -3.0), (0.0, -10.0), (-14.0, -10.0), (-10.0, 0.0)]


def test_solve_half_recovers_synthetic_homography():
    lms = click_landmarks()
    world = np.array([(0, 10), (0, -3), (0, -10), (-13.0, -10), (-10, 0)], float)
    img, _ = project(G, world)
    H, _, err = solve_half(img, lms, "east")
    assert err.max() < 1e-3
    assert np.allclose(cv2.perspectiveTransform(img[None], H)[0], world, atol=1e-3)


def test_pixel_to_court_v1_needs_no_time():
    img, _ = project(G, np.array([[-5.0, 2.0]]))
    assert np.allclose(pixel_to_court(anchor(), img), [[-5.0, 2.0]], atol=1e-3)


def v2(shifted):
    H = np.linalg.inv(translation(5, 0) @ G)
    return {"version": 2, "half": "east", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist(), "segments": [
        {"t0": 0.0, "t1": 10.0, "source": "manual", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist(), "score": 1},
        {"t0": 10.0, "t1": 20.0, "source": "unresolved", "H_near": None, "H_far": None, "score": 0},
        {"t0": 20.0, "t1": 30.0, "source": "tracked", "H_near": H.tolist(), "H_far": H.tolist(), "score": 1},
    ]} if shifted else None


def test_pixel_to_court_v2_selects_segment_by_time():
    cal = v2(True)
    court = np.array([[-5.0, 2.0]])
    still, _ = project(G, court)
    moved, _ = project(translation(5, 0) @ G, court)
    assert np.allclose(pixel_to_court(cal, still, t=5.0), court, atol=1e-3)
    assert np.isnan(pixel_to_court(cal, still, t=15.0)).all()
    assert np.allclose(pixel_to_court(cal, moved, t=25.0), court, atol=1e-3)
    assert np.isnan(pixel_to_court(cal, still, t=31.0)).all()
    both = pixel_to_court(cal, np.r_[still, moved], t=[5.0, 25.0])
    assert np.allclose(both, np.r_[court, court], atol=1e-3)


def test_pixel_to_court_v2_requires_time():
    with pytest.raises(ValueError):
        pixel_to_court(v2(True), [[1.0, 2.0]])


def test_to_sync_time_shifts_and_clips_segments():
    cal = {**v2(True), "t_sec": 5.0, "video": "a.mp4"}
    out = to_sync_time(cal, "a_sync.mp4", 8.0, 15.0)
    assert [(s["t0"], s["t1"], s["source"]) for s in out["segments"]] == [
        (0.0, 2.0, "manual"), (2.0, 12.0, "unresolved"), (12.0, 15.0, "tracked")]
    assert out["video"] == "a_sync.mp4" and out["t_sec"] == -3.0
    assert cal["segments"][0]["t0"] == 0.0  # 원본은 그대로
    court = np.array([[-5.0, 2.0]])
    moved, _ = project(translation(5, 0) @ G, court)
    assert np.allclose(pixel_to_court(out, moved, t=13.0), court, atol=1e-3)


def test_load_uses_sync_time_when_sync_file_exists(tmp_path):
    cal = {**v2(True), "t_sec": 25.0, "video": "/videos/camera1.mp4"}
    (tmp_path / "calib_cam1.json").write_text(json.dumps(cal))
    assert load(1, tmp_path)["segments"][2]["t0"] == 20.0  # sync.json이 없으면 원본 시각
    sync = {"cameras": {"1": "a_sync.mp4"}, "videos": {"a_sync.mp4": {"source_time_of_output_0_s": 8.0, "duration_s": 15.0}}}
    (tmp_path / "sync.json").write_text(json.dumps(sync))
    out = load(1, tmp_path)
    assert [s["t0"] for s in out["segments"]] == [0.0, 2.0, 12.0]
    assert out["video"] == "/videos/a_sync.mp4"
    assert load_source(1, tmp_path)["segments"][2]["t0"] == 20.0


def S(t, dx=0.0, dy=0.0, ok=True):
    return Sample(t, dx, dy, 0.9 if ok else 0.0, ok)


def test_group_ignores_single_outlier():
    samples = [S(0), S(5), S(10, 20, 0), S(15), S(20)]
    runs = group_samples(samples, 3.0)
    assert len(runs) == 1 and len(runs[0]["samples"]) == 5
    assert runs[0]["dx"] == [0.0, 0.0, 0.0, 0.0]


def test_group_splits_on_lasting_shift():
    runs = group_samples([S(0), S(5), S(10, 0, -10), S(15, 0, -10)], 3.0)
    assert [len(r["samples"]) for r in runs] == [2, 2]


def test_group_fills_short_gap_only_when_shift_matches():
    assert len(group_samples([S(0), S(5, ok=False), S(10)], 3.0)) == 1
    assert len(group_samples([S(0), S(5, ok=False), S(10, 15, 0)], 3.0)) == 3
    assert len(group_samples([S(0), S(5, ok=False), S(10, ok=False), S(15, ok=False), S(20)], 3.0)) == 3


def test_track_static_camera_is_one_anchor_segment():
    result = track_camera(FakeSource(60, lambda t: (0, 0)), anchor(), CFG)
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["source"] == "manual" and (seg["t0"], seg["t1"]) == (0.0, 60)
    assert result["version"] == 2


def test_track_finds_drift_and_maps_back_to_court():
    moved_at = 37.0
    source = FakeSource(70, lambda t: (0, 0) if t < moved_at else (-2, -10))
    result = track_camera(source, anchor(), CFG)
    segs = result["segments"]
    assert [s["source"] for s in segs] == ["manual", "tracked"]
    assert abs(segs[0]["t1"] - moved_at) <= 1.0
    assert np.allclose(segs[1]["shift"], (-2, -10), atol=1.5)
    world = np.array([[-5.0, 0.0], [-10.0, 3.0], [-3.0, -7.0]])
    px, _ = project(translation(-2, -10) @ G, world)
    err = np.linalg.norm(pixel_to_court(result, px, t=60.0) - world, axis=1)
    assert err.max() < 0.25


def test_track_marks_lost_period_unresolved():
    source = FakeSource(70, lambda t: None if t < 12 else (0, 0))
    result = track_camera(source, anchor(40.0), CFG)
    segs = result["segments"]
    assert segs[0]["source"] == "unresolved" and segs[0]["t0"] == 0.0
    assert segs[-1]["source"] == "manual"
    assert 10.0 <= segs[0]["t1"] <= 15.0


def test_track_rejects_wrong_anchor():
    bad = anchor()
    bad["H_near"] = np.linalg.inv(translation(300, 200) @ G).tolist()
    with pytest.raises(ValueError):
        track_camera(FakeSource(30, lambda t: (0, 0)), bad, CFG)
