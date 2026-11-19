"""합성 코트 영상으로 코트 치수, 픽셀 -> 코트 변환, 경기 전체 추적을 검증한다."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

from calib_core import ROOT, click_landmarks, front_facing, match_dir, solve_half  # noqa: E402
from court import CourtSpec, court_lines, landmarks  # noqa: E402
from court_map import load, load_source, pixel_to_court, to_sync_time  # noqa: E402
from refine import project, sample_model  # noqa: E402
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
            if len(run) > 1:  # 1/16 px 고정소수점: 서브픽셀 밀림도 그림에 나타난다
                cv2.polylines(frame, [np.rint(px[run] * 16).astype(np.int32)], False, (235, 235, 235), 5, cv2.LINE_AA, shift=4)
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
    assert [lm["xy"] for lm in lms[:7]] == [(0.0, 10.0), (0.0, -3.0), (0.0, -10.0), (-14.0, -10.0), (-10.0, 0.0), (20.0, 10.0), (20.0, -10.0)]
    assert not any(lm.get("optional") for lm in lms[:7])  # 기존 7클릭 파일과 순서가 같다
    assert [lm["xy"] for lm in lms[7:]] == [(-14.0, 0.0), (-18.0, -10.0)] and all(lm["optional"] for lm in lms[7:])


def test_solve_half_recovers_synthetic_homography():
    lms = click_landmarks()[:5]
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
    (tmp_path / "calib_cam1.json").write_text(json.dumps(cal), encoding="utf-8")
    assert load(1, tmp_path)["segments"][2]["t0"] == 20.0  # sync.json이 없으면 원본 시각
    sync = {"cameras": {"1": "a_sync.mp4"}, "videos": {"a_sync.mp4": {"source_time_of_output_0_s": 8.0, "duration_s": 15.0}}}
    (tmp_path / "sync.json").write_text(json.dumps(sync), encoding="utf-8")
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


# ---- calib_service: 자동 인식은 느리므로 결과를 흉내 내고, 나머지 흐름은 실제 코드로 검증한다


import calib_service as svc  # noqa: E402

CLICK_WORLD = np.array([(0, 10), (0, -3), (0, -10), (-13.0, -10), (-10, 0), (20, 10), (20, -10)], float)
RIGHT_CLICKS = project(G, CLICK_WORLD)[0].tolist()


@pytest.fixture
def service(monkeypatch, tmp_path):
    source = FakeSource(60, lambda t: (0, 0))
    monkeypatch.setattr(svc, "_open", lambda video: source)
    return str(tmp_path)


def auto_stub(monkeypatch, status="ok", H=H_NEAR):
    result = {"status": status, "t_sec": 20.0, "votes": 3, "windows": 5, "confidence": 0.6, "details": [],
              "H_near": H.tolist() if status == "ok" else None, "H_far": H.tolist() if status == "ok" else None}
    monkeypatch.setattr(svc, "find_court", lambda *a, **k: result)


def test_service_auto_candidate_needs_approval_then_tracks(service, monkeypatch):
    auto_stub(monkeypatch)
    view = svc.propose("v.mp4", 1, half="east", root=service)
    assert view["state"] == "needs_review" and view["next"] == "review_overlay" and view["source"] == "auto"
    assert os.path.exists(view["overlay"]) and os.path.exists(view["frame"])
    done = svc.accept(view["session"], root=service, cfg=CFG)
    assert done["state"] == "done" and done["flags"] == []
    cal = load(1, os.path.dirname(done["result"]))
    px, _ = project(G, np.array([[-5.0, 2.0]]))
    assert np.allclose(pixel_to_court(cal, px, t=30.0), [[-5.0, 2.0]], atol=0.05)
    assert cal["segments"][0]["source"] == "auto"


def test_service_low_confidence_goes_to_manual_then_accepts(service, monkeypatch):
    auto_stub(monkeypatch, status="low_confidence")
    view = svc.propose("v.mp4", 1, half="east", root=service)
    assert view["state"] == "needs_manual" and view["next"] == "click_landmarks" and view["overlay"] is None
    with pytest.raises(svc.SessionError):
        svc.accept(view["session"], root=service, cfg=CFG)
    view = svc.submit_manual(view["session"], RIGHT_CLICKS, root=service)
    assert view["state"] == "needs_review" and view["source"] == "manual" and view["warnings"] == []
    assert view["confidence"] > 0.8
    done = svc.accept(view["session"], root=service, cfg=CFG)
    assert done["state"] == "done"
    assert load(1, os.path.dirname(done["result"]))["segments"][0]["source"] == "manual"


def test_service_reject_clears_auto_candidate(service, monkeypatch):
    auto_stub(monkeypatch)
    view = svc.propose("v.mp4", 1, half="east", root=service)
    view = svc.reject(view["session"], root=service)
    assert view["state"] == "needs_manual" and view["H_near"] is None


def test_service_warns_on_wrong_click_order_and_requires_confirmation(service, monkeypatch):
    auto_stub(monkeypatch, status="failed")
    view = svc.propose("v.mp4", 1, half="east", root=service)
    swapped = [RIGHT_CLICKS[2], RIGHT_CLICKS[1], RIGHT_CLICKS[0]] + RIGHT_CLICKS[3:]
    view = svc.submit_manual(view["session"], swapped, root=service)
    assert view["warnings"] and view["status"] == "suspect"
    with pytest.raises(svc.SessionError):
        svc.accept(view["session"], root=service, cfg=CFG)


def test_service_rejects_bad_click_input(service, monkeypatch):
    auto_stub(monkeypatch, status="failed")
    sid = svc.propose("v.mp4", 1, half="east", root=service)["session"]
    with pytest.raises(svc.SessionError):
        svc.submit_manual(sid, RIGHT_CLICKS[:4], root=service)
    with pytest.raises(svc.SessionError):
        svc.submit_manual(sid, [[5000, 10]] + RIGHT_CLICKS[1:], root=service)
    with pytest.raises(svc.SessionError):
        svc.frame("../etc", 1.0, root=service)


def test_service_unresolved_segment_flag_and_manual_fix(monkeypatch, tmp_path):
    source = FakeSource(70, lambda t: None if t < 12 else (0, 0))
    monkeypatch.setattr(svc, "_open", lambda video: source)
    auto_stub(monkeypatch)
    root = str(tmp_path)
    sid = svc.propose("v.mp4", 1, half="east", root=root, times=None)["session"]
    svc.accept(sid, root=root, cfg=CFG)
    flags = svc.review_flags(sid, root=root)["flags"]
    assert len(flags) == 1 and flags[0]["t0"] == 0.0
    source.shift_at = lambda t: (0, 0)  # 사용자가 선이 보이는 프레임을 지정한 상황
    out = svc.resolve_segment(sid, flags[0]["index"], RIGHT_CLICKS, root=root)
    assert out["resolved"] and out["flags"] == []
    with pytest.raises(svc.SessionError):
        svc.resolve_segment(sid, flags[0]["index"], RIGHT_CLICKS, root=root)


def test_service_returns_homography_in_original_pixels(monkeypatch, tmp_path):
    class Big(FakeSource):
        def frame_at(self, t):
            f = super().frame_at(t)
            return None if f is None else cv2.resize(f, None, fx=2, fy=2)

    source = Big(60, lambda t: (0, 0))
    monkeypatch.setattr(svc, "_open", lambda video: source)
    auto_stub(monkeypatch)
    root = str(tmp_path)
    view = svc.propose("v.mp4", 1, half="east", root=root)
    done = svc.accept(view["session"], root=root, cfg=CFG)
    cal = load(1, os.path.dirname(done["result"]))
    px, _ = project(G, np.array([[-5.0, 2.0]]))
    assert np.allclose(pixel_to_court(cal, px * 2, t=30.0), [[-5.0, 2.0]], atol=0.05)


def test_service_refuses_portrait_video(monkeypatch, tmp_path):
    class Tall:
        duration = 10.0

        def frame_at(self, t):
            return np.zeros((1920, 1080, 3), np.uint8)

    monkeypatch.setattr(svc, "_open", lambda video: Tall())
    with pytest.raises(svc.SessionError):
        svc.propose("v.mp4", 1, half="east", root=str(tmp_path))


def test_service_cli_prints_only_json(service, monkeypatch, capsys):
    auto_stub(monkeypatch, status="failed")
    assert svc.main(["--root", service, "propose", "v.mp4", "--cam", "1", "--half", "east"]) == 0
    sid = json.loads(capsys.readouterr().out)["session"]
    assert svc.main(["--root", service, "manual", sid, "--clicks", json.dumps(RIGHT_CLICKS)]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "needs_review"
    assert svc.main(["--root", service, "flags", sid]) == 1  # 추적 전: 오류도 JSON으로
    assert "error" in json.loads(capsys.readouterr().out)


# ---- 깨끗한 2차원 코트 맵


def test_clean_map_removes_moving_players_and_fills_court():
    import topdown

    ROT = np.diag([-1.0, -1.0, 1.0])
    H2 = ROT @ H_NEAR  # 서쪽을 보는 카메라는 코트가 180도 돌아가 보인다

    class Moving:
        duration = 100.0

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def frame_at(self, t):
            frame = render(G, seed=int(t))
            x, y = self.rng.integers(300, 1600), self.rng.integers(300, 900)
            frame[y:y + 160, x:x + 70] = (255, 0, 255)  # 프레임마다 다른 곳에 있는 선수
            return frame

    def cal(H, half):
        seg = {"t0": 0.0, "t1": 100.0, "source": "manual", "H_near": H.tolist(), "H_far": H.tolist()}
        return {"half": half, "video": half, "segments": [seg]}

    sources = {"east": Moving(1), "west": Moving(2)}
    canvas, geo = topdown.build({1: cal(H_NEAR, "east"), 2: cal(H2, "west")}, open_source=lambda v: sources[v], frames=9)
    assert canvas.shape[:2] == (geo.size[1], geo.size[0])
    magenta = (canvas[..., 0] > 200) & (canvas[..., 1] < 80) & (canvas[..., 2] > 200)
    assert magenta.mean() < 0.001
    assert canvas.any(2).mean() > 0.8  # 코트 둘레 여백은 비어 있다


def test_synthetic_map_draws_court_at_true_coordinates():
    import plot_players
    import topdown

    spec = CourtSpec(length=38, width=18)
    canvas, geo = topdown.render_court(spec)
    assert (geo.size[1], geo.size[0]) == canvas.shape[:2] == (600, 1200)
    cx, cy = plot_players.to_canvas((0, 0), geo)
    assert (cx, cy) == (600, 300)
    assert (canvas[cy + 150, cx] > 200).all()  # 하프라인 위
    assert (canvas[cy + 150, cx + 200] == topdown.GRASS).all()  # 빈 잔디
    corner = plot_players.to_canvas((19, 9), geo)
    assert corner == (1170, 30) and (canvas[corner[1] + 40, corner[0]] > 200).all()  # 동쪽 골라인 위
    assert (canvas[10, 10] == topdown.GRASS).all()  # 여백


# ---- 검증 점 라벨링과 채점


def _clicks_for(points, Gm, half):
    """검증 지점을 호모그래피로 화면에 찍은 가상의 사람 클릭. 선 지점은 표시된 xy를 쓴다."""
    from calib_core import to_world

    world = np.array([to_world(lm["xy"], half) for lm in points], float)
    return project(Gm, world)[0].tolist()


def test_validation_points_do_not_reuse_calibration_clicks():
    import label_points as lp

    fit = {lm["xy"] for lm in click_landmarks() if lm["kind"] == "point"}
    pts = lp.validation_points()
    assert not fit & {lm["xy"] for lm in pts if lm["kind"] == "point"}
    assert len({lm["id"] for lm in pts}) == len(pts)
    assert {lm["region"] for lm in pts} == {"near", "center", "far"}


def test_score_is_zero_for_exact_clicks_and_measures_a_known_offset():
    import label_points as lp
    import score_points as sp

    cal = {"half": "east", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist()}
    pts = lp.validation_points()
    clicks = _clicks_for(pts, G, "east")
    rows = sp.score([lp.make_label(1, 300.0, "v", "east", pts, clicks)], lambda n: cal)
    assert len(rows) == len(pts) and max(r["error"] for r in rows) < 0.02

    world = np.array([[-5.0, 3.0]])
    px = project(G, world + [0.5, 0.0])[0][0].tolist()
    label = {"cam": 1, "t": 1.0, "half": "east", "points": [{"id": "x", "kind": "point", "region": "near", "px": px, "world": world[0].tolist()}]}
    assert sp.score([label], lambda n: cal)[0]["error"] == pytest.approx(0.5, abs=0.01)


def test_label_curves_follow_camera_direction():
    import label_points as lp

    east = lp.curve_points("right_sideline", "east")
    assert (east[:, 1] == -10).all()
    assert (lp.curve_points("right_sideline", "west")[:, 1] == 10).all()
    assert lp.curve_points("near_arc", "east")[:, 0].max() < 0
    assert lp.curve_points("near_arc", "west")[:, 0].min() > 0


def test_score_marks_unresolved_time_without_failing():
    import label_points as lp
    import score_points as sp

    pts = lp.validation_points()[:2]
    label = lp.make_label(1, 15.0, "v", "east", pts, _clicks_for(pts, G, "east"))
    rows = sp.score([label], lambda n: {**v2(True), "spec": None})
    assert [r["error"] for r in rows] == [None, None]
    assert sp.summarize(rows) == []


def test_labeler_skip_undo_and_save_format(tmp_path):
    import label_points as lp

    tool = lp.Labeler(np.zeros((1080, 1920, 3), np.uint8), "west", CourtSpec())
    tool.click(100, 200)
    tool.skip()
    tool.click(300, 400)
    tool.undo()
    assert tool.clicks == [(100, 200), None]
    assert tool.render().shape == (1080, 1920, 3)
    while not tool.done:
        tool.skip()
    label = lp.make_label(2, 300.0, "v", "west", tool.lms, tool.clicks)
    assert len(label["points"]) == 1 and label["points"][0]["world"] == [-0.0, -3.0]  # 서쪽 카메라는 좌표가 뒤집힌다
    path = str(tmp_path / "labels.json")
    lp.write_labels([label], path)
    assert lp.read_labels(path) == [label]


# ---- 코트 키포인트 데이터셋


def test_flip_index_matches_left_right_mirror():
    import make_dataset as md

    kps = [xy for _, xy in md.keypoints()]
    assert len(md.FLIP_IDX) == len(kps)
    for i, j in enumerate(md.FLIP_IDX):
        assert kps[j][0] == kps[i][0] and kps[j][1] == pytest.approx(-kps[i][1])


def seg_of(H):
    return {"H_near": H.tolist(), "H_far": H.tolist()}


def test_keypoints_project_to_same_pixels_for_facing_cameras():
    import make_dataset as md

    east = md.project_keypoints(seg_of(H_NEAR), "east", 1)
    west = md.project_keypoints(seg_of(np.diag([-1.0, -1.0, 1.0]) @ H_NEAR), "west", 1)
    assert np.allclose(east, west, atol=1e-6)
    assert (east[:, 2] > 0).sum() >= 4


def test_unreliable_cam2_far_keypoints_are_not_labeled():
    import make_dataset as md

    names = [n for n, _ in md.keypoints()]
    far = [names.index(n) for n in md.UNRELIABLE["1008"][2]]
    assert (md.project_keypoints(seg_of(H_NEAR), "east", 1)[far, 2] > 0).any()
    cam2 = md.project_keypoints(seg_of(H_NEAR), "east", 2)
    assert (cam2[far, 2] == 0).all() and (cam2[names.index("far_10m"), 2] > 0)
    other = md.project_keypoints(seg_of(H_NEAR), "east", 2, match="1103")  # 회차마다 따로 정한다
    assert (other[far, 2] > 0).any()


def test_yolo_line_format_and_too_few_points():
    import make_dataset as md

    kp = md.project_keypoints(seg_of(H_NEAR), "east", 1)
    fields = md.yolo_line(kp).split()
    assert len(fields) == 5 + 3 * len(kp) and fields[0] == "0"
    assert all(0.0 <= float(v) <= 2.0 for v in fields[1:])
    few = kp.copy()
    few[3:] = 0
    assert md.yolo_line(few) is None


def test_dataset_skips_unresolved_and_moving_periods():
    import make_dataset as md

    cal = v2(True)
    assert md.usable_segment(cal, 5.0) is not None
    assert md.usable_segment(cal, 15.0) is None
    assert md.usable_segment(cal, 9.5) is None  # 구간 경계 근처
    assert md.split_of(10.0) == "val" and md.split_of(130.0) == "train"


# ---- 먼 쪽 코너 2점


def _with_far_bend(v):
    return G @ (np.eye(3) + np.outer(v, [1.0, 0.0, 0.0]))


def test_far_corners_set_far_homography_and_leave_near_and_halfway_unchanged():
    from calib_core import manual_result

    lms = click_landmarks()
    true_far = _with_far_bend(np.array([0.002, 0.004, 0.0006]))
    clicks = project(G, CLICK_WORLD[:5])[0].tolist() + project(true_far, CLICK_WORLD[5:])[0].tolist()
    res = manual_result(clicks, lms, "east")
    near, far = res["near"][0], res["far"][0]
    assert np.allclose(near / near[2, 2], H_NEAR / H_NEAR[2, 2], atol=1e-3)
    assert np.allclose(project(front_facing(np.linalg.inv(far)), CLICK_WORLD[5:])[0], clicks[5:], atol=0.05)
    halfway = np.array([[0.0, -8.0], [0.0, 5.0]])
    assert np.allclose(project(front_facing(np.linalg.inv(far)), halfway)[0], project(G, halfway)[0], atol=0.05)


def test_old_five_click_files_keep_working():
    from refine import far_anchors

    old = {"half": "east", "img_pts": [[1.0, 2.0]] * 5, "world": {"far": {"0": [0.0, 10.0], "2": [0.0, -10.0]}}}
    world, img = far_anchors(old)
    assert world.shape == (0, 2) and img.shape == (0, 2)


def test_far_fit_keeps_anchor_when_white_lines_are_missing():
    from refine import icp_align_far

    true_far = _with_far_bend(np.array([0.002, 0.004, 0.0006]))
    img = project(true_far, CLICK_WORLD[5:])[0]
    H, _ = icp_align_far(np.zeros((1080, 1920), np.float32), H_NEAR, CLICK_WORLD, anchor_world=CLICK_WORLD[5:], anchor_img=img)
    assert np.allclose(project(front_facing(np.linalg.inv(H)), CLICK_WORLD[5:])[0], img, atol=0.05)


def test_tracking_keeps_far_corner_constraint_after_camera_moves():
    from calib_core import manual_json, manual_result

    lms = click_landmarks()
    d = manual_json(manual_result(RIGHT_CLICKS, lms, "east"), RIGHT_CLICKS, lms, "east", video="v", t=20.0)
    source = FakeSource(70, lambda t: (0, 0) if t < 37 else (-2, -10))
    result = track_camera(source, d, CFG)
    moved = result["segments"][-1]
    assert moved["source"] == "tracked"
    px = project(translation(-2, -10) @ G, CLICK_WORLD[5:])[0]
    assert np.allclose(cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.array(moved["H_far"]))[:, 0], CLICK_WORLD[5:], atol=0.6)


def test_service_warns_when_far_corners_are_swapped(service, monkeypatch):
    auto_stub(monkeypatch, status="failed")
    sid = svc.propose("v.mp4", 1, half="east", root=service)["session"]
    view = svc.submit_manual(sid, RIGHT_CLICKS[:5] + [RIGHT_CLICKS[6], RIGHT_CLICKS[5]], root=service)
    assert any("먼 쪽 코너" in w for w in view["warnings"])


def test_score_excludes_fit_corners_from_existing_labels():
    import score_points as sp

    cal = {"half": "east", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist()}
    label = {"cam": 1, "t": 1.0, "half": "east", "points": [
        {"id": "far_corner_left", "kind": "point", "region": "far", "px": [1.0, 1.0], "world": [20.0, 10.0]},
        {"id": "far_10m", "kind": "point", "region": "far", "px": project(G, np.array([[10.0, 0.0]]))[0][0].tolist(), "world": [10.0, 0.0]}]}
    assert [r["id"] for r in sp.score([label], lambda n: cal)] == ["far_10m"]


def test_load_does_not_shift_results_tracked_on_the_synced_video(tmp_path):
    cal = {**v2(True), "t_sec": 25.0, "video": "/videos/a_sync.mp4"}
    (tmp_path / "calib_cam1.json").write_text(json.dumps(cal), encoding="utf-8")
    sync = {"cameras": {"1": "a_sync.mp4"}, "videos": {"a_sync.mp4": {"source_time_of_output_0_s": 8.0, "duration_s": 15.0}}}
    (tmp_path / "sync.json").write_text(json.dumps(sync), encoding="utf-8")
    assert [s["t0"] for s in load(1, tmp_path)["segments"]] == [0.0, 10.0, 20.0]


def test_line_matching_finds_subpixel_center():
    from refine import match_lines

    ys, xs = np.mgrid[0:300, 0:400].astype(np.float32)
    ridge = np.exp(-((ys - 150.3) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
    pts = np.c_[np.arange(50.0, 350.0, 0.2), np.full(1500, 150.0)]
    _, dst = match_lines(ridge, np.eye(3), pts, 6)
    assert abs(np.median(dst[:, 1]) - 150.3) < 0.1


def test_anchor_segment_far_homography_is_refined_without_changing_near():
    from calib_core import manual_json, manual_result

    lms = click_landmarks()
    d = manual_json(manual_result(RIGHT_CLICKS, lms, "east"), RIGHT_CLICKS, lms, "east", video="v", t=20.0)
    d["H_far"] = np.linalg.inv(translation(3, 2) @ G).tolist()  # 일부러 어긋난 먼 쪽
    result = track_camera(FakeSource(40, lambda t: (0, 0)), d, CFG)
    seg = result["segments"][0]
    assert seg["source"] == "manual" and seg["H_near"] == d["H_near"]
    world = CLICK_WORLD[5:]
    assert np.allclose(project(front_facing(np.linalg.inv(np.array(seg["H_far"]))), world)[0], project(G, world)[0], atol=1.0)



# ---- 구간 안의 느린 밀림 (tol_px보다 작아 구간을 나누지 않는 이동)


def with_drift(drift):
    cal = v2(True)
    cal["segments"][0]["drift"] = drift
    return cal


def test_pixel_to_court_interpolates_drift_within_segment():
    cal = with_drift([[0.0, 0.0, 0.0], [8.0, 2.0, -1.0]])
    court = np.array([[-5.0, 2.0], [-12.0, -6.0]])
    moved, _ = project(translation(1.0, -0.5) @ G, court)  # t = 4: 절반만큼 밀림
    assert np.allclose(pixel_to_court(cal, moved, t=4.0), court, atol=1e-3)
    end, _ = project(translation(2.0, -1.0) @ G, court)  # 마지막 측정 뒤는 마지막 값을 유지
    assert np.allclose(pixel_to_court(cal, end, t=[8.0, 9.5]), court, atol=1e-3)
    still, _ = project(G, court)  # 밀림이 없는 구간은 그대로
    assert np.allclose(pixel_to_court(v2(True), still, t=5.0), court, atol=1e-3)


def test_homographies_at_matches_pixel_to_court():
    from court_map import homographies_at

    cal = with_drift([[0.0, 1.0, 0.5], [10.0, 3.0, -1.5]])
    seg = cal["segments"][0]
    px, _ = project(translation(2.0, -0.5) @ G, np.array([[-8.0, 1.0]]))
    H_near, _ = homographies_at(seg, 5.0)
    via_H = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H_near)[:, 0]
    assert np.allclose(via_H, pixel_to_court(cal, px, t=5.0), atol=1e-6)


def test_to_sync_time_moves_drift_times():
    cal = {**with_drift([[1.0, 0.0, 0.0], [9.0, 2.0, 0.0]]), "t_sec": 5.0, "video": "a.mp4"}
    out = to_sync_time(cal, "a_sync.mp4", 0.5, 30.0)
    assert out["segments"][0]["drift"] == [[0.5, 0.0, 0.0], [8.5, 2.0, 0.0]]
    assert cal["segments"][0]["drift"][0][0] == 1.0  # 원본은 그대로


def test_track_measures_slow_drift_below_segment_tolerance():
    def shift(t):  # 1 px씩 두 번 밀린다: 합쳐도 tol_px(3) 안이라 구간은 하나
        return (0, 0) if t < 25 else (1, 0) if t < 50 else (2, 0)

    result = track_camera(FakeSource(75, shift), anchor(), TrackConfig(step=5.0, median_frames=3, drift_step=10.0, drift_frames=3))
    segs = result["segments"]
    assert len(segs) == 1 and segs[0]["source"] == "manual"
    drift = np.array(segs[0]["drift"])
    assert np.allclose(drift[drift[:, 0] < 20, 1:], 0.0, atol=0.3)
    assert np.allclose(drift[drift[:, 0] > 55, 1:], [2.0, 0.0], atol=0.3)
    world = np.array([[15.0, 0.0], [18.0, 5.0], [10.0, -6.0]])  # 먼 쪽: 1 px가 큰 거리
    px, _ = project(translation(2, 0) @ G, world)
    err_fixed = np.linalg.norm(pixel_to_court({**result, "segments": [{**segs[0], "drift": None}]}, px, t=65.0) - world, axis=1)
    err = np.linalg.norm(pixel_to_court(result, px, t=65.0) - world, axis=1)
    assert err.max() < 0.3 * err_fixed.max()


def test_dataset_labels_follow_drift_within_segment():
    import make_dataset as md

    seg = {**seg_of(H_NEAR), "drift": [[0.0, 0.0, 0.0], [10.0, 2.0, -1.0]]}
    still = md.project_keypoints(seg, "east", 1)
    moved = md.project_keypoints(seg, "east", 1, t=10.0)
    vis = (still[:, 2] > 0) & (moved[:, 2] > 0)
    assert vis.sum() >= 4
    assert np.allclose(moved[vis, :2] - still[vis, :2], [2.0, -1.0], atol=1e-6)


# ---- 키포인트 모델 기반 자동 초기화 (모델 대신 정답 지점을 넣어 기하만 검증)


def fake_detect(names):
    import make_dataset as md

    def detect(frame, model=None):
        return {k: project(G, np.array([xy]))[0][0] for k, xy in md.keypoints() if k in names}
    return detect


def test_keypoint_init_recovers_homography_from_halfway_and_far_points(monkeypatch):
    import keypoint_init as ki

    monkeypatch.setattr(ki, "detect", fake_detect({"halfway_left", "halfway_right", "circle_halfway_left",
                                                   "circle_halfway_right", "near_10m", "far_10m"}))
    r = ki.keypoint_calibrate(render(G), "east")
    assert r["status"] == "ok"
    world = np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0]])
    px, _ = project(G, world)
    assert np.linalg.norm(pixel_to_court({"half": "east", **r}, px) - world, axis=1).max() < 0.2


def test_keypoint_init_refuses_points_that_do_not_fix_a_homography(monkeypatch):
    import keypoint_init as ki

    # 하프라인 4점 + 가까운 10 m 지점: 제약 7개라 호모그래피가 정해지지 않는다
    monkeypatch.setattr(ki, "detect", fake_detect({"halfway_left", "halfway_right", "circle_halfway_left",
                                                   "circle_halfway_right", "near_10m"}))
    assert ki.keypoint_calibrate(render(G), "east")["status"] == "failed"


def test_keypoint_video_needs_majority_and_line_match(monkeypatch):
    import keypoint_init as ki

    monkeypatch.setattr(ki, "detect", fake_detect({"halfway_left", "halfway_right", "circle_halfway_left",
                                                   "circle_halfway_right", "near_10m", "far_10m"}))
    monkeypatch.setattr(ki, "median_of", lambda source, t: source.frame_at(t))
    r = ki.keypoint_calibrate_video(FakeSource(60, lambda t: (0, 0)), "east", times=[10.0, 20.0, 30.0])
    assert r["status"] == "ok" and r["votes"] == 3 and r["method"] == "keypoint" and r["confidence"] >= ki.ACCEPT_SCORE
    # 지점은 맞지만 화면에 흰 선이 없으면 일치율이 낮아 자동 후보로 내놓지 않는다
    blank = ki.keypoint_calibrate_video(FakeSource(60, lambda t: None), "east", times=[10.0, 20.0, 30.0])
    assert blank["status"] == "low_confidence" and blank["confidence"] < ki.ACCEPT_SCORE


def test_service_uses_keypoint_model_only_when_weights_exist(monkeypatch, tmp_path):
    import keypoint_init as ki

    called = []
    monkeypatch.setattr(svc, "auto_calibrate_video", lambda *a, **k: called.append("pose_grid") or {"status": "failed"})
    monkeypatch.setattr(ki, "keypoint_calibrate_video", lambda *a, **k: called.append("keypoint") or {"status": "failed"})
    monkeypatch.setattr(svc, "SEG_MODELS", (str(tmp_path / "missing.ts"),))
    monkeypatch.setattr(svc, "KEYPOINT_MODEL", str(tmp_path / "missing.pt"))
    svc.find_court(None, "east", CourtSpec())
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    monkeypatch.setattr(svc, "KEYPOINT_MODEL", str(weights))
    svc.find_court(None, "east", CourtSpec())
    assert called == ["pose_grid", "keypoint"]


def test_service_prefers_segmentation_model(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import keypoint_init as ki
    import seg_init as si

    called = []
    monkeypatch.setattr(ki, "keypoint_calibrate_video", lambda *a, **k: called.append("keypoint") or {"status": "failed"})
    monkeypatch.setattr(si, "seg_calibrate_video", lambda *a, **k: called.append("seg") or {"status": "failed"})
    seg = tmp_path / "seg_best.ts"
    seg.write_bytes(b"")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    monkeypatch.setattr(svc, "SEG_MODELS", (str(seg),))
    monkeypatch.setattr(svc, "KEYPOINT_MODEL", str(weights))
    svc.find_court(None, "east", CourtSpec())
    assert called == ["seg"]


def test_seg_video_picks_mirror_once_and_reuses_it(monkeypatch):
    import seg_init as si

    Gi = np.array([[60.0, 0.0, 960.0], [0.0, 30.0, 700.0], [0.0, 0.0, 1.0]])  # 판정(decide)은 바꿔 끼운다
    calls = []

    def fake(frame, half, spec, model=None, mirror=False, far_model="shared", **k):
        calls.append((mirror, far_model))
        if mirror:  # 이 영상은 반전 배치: 맞는 배치만 풀린다
            return {"status": "ok", "H_near": np.linalg.inv(Gi).tolist(), "H_far": np.linalg.inv(Gi).tolist(),
                    "confidence": 0.9, "seg_score": np.float64(0.95), "n_near": np.int64(4), "_G": Gi, "_size": (1920, 1080)}
        return {"status": "failed", "H_near": None, "confidence": 0.0, "seg_score": 0.2}

    monkeypatch.setattr(si, "seg_calibrate", fake)
    monkeypatch.setattr("auto_init.median_of", lambda source, t: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr("keypoint_init.decide", lambda results, spec: (0, [0, 1, 2], "ok"))
    r = si.seg_calibrate_video(FakeSource(60, lambda t: None), "east", CourtSpec(), times=[10.0, 20.0, 30.0], model=object())
    assert [m for m, _ in calls] == [False, True, True, True]  # 첫 시점만 두 배치, 이후는 정해진 배치
    assert all(f == "free" for _, f in calls)
    assert r["status"] == "ok" and r["mirror"] is True and r["far_model"] == "free" and r["method"] == "segmentation"
    json.dumps(r)  # 세션 파일에 저장할 수 있다 (numpy 값 없음)


def test_accept_passes_auto_far_model_and_mirror_to_tracking(monkeypatch, tmp_path):
    from dataclasses import asdict

    seen = {}
    H = np.eye(3).tolist()
    s = {"id": "cam1-test", "video": "v.mp4", "cam": 1, "half": "east", "spec": asdict(CourtSpec()), "scale": 1.0,
         "state": "needs_review", "warnings": [],
         "candidate": {"source": "auto", "t_sec": 5.0, "H_near": H, "H_far": H, "confidence": 0.9,
                       "far_model": "free", "mirror": True}}
    os.makedirs(tmp_path / s["id"])
    svc._save(s, str(tmp_path))
    monkeypatch.setattr(svc, "_open_scaled", lambda video: None)

    def fake_track(source, anchor, cfg, log):
        seen.update(anchor)
        return {**anchor, "segments": []}

    monkeypatch.setattr(svc, "track_camera", fake_track)
    svc.accept(s["id"], root=str(tmp_path))
    assert seen["far_model"] == "free" and seen["mirror"] is True
    with open(tmp_path / s["id"] / "calib_cam1.json", encoding="utf-8") as f:
        assert json.load(f)["far_model"] == "free"


def test_match_dir_picks_match_from_arg_env_default(monkeypatch):
    monkeypatch.delenv("FUTSAL_MATCH", raising=False)
    assert match_dir() == os.path.join(ROOT, "out", "1008")
    monkeypatch.setenv("FUTSAL_MATCH", "1103")
    assert match_dir() == os.path.join(ROOT, "out", "1103")
    assert match_dir("0901") == os.path.join(ROOT, "out", "0901")


def test_load_reads_the_selected_match(tmp_path, monkeypatch):
    import calib_core
    for match, video in (("1008", "a.mp4"), ("1103", "b.mp4")):
        os.makedirs(tmp_path / "out" / match)
        with open(tmp_path / "out" / match / "H_cam1.json", "w", encoding="utf-8") as f:
            json.dump({"half": "east", "video": video}, f)
    monkeypatch.setattr(calib_core, "ROOT", str(tmp_path))
    monkeypatch.delenv("FUTSAL_MATCH", raising=False)
    assert load(1)["video"] == "a.mp4"
    monkeypatch.setenv("FUTSAL_MATCH", "1103")
    assert load(1)["video"] == "b.mp4"


def test_mirrored_clicks_recover_camera_on_the_other_corner():
    """1103 배치: 카메라가 반대쪽 모서리라 화면이 1008의 좌우 반전. f 모드 클릭으로 같은 코트를 풀어야 한다."""
    from calib_core import manual_result
    F = np.array([[-1.0, 0, 1919], [0, 1, 0], [0, 0, 1]])
    M = np.diag([1.0, -1.0, 1.0])
    H_true = M @ H_NEAR @ F  # 반전된 화면의 이미지 -> 코트 (보이는 사이드라인 y = +10)
    G_m = front_facing(np.linalg.inv(H_true))
    lms = click_landmarks(mirror=True)
    world = np.array([lm["xy"] for lm in lms], float)
    world[3, 0] = -13.0  # 사이드라인 위 아무 점
    clicks = project(G_m, world)[0]
    assert clicks[3, 0] < 960  # 보이는 사이드라인은 화면 왼쪽
    res = manual_result(clicks.tolist(), lms, "east")
    probe = np.array([[-15.0, 8.0], [-5.0, -6.0], [0.0, 0.0]])
    px = project(G_m, probe)[0]
    assert np.allclose(cv2.perspectiveTransform(px[None], res["near"][0])[0], probe, atol=1e-3)
    assert np.allclose(res["near"][1][3], [-13.0, 10.0], atol=1e-3)


def test_mirror_swaps_names_and_sides_only():
    a, b = click_landmarks(), click_landmarks(mirror=True)
    assert [x["kind"] for x in a] == [x["kind"] for x in b]
    assert b[0]["name"] == "halfway line, right end" and b[3]["name"].startswith("left sideline")
    assert all(x["xy"] == (y["xy"][0], -y["xy"][1]) for x, y in zip(b, a))


def test_keypoint_init_solves_mirrored_camera_through_flipped_frame(monkeypatch):
    """반대쪽 모서리 카메라(1103): 반전 프레임은 1008 화면처럼 보이므로 모델은 반전 프레임에서 지점을 찾는다."""
    import keypoint_init as ki

    F = np.array([[-1.0, 0, 1919], [0, 1, 0], [0, 0, 1]])
    G_m = front_facing(np.linalg.inv(np.diag([1.0, -1.0, 1.0]) @ H_NEAR @ F))
    # 가짜 모델은 어떤 프레임이든 1008 배치(G)의 지점을 낸다 = 반전 프레임에서만 맞는 답
    monkeypatch.setattr(ki, "detect", fake_detect({"halfway_left", "halfway_right", "circle_halfway_left",
                                                   "circle_halfway_right", "near_10m", "far_10m"}))
    r = ki.keypoint_calibrate(render(G_m), "east")
    assert r["status"] == "ok" and r["flipped"] and r["confidence"] >= ki.ACCEPT_SCORE
    world = np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0]])
    px, _ = project(G_m, world)
    assert np.linalg.norm(pixel_to_court({"half": "east", **r}, px) - world, axis=1).max() < 0.2
    assert not ki.keypoint_calibrate(render(G), "east")["flipped"]  # 1008 배치는 그대로


def _scene(seed=3, sky=1080):
    """무늬가 있는 배경(위쪽 sky 줄). 그 아래는 선 없는 잔디."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920)).astype(np.uint8), (0, 0), 2)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img[sky:] = (60, 130, 50)
    return img


def test_background_shift_measures_camera_translation_and_refuses_zoom():
    import track as tr

    img = _scene()
    moved = cv2.warpAffine(img, np.float32([[1, 0, 7], [0, 1, -5]]), (1920, 1080), borderMode=cv2.BORDER_REFLECT)
    d = tr.background_shift(tr.describe(img), tr.describe(moved))
    assert d is not None and abs(d[0] - 7) < 0.5 and abs(d[1] + 5) < 0.5
    zoom = cv2.warpAffine(img, cv2.getRotationMatrix2D((960, 540), 0, 1.1), (1920, 1080), borderMode=cv2.BORDER_REFLECT)
    assert tr.background_shift(tr.describe(img), tr.describe(zoom)) is None
    assert tr.background_shift(tr.describe(img), None) is None


def test_walk_keeps_feature_confirmed_position_when_lines_fade(monkeypatch):
    """배경으로 카메라가 그대로임이 확인되면, 흰 선이 흐린 프레임(야간 조명)도 잃지 않고 같은 자리로 둔다."""
    import track as tr

    class Faded:
        duration = 30.0

        def frame_at(self, t):
            return _scene(sky=200)  # 펜스 너머 배경은 그대로, 코트에는 흰 선이 없음

    pts = sample_model(CourtSpec())
    s = tr.walk(Faded(), [5.0, 10.0], G, pts, (0.0, 0.0), CFG, 0.4, lambda *_: None, start_frame=_scene(sky=200))
    assert all(x.ok and x.by == "features" and abs(x.dx) < 0.5 and abs(x.dy) < 0.5 for x in s)
    no_feat = tr.walk(Faded(), [5.0], G, pts, (0.0, 0.0), TrackConfig(use_features=False), 0.4, lambda *_: None, start_frame=_scene(sky=200))
    assert not no_feat[0].ok or abs(no_feat[0].dx) + abs(no_feat[0].dy) > 10  # 흰 선만으로는 엉뚱한 곳에 맞추거나 놓친다


def test_mirrored_match_labels_become_the_1008_layout():
    """1103(좌우 반전 배치)의 라벨은 화면과 함께 뒤집으면 1008 배치 카메라의 라벨과 같아야 한다 (모델은 한 배치만 배운다)."""
    import make_dataset as md

    F = np.array([[-1.0, 0, 1919], [0, 1, 0], [0, 0, 1]])
    mirrored = seg_of(np.diag([1.0, -1.0, 1.0]) @ H_NEAR @ F)
    plain = md.project_keypoints(seg_of(H_NEAR), "east", 1)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, :10] = 255
    flipped, kp = md.canonical(frame, md.project_keypoints(mirrored, "east", 1))
    assert np.allclose(kp[:, 2], plain[:, 2]) and np.allclose(kp[plain[:, 2] > 0], plain[plain[:, 2] > 0], atol=1e-6)
    assert flipped[0, -1, 0] == 255 and flipped[0, 0, 0] == 0


# ---- 평가 하네스 (eval/harness.py)


def test_harness_zones_follow_camera_direction():
    import harness as hz

    assert [hz.zone_of((x, 0), "east") for x in (-10, -2, 2, 10, 18)] == ["near", "center", "center", "far_mid", "goal"]
    assert [hz.zone_of((x, 0), "west") for x in (10, -10, -18)] == ["near", "far_mid", "goal"]


def test_harness_scores_points_by_zone_with_components_and_ci():
    import harness as hz

    cal = {"half": "east", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist()}
    world = np.array([[-10.0, 0.0], [10.0, 0.0], [18.0, 5.0]])
    px = project(G, world)[0]
    px[1, 0] += 3  # 먼 쪽 점 하나만 3 px 어긋나게 찍음
    label = {"cam": 1, "t": 0.0, "half": "east", "source": "tape",
             "points": [{"id": f"p{i}", "kind": "point", "px": px[i].tolist(), "world": world[i].tolist()} for i in range(3)]}
    rows = hz.score_rows([label], lambda n: cal)
    by = {r["id"]: r for r in rows}
    assert by["p0"]["zone"] == "near" and by["p0"]["error"] < 1e-3
    assert by["p1"]["zone"] == "far_mid" and by["p1"]["error"] > 0.02 and by["p1"]["ex"] is not None
    assert by["p2"]["zone"] == "goal" and hz.gate_g1(rows)
    s = hz.summarize(rows)
    assert all(x["reference_only"] for x in s)  # n < 5
    assert not hz.gate_g1([{**r, "source": "video_click"} for r in rows])


def test_harness_median_ci_and_split_guard(tmp_path):
    import harness as hz

    lo, hi = hz.median_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo <= 3.0 <= hi and hz.median_ci([1.0]) != hz.median_ci([1.0])  # nan
    splits = {"train": ["1103"], "tune": [], "val": ["1008"], "blind": ["1201"]}
    assert hz.check_purpose("1103", "train", splits) == "train"
    assert hz.check_purpose("1008", "val", splits) == "val"
    for m, p in (("1008", "tune"), ("1201", "train"), ("1008", "train")):
        with pytest.raises(ValueError):
            hz.check_purpose(m, p, splits)
    log = tmp_path / "log.jsonl"
    rec = hz.log_usage("1008", "val", [str(log)], {"labels_n": 3}, path=str(log))
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["purpose"] == "val" and rec["labels_n"] == 3


def _true_clicks(G_cam, lms):
    world = np.array([lm["xy"] for lm in lms], float)
    world[3, 0], world[8, 0] = -13.0, -18.0  # 선 클릭은 선 위 아무 곳
    return project(G_cam, world)[0]


def test_check_clicks_reveal_a_misclicked_10m_spot():
    """10 m 지점은 7클릭만으로는 틀려도 잔차가 0이다. 확인 클릭 2개를 더하면 5 px 오클릭이 드러난다."""
    from calib_core import MISCLICK_PX, click_residuals_px, manual_result

    lms = click_landmarks()
    clicks = _true_clicks(G, lms)
    clean = manual_result(clicks.tolist(), lms, "east")
    assert max(click_residuals_px(clean, clicks.tolist(), lms).values()) < 0.1
    bad = clicks.copy()
    bad[4] += (0, 5)  # 10 m 지점을 5 px 아래에
    seven = manual_result(bad[:7].tolist(), lms, "east")
    assert max(click_residuals_px(seven, bad[:7].tolist(), lms).values()) < 0.1  # 7클릭: 드러나지 않음
    nine = manual_result(bad.tolist(), lms, "east")
    assert max(click_residuals_px(nine, bad.tolist(), lms).values()) > MISCLICK_PX


def test_calibrator_skips_optional_checks_and_flags_misclicks():
    from calibrate import Calibrator

    cal = Calibrator(render(G), "east")
    clicks = _true_clicks(G, cal.lms)
    for x, y in clicks[:7]:
        cal.click(x, y)
    assert cal.ready and cal.result is not None and not cal.done and not cal.suspicious
    cal.skip()
    cal.skip()
    assert cal.done and cal.clicks[7:] == [None, None]
    d = cal.to_json("v.mp4")
    assert d["img_pts"][7] is None and "7" not in d["world"]["near"]
    cal.undo(); cal.undo()
    cal.clicks[4] = (clicks[4][0], clicks[4][1] + 6)
    cal.click(*clicks[7]); cal.click(*clicks[8])
    assert cal.suspicious  # 10 m 지점 오클릭이 확인 클릭으로 드러남
    cal.render()
    for _ in range(9):
        cal.undo()
    cal.click(*clicks[0])
    cal.skip()  # 필수 클릭은 건너뛸 수 없다
    assert len(cal.clicks) == 1


def test_resolve_video_finds_moved_file(tmp_path, monkeypatch):
    """JSON에 다른 PC(WSL) 경로가 적혀 있어도 FUTSAL_VIDEOS 아래에서 같은 파일 이름을 찾는다."""
    import calib_core

    (tmp_path / "1103").mkdir()
    moved = tmp_path / "1103" / "20261103_165422_sync-001.mp4"
    moved.write_bytes(b"")
    monkeypatch.setenv("FUTSAL_VIDEOS", str(tmp_path))
    calib_core._video_index.clear()
    stale = "/mnt/d/02_Projects/캡스톤/1103/20261103_165422_sync-001.mp4"
    assert calib_core.resolve_video(stale) == str(moved)
    assert calib_core.resolve_video("C:\\old\\20261103_165422_sync-001.mp4") == str(moved)
    assert calib_core.resolve_video(str(moved)) == str(moved)  # 있는 경로는 그대로
    assert calib_core.resolve_video("/nowhere/unknown.mp4") == "/nowhere/unknown.mp4"  # 못 찾으면 원래 경로
    assert calib_core.resolve_video(None) is None


def _render_bent(G_near, G_far, spec=CourtSpec(), seed=0):
    """가까운 하프(x <= 0)는 G_near, 먼 하프는 G_far로 그린 프레임: 하프라인에서 바닥이 꺾인 코트."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = (60, 130, 50)
    frame = np.clip(frame.astype(np.int16) + rng.integers(-12, 12, frame.shape), 0, 255).astype(np.uint8)
    for line in court_lines(spec):
        dense = np.concatenate([np.linspace(a, b, 60, endpoint=False) for a, b in zip(line[:-1], line[1:])])
        px_n, d_n = project(G_near, dense)
        px_f, d_f = project(G_far, dense)
        far = dense[:, 0] > 0
        px, depth = np.where(far[:, None], px_f, px_n), np.where(far, d_f, d_n)
        keep = (depth > 0) & (np.abs(px) < 1e4).all(1)
        for run in np.split(np.arange(len(dense)), np.where(~keep)[0]):
            run = run[keep[run]]
            if len(run) > 1:
                cv2.polylines(frame, [np.rint(px[run] * 16).astype(np.int32)], False, (235, 235, 235), 5, cv2.LINE_AA, shift=4)
    return frame


def test_free_far_model_fits_a_bend_the_shared_model_cannot():
    """먼 하프가 H_near o K 꼴이 아닌 경우(하프라인 방향 원근이 다름): "free"가 "shared"보다 먼 쪽을 잘 맞추고
    하프라인에서는 H_near와 거의 이어진다. 기본값 "shared"는 그대로다."""
    from refine import FAR_MARGIN, align_far, halfway_gap_m, ridge_map, sample_model

    G_far_true = G @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.003, 1.0]])  # 3자유도 K로는 표현할 수 없는 꺾임
    frame = _render_bent(G, G_far_true)
    ridge = cv2.GaussianBlur(ridge_map(frame), (0, 0), 1)
    model = sample_model()
    pts = model[model[:, 0] > -FAR_MARGIN]
    corners = CLICK_WORLD[5:]
    corner_px = project(G_far_true, corners)[0]
    xs, ys = np.meshgrid(np.linspace(3, 19, 9), np.linspace(-9, 9, 7))
    grid = np.c_[xs.ravel(), ys.ravel()]
    grid_px = project(G_far_true, grid)[0]

    def far_error(H):
        return np.median(np.linalg.norm(cv2.perspectiveTransform(grid_px.reshape(-1, 1, 2), H)[:, 0] - grid, axis=1))

    H_shared, _ = align_far(ridge, H_NEAR, pts, "shared", anchor_world=corners, anchor_img=corner_px, half="east")
    H_free, _ = align_far(ridge, H_NEAR, pts, "free", anchor_world=corners, anchor_img=corner_px, half="east")
    # 측정 (HALFWAY_LAM 3): shared 0.21 m, free 0.12 m, free 하프라인 차이 0.23 m (참값의 꺾임은 하프라인 끝에서 0.29 m)
    assert far_error(H_free) < 0.7 * far_error(H_shared)
    assert far_error(H_free) < 0.15
    assert halfway_gap_m(H_NEAR, H_shared) < 1e-6  # shared는 하프라인을 정확히 공유
    assert halfway_gap_m(H_NEAR, H_free) < 0.35
    with pytest.raises(ValueError):
        align_far(ridge, H_NEAR, pts, "bogus")


def test_free_far_model_blends_across_halfway_and_shared_is_unchanged():
    """far_model "free": 하프라인에서는 H_near, FAR_BLEND_M 너머는 H_far, 그 사이는 연속. "shared"는 섞지 않는다."""
    from court_map import FAR_BLEND_M

    H_far = np.linalg.inv(translation(0, -6) @ G)  # 먼 쪽만 6 px 어긋난 H_far: 하프라인에서 H_near와 다르다
    shared = {"half": "east", "H_near": H_NEAR.tolist(), "H_far": H_far.tolist()}
    free = {**shared, "far_model": "free"}
    ys = np.linspace(-8, 8, 5)

    def px_at(x):
        return project(G, np.c_[np.full(len(ys), x), ys])[0]

    for x in (0.01, 0.5 * FAR_BLEND_M, FAR_BLEND_M + 2.0):
        near = cv2.perspectiveTransform(px_at(x).reshape(-1, 1, 2), H_NEAR)[:, 0]
        far = cv2.perspectiveTransform(px_at(x).reshape(-1, 1, 2), H_far)[:, 0]
        got = pixel_to_court(free, px_at(x))
        w = min(x / FAR_BLEND_M, 1.0)
        assert np.allclose(got, (1 - w) * near + w * far, atol=1e-6)
        assert np.allclose(pixel_to_court(shared, px_at(x)), far, atol=1e-9)  # shared: 하프라인 너머는 H_far 그대로
    # 하프라인을 넘을 때 free는 튀지 않는다 (shared에 같은 H_far를 넣으면 튄다)
    step_free = np.linalg.norm(pixel_to_court(free, px_at(0.01)) - pixel_to_court(free, px_at(-0.01)), axis=1).max()
    step_shared = np.linalg.norm(pixel_to_court(shared, px_at(0.01)) - pixel_to_court(shared, px_at(-0.01)), axis=1).max()
    assert step_free < 0.05 < step_shared
    assert np.allclose(pixel_to_court({**shared, "far_blend_m": 9.0}, px_at(3.0)), pixel_to_court(shared, px_at(3.0)))


def test_mirrored_validation_points_match_the_screen_side_and_score_zero():
    """1103처럼 보이는 사이드라인이 화면 왼쪽인 배치: 이름/id의 left는 화면 왼쪽에 투영되고, 보이는 사이드라인은
    left_sideline이며, 정답 위치를 찍은 라벨은 오차 0으로 채점된다."""
    import label_points as lp
    import score_points as sp

    pts = {p["id"]: p for p in lp.validation_points(mirror=True)}
    assert "left_sideline" in pts and "right_sideline" not in pts
    assert pts["left_sideline"]["curve"] == "left_sideline"
    for base in ("far_arc", "far_post"):
        left = project(G, np.array([pts[f"{base}_left"]["xy"]]))[0][0]
        right = project(G, np.array([pts[f"{base}_right"]["xy"]]))[0][0]
        assert left[0] < right[0] and "left" in pts[f"{base}_left"]["name"]
    sideline_px = project(G, np.array([pts["left_sideline"]["xy"]]))[0][0]
    assert sideline_px[0] < project(G, np.array([[-16.0, -10.0]]))[0][0][0]  # 보이는 쪽 = 화면 왼쪽(y = +10)
    assert (lp.curve_points("left_sideline", "east")[:, 1] == 10).all()
    assert (lp.curve_points("left_sideline", "west")[:, 1] == -10).all()

    lms = lp.validation_points(mirror=True)
    clicks = [project(G, np.array([lm["xy"]]))[0][0] for lm in lms]
    label = lp.make_label(1, 5.0, "v.mp4", "east", lms, clicks, mirror=True)
    assert label["mirror"] is True
    cal = {"half": "east", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist()}
    errors = [r["error"] for r in sp.score([label], lambda n: cal)]
    assert len(errors) == len(lms) and max(errors) < 0.05
    assert lp.validation_points() == lp._validation_points()  # 기본(반전 없음)은 그대로


# ---- 코트 라인 분할 데이터셋 (train/make_seg_dataset.py)


def test_seg_mask_follows_projected_lines_flip_and_ignore():
    import make_seg_dataset as sd

    size, s = (1280, 720), 1280 / 1920
    mask = sd.render_mask(H_NEAR, H_NEAR, "east", size=size)
    assert mask.shape == (720, 1280) and mask.dtype == np.uint8

    def px(xy):  # 원본 1080p 투영 -> 저장 해상도 (픽셀 중심 기준)
        p = project(G, np.array(xy, float))[0]
        return (p + 0.5) * s - 0.5

    # 하프라인 위의 점(서클과 스폿 밖)은 클래스 1, 하프라인 픽셀은 모두 x=0 투영선에서 2 px 이내
    ys = np.r_[np.linspace(-9.5, -3.5, 13), np.linspace(3.5, 9.5, 13)]
    on = px(np.c_[np.zeros_like(ys), ys])
    vis = (on[:, 0] >= 0) & (on[:, 0] < 1280) & (on[:, 1] >= 0) & (on[:, 1] < 720)
    assert vis.sum() >= 5
    assert (mask[np.rint(on[vis, 1]).astype(int), np.rint(on[vis, 0]).astype(int)] == sd.CLS["halfway_line"]).all()
    rr, cc = np.nonzero(mask == sd.CLS["halfway_line"])
    a, b = px([[0.0, -10.0]])[0], px([[0.0, 10.0]])[0]
    e, q = b - a, np.c_[cc, rr] - a
    dist = np.abs(e[0] * q[:, 1] - e[1] * q[:, 0]) / np.linalg.norm(e)
    assert dist.max() < 2.0
    # 먼 선도 최소 굵기로 남고, 가까운 선은 실제 폭이라 더 두껍다
    for cls in ("far_goal_line", "sideline_right", "center_circle", "near_penalty_area", "spots"):
        assert (mask == sd.CLS[cls]).any(), cls
    near_side = px([[-14.0, -10.0]])[0]
    r, c = int(round(near_side[1])), int(round(near_side[0]))
    far_goal = px([[20.0, 0.0]])[0]
    near_w = (mask[r - 30:r + 30, c] == sd.CLS["sideline_right"]).sum()
    far_w = (mask[:, int(round(far_goal[0]))] == sd.CLS["far_goal_line"]).sum()
    assert near_w > far_w >= sd.MIN_WIDTH_PX
    assert sd.halfway_gap_px(_seg_proj(sd, H_NEAR, H_NEAR)) == 0.0

    # 서쪽을 보는 카메라(코트 180도 회전)는 같은 화면 마스크
    west = sd.render_mask(np.diag([-1.0, -1.0, 1.0]) @ H_NEAR, np.diag([-1.0, -1.0, 1.0]) @ H_NEAR, "west", size=size)
    assert (west == mask).mean() > 0.9999

    # canonical: 좌우 반전 + sideline_left/right 교환, 나머지 클래스는 그대로
    flipped = sd.canonical_mask(mask)
    assert mask[r, c] == sd.CLS["sideline_right"] and flipped[r, 1279 - c] == sd.CLS["sideline_left"]
    for k in range(10):
        swap = {3: 4, 4: 3}.get(k, k)
        assert (flipped == swap).sum() == (mask == k).sum()

    # UNRELIABLE (1008 cam2 먼 골대 구역) -> 255, 다른 카메라/회차는 그대로
    cam2 = sd.render_mask(H_NEAR, H_NEAR, "east", cam=2, match="1008", size=size)
    for cls in ("far_goal_line", "far_penalty_area"):
        assert not (cam2 == sd.CLS[cls]).any() and (mask == sd.CLS[cls]).any()
    assert (cam2 == sd.IGNORE).sum() > 0 and not (mask == sd.IGNORE).any()
    assert (cam2 == sd.CLS["halfway_line"]).sum() == (mask == sd.CLS["halfway_line"]).sum()
    assert not (sd.render_mask(H_NEAR, H_NEAR, "east", cam=2, match="1103", size=size) == sd.IGNORE).any()

    # shared 모델에서 H_far가 하프라인에서 어긋나면 하프라인 둘레가 무시(255)
    bent = np.linalg.inv(translation(6.0, 0.0) @ G)
    gap = sd.render_mask(H_NEAR, bent, "east", size=size)
    assert (gap == sd.IGNORE).any() and (gap == sd.CLS["halfway_line"]).any()


def _seg_proj(sd, H_near, H_far):
    proj = sd.Projector(H_near, H_far, "east", (1280 / 1920, 720 / 1080))
    proj.size = (1280, 720)
    return proj


# ---- 세그멘테이션 기반 자동 초기화 (seg_init.py): 모델 대신 정답 마스크(one-hot)를 넣어 기하만 검증


def _seg_input(H, mirror=False, drop=()):
    """정답 클래스 마스크를 모델 출력 꼴 (labels, conf)로. mirror면 모델처럼 sideline_left/right 이름이 바뀐다."""
    import make_seg_dataset as sd

    mask = sd.render_mask(H, H, "east", size=(1920, 1080), src_size=(1920, 1080))
    for name in drop:
        mask[mask == sd.CLS[name]] = 0
    if mirror:
        lut = np.arange(256, dtype=np.uint8)
        for a, b in sd.FLIP_CLASS.items():
            lut[a] = b
        mask = lut[mask]
    return mask, np.ones(mask.shape, np.float32)


def _probe_error(r, Gc, world):
    px, _ = project(Gc, world)
    return np.linalg.norm(pixel_to_court({"half": "east", **r}, px) - world, axis=1).max()


def test_seg_init_recovers_homography_from_class_map():
    import keypoint_init as ki
    import seg_init as si

    r = si.seg_calibrate(render(G), "east", seg_probs=_seg_input(H_NEAR))
    assert r["status"] == "ok" and r["confidence"] >= ki.ACCEPT_SCORE
    assert {"halfway_line", "sideline_left", "sideline_right", "far_goal_line"} <= set(r["lines"])
    assert r["_size"] == (1920, 1080) and r["n_near"] >= 4 and r["n_far"] >= 2
    assert _probe_error(r, G, np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0], [0.0, 0.0]])) < 0.1
    assert _probe_error(r, G, np.array([[15.0, 5.0], [8.0, -6.0], [19.0, 9.0]])) < 0.15
    # 먼 골라인/먼 페널티 에어리어가 없어도 센터 서클로 x 방향 축척을 정한다
    r2 = si.seg_calibrate(render(G), "east", seg_probs=_seg_input(H_NEAR, drop=("far_goal_line", "far_penalty_area")))
    assert r2["status"] == "ok" and "far_goal_line" not in r2["lines"]
    assert _probe_error(r2, G, np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0]])) < 0.1
    # 서쪽을 보는 카메라: 같은 화면이면 코트 좌표만 180도 돌아간다
    west = si.seg_calibrate(render(G), "west", seg_probs=_seg_input(H_NEAR))
    probe = np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0]])
    got = pixel_to_court({"half": "west", **west}, project(G, probe)[0])
    assert west["status"] == "ok" and np.linalg.norm(got + probe, axis=1).max() < 0.1


def test_seg_init_swaps_sidelines_for_mirrored_camera():
    """좌우 반전 배치(1103): court_seg.predict(mirror=True)는 픽셀만 되돌리고 이름은 정규 배치 그대로라, 원래 화면의
    sideline_left는 코트 y<0 사이드라인이다. mirror=True면 바로 풀리고, 이름을 그대로 믿으면(mirror=False) 코트가 y로 뒤집힌다."""
    import seg_init as si

    F = np.array([[-1.0, 0, 1919], [0, 1, 0], [0, 0, 1]])
    H_m = np.diag([1.0, -1.0, 1.0]) @ H_NEAR @ F
    G_m = front_facing(np.linalg.inv(H_m))
    frame, seg = render(G_m), _seg_input(H_m, mirror=True)
    probe = np.array([[-15.0, 5.0], [-8.0, -6.0], [-3.0, 2.0]])
    r = si.seg_calibrate(frame, "east", seg_probs=seg, mirror=True)
    assert r["status"] == "ok" and _probe_error(r, G_m, probe) < 0.1
    wrong = si.seg_calibrate(frame, "east", seg_probs=seg, mirror=False)
    px, _ = project(G_m, probe)
    got = pixel_to_court({"half": "east", **wrong}, px)
    assert np.allclose(got, probe * [1.0, -1.0], atol=0.3)  # 사이드라인 이름이 바뀐 만큼 y가 뒤집힌 답
    # 클래스 지도 처리: 확률 문턱과 좌우 교환
    labels, conf = si.seg_labels((np.array([[3, 4, 1]], np.uint8), np.array([[0.9, 0.9, 0.2]], np.float32)))
    assert labels.tolist() == [[3, 4, 0]]
    assert si.true_classes(labels, True).tolist() == [[4, 3, 0]] and si.true_classes(labels, False) is labels
    probs = np.zeros((10, 1, 2), np.float32)
    probs[0, 0, 0], probs[5, 0, 1] = 1.0, 0.8
    assert si.seg_labels(probs)[0].tolist() == [[0, 5]]


def test_seg_init_fails_with_reason_when_classes_are_missing():
    import seg_init as si

    frame = render(G)
    no_half = si.seg_calibrate(frame, "east", seg_probs=_seg_input(H_NEAR, drop=("halfway_line",)))
    assert no_half["status"] == "failed" and "하프라인" in no_half["reason"] and no_half["H_near"] is None
    # 하프라인 + 사이드라인 하나뿐: 직선 두 개로는 코트 점이 하나라 초기 행렬을 만들 수 없다
    lonely = _seg_input(H_NEAR, drop=("center_circle", "sideline_left", "far_goal_line", "far_penalty_area",
                                      "near_penalty_area", "spots"))
    r = si.seg_calibrate(frame, "east", seg_probs=lonely)
    assert r["status"] == "failed" and r["reason"] and r["confidence"] == 0.0
    empty = si.seg_calibrate(frame, "east", seg_probs=(np.zeros((1080, 1920), np.uint8), np.ones((1080, 1920), np.float32)))
    assert empty["status"] == "failed"


def test_camera_check_accepts_real_calibrations_and_rejects_impossible_cameras():
    """자동 통과의 물리 검사: 저장소의 정답 캘리브레이션 4개(1008·1103 x 2대)는 통과하고, 코트를 한 방향으로 늘린 해
    (1008에서 틀린 자동 통과의 모양), 바닥에 붙은 카메라, 코트 안의 카메라는 걸린다. decide는 그런 대표를 통과시키지 않는다."""
    import keypoint_init as ki
    from auto_init import ROT180

    for match in ("1008", "1103"):
        for n in (1, 2):
            cal = load(n, os.path.join(ROOT, "out", match))
            seg = next(s for s in cal["segments"] if s["source"] == "manual")
            G = front_facing(np.linalg.inv((np.eye(3) if cal["half"] == "east" else ROT180) @ np.array(seg["H_near"])))
            ok, why = ki.camera_check({"_G": G, "_size": (1920, 1080)})
            assert ok, f"{match} cam{n}: {why}"

    stretched = G @ np.diag([1.0, 1.4, 1.0])  # 코트 y 방향으로 1.4배 늘린 해
    assert not ki.camera_check({"_G": stretched, "_size": (1920, 1080)})[0]
    ok_g = front_facing(np.linalg.inv(H_NEAR))
    assert ki.camera_check({"_G": ok_g, "_size": (1920, 1080)})[0]

    r_ok = {"_G": ok_g, "_size": (1920, 1080), "confidence": 0.9, "H_near": H_NEAR.tolist()}
    r_bad = {"_G": ok_g @ np.diag([1.0, 1.4, 1.0]), "_size": (1920, 1080), "confidence": 0.9, "H_near": H_NEAR.tolist()}
    assert ki.decide([r_ok] * 5)[2] == "ok"
    assert ki.decide([r_bad] * 5)[2] == "low_confidence"
    assert ki.decide([r_bad] * 5, check_camera=False)[2] == "ok"


def test_auto_match_virtual_clicks_reproject_exactly(monkeypatch, tmp_path):
    from dataclasses import asdict

    import auto_match
    from calib_core import to_world

    spec = CourtSpec()
    for half, mirror in (("east", False), ("west", True)):
        S = np.eye(3) if half == "east" else np.diag([-1.0, -1.0, 1.0])
        H = np.linalg.inv(G @ S)  # 이미지 -> 실제 코트 (west는 코트를 180도 돌려 본다)
        cal = {"half": half, "mirror": mirror, "spec": asdict(spec), "video": "v.mp4",
               "t_sec": 5.0, "H_near": (H / H[2, 2]).tolist(), "H_far": (H / H[2, 2]).tolist(), "source": "auto",
               "far_model": "free", "version": 2, "segments": [{"t0": 0.0, "t1": 10.0, "source": "auto"}]}
        names, img_pts, world = auto_match.virtual_clicks(cal, (1920, 1080))
        assert len(names) == len(img_pts) == 7 and any(p is not None for p in img_pts)
        Gi = front_facing(np.linalg.inv(np.array(cal["H_near"])))
        for i, xy in world["near"].items():  # 팀 nearfar.from_team이 하는 재투영: 잔차 0
            assert np.allclose(project(Gi, np.array([xy]))[0][0], img_pts[int(i)], atol=0.02)
        assert all(img_pts[int(i)] is not None for region in world.values() for i in region)
        assert world["near"]["0"] == list(map(float, to_world((0.0, -spec.width / 2 if mirror else spec.width / 2), half)))

        monkeypatch.setattr(auto_match, "video_size", lambda path: (1920, 1080))
        paths, n_pts = auto_match.write_outputs(cal, str(tmp_path), 1)
        with open(paths["anchor"], encoding="utf-8") as f:
            anchor = json.load(f)
        assert "segments" not in anchor and anchor["clicks"] == "virtual" and anchor["far_model"] == "free"
        with open(paths["calib"], encoding="utf-8") as f:
            assert json.load(f)["segments"] == cal["segments"]


def test_auto_match_sync_record_round_trips_through_court_map_and_team_offsets(monkeypatch, tmp_path):
    import auto_match
    import court_map

    monkeypatch.setattr(auto_match.cv2, "VideoCapture", lambda path: _FakeCap(3000 if "a" in os.path.basename(path) else 2900))
    monkeypatch.setattr(auto_match, "frames_of", lambda path: 0)
    for offset in (12.5, -7.25):  # cam1 시각 = cam2 시각 + offset
        rec = auto_match.sync_record({1: str(tmp_path / "20261120_100000a.mp4"), 2: str(tmp_path / "20261120_100012b.mp4")},
                                     offset, {"method": "test"})
        s = {int(n): rec["videos"][name]["source_time_of_output_0_s"] for n, name in rec["cameras"].items()}
        assert min(s.values()) == 0.0 and s[1] - s[2] == pytest.approx(offset)  # 팀 trackview: 원본끼리 시간 차 = 두 값의 차
        assert rec["cameras"]["1"] == "20261120_100000a_sync.mp4" and all(v["virtual"] for v in rec["videos"].values())
        # 원본 시각의 보정 -> court_map.load가 사본 시각으로 옮긴다 (같은 순간 = 사본 시각 0)
        out = tmp_path / f"m{offset}"
        os.makedirs(out)
        for n in (1, 2):
            cal = {"half": "east", "video": str(tmp_path / ("20261120_100000a.mp4" if n == 1 else "20261120_100012b.mp4")),
                   "t_sec": 100.0, "H_near": np.eye(3).tolist(), "H_far": np.eye(3).tolist(), "version": 2,
                   "segments": [{"t0": 0.0, "t1": 200.0, "source": "auto", "H_near": np.eye(3).tolist(),
                                 "H_far": np.eye(3).tolist()}]}
            with open(out / f"calib_cam{n}.json", "w", encoding="utf-8") as f:
                json.dump(cal, f)
        with open(out / "sync.json", "w", encoding="utf-8") as f:
            json.dump(rec, f)
        for n in (1, 2):
            assert court_map.load(n, str(out))["t_sec"] == pytest.approx(100.0 - s[n])


class _FakeCap:
    def __init__(self, frames):
        self.frames = frames

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: self.frames, cv2.CAP_PROP_FPS: 10.0}.get(prop, 0)

    def release(self):
        pass


def test_solve_half_degenerate_clicks_raise_value_error():
    """한 직선 위의 클릭은 TypeError가 아니라 ValueError (calib_service가 사용자 메시지로 바꾼다)."""
    lms = click_landmarks()[:5]
    img = np.array([[100.0, 100.0], [200, 200], [300, 300], [400, 400], [500, 500]])
    with pytest.raises(ValueError):
        solve_half(img, lms, "east")


def test_result_to_original_keeps_pixel_to_court():
    """작업 해상도 결과를 원본 픽셀로 바꾼 뒤에도 같은 지점이 같은 코트 좌표로 간다 (drift 포함)."""
    scale = 0.5  # 4K -> 1080 작업 해상도
    drift = [[0.0, 0.0, 0.0], [10.0, 3.0, -2.0]]
    work = {"version": 2, "half": "east", "spec": json.loads(json.dumps(CourtSpec().__dict__)),
            "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist(), "img_pts": [[100.0, 50.0], None],
            "segments": [{"t0": 0.0, "t1": 10.0, "source": "anchor", "H_near": H_NEAR.tolist(), "H_far": H_NEAR.tolist(),
                          "shift": [4.0, -6.0], "drift": drift, "score": 1.0}]}
    work_px = np.array([[900.0, 600.0], [1200.0, 700.0]])
    expect = pixel_to_court(json.loads(json.dumps(work)), work_px, t=7.0)
    orig = svc._result_to_original(json.loads(json.dumps(work)), scale)
    got = pixel_to_court(orig, work_px / scale, t=7.0)
    assert np.allclose(got, expect, atol=1e-6)
    assert orig["img_pts"] == [[200.0, 100.0], None]
    assert orig["segments"][0]["shift"] == [8.0, -12.0]
    assert orig["segments"][0]["drift"][1] == [10.0, 6.0, -4.0]
