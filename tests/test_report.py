"""report/make_report.py 의 계산을 답을 아는 손수 만든 궤적으로 확인한다."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "report"))

import make_report as mr  # noqa: E402
from synth_match import synthetic_match  # noqa: E402

FPS = 10.0


def straight(n, speed, x0=2.0, y=10.0, fps=FPS):
    t = np.arange(n) / fps
    return np.stack([x0 + speed * t, np.full(n, y)], 1)


def as_tracks(players, n, fps=FPS, ball=None):
    def col(v):
        return [None if not np.isfinite(a) else float(a) for a in v]
    d = {"version": 1, "court": {"length": 40, "width": 20}, "fps": fps, "start": 0.0, "n_frames": n,
         "players": [{"id": i + 1, "team": t, "name": None, "x": col(xy[:, 0]), "y": col(xy[:, 1])}
                     for i, (t, xy) in enumerate(players)]}
    if ball is not None:
        d["ball"] = {"x": col(ball[:, 0]), "y": col(ball[:, 1]), "cam": [1] * n}
    return d


def test_straight_run_distance_and_speed():
    xy = straight(401, 0.8)                       # 40 s, 0.8 m/s → 32 m
    m = mr.player_metrics(xy, FPS)
    assert m["distance_m"] == pytest.approx(32.0, abs=1e-6)
    assert m["mean_speed_ms"] == pytest.approx(0.8, rel=1e-3)
    assert m["top_speed_est_ms"] == pytest.approx(0.8, abs=1e-6)
    assert m["hi_distance_m"] == 0 and m["sprints"] == 0
    assert m["tracked_s"] == pytest.approx(40.1)


def test_noise_is_smoothed_out():
    rng = np.random.default_rng(0)
    xy = straight(601, 0.5) + rng.normal(0, 0.25, (601, 2))   # 60 s, 30 m, 25 cm 흔들림
    m = mr.player_metrics(xy, FPS)
    raw = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    assert raw > 100                               # 그대로 더하면 몇 배로 부풂
    assert m["distance_m"] == pytest.approx(30.0, rel=0.15)
    assert m["top_speed_est_ms"] < 1.2
    assert m["raw_max_speed_ms"] > 5              # 원시 최댓값은 믿을 수 없음


def test_short_gap_bridged_long_gap_not():
    xy = straight(401, 1.0)
    short = xy.copy(); short[100:110] = np.nan     # 1 s 빈칸 → 직선으로 메움, 거리 그대로
    assert mr.player_metrics(short, FPS)["distance_m"] == pytest.approx(40.0, abs=1e-6)
    long_ = xy.copy(); long_[100:150] = np.nan     # 5 s 빈칸 → 그 사이 이동은 세지 않음
    m = mr.player_metrics(long_, FPS)
    assert m["distance_m"] == pytest.approx(40.0 - 5.1, abs=1e-6)
    assert m["tracked_s"] == pytest.approx(35.1)
    assert m["seen_s"] == pytest.approx(35.1)


def test_teleport_is_cut_from_distance_and_speed():
    xy = straight(401, 1.0)
    xy[200:203] += [8.0, 5.0]                      # 0.3 s 동안 다른 사람에게 잘못 이어짐
    m = mr.player_metrics(xy, FPS)
    assert m["jumps"] == 2
    assert m["raw_max_speed_ms"] > 50
    assert m["top_speed_est_ms"] == pytest.approx(1.0, abs=1e-6)
    assert m["distance_m"] == pytest.approx(40.0 - 0.4, abs=1e-6)    # 끊긴 4 프레임 사이 이동만 빠짐


def test_sprint_and_high_intensity():
    a = straight(100, 2.0)                                     # 10 s 조깅
    b = straight(31, 6.0, x0=a[-1, 0] + 0.2)                   # 3 s 스프린트
    c = straight(100, 2.0, x0=b[-1, 0] + 0.2)
    m = mr.player_metrics(np.vstack([a, b, c]), FPS)
    assert m["sprints"] == 1
    assert 15 < m["hi_distance_m"] < 22                        # ~18 m (평활 때문에 가장자리 조금)


def test_fragment_threshold_and_filtering():
    assert mr.fragment_threshold_s(40 * 60) == 120
    assert mr.fragment_threshold_s(10 * 60) == 60
    assert mr.fragment_threshold_s(60) == 30
    n = 6000                                                    # 10분 → 기준 60 s
    full = np.tile([[10.0, 10.0]], (n, 1)) + np.stack([np.sin(np.arange(n) / 30), np.zeros(n)], 1)
    frag = np.full((n, 2), np.nan); frag[1000:1300] = full[1000:1300] + [20, 0]     # 30 s
    res = mr.analyze(as_tracks([("A", full), ("A", frag)], n))
    p1, p2 = res["players"]
    assert not p1["fragment"] and p2["fragment"]
    assert res["quality"]["fragment_ids"] == 1
    t = res["teams"]["A"]
    # 조각은 팀 집계 어디에도 들어가지 않는다 (품질 섹션에만)
    assert t["distance_total_m"] == pytest.approx(p1["distance_m"])
    assert t["distance_per_main_player_m"] == pytest.approx(p1["distance_m"])
    assert sum(t["distance_blocks_m"]) == pytest.approx(p1["distance_m"])
    assert t["fragment_ids"] == 1 and t["fragment_distance_m"] == pytest.approx(p2["distance_m"])
    assert res["quality"]["fragment_by_team"]["A"]["ids"] == 1


def test_fragments_excluded_from_heatmap_and_zones():
    n = 6000
    main = np.tile([[5.0, 10.0]], (n, 1)) + np.stack([np.sin(np.arange(n) / 30), np.zeros(n)], 1)   # 왼쪽 1/3
    frag = np.full((n, 2), np.nan); frag[2000:2400] = [35.0, 10.0]                                   # 오른쪽, 40 s
    res = mr.analyze(as_tracks([("A", main), ("A", frag)], n))
    t = res["teams"]["A"]
    assert t["zones_abs"]["thirds"] == pytest.approx([1.0, 0.0, 0.0])
    assert t["_heat"][:, 30:].sum() == 0                     # 오른쪽 칸에는 아무것도 없음
    assert t["shape"]["frames"] == 0
    # 조각만 있는 팀은 팀 목록에 나오지 않는다
    res2 = mr.analyze(as_tracks([("A", main), ("B", frag)], n))
    assert "B" not in res2["teams"] and res2["quality"]["fragment_by_team"]["B"]["ids"] == 1


def test_zone_shares():
    xy = np.array([[1, 1], [1, 19], [20, 10], [39, 10], [39, 10], [np.nan, 5]], float)
    z = mr.zone_shares(xy, 40, 20)
    assert z["n"] == 5
    assert z["thirds"] == pytest.approx([0.4, 0.2, 0.4])
    assert z["channels"] == pytest.approx([0.2, 0.6, 0.2])
    assert z["grid"][1][2] == pytest.approx(0.4)


def test_attack_right_rotation():
    s = np.array([[5.0, 3.0], [5.0, 3.0]])
    out = mr.to_attack_right(s, np.array([1.0, -1.0]), 40, 20)
    assert out.tolist() == [[5.0, 3.0], [35.0, 17.0]]


def test_half_time_switch_detected():
    n = int(20 * 60 * FPS)
    t = np.arange(n)
    ax = np.where(t < n // 2, 12.0, 28.0) + np.sin(t / 50)
    bx = 40 - ax
    A = np.stack([ax, np.full(n, 10.0)], 1)
    B = np.stack([bx, np.full(n, 10.0)], 1)
    res = mr.analyze(as_tracks([("A", A), ("B", B)], n))
    sd = res["sides"]
    assert sd["a_left_first"] is True
    assert abs(sd["switch_frame"] - n // 2) <= 60 * FPS
    za = res["teams"]["A"]["zones_rel"]["thirds"]
    assert za[0] > 0.95                         # 공격 방향 기준으로 A는 계속 수비 1/3에 있었음


def test_no_switch_when_teams_stay():
    n = int(10 * 60 * FPS)
    A = np.tile([[10.0, 10.0]], (n, 1)); B = np.tile([[30.0, 10.0]], (n, 1))
    sd = mr.analyze(as_tracks([("A", A), ("B", B)], n))["sides"]
    assert sd["switch_frame"] is None and sd["a_left_first"] is True


def test_report_files_written(tmp_path):
    d = synthetic_match(minutes=6, seed=3, ball=True)
    src = tmp_path / "tracks.json"
    src.write_text(json.dumps(d), encoding="utf-8")
    out = tmp_path / "rep"
    mr.main([str(src), "--out", str(out), "--title", "테스트 경기"])
    h = (out / "report.html").read_text(encoding="utf-8")
    assert h.startswith("<!doctype html>") and "테스트 경기" in h
    assert "http://" not in h and "https://" not in h          # 외부 리소스 없음
    for sec in ("경기 개요", "선수별 기록", "히트맵 (팀)", "구역 점유", "평균 위치와 팀 형태", "공 위치", "데이터 품질",
                "짧게 잡힌 ID"):
        assert sec in h
    r = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert r["match"]["duration_s"] == pytest.approx(360)
    assert set(r["main_teams"]) == {"A", "B"}
    assert r["ball"]["coverage"] > 0.3
    assert len(r["teams"]["A"]["heat"]) == 20 and len(r["teams"]["A"]["heat"][0]) == 40


def test_without_ball_and_one_team(tmp_path):
    n = 3000
    d = as_tracks([("A", straight(n, 0.01)), ("A", straight(n, 0.01, y=5))], n)
    res = mr.analyze(d)
    assert res["ball"] is None and res["sides"]["a_left_first"] is None
    h = mr.render_html(res)
    assert "공 위치" not in h and "팀 A" in h


def test_roster_parsing_sources():
    assert mr.parse_roster(["A=5", "B=6"]) == {"A": 5, "B": 6}
    assert mr.parse_roster("Red=5, Blue=5") == {"Red": 5, "Blue": 5}
    assert mr.parse_roster({"A": "5"}) == {"A": 5}
    with pytest.raises(ValueError):
        mr.parse_roster(["A5"])
    with pytest.raises(ValueError):
        mr.parse_roster(["A=0"])
    tr, sm = {"roster": {"A": 5, "B": 5}}, {"roster": {"A": 4, "B": 4}}
    assert mr.resolve_roster(["A=6"], tr, sm) == ({"A": 6}, "명령줄 --players")
    assert mr.resolve_roster(None, tr, sm) == ({"A": 5, "B": 5}, "tracks.json")
    assert mr.resolve_roster(None, {}, sm) == ({"A": 4, "B": 4}, "summary.json")
    assert mr.resolve_roster(None, {}, None) == ({}, None)


def _two_team_match(n, ids_a, ids_b):
    rng = np.random.default_rng(1)
    pl = []
    for t, k, x in (("A", ids_a, 10.0), ("B", ids_b, 30.0)):
        for i in range(k):
            pl.append((t, np.tile([[x, 3.0 + 3 * i]], (n, 1)) + rng.normal(0, 0.05, (n, 2))))
    return as_tracks(pl, n)


def test_roster_average_and_mismatch_warning(tmp_path):
    n = 3000
    d = _two_team_match(n, 2, 3)
    d["roster"] = {"A": 2, "B": 2}                          # B: 주요 ID 3개, 실제 2명
    res = mr.analyze(d)
    assert res["roster_source"] == "tracks.json"
    tb = res["teams"]["B"]
    assert tb["distance_per_roster_player_m"] == pytest.approx(tb["distance_total_m"] / 2)
    chk = {c["team"]: c for c in res["quality"]["roster_checks"]}
    assert chk["A"]["status"] == "ok" and chk["B"]["status"] == "more_ids"
    assert "주요 ID 3개, 실제 선수 2명" in chk["B"]["message"]
    h = mr.render_html(res)
    assert "실제 선수 2명" in h and "1명당" in h
    assert len(res["players"]) == 5                        # ID를 합치거나 버리지 않음
    # summary.json 에서 읽기, CLI 가 우선
    src = tmp_path / "t.json"; src.write_text(json.dumps(_two_team_match(n, 2, 2)), encoding="utf-8")
    sm = tmp_path / "s.json"; sm.write_text(json.dumps({"roster": {"A": 3, "B": 2}}), encoding="utf-8")
    r = mr.make_report(str(src), str(tmp_path / "o"), str(sm))
    assert r["roster"] == {"A": 3, "B": 2} and r["roster_source"] == "summary.json"
    assert {c["team"]: c["status"] for c in r["quality"]["roster_checks"]} == {"A": "fewer_ids", "B": "ok"}
    mr.main([str(src), "--out", str(tmp_path / "o2"), "--summary", str(sm), "--players", "A=2", "B=2"])
    r2 = json.loads((tmp_path / "o2" / "report.json").read_text(encoding="utf-8"))
    assert r2["roster"] == {"A": 2, "B": 2} and r2["roster_source"] == "명령줄 --players"


def test_no_roster_no_check():
    res = mr.analyze(_two_team_match(3000, 2, 2))
    assert res["roster"] is None and res["quality"]["roster_checks"] == []
    assert res["teams"]["A"]["distance_per_roster_player_m"] is None
    h = mr.render_html(res)
    assert "1명당" not in h and "실제 인원" not in h


def test_high_intensity_hidden_by_default(tmp_path):
    src = tmp_path / "t.json"
    src.write_text(json.dumps(synthetic_match(minutes=4, seed=2, ball=False)), encoding="utf-8")
    mr.main([str(src), "--out", str(tmp_path / "a")])
    h = (tmp_path / "a" / "report.html").read_text(encoding="utf-8")
    r = json.loads((tmp_path / "a" / "report.json").read_text(encoding="utf-8"))
    assert "고강도" not in h and "스프린트" in h
    assert "hi_distance_m" in r["players"][0] and "hi_distance_m" in r["teams"]["A"]
    mr.main([str(src), "--out", str(tmp_path / "b"), "--high-intensity"])
    assert "고강도" in (tmp_path / "b" / "report.html").read_text(encoding="utf-8")


def test_ball_shorter_than_players_is_padded():
    """공 추적이 선수보다 먼저 끝나도(공 길이 < n_frames) 리포트가 멈추지 않는다."""
    d = synthetic_match(minutes=4, seed=1, ball=True)
    d["ball"] = {k: v[:-100] for k, v in d["ball"].items()}
    T = mr.load_tracks(d)
    assert T["ball"].shape == (T["n"], 2) and np.isnan(T["ball"][-100:]).all()
    res = mr.analyze(d)
    assert res["ball"] is not None
