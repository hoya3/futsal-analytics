"""앱/서버용 하이브리드 코트 캘리브레이션. 화면 없이 돌고, 입력과 출력은 JSON으로 다룰 수 있는 dict다.

흐름:  propose -> (자동 후보가 있으면) 오버레이 확인 -> accept
                -> (없거나 reject) frame으로 프레임 고르기 -> submit_manual -> 오버레이 확인 -> accept
       accept 뒤에는 review_flags로 코트를 확정 못 한 구간을 보고, resolve_segment로 그 구간만 수동 지정한다.

자동 후보는 다수결을 통과해도 항상 사용자 승인을 받는다(accept). 세션은 <root>/<id>/session.json에 저장되어 서버가 재시작돼도
이어서 진행할 수 있다. 결과는 세션 폴더의 calib_cam<N>.json이고, 시각은 입력한 영상의 자체 시각이다.
    court_map.load(cam, out_dir=세션 폴더)로 읽는다.

클릭 좌표는 돌려준 frame 이미지의 픽셀이다(세로 1080으로 정규화됨). 결과 호모그래피는 원본 영상 픽셀 기준이다.

CLI: python calib_service.py propose VIDEO --cam 1 [--court 38x18]
     python calib_service.py frame|manual|accept|reject|flags|resolve SESSION ...  (--help 참고)
"""
import argparse
import json
import os
import re
import uuid
from dataclasses import asdict

import cv2
import numpy as np

from auto_init import ROT180, AutoConfig, auto_calibrate_video, camera_from, median_of, plausible
from calib_core import Source, click_landmarks, draw_overlay, front_facing, in_half, manual_json, manual_result
from court import DEFAULT_SPEC, CourtSpec, spec_of
from refine import CIRCLE_PAD, far_anchors, project, refine_camera, sample_model
from track import TrackConfig, line_score, to_ridge, track_camera

WORK_HEIGHT = 1080
DEFAULT_ROOT = os.path.join("out", "sessions")
DEFAULT_HALF = {1: "east", 2: "west"}
MANUAL_MIN_SCORE = 0.6  # 실제 영상의 올바른 캘리브레이션은 0.8 이상
FAR_ANCHOR_MAX_PX = 15.0  # 먼 쪽 코너 클릭과 결과의 허용 차이: 좌우를 바꿔 찍으면 훨씬 크다
KEYPOINT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train", "best.pt")


class SessionError(ValueError):
    """현재 세션 상태에서 할 수 없는 요청."""


class ScaledSource:
    """프레임을 세로 WORK_HEIGHT로 맞춰 돌려준다. 선 검출과 자세 탐색의 파라미터가 이 해상도 기준이다."""

    def __init__(self, source, scale):
        self.source, self.scale, self.duration = source, scale, source.duration

    def frame_at(self, t):
        frame = self.source.frame_at(t)
        if frame is None or self.scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)


def _open(video):
    return Source(video)


def _open_scaled(video):
    source = _open(video)
    probe = source.frame_at(0.0)
    if probe is None:
        raise SessionError(f"영상의 프레임을 읽을 수 없습니다: {video}")
    h, w = probe.shape[:2]
    if h > w:
        raise SessionError("세로로 촬영한 영상은 지원하지 않습니다. 폰을 가로로 두고 촬영해 주세요.")
    return ScaledSource(source, WORK_HEIGHT / h)


def _dir(session_id, root):
    if not re.fullmatch(r"[\w-]+", session_id):
        raise SessionError(f"잘못된 세션 id: {session_id!r}")
    return os.path.join(root, session_id)


def _load(session_id, root):
    path = os.path.join(_dir(session_id, root), "session.json")
    if not os.path.exists(path):
        raise SessionError(f"세션이 없습니다: {session_id}")
    with open(path) as f:
        return json.load(f)


def _save(s, root):
    with open(os.path.join(_dir(s["id"], root), "session.json"), "w") as f:
        json.dump(s, f, indent=2)


def _view(s):
    """클라이언트에 돌려주는 현재 상태. next = 사용자가 다음에 할 일."""
    next_step = {"needs_review": "review_overlay", "needs_manual": "click_landmarks", "done": "check_flags"}[s["state"]]
    cand = s.get("candidate") or {}
    return {"session": s["id"], "state": s["state"], "next": next_step, "status": s.get("status"),
            "t": cand.get("t_sec", s.get("frame_t")), "frame": s.get("frame"), "overlay": s.get("overlay") if cand else None,
            "confidence": cand.get("confidence"), "source": cand.get("source"), "warnings": s.get("warnings", []),
            "H_near": cand.get("H_near"), "H_far": cand.get("H_far"), "votes": s.get("votes"), "windows": s.get("windows"),
            "method": s.get("method"),
            "landmarks": [lm["name"] for lm in click_landmarks(spec_of(s))]}


def _write_frame(s, frame, t, root):
    path = os.path.join(_dir(s["id"], root), "frame.jpg")
    cv2.imwrite(path, frame)
    s["frame"], s["frame_t"] = path, float(t)


def _write_overlay(s, frame, cand, root):
    path = os.path.join(_dir(s["id"], root), "overlay.jpg")
    cv2.imwrite(path, draw_overlay(frame, np.array(cand["H_near"]), np.array(cand["H_far"]), s["half"], spec_of(s)))
    s["overlay"] = path


def find_court(source, half, spec, times=None, cfg=AutoConfig(), log=lambda *_: None):
    """자동 인식: 코트 키포인트 모델이 있으면 그것으로(시점당 약 1초), 없으면 카메라 자세 격자 탐색(시점당 약 30초).
    모델(train/best.pt)은 저장소에 없으므로 학습한 가중치를 넣은 서버에서만 키포인트 방식이 쓰인다."""
    if os.path.exists(KEYPOINT_MODEL):
        from keypoint_init import keypoint_calibrate_video  # ultralytics는 이때만 불러온다
        return keypoint_calibrate_video(source, half, spec, times, log=log)
    return auto_calibrate_video(source, half, spec, times, cfg, log=log)


def propose(video, cam, spec=DEFAULT_SPEC, half=None, root=DEFAULT_ROOT, times=None, cfg=AutoConfig(), log=lambda *_: None):
    """새 세션을 만들고 자동 인식을 시도한다(find_court, 기본 5개 시점). 자동 후보도 accept로 사용자 승인을 받는다."""
    source = _open_scaled(video)
    session_id = f"cam{cam}-{uuid.uuid4().hex[:8]}"
    os.makedirs(_dir(session_id, root))
    half = half or DEFAULT_HALF[cam]
    s = {"id": session_id, "video": video, "cam": cam, "half": half, "spec": asdict(spec), "scale": source.scale,
         "state": "needs_manual", "candidate": None}

    result = find_court(source, half, spec, times, cfg, log=log)
    s.update(status=result["status"], method=result.get("method", "pose_grid"), votes=result.get("votes"),
             windows=result.get("windows"), details=result.get("details"))
    t = float(result.get("t_sec", source.duration * 0.5))
    frame = median_of(source, t)
    if frame is None:
        raise SessionError(f"t={t:.1f}초의 프레임을 읽을 수 없습니다")
    _write_frame(s, frame, t, root)
    if result["status"] == "ok":
        cand = {"source": "auto", "t_sec": t, "H_near": result["H_near"], "H_far": result["H_far"], "confidence": result["confidence"]}
        s.update(state="needs_review", candidate=cand)
        _write_overlay(s, frame, cand, root)
    _save(s, root)
    return _view(s)


def frame(session_id, t, root=DEFAULT_ROOT):
    """수동 지정에 쓸 프레임을 t초로 옮긴다. 선수를 지운 짧은 구간의 중앙값 이미지를 돌려준다."""
    s = _load(session_id, root)
    img = median_of(_open_scaled(s["video"]), float(t))
    if img is None:
        raise SessionError(f"t={t}초의 프레임을 읽을 수 없습니다")
    _write_frame(s, img, t, root)
    _save(s, root)
    return _view(s)


def reject(session_id, root=DEFAULT_ROOT):
    """현재 후보를 버리고 수동 지정으로 넘어간다."""
    s = _load(session_id, root)
    s.update(state="needs_manual", candidate=None, warnings=[])
    s["overlay"] = None
    _save(s, root)
    return _view(s)


def line_quality(frame_img, H_near, half, spec):
    model = sample_model(spec)
    pts = model[in_half(model, half, "near", margin=spec.center_r + CIRCLE_PAD)]
    return line_score(to_ridge(frame_img), front_facing(np.linalg.inv(np.array(H_near))), pts)


def _solve_manual(frame_img, clicks, half, spec, video, t):
    """클릭 -> 정밀화 -> 품질 검사. (캘리브레이션 dict, 일치율, 경고 목록)을 돌려준다."""
    lms = click_landmarks(spec)
    h, w = frame_img.shape[:2]
    if len(clicks) != len(lms):
        raise SessionError(f"클릭은 {len(lms)}개여야 합니다 (받은 개수: {len(clicks)})")
    if any(not (0 <= x < w and 0 <= y < h) for x, y in clicks):
        raise SessionError("화면 밖의 클릭이 있습니다")
    try:
        d = manual_json(manual_result(clicks, lms, half, spec), clicks, lms, half, spec, video, t)
    except (cv2.error, np.linalg.LinAlgError, ValueError) as e:
        raise SessionError(f"클릭으로 코트를 계산할 수 없습니다. 점의 순서와 위치를 확인해 주세요. ({e})") from e

    warnings = []
    try:
        d = refine_camera(d, frame_img, log=lambda *_: None)
    except (cv2.error, np.linalg.LinAlgError):
        warnings.append("흰 선 정밀화에 실패해 클릭만으로 계산한 결과를 씁니다.")
    score = line_quality(frame_img, d["H_near"], half, spec)
    if score < MANUAL_MIN_SCORE:
        warnings.append(f"코트 선과 흰 선이 잘 맞지 않습니다 (일치율 {score:.2f}). 클릭 순서와 위치를 확인해 주세요.")
    far_world, far_img = far_anchors(d)
    if len(far_world):
        px = project(front_facing(np.linalg.inv(np.array(d["H_far"]))), far_world)[0]
        if np.linalg.norm(px - far_img, axis=1).max() > FAR_ANCHOR_MAX_PX:
            warnings.append("먼 쪽 코너가 코트 선과 맞지 않습니다. 왼쪽과 오른쪽을 바꿔 찍지 않았는지 확인해 주세요.")
    S = np.eye(3) if half == "east" else ROT180
    G = front_facing(np.linalg.inv(np.array(d["H_near"])) @ S)
    if not plausible(G, (w, h), spec) or camera_from(G, (w, h)) is None:
        warnings.append("계산된 카메라 위치가 현실적이지 않습니다. 클릭 순서를 확인해 주세요.")
    d["source"] = "manual"
    return d, score, warnings


def submit_manual(session_id, clicks, t=None, root=DEFAULT_ROOT):
    """7클릭을 받아 후보를 만든다. clicks: frame 이미지 기준 [[x, y], ...] (landmarks 순서)."""
    s = _load(session_id, root)
    if s["state"] == "done":
        raise SessionError("이미 끝난 세션입니다. 코트를 확정 못 한 구간은 resolve_segment를 쓰세요.")
    t = float(s["frame_t"] if t is None else t)
    source = _open_scaled(s["video"])
    img = median_of(source, t)
    if img is None:
        raise SessionError(f"t={t}초의 프레임을 읽을 수 없습니다")
    d, score, warnings = _solve_manual(img, clicks, s["half"], spec_of(s), s["video"], t)
    cand = {"source": "manual", "t_sec": t, "H_near": d["H_near"], "H_far": d["H_far"], "confidence": round(score, 3), "calibration": d}
    s.update(state="needs_review", candidate=cand, warnings=warnings, status="suspect" if warnings else "ok")
    _write_frame(s, img, t, root)
    _write_overlay(s, img, cand, root)
    _save(s, root)
    return _view(s)


def _to_original(H, scale):
    """작업 해상도 픽셀 기준 호모그래피 -> 원본 영상 픽셀 기준."""
    out = np.array(H) @ np.diag([scale, scale, 1.0])
    return (out / out[2, 2]).tolist()


def _result_path(s, root):
    return os.path.join(_dir(s["id"], root), f"calib_cam{s['cam']}.json")


def accept(session_id, root=DEFAULT_ROOT, confirm_warnings=False, cfg=TrackConfig(), log=lambda *_: None):
    """현재 후보를 승인하고 경기 전체를 추적한다. 경기 길이에 따라 수 분 이상 걸린다."""
    s = _load(session_id, root)
    if s["state"] != "needs_review":
        raise SessionError(f"승인할 후보가 없습니다 (상태: {s['state']})")
    if s.get("warnings") and not confirm_warnings:
        raise SessionError("경고가 있는 후보입니다. 다시 클릭하거나 confirm_warnings=True로 승인하세요.")
    cand = s["candidate"]
    anchor = dict(cand.get("calibration") or {})
    anchor.update(half=s["half"], video=s["video"], t_sec=cand["t_sec"], spec=s["spec"], source=cand["source"],
                  H_near=cand["H_near"], H_far=cand["H_far"])
    source = _open_scaled(s["video"])
    try:
        result = track_camera(source, anchor, cfg, log)
    except ValueError as e:
        raise SessionError(f"후보로 추적을 시작할 수 없습니다: {e}") from e

    scale = s["scale"]
    for holder in [result] + result["segments"]:
        if holder.get("H_near") is not None:
            holder["H_near"], holder["H_far"] = _to_original(holder["H_near"], scale), _to_original(holder["H_far"], scale)
    with open(_result_path(s, root), "w") as f:
        json.dump(result, f, indent=2)
    s.update(state="done", result=_result_path(s, root))
    _save(s, root)
    out = _view(s)
    out.update(result=s["result"], flags=_flags(result))
    return out


def _flags(result):
    return [{"index": i, "t0": seg["t0"], "t1": seg["t1"], "duration": seg["t1"] - seg["t0"], "suggested_t": (seg["t0"] + seg["t1"]) / 2}
            for i, seg in enumerate(result["segments"]) if seg["source"] == "unresolved"]


def _result(s, root):
    if s["state"] != "done":
        raise SessionError(f"아직 추적이 끝나지 않았습니다 (상태: {s['state']})")
    with open(_result_path(s, root)) as f:
        return json.load(f)


def review_flags(session_id, root=DEFAULT_ROOT):
    """코트를 확정하지 못한 구간 목록."""
    return {"session": session_id, "flags": _flags(_result(_load(session_id, root), root))}


def resolve_segment(session_id, index, clicks, t=None, root=DEFAULT_ROOT, confirm_warnings=False):
    """확정 못 한 구간 하나를 수동으로 지정한다. t를 주지 않으면 구간 가운데의 프레임을 쓴다."""
    s = _load(session_id, root)
    result = _result(s, root)
    segments = result["segments"]
    if not 0 <= index < len(segments) or segments[index]["source"] != "unresolved":
        raise SessionError(f"구간 {index}은 수동 지정 대상이 아닙니다")
    seg = segments[index]
    t = (seg["t0"] + seg["t1"]) / 2 if t is None else float(t)
    if not seg["t0"] <= t <= seg["t1"]:
        raise SessionError(f"t={t}는 구간 [{seg['t0']:.1f}, {seg['t1']:.1f}] 밖입니다")
    img = _open_scaled(s["video"]).frame_at(t)  # 카메라가 움직이는 구간일 수 있어 중앙값이 아닌 그 순간의 프레임을 쓴다
    if img is None:
        raise SessionError(f"t={t}초의 프레임을 읽을 수 없습니다")
    d, score, warnings = _solve_manual(img, clicks, s["half"], spec_of(s), s["video"], t)
    if warnings and not confirm_warnings:
        return {"session": s["id"], "resolved": False, "confidence": round(score, 3), "warnings": warnings}
    seg.update(source="manual", H_near=_to_original(d["H_near"], s["scale"]), H_far=_to_original(d["H_far"], s["scale"]), score=score)
    with open(_result_path(s, root), "w") as f:
        json.dump(result, f, indent=2)
    return {"session": s["id"], "resolved": True, "confidence": round(score, 3), "warnings": warnings, "flags": _flags(result)}


def _parse_court(text):
    return CourtSpec(*map(float, text.lower().split("x"))) if text else DEFAULT_SPEC


def main(argv=None):
    p = argparse.ArgumentParser(description="하이브리드 코트 캘리브레이션 (결과는 JSON으로 출력)")
    p.add_argument("--root", default=DEFAULT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("propose")
    a.add_argument("video")
    a.add_argument("--cam", type=int, required=True)
    a.add_argument("--court", help="길이x폭 (미터), 예: 38x18")
    a.add_argument("--half", choices=("east", "west"))
    for name in ("frame", "manual", "accept", "reject", "flags", "resolve"):
        c = sub.add_parser(name)
        c.add_argument("session")
        if name == "frame":
            c.add_argument("t", type=float)
        if name in ("manual", "resolve"):
            c.add_argument("--clicks", required=True, help='JSON, 예: "[[385,274],[1061,316],[1582,350],[1199,735],[377,398],[905,217],[1668,244]]"')
            c.add_argument("--t", type=float)
        if name == "resolve":
            c.add_argument("--index", type=int, required=True)
        if name in ("accept", "resolve"):
            c.add_argument("--confirm-warnings", action="store_true")
    args = p.parse_args(argv)
    r = args.root
    try:
        if args.cmd == "propose":
            out = propose(args.video, args.cam, _parse_court(args.court), args.half, r)
        elif args.cmd == "frame":
            out = frame(args.session, args.t, r)
        elif args.cmd == "manual":
            out = submit_manual(args.session, json.loads(args.clicks), args.t, r)
        elif args.cmd == "accept":
            out = accept(args.session, r, args.confirm_warnings)
        elif args.cmd == "reject":
            out = reject(args.session, r)
        elif args.cmd == "flags":
            out = review_flags(args.session, r)
        else:
            out = resolve_segment(args.session, args.index, json.loads(args.clicks), args.t, r, args.confirm_warnings)
    except SessionError as e:
        out = {"error": str(e)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if "error" in out else 0


if __name__ == "__main__":
    raise SystemExit(main())
