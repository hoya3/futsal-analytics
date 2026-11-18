"""새 경기 영상 2개 -> 클릭 없이 경기 전체 캘리브레이션 (out/<경기>/). 팀 파이프라인(futsal.trackview)이 바로 읽는다.

    python auto_match.py cam1.mp4 cam2.mp4 --match 1120            # 오버레이를 보고 터미널에서 승인
    python auto_match.py cam1.mp4 cam2.mp4 --match 1120 --yes      # 확인 없이 자동 승인
    python auto_match.py - cam2.mp4 --match 1120                   # "-" = 그 카메라 건너뜀

카메라마다
  1. calib_service.propose: 자동 코트 인식 (세그멘테이션 모델, 5개 시점 다수결). 확인용 오버레이를
     out/<경기>/auto_overlay_cam<N>.jpg에 저장
  2. 승인(사람 또는 --yes) -> calib_service.accept: 경기 전체 추적 (CPU 카메라당 약 7~12분, 두 카메라를 동시에)
  3. out/<경기>/calib_cam<N>.json  경기 전체 시간별 캘리브레이션 (court_map.load, 팀 nearfar.team_timeline)
     out/<경기>/H_cam<N>.json      기준 시점 하나의 캘리브레이션 (court_map.load_anchor, 팀 trackview가 있는지 확인)
자동 인식이 실패한 카메라는 calibrate.py(7클릭)로 수동 보정하라고 안내한다.

H_cam<N>.json과 calib_cam<N>.json의 img_pts / world는 사람이 찍은 점이 아니라, 7클릭 지점을 자동 결과 행렬로 화면에
투영한 가상 클릭이다 (clicks: "virtual"). 팀 코드(nearfar.NearFarCalibration.from_team)가 클릭 정보를 읽기 때문이며,
그 재투영 잔차는 0 px이다. 화면 밖이거나 카메라 뒤에 떨어지는 지점은 None.

세션(calib_service)은 out/sessions/에 남고 out/<경기>/auto_match.json에 기록된다. --yes 없이 승인하지 않고 끝냈다면
같은 명령을 다시 실행할 때 자동 인식을 다시 하지 않고 그 세션을 이어서 쓴다.

두 카메라 시간 차 (out/<경기>/sync.json):
  두 영상의 소리를 팀 저장소의 futsal.sync.align_segments(1분 구간마다 소리 패턴을 맞춰 중앙값, 손뼉 없어도 됨)로
  맞춰 "cam1 시각 = cam2 시각 + offset"을 구하고, 10/8·11/3과 같은 sync.json 형식으로 쓴다. 다만 시작을 맞춘 사본
  (<이름>_sync.mp4)은 실제로 만들지 않는다 (가상 사본, "virtual": true): 두 영상이 같은 순간이 되는 원본 시각을
  source_time_of_output_0_s로 적으면 court_map.load(사본 시각)와 팀 nearfar.find_video / trackview(원본 영상의
  시간 차 = 두 값의 차)가 이 파일만으로 맞게 돈다. 시계 속도 차(drift)는 경기 가운데의 값으로 접는다.
  소리는 ffmpeg로 읽는다 (PATH에 없으면 imageio-ffmpeg에 든 것). futsal.sync는 설치된 futsal 패키지, 또는 이 폴더의
  ../src (팀 저장소 안의 calibration/)나 ../team/src에서 찾는다. 찾지 못하거나 소리가 없으면 --offset 초로 직접 넣는다.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import calib_service as svc
from calib_core import click_landmarks, front_facing, match_dir, to_world
from refine import project

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_ROOT = os.path.join(HERE, "out", "sessions")
ANCHOR_KEYS = ("half", "video", "t_sec", "spec", "mirror", "far_model", "source", "H_near", "H_far")


def virtual_clicks(cal, size):
    """7클릭 지점을 결과 행렬로 화면에 투영한 가상 클릭: (names, img_pts, world). manual_json과 같은 형식.
    가까운 하프 지점은 H_near, 먼 쪽 코너는 H_far로 투영하고, 화면 밖이나 카메라 뒤면 None (world에서도 뺀다)."""
    spec = svc.spec_of(cal)
    lms = [lm for lm in click_landmarks(spec, bool(cal.get("mirror"))) if not lm.get("optional")]
    G = {region: front_facing(np.linalg.inv(np.array(cal[f"H_{region}"], float))) for region in ("near", "far")}
    w, h = size
    img_pts, world = [], {"near": {}, "far": {}}
    for i, lm in enumerate(lms):
        xy = np.array(to_world(lm["xy"], cal["half"]), float)
        region = "near" if "near" in lm["halves"] else "far"
        px, depth = project(G[region], xy[None])
        px = px[0]
        if depth[0] <= 0 or not (0 <= px[0] < w and 0 <= px[1] < h):
            img_pts.append(None)
            continue
        img_pts.append([round(float(px[0]), 2), round(float(px[1]), 2)])
        for r in lm["halves"]:
            world[r][str(i)] = [float(xy[0]), float(xy[1])]
    return [lm["name"] for lm in lms], img_pts, world


def video_size(path):
    cap = cv2.VideoCapture(path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if min(size) <= 0:
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    return size


def write_outputs(result, out_dir, n):
    """accept 결과(시간별, 원본 영상 픽셀) -> calib_cam<N>.json + H_cam<N>.json (가상 클릭 포함)."""
    names, img_pts, world = virtual_clicks(result, video_size(result["video"]))
    clicks = {"names": names, "img_pts": img_pts, "world": world, "clicks": "virtual"}
    timeline = {**result, **clicks}
    anchor = {**{k: result[k] for k in ANCHOR_KEYS if k in result}, **clicks}
    paths = {"calib": os.path.join(out_dir, f"calib_cam{n}.json"), "anchor": os.path.join(out_dir, f"H_cam{n}.json")}
    for key, d in (("calib", timeline), ("anchor", anchor)):
        with open(paths[key], "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
    return paths, sum(p is not None for p in img_pts)


SYNC_RATE = 16000  # 소리 표본 (Hz), futsal.sync와 같게
SYNC_MIN_WINDOWS = 5  # 일치한 1분 구간이 이보다 적거나
SYNC_MAX_STDERR = 0.05  # 표준오차가 이보다 크면 (초) 경고: futsal.sync main과 같은 기준


def ffmpeg_exe():
    """ffmpeg 실행 파일: PATH, 없으면 imageio-ffmpeg에 든 것. 둘 다 없으면 None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def read_audio(path, rate=SYNC_RATE):
    """영상의 소리 -> 모노 float32 (futsal.sync.read_audio와 같은 변환, ffmpeg 위치만 ffmpeg_exe)."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg가 없습니다 (pip install imageio-ffmpeg)")
    raw = subprocess.run([exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def team_sync():
    """팀 저장소의 futsal.sync 모듈. 설치된 패키지, 또는 ../src, ../team/src. 없으면 None."""
    try:
        from futsal import sync
        return sync
    except ImportError:
        pass
    for src in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "..", "team", "src")):
        if os.path.isfile(os.path.join(src, "futsal", "sync.py")):
            sys.path.insert(0, os.path.abspath(src))
            try:
                from futsal import sync
                return sync
            except ImportError:
                sys.path.pop(0)
    return None


def measure_offset(video1, video2, log=print):
    """소리로 잰 두 영상의 시간 차 (cam1 시각 = cam2 시각 + offset)와 진단 dict. 시계 속도 차는 cam2 가운데 시각의 값으로 접는다."""
    sync = team_sync()
    if sync is None:
        raise RuntimeError("시간 차를 잴 futsal.sync(팀 저장소 src/futsal/sync.py)를 찾지 못했습니다")
    hint = sync.start_hint(video1, video2)
    if all("_sync" in os.path.basename(v) for v in (video1, video2)):
        hint = None  # 이미 시작을 맞춘 사본끼리: 파일 이름의 녹화 시각 차는 잘라 낸 뒤라 맞지 않는다
    lag = 5.0 if hint is not None else 30.0
    log(f"두 카메라 시간 차: 소리 비교 중 (대략 {hint or 0.0:+.0f} s 주변 +-{lag:.0f} s, 긴 영상은 몇 분)")
    a, b = read_audio(video1), read_audio(video2)
    if not len(a) or not len(b):
        raise RuntimeError("소리가 없는 영상입니다")
    res = sync.align_segments(a, b, SYNC_RATE, hint or 0.0, lag)
    mid_b = len(b) / SYNC_RATE / 2
    offset = float(res["offset"] + res["drift"] * mid_b)
    info = {"method": "audio: futsal.sync.align_segments (1분 구간마다 소리 패턴 맞춤, 중앙값 + 직선)",
            "hint_s": hint, "n_used": res.get("n_used"), "n_windows": res.get("n_windows"),
            "stderr_s": res.get("stderr"), "drift_ppm": round(float(res["drift"]) * 1e6, 1),
            "offset_at_cam2_start_s": float(res["offset"])}
    weak = (res.get("n_used") or 0) < SYNC_MIN_WINDOWS or (res.get("stderr") or 1.0) > SYNC_MAX_STDERR
    info["warning"] = ("일치하는 구간이 적거나 흩어져 있음: python -m futsal.syncview로 장면을 보고 확인" if weak else None)
    return offset, info


def sync_record(videos, offset, info):
    """sync.json (10/8, 11/3과 같은 형식, 가상 사본). videos = {1: 원본 경로, 2: 원본 경로}, cam1 시각 = cam2 시각 + offset.
    두 영상이 같은 순간이 되는 원본 시각: cam1 s1, cam2 s2 (s1 = s2 + offset, 둘 다 0 이상, 하나는 0)."""
    shifts = {1: max(offset, 0.0), 2: max(-offset, 0.0)}
    entries, cameras, durations = {}, {}, {}
    for n, path in videos.items():
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        durations[n] = frames / fps if fps else 0.0
    duration = min(durations[n] - shifts[n] for n in videos)
    for n, path in videos.items():
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0].split("_sync")[0]
        copy = f"{stem}_sync.mp4"
        cameras[str(n)] = copy
        entries[copy] = {"source": name, "source_time_of_output_0_s": round(shifts[n], 4),
                         "duration_s": round(duration, 3), "source_frames": frames_of(path), "virtual": True}
    return {"offset_a_minus_b_s": round(offset, 4), **info,
            "cut": "virtual: 사본 파일은 만들지 않음. 보정은 원본 영상으로 했고 court_map.load가 이 표로 사본 시각으로 옮긴다",
            "videos": entries, "cameras": cameras, "made_by": "auto_match.py"}


def frames_of(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def _accept(session_id, log_path):
    """다른 프로세스에서 경기 전체 추적 (두 카메라를 동시에). 진행 기록은 log_path에."""
    with open(log_path, "w", encoding="utf-8") as f:
        out = svc.accept(session_id, root=SESSION_ROOT, log=lambda m: print(m, file=f, flush=True))
    return out["result"]


def _ask(question):
    try:
        return input(question).strip().lower() in ("y", "yes", "ㅛ")
    except EOFError:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description="새 경기 영상 2개 -> 클릭 없는 경기 전체 캘리브레이션 (out/<경기>/)")
    p.add_argument("videos", nargs=2, metavar="VIDEO", help='cam1 영상, cam2 영상 ("-"는 그 카메라 건너뜀)')
    p.add_argument("--match", required=True, help="경기 이름 (결과 폴더 out/<경기>), 예: 1120")
    p.add_argument("--yes", action="store_true", help="오버레이 확인 없이 자동 후보를 바로 승인")
    p.add_argument("--force", action="store_true", help="이미 있는 calib_cam<N>.json을 덮어쓴다")
    p.add_argument("--court", help="길이x폭 (미터), 기본 40x20")
    p.add_argument("--offset", type=float, help="두 카메라 시간 차를 직접 지정 (초, cam1 시각 = cam2 시각 + OFFSET). 주면 소리로 재지 않는다")
    p.add_argument("--no-sync", action="store_true", help="sync.json을 만들지 않는다")
    p.add_argument("--sync-only", action="store_true", help="보정은 하지 않고 두 카메라 시간 차(sync.json)만 잰다")
    a = p.parse_args(argv)

    out_dir = match_dir(a.match)
    os.makedirs(out_dir, exist_ok=True)
    spec = svc._parse_court(a.court)
    state_path = os.path.join(out_dir, "auto_match.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

    cams = []
    for n, video in enumerate(a.videos, start=1):
        if video == "-":
            continue
        if not os.path.exists(video):
            raise SystemExit(f"cam{n} 영상이 없습니다: {video}")
        if os.path.exists(os.path.join(out_dir, f"calib_cam{n}.json")) and not (a.force or a.sync_only):
            raise SystemExit(f"{out_dir}에 이미 cam{n} 보정이 있습니다. 덮어쓰려면 --force (다른 경기면 --match를 확인)")
        cams.append((n, os.path.abspath(video)))

    # 1. 자동 인식 (이전 실행에서 승인만 안 한 세션은 다시 쓴다)
    ready = []
    for n, video in ([] if a.sync_only else cams):
        prev = state.get(f"cam{n}", {})
        view = None
        if prev.get("video") == video and prev.get("session"):
            try:
                s = svc._load(prev["session"], SESSION_ROOT)
                if s["state"] == "needs_review":
                    view = svc._view(s)
                    print(f"cam{n}: 이전 자동 인식 결과를 다시 씁니다 (세션 {prev['session']})")
            except svc.SessionError:
                view = None
        if view is None:
            print(f"cam{n}: 자동 코트 인식 중 ({os.path.basename(video)})", flush=True)
            view = svc.propose(video, n, spec, root=SESSION_ROOT, log=lambda m: print(m, flush=True))
        state[f"cam{n}"] = {"video": video, "session": view["session"], "method": view.get("method"),
                            "status": view.get("status")}
        with open(state_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, indent=1, ensure_ascii=False)
        if view["state"] != "needs_review":
            print(f"cam{n}: 자동 인식 실패 ({view.get('status')}, 방법 {view.get('method')}). "
                  f"수동 보정: FUTSAL_MATCH={a.match} python calibrate.py 로 7클릭 후 track.py", flush=True)
            continue
        overlay = os.path.join(out_dir, f"auto_overlay_cam{n}.jpg")
        shutil.copyfile(view["overlay"], overlay)
        print(f"cam{n}: 자동 후보 ({view.get('method')}, {view.get('votes')}/{view.get('windows')} 시점 일치, "
              f"흰 선 일치율 {view.get('confidence'):.2f}) -> 확인: {overlay}", flush=True)
        ready.append((n, view["session"], overlay))

    # 1-2. 두 카메라 시간 차 -> sync.json (추적 전에: 실패해도 보정은 계속한다)
    sync_path = os.path.join(out_dir, "sync.json")
    if len(cams) == 2 and not a.no_sync:
        if os.path.exists(sync_path) and not (a.force or a.sync_only):
            print(f"sync.json이 이미 있어 그대로 둡니다 ({sync_path}, 다시 재려면 --force)")
        else:
            videos = dict(cams)
            try:
                if a.offset is not None:
                    offset, info = a.offset, {"method": "직접 지정 (auto_match.py --offset)"}
                else:
                    offset, info = measure_offset(videos[1], videos[2], log=lambda m: print(m, flush=True))
                with open(sync_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(sync_record(videos, offset, info), f, indent=1, ensure_ascii=False)
                extra = (f", 1분 구간 {info['n_used']}/{info['n_windows']}개 일치, 표준오차 {info['stderr_s']:.3f} s"
                         if info.get("n_used") is not None and info.get("stderr_s") is not None else "")
                print(f"두 카메라 시간 차: cam1 시각 = cam2 시각 {offset:+.3f} s{extra} -> {sync_path}", flush=True)
                if info.get("warning"):
                    print(f"주의: {info['warning']}", flush=True)
            except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
                print(f"두 카메라 시간 차를 재지 못했습니다 ({e}). 나중에 --offset 초 로 다시 실행하거나 "
                      "python -m futsal.sync 결과를 trackview --offset에 넣으세요", flush=True)

    if a.sync_only:
        return 0 if os.path.exists(sync_path) else 1

    # 2. 승인
    approved = []
    for n, session_id, overlay in ready:
        if a.yes or _ask(f"cam{n}: {overlay} 의 빨간 선이 코트 흰 선과 맞습니까? 승인하면 경기 전체를 추적합니다 [y/N] "):
            approved.append((n, session_id))
        else:
            print(f"cam{n}: 승인하지 않았습니다. 다시 실행하면 이 후보로 다시 묻고, 틀렸으면 calibrate.py로 수동 보정하세요")
    if not approved:
        return 1 if cams else 0

    # 3. 경기 전체 추적 (카메라마다 다른 프로세스) -> out/<경기>/
    print("경기 전체 추적 중 (CPU 카메라당 약 7~12분). 진행 기록: "
          + ", ".join(os.path.join(out_dir, f"auto_track_cam{n}.log") for n, _ in approved), flush=True)
    with ProcessPoolExecutor(max_workers=len(approved)) as pool:
        jobs = {n: pool.submit(_accept, sid, os.path.join(out_dir, f"auto_track_cam{n}.log")) for n, sid in approved}
        failed = False
        for n, job in jobs.items():
            try:
                with open(job.result(), encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as e:  # 한 카메라가 실패해도 다른 카메라 결과는 저장한다
                print(f"cam{n}: 추적 실패: {e}", flush=True)
                failed = True
                continue
            paths, n_pts = write_outputs(result, out_dir, n)
            segs = result["segments"]
            lost = sum(s["t1"] - s["t0"] for s in segs if s["source"] == "unresolved")
            state[f"cam{n}"].update(accepted=True, calib=os.path.basename(paths["calib"]))
            print(f"cam{n}: 완료 -> {paths['calib']}, {paths['anchor']} (구간 {len(segs)}개, 코트를 못 찾은 시간 "
                  f"{lost:.0f}s / {segs[-1]['t1']:.0f}s, 먼 하프 {result.get('far_model', 'shared')}, "
                  f"좌우 반전 {bool(result.get('mirror'))}, 가상 클릭 {n_pts}/7)", flush=True)
    with open(state_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    if os.path.exists(sync_path):
        print(f"팀 파이프라인: python -m futsal.trackview --video <cam1> <cam2> ... (팀 저장소의 calibration/out/{a.match}에 "
              "있으면 보정과 시간 차를 sync.json에서 알아서 찾음, 다른 곳이면 --team-calib "
              f"{out_dir} --offset 0 <offset_a_minus_b_s>)")
    else:
        print(f"팀 파이프라인: python -m futsal.trackview --video <cam1> <cam2> --team-calib {out_dir} "
              "--offset 0 <python -m futsal.sync 결과> ...")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
