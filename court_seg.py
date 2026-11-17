"""코트 라인 세그멘테이션 모델(train/seg_best.ts)로 한 프레임의 흰 선을 클래스별로 찾는다.

학습은 train/train_court_seg_colab.ipynb (segmentation_models_pytorch U-Net). 노트북이 내보낸 TorchScript
(seg_best.ts, 메타데이터 내장)를 쓰면 로컬에는 torch / numpy / opencv만 있으면 된다. TorchScript가 없고
state_dict(seg_best.pt + seg_meta.json)만 있으면 segmentation_models_pytorch로 모델을 다시 만든다
(pip install segmentation-models-pytorch==0.5.0).

사용 예
    import cv2
    from court_seg import load_seg_model, predict, line_strength, class_map

    frame = cv2.imread("frame.jpg")                 # BGR, 크기 자유 (1920x1080 등)
    model = load_seg_model()                        # 기본: train/seg_best.ts (없으면 train/seg_best.pt)
    prob = predict(frame, model)                    # (C, H, W) float32 클래스별 확률, 원본 해상도
    ridge = line_strength(frame, model)             # (H, W) float32 [0, 1]: refine.ridge_map(frame) 대신 ICP에 넣는다
    labels, conf = class_map(frame, model)          # (H, W) uint8 클래스 번호, (H, W) float32 그 확률
    model.names                                     # 클래스 이름 (classes.json 순서)

    # 미러 카메라 (calibration 키 mirror: true): 데이터셋이 그런 경기를 좌우 반전해 1008 배치로 맞췄으므로
    # 입력을 반전해 예측하고 결과를 다시 반전해 원래 픽셀 위치로 돌려준다.
    ridge = line_strength(frame, model, mirror=True)

클래스 이름(sideline_left 등)은 항상 정규 배치(1008) 기준이다. mirror=True면 픽셀 위치는 원래 프레임으로 돌아오지만
이름은 정규 배치 기준 그대로라, 원래 화면에서 sideline_left는 오른쪽에 있다.

모델은 처음 쓸 때 읽는다 (모델 파일이 없어도 import는 된다).
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TS_PATH = os.path.join(HERE, "train", "seg_best.ts")  # 노트북 13번 셀이 만드는 TorchScript (메타데이터 내장)
PT_PATH = os.path.join(HERE, "train", "seg_best.pt")  # state_dict: 같은 폴더의 seg_meta.json이 필요하다
SMP_PIN = "segmentation-models-pytorch==0.5.0"  # 노트북과 같은 버전
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

_cache = {}


class SegModel:
    """불러온 네트워크와 전처리 정보. net(x)는 (N, C, H_pad, W_pad) 로짓을 낸다."""

    def __init__(self, net, meta, device, path):
        self.net, self.meta, self.device, self.path = net, meta, device, path
        self.names = list(meta["names"])
        self.input_size = tuple(meta.get("input_size", (1280, 720)))  # (W, H)
        self.pad_to = int(meta.get("pad_to", 32))
        self.mean = np.array(meta.get("mean", IMAGENET_MEAN), np.float32) * 255
        self.std = np.array(meta.get("std", IMAGENET_STD), np.float32) * 255
        self.rgb = meta.get("color", "RGB").upper() == "RGB"
        bg = [i for i, n in enumerate(self.names) if n == "background"]
        self.line_classes = list(meta.get("line_classes") or [i for i in range(len(self.names)) if i not in bg])

    def __repr__(self):
        return f"SegModel({os.path.basename(self.path)}, {self.meta.get('arch')}/{self.meta.get('encoder')}, " \
               f"{len(self.names)} classes, input {self.input_size})"


def load_seg_model(path=None, device="cpu"):
    """모델을 한 번만 읽어 둔다. path: .ts(TorchScript) 또는 .pt(state_dict). None이면 TS_PATH, 없으면 PT_PATH."""
    if path is None:
        path = TS_PATH if os.path.exists(TS_PATH) or not os.path.exists(PT_PATH) else PT_PATH
    path = os.path.abspath(path)
    key = (path, str(device))
    if key not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"세그멘테이션 모델이 없습니다: {path}\n"
                "Colab 노트북(train/train_court_seg_colab.ipynb)이 Drive court_seg/baseline_seg/에 만든 "
                "seg_best.ts를 calibration/train/seg_best.ts로 복사하세요.")
        _cache[key] = _load_torchscript(path, device) if _is_torchscript(path) else _load_state_dict(path, device)
    return _cache[key]


def _is_torchscript(path):
    if path.endswith((".ts", ".torchscript")):
        return True
    import zipfile
    try:  # TorchScript zip에는 code/ 폴더가 있다 (state_dict에는 없다)
        with zipfile.ZipFile(path) as z:
            return any("/code/" in n for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def _load_torchscript(path, device):
    import torch
    extra = {"meta.json": ""}
    net = torch.jit.load(path, map_location=device, _extra_files=extra)
    raw = extra["meta.json"]
    meta = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw) if raw else _read_meta_json(path)
    return SegModel(net.eval(), meta, device, path)


def _read_meta_json(path):
    folder, stem = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    for cand in (os.path.join(folder, stem + ".json"), os.path.join(folder, "seg_meta.json")):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"{path}의 메타데이터(seg_meta.json)가 없습니다: 노트북이 만든 seg_meta.json을 같은 폴더에 두세요")


def _load_state_dict(path, device):
    import torch
    meta = _read_meta_json(path)
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            f"{os.path.basename(path)}(state_dict)를 읽으려면 segmentation_models_pytorch가 필요합니다.\n"
            f"  .venv\\Scripts\\python.exe -m pip install {SMP_PIN}\n"
            "또는 노트북이 내보낸 seg_best.ts(TorchScript)를 train/에 두면 추가 패키지 없이 읽습니다.") from e
    net = getattr(smp, meta["arch"])(encoder_name=meta["encoder"], encoder_weights=None, in_channels=3,
                                     classes=len(meta["names"]))
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return SegModel(net.to(device).eval(), meta, device, path)


def _net_probs(frame, model, mirror):
    """(C, h, w) float32 softmax 확률, 네트워크 입력 해상도 (패딩 제외). mirror면 반전 프레임으로 예측해 되돌린다."""
    import torch
    model = model or load_seg_model()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if mirror:
        frame = np.ascontiguousarray(frame[:, ::-1])
    w, h = model.input_size
    if frame.shape[:2] != (h, w):
        shrink = frame.shape[1] > w
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if model.rgb else np.ascontiguousarray(frame)
    x = (img.astype(np.float32) - model.mean) / model.std  # 학습과 같은 정규화 (0~255 기준 mean/std)
    ph, pw = (-h) % model.pad_to, (-w) % model.pad_to
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)))  # 학습처럼 정규화 뒤 아래/오른쪽을 0으로 채운다
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(model.device)
    with torch.inference_mode():
        prob = torch.softmax(model.net(t).float(), dim=1)[0, :, :h, :w].cpu().numpy()
    return prob[:, :, ::-1] if mirror else prob


def _resize_maps(maps, size):
    """(C, h, w) -> (C, H, W). cv2.resize는 채널 512개까지 HWC로 한 번에 처리한다."""
    W, H = size
    if maps.shape[1:] == (H, W):
        return np.ascontiguousarray(maps)
    out = cv2.resize(np.ascontiguousarray(maps.transpose(1, 2, 0)), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out.reshape(H, W, -1).transpose(2, 0, 1))


def predict(frame_bgr, model=None, mirror=False):
    """클래스별 확률 (C, H, W) float32, 원본 프레임 해상도. 클래스 순서는 model.names."""
    prob = _net_probs(frame_bgr, model, mirror)
    return _resize_maps(prob, (frame_bgr.shape[1], frame_bgr.shape[0]))


def class_ids(classes, model=None):
    """클래스 이름/번호 목록 -> 번호 목록. None이면 배경을 뺀 모든 선 클래스."""
    model = model or load_seg_model()
    if classes is None:
        return list(model.line_classes)
    ids = []
    for c in classes:
        if isinstance(c, str):
            if c not in model.names:
                raise KeyError(f"모르는 클래스 {c!r}: {model.names}")
            ids.append(model.names.index(c))
        else:
            ids.append(int(c))
    return ids


def line_strength(frame, model=None, classes=None, mirror=False):
    """흰 선 강도 (H, W) float32 [0, 1]: 고른 선 클래스 확률의 최댓값. refine.ridge_map(frame)과 같은 형식이라
    ICP(match_lines / icp_align / icp_align_far)에 그대로 넣는다. classes: 이름 또는 번호 목록 (None이면 모든 선 클래스)."""
    model = model or load_seg_model()
    prob = _net_probs(frame, model, mirror)
    s = prob[class_ids(classes, model)].max(axis=0)
    s = _resize_maps(s[None], (frame.shape[1], frame.shape[0]))[0]
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def class_map(frame, model=None, mirror=False):
    """(labels, conf): 픽셀마다 가장 확률이 높은 클래스 번호 (H, W) uint8와 그 확률 (H, W) float32, 원본 해상도.
    클래스별 매칭(예: 사이드라인 점은 사이드라인 픽셀에만)을 할 때 쓴다. 배경은 0번 (model.names 참고)."""
    prob = predict(frame, model, mirror)
    return prob.argmax(axis=0).astype(np.uint8), prob.max(axis=0)


def colorize(labels, names=None):
    """클래스 번호 맵 -> BGR 미리보기 (배경 검정). 확인용."""
    n = int(labels.max()) + 1 if names is None else len(names)
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.arange(n) * 180 // max(n - 1, 1)) % 180
    hsv[:, 0, 1:] = 255
    lut = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    lut[0] = 0
    return lut[np.minimum(labels, n - 1)]


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("사용: python court_seg.py 프레임.jpg [모델 경로] [--mirror]")
        sys.exit(1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    img = cv2.imread(args[0])
    m = load_seg_model(args[1] if len(args) > 1 else None)
    t0 = time.time()
    lab, _ = class_map(img, m, mirror="--mirror" in sys.argv)
    print(m, f"{time.time() - t0:.2f}s")
    for i, name in enumerate(m.names):
        print(f"  {i:2d} {name:20s} {int((lab == i).sum()):8d} px")
    out = os.path.splitext(args[0])[0] + "_seg.jpg"
    cv2.imwrite(out, cv2.addWeighted(img, 0.5, colorize(lab, m.names), 0.8, 0))
    print("저장:", out)
