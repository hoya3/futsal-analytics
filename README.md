# futsal-analytics

야외 풋살장을 스마트폰 2대로 촬영한 영상에서 **영상 픽셀 → 코트 좌표(미터)** 로 변환하는 캘리브레이션 도구입니다.
선수 검출 결과(발 위치 픽셀)를 넣으면 40×20 m 코트 위의 위치로 바꿔 줍니다.

## 카메라 구성

- 폰 2대를 각자 자기 쪽 골라인 근처 낮은 위치에 두고, 반대편 골대 방향으로 촬영
  - cam1: 동쪽 골대(x = +20) 방향을 바라봄
  - cam2: 서쪽 골대(x = −20) 방향을 바라봄
- 카메라가 낮고 야외 코트 바닥이 고르지 않아 호모그래피 하나로는 코트 전체가 맞지 않습니다. 그래서 카메라마다 두 개를 씁니다.
  - `H_near`: 카메라 쪽 절반 (선수 추적용, 정확도 높음)
  - `H_far`: 하프라인 너머 (반대편 카메라의 사각지대를 채우는 용도)
  - 두 행렬은 하프라인 위에서 정확히 같은 값을 내도록 묶여 있어서(`H_far = H_near ∘ K`), 선수가 하프라인을 넘어도 위치가 튀지 않습니다.

## 좌표계

```
        y = +10 (북)
   ┌──────────┬──────────┐
   │          │          │
 서│    ◯     ●     ◯    │동     x: −20 … +20 (코트 길이 방향)
   │          │          │     y: −10 … +10 (코트 폭 방향)
   └──────────┴──────────┘     원점 = 센터 스팟, 단위 = m
        y = −10 (남)
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사용법

모든 스크립트는 프로젝트 폴더에서 실행합니다.

### 촬영 회차 (`out/<회차>/`)

촬영마다 카메라 배치가 달라 결과를 회차 폴더에 따로 둡니다. 회차 이름은 촬영 날짜(MMDD)입니다.

| 회차 | 영상 | 비고 |
|---|---|---|
| `1008` (기본값) | `D:\02_Projects\캡스톤\1008\` | 보정·추적·검증 점 완료 |
| `1103` | `D:\02_Projects\캡스톤\1103\` | 카메라가 반대쪽 모서리, 더 높은 곳 (화면 좌우가 1008과 반대) |

환경변수 `FUTSAL_MATCH`로 회차를 고릅니다. 지정하지 않으면 `1008`이라 기존 코드의 `load(1)`은 그대로 동작합니다.

```bash
FUTSAL_MATCH=1103 python calibrate.py /mnt/d/02_Projects/캡스톤/1103/20261103_165422_sync-001.mp4 /mnt/d/02_Projects/캡스톤/1103/20261103_165851_sync-002.mp4
FUTSAL_MATCH=1103 python track.py 1 2
```

```python
from court_map import load
cam1 = load(1)                                      # 1008 (기본)
cam1 = load(1, "/path/to/futsal-analytics/out/1103")  # 또는 FUTSAL_MATCH=1103
```

아래 설명의 `out/`은 모두 `out/<회차>/`를 뜻합니다.

### 1. 캘리브레이션 (`calibrate.py`)

```bash
python calibrate.py camera1.mp4 camera2.mp4   # "-"는 해당 카메라 건너뜀 (예: - camera2.mp4)
```

영상의 60초 지점에서 시작합니다. 랜드마크가 사람에 가리지 않는 프레임을 고른 뒤, 7개 지점을 클릭합니다.

1. 하프라인 왼쪽 끝
2. 센터서클과 하프라인의 오른쪽 교차점
3. 하프라인 오른쪽 끝
4. 오른쪽 사이드라인 위 아무 점 (화면 아래쪽, 표시가 없어도 됨)
5. 10 m 지점 (작은 흰 점)
6. 먼 쪽 왼쪽 코너
7. 먼 쪽 오른쪽 코너

| 키 | 동작 |
|---|---|
| 왼쪽 클릭 | 점 찍기 (왼쪽 아래 미니맵에 다음 점이 빨간색으로 표시) |
| `a` / `d` | −/+ 10초 이동 (첫 클릭 전에만) |
| `z` / `c` | −/+ 1초 이동 (첫 클릭 전에만) |
| `u` | 실행 취소 |
| `h` | 바라보는 골대 방향 뒤집기 (첫 클릭 전에만) |
| `f` | 좌우 반전 (첫 클릭 전에만): 보이는 사이드라인이 화면 **왼쪽**인 배치(1103). 위 목록의 왼쪽/오른쪽이 서로 바뀌고 미니맵도 바뀜 |
| `m` | 도움말 숨기기/보이기 |
| `Enter` | 저장 |
| `q` | 종료 |

결과: `out/H_cam<N>.json`, `out/frame_cam<N>.jpg`, `out/H_cam<N>_overlay.jpg`

### 2. 흰 선에 맞춰 보정 (`refine.py`)

```bash
python refine.py
```

이미지에서 흰 선을 찾아 ICP와 RANSAC으로 두 호모그래피를 다듬습니다. 결과는 `out/H_cam<N>_refined.json`과 overlay 이미지입니다.
이후 스크립트들은 refined 파일이 있으면 자동으로 그것을 씁니다.

> 저장소의 `out/H_cam<N>_refined.json`은 지금 코드로 `refine.py`를 다시 돌려 나오는 결과보다 정확합니다(검증 점 기준 cam2 가까운 쪽 0.02 m vs 0.17 m). 클릭을 새로 하지 않았다면 다시 돌리지 마세요.

### 2-1. 경기 전체로 전파 (`track.py`)

```bash
python track.py 1 2                 # out/calib_cam<N>.json: 시간 구간별 호모그래피 + 구간 안 밀림
python track.py 1 2 --drift-only    # 구간은 그대로 두고 밀림만 다시 잰다 (카메라당 약 10분)
```

폰이 3 px 넘게 움직이면 구간을 나눠 호모그래피를 다시 맞춥니다. 그보다 작은 느린 밀림(두 카메라 모두 경기 중 약 3 px, 먼 쪽에서는 1 px ≈ 0.3 m)은 구간마다 30초 간격으로 서브픽셀 단위로 재서 `drift`로 저장하고, `court_map`이 시각에 맞춰 보간해 반영합니다.

카메라 이동은 **배경 특징점**(펜스, 건물, 차; SIFT + 밝기 평준화 CLAHE)으로 먼저 재고, 흰 선은 그 자리 ±4 px에서만 맞춥니다. 흰 선만 보면 긴 사이드라인을 따라 미끄러진 위치도 비슷한 점수를 받기 때문입니다(1103 cam1: 실제로는 고정인데 흰 선만으로는 31개 구간으로 잘못 나뉘고 야간 12분을 잃음 → 배경 특징점으로 구간 1개, 야간 포함). 배율·회전이 섞였거나 근거가 약하면 흰 선 탐색으로 넘어갑니다. 1008 결과는 이 변경 전 방식으로 만든 것입니다.

### 3. 코드에서 픽셀 → 코트 좌표 변환 (`court_map.py`)

```python
from court_map import load, pixel_to_court

cam1 = load(1)                                        # refined 캘리브레이션 우선
xy = pixel_to_court(cam1, [[960, 700], [1200, 300]], t=61.2)  # (N, 2) 코트 좌표, 미터
```

`t`는 **동기화 영상(`*_sync.mp4`)의 시각(초)** 입니다. 경기 중 카메라가 밀리므로 `track.py`가 구간별 호모그래피를 만들고, `t`로 구간을 고릅니다. 코트를 확정하지 못한 구간의 점은 NaN입니다. 원본 영상 시각이 필요하면 `load_source`를 씁니다.

### 4. 선수 위치 시각화 (`plot_players.py`)

입력 CSV: 프레임마다 검출된 선수 한 명당 한 줄, 해당 카메라 영상(1920×1080)에서 **발** 위치 픽셀

```csv
cam,t,id,x,y
1,61.20,7,812.5,640.0
2,61.20,3,1102.0,415.5
```

```bash
python plot_players.py detections.csv            # 전체 행 → 이동 경로
python plot_players.py detections.csv --t 61.2   # 해당 시각 ±0.1초 → 한 순간
python plot_players.py detections.csv --bg out/topdown_cam1.jpg
```

결과: `out/players.jpg`(코트 위 점), `out/players_court.csv`(입력에 `X_m`, `Y_m` 열 추가)

### 5. 2차원 코트 지도 (`topdown.py`)

```bash
python topdown.py                  # out/topdown.jpg: 코트 치수로 그린 합성 지도
python topdown.py --photo --lines  # out/topdown_photo.jpg: 실제 영상 합성 + 모델 코트 선 (정렬 확인용)
```

합성 지도는 `plot_players.py`의 기본 배경이자 히트맵 배경입니다. 코트 둘레에 1 m 여백이 있으며 좌표 변환은 `topdown.Canvas`가 담당합니다. 시각별 호모그래피를 카메라 화면에서 확인하려면 `python view_calib.py --t 1000` 또는 `--play`를 씁니다.

### 6. 클릭 없는 자동 초기화 (`keypoint_init.py`)

```python
from keypoint_init import keypoint_calibrate
r = keypoint_calibrate(frame, "east")   # {"status", "H_near", "H_far", ...}
```

코트 키포인트 모델(`train/best.pt`, YOLO pose)이 찾은 지점으로 호모그래피를 풀고 흰 선 ICP로 다듬습니다. 하프라인 위 4점은 한 직선 위라 제약이 부족하므로 첫 행렬에는 먼 10 m 지점까지 더합니다(먼 골라인 지점은 꺾인 바닥 때문에 가까운 쪽을 비틀어 쓰지 않음). 지점이 화면 위쪽에 몰려 있어 ICP는 앵커를 강하게 걸고 넓은 반경으로 화면 아래 선을 잡습니다. 기존 캘리브레이션과의 차이는 가까운 쪽 cam1 0.09 m, cam2 0.08 m(15개 시점 중앙값)이고, 검증 점 기준 먼 쪽은 클릭 보정보다 나빠(cam1 0.63 m vs 0.41 m) 사람이 확인할 자동 후보 용도입니다. 평가: `python eval/eval_keypoint_init.py`.

`keypoint_calibrate_video`는 5개 시점에서 이를 돌려 과반이 같은 답을 내고 가까운 쪽 흰 선 일치율이 0.6 이상이면 `ok`(자동 후보), 아니면 `low_confidence`(수동 클릭 안내)를 돌려줍니다. `calib_service.propose`가 이것을 쓰며, 자동 후보도 화면 확인 뒤 `accept`로 승인해야 진행됩니다.

모델 학습 데이터는 `python train/make_dataset.py`로 시간별 캘리브레이션(밀림 반영)에서 만들고, `train/train_court_pose.ipynb`(Colab) 또는 `_kaggle.ipynb`로 학습합니다. 좌우 반전 증강은 끕니다(`fliplr=0.0`): 두 카메라 모두 대각선 사이드라인이 화면 오른쪽이라 반전 영상은 실제로 나오지 않고, 켜면 먼 쪽 오른쪽 지점을 잘못 배웁니다.

### 정확도 확인

```bash
python eval/label_points.py --cam 1   # 보정에 쓰지 않은 지점을 사람이 찍어 독립 기준을 만든다
python eval/score_points.py           # 그 지점으로 채점 (미터)
python line_error.py                  # 흰 선과 모델 선의 오차 중앙값 (보정과 같은 매칭이라 낙관적)
python measure_circle.py              # 센터서클 반지름/중심 실측
```

## 현재 정확도 (`eval/score_points.py`, 사람이 찍은 검증 점, 중앙값 / 90%)

| | 가까운 쪽 (선) | 센터 (선) | 먼 쪽 (점) |
|---|---|---|---|
| cam1 | 0.01 / 0.01 m | 0.24 / 0.50 m | 0.41 / 0.62 m (10점) |
| cam2 | 0.02 / 0.02 m | 0.06 / 0.07 m | 0.46 / 1.32 m (5점) |

구간 안 밀림 보정 전에는 cam1 먼 쪽 0.61 m, 센터 0.39 m였습니다. 먼 쪽은 화면 1 px가 0.3~0.4 m라 오차가 큽니다.
렌즈 왜곡(직선이 화면 가장자리에서도 1 px 안으로 곧음)과 반대편 카메라 정합(오차가 큰 골대 근처가 반대편의 사각지대)은 효과가 없어 쓰지 않습니다.
검증 점이 적으므로, 최종 정확도는 줄자로 잰 지점 같은 독립 기준점으로 확인해야 합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `court.py` | 공식 풋살 코트 모델 (40×20 m, 센터서클 r = 3 m, 페널티 아크 r = 6 m) |
| `calibrate.py` | 마우스 클릭 캘리브레이션 GUI |
| `refine.py` | 흰 선 기반 호모그래피 보정 |
| `track.py` | 경기 전체로 전파: 시간 구간별 호모그래피, 구간 안 밀림 |
| `court_map.py` | 픽셀 → 코트 좌표 API |
| `topdown.py` | 위에서 본 코트 이미지 생성 |
| `view_calib.py` | 시각별 호모그래피를 카메라 화면에서 확인 |
| `plot_players.py` | 선수 검출 결과를 코트 위에 표시 |
| `calib_service.py` | 자동 후보 → 승인/수동 보정 → 추적 흐름 (서버용). 자동 후보는 `train/best.pt`가 있으면 키포인트 모델, 없으면 `auto_init` |
| `auto_init.py` | 카메라 자세 격자 탐색으로 클릭 없이 코트 찾기 (실험) |
| `keypoint_init.py` | 키포인트 모델로 클릭 없이 코트 찾기 |
| `eval/` | 검증 점 라벨링·채점, 자동 초기화 평가 |
| `train/` | 키포인트 학습 데이터 생성과 학습 노트북 (데이터와 가중치는 저장소에 넣지 않음) |
| `line_error.py`, `measure_circle.py` | 정확도 진단 |
| `out/<회차>/H_cam*.json`, `calib_cam*.json`, `sync.json`, `labels.json` | 회차별 캘리브레이션 결과와 검증 점 (저장소에 포함) |

`out/<회차>/`의 jpg 이미지는 영상 속 사람이 찍혀 있어 저장소에 넣지 않았습니다.
`refine.py`를 실행하려면 `out/frame_cam1.jpg`, `out/frame_cam2.jpg`가 필요합니다(`calibrate.py`가 생성). `topdown.py --photo`는 `out/calib_cam*.json`과 동기화 영상이 필요합니다.

## 앞으로 할 일

- 두 카메라가 같은 선수를 동시에 볼 때 중복 제거와 위치 융합
- 줄자로 잰 기준점으로 실제 정확도 검증 (골대가 규격 위치에 있는지도 확인: 골포스트는 검증 점과 약 0.5 m 어긋남)
- cam2 먼 쪽 골대 주변 지점: 캘리브레이션이 1.7 m까지 어긋나 키포인트 학습에서 빠져 있음
