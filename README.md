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

모든 스크립트는 프로젝트 폴더에서 실행합니다(결과는 `out/`에 저장).

### 1. 캘리브레이션 (`calibrate.py`)

```bash
python calibrate.py camera1.mp4 camera2.mp4   # "-"는 해당 카메라 건너뜀 (예: - camera2.mp4)
```

영상의 60초 지점에서 시작합니다. 랜드마크가 사람에 가리지 않는 프레임을 고른 뒤, 카메라 쪽 절반에서 5개 지점을 클릭합니다.

1. 하프라인 왼쪽 끝
2. 센터서클과 하프라인의 오른쪽 교차점
3. 하프라인 오른쪽 끝
4. 오른쪽 사이드라인 위 아무 점 (화면 아래쪽, 표시가 없어도 됨)
5. 10 m 지점 (작은 흰 점)

| 키 | 동작 |
|---|---|
| 왼쪽 클릭 | 점 찍기 (왼쪽 아래 미니맵에 다음 점이 빨간색으로 표시) |
| `a` / `d` | −/+ 10초 이동 (첫 클릭 전에만) |
| `z` / `c` | −/+ 1초 이동 (첫 클릭 전에만) |
| `u` | 실행 취소 |
| `h` | 바라보는 골대 방향 뒤집기 (첫 클릭 전에만) |
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

### 3. 코드에서 픽셀 → 코트 좌표 변환 (`court_map.py`)

```python
from court_map import load, pixel_to_court

cam1 = load(1)                                        # refined 캘리브레이션 우선
xy = pixel_to_court(cam1, [[960, 700], [1200, 300]])  # (N, 2) 코트 좌표, 미터
```

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

### 5. 위에서 본 코트 이미지 (`topdown.py`)

```bash
python topdown.py          # out/topdown.jpg: 두 카메라의 near 쪽 + 사각지대(노란 테두리)
python topdown.py --cam 1  # out/topdown_cam1.jpg: 카메라 1이 본 코트 전체
```

### 정확도 확인

```bash
python line_error.py        # 흰 선과 모델 선의 오차 중앙값 (refined 기준, ""를 주면 클릭만 한 결과)
python measure_circle.py    # 센터서클 반지름/중심 실측
```

## 현재 정확도 (t = 60초 프레임)

| | near | far |
|---|---|---|
| cam1 | 0.01 m | 0.36 m |
| cam2 | 0.08 m | 0.61 m |

센터서클 실측: cam1 반지름 3.01 m(잔차 0.22 m), cam2 반지름 2.95 m(잔차 0.06 m)

`line_error.py`는 보정에 쓴 것과 같은 선 매칭으로 평가하므로 실제보다 좋게 나옵니다.
실제 정확도는 줄자로 잰 지점 같은 독립 기준점으로 확인해야 합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `court.py` | 공식 풋살 코트 모델 (40×20 m, 센터서클 r = 3 m, 페널티 아크 r = 6 m) |
| `calibrate.py` | 마우스 클릭 캘리브레이션 GUI |
| `refine.py` | 흰 선 기반 호모그래피 보정 |
| `court_map.py` | 픽셀 → 코트 좌표 API |
| `topdown.py` | 위에서 본 코트 이미지 생성 |
| `plot_players.py` | 선수 검출 결과를 코트 위에 표시 |
| `line_error.py`, `measure_circle.py` | 정확도 진단 |
| `out/H_cam*.json` | 캘리브레이션 결과 (저장소에 포함) |

`out/`의 jpg 이미지는 영상 속 사람이 찍혀 있어 저장소에 넣지 않았습니다.
`refine.py`, `topdown.py`를 실행하려면 `out/frame_cam1.jpg`, `out/frame_cam2.jpg`가 필요합니다(`calibrate.py`가 생성).

## 앞으로 할 일

- 두 카메라 영상의 시간 동기화 (두 폰의 녹화 시작 시점이 다름)
- 두 카메라가 같은 선수를 동시에 볼 때 중복 제거와 위치 융합
- 클릭 실수 검출을 위한 여분 랜드마크 (현재 5번 클릭이 자유도 8과 딱 맞아 오차를 알 수 없음)
