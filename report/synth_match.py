"""개발용 합성 경기 tracks.json 생성기 (팀 코드에 의존하지 않음).

팀 시뮬레이터(futsal.sim.scenario)와 같은 방식(웨이포인트 + 가속 제한)으로 움직이되,
리포트가 다뤄야 할 실제 데이터의 문제를 일부러 넣는다:
  - 전·후반 (후반에 진영 교체), 팀마다 교체 선수 1명
  - 위치 잡음 (먼 쪽일수록 큼, 0.12~0.30 m), 짧은 미검출 구간 (NaN)
  - ID 조각: 한 선수의 일부 구간이 새 ID로 잡힘 + 몇 초짜리 가짜 ID
  - 잘못 이어 붙인 구간 (순간이동 → 원시 최고속도 튐)
  - 공 (선택): 일부만 보이고, 가끔 코트 밖으로 잘못 투영된 위치

python report/synth_match.py out.json --minutes 40 --seed 1 [--no-ball]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

L, W = 40.0, 20.0
NAMES = ["김민준", "이서준", "박도윤", "최예준", "정시우", "강하준", "조주원", "윤지호", "장지후", "임준서", "한지성", "오태윤"]
GAITS = [(1.2, 0.22), (2.8, 0.45), (4.5, 0.24), (6.5, 0.09)]   # (속도 m/s, 확률)
MAX_ACCEL = 3.0


def _ball_path(n: int, dt: float, rng) -> np.ndarray:
    t = np.arange(n) * dt
    ph = rng.uniform(0, 2 * np.pi, 4)
    return np.stack([L / 2 + 0.42 * L * np.sin(t / 4.1 + ph[0]) * np.cos(t / 11.0 + ph[1]),
                     W / 2 + 0.40 * W * np.sin(t / 2.7 + ph[2]) * np.cos(t / 7.3 + ph[3])], 1)


def _run(n: int, dt: float, home: np.ndarray, ball: np.ndarray, gk: bool, rng, active: np.ndarray) -> np.ndarray:
    """active 인 프레임에서만 움직이는 한 선수의 실제 궤적 (비활성 = NaN)."""
    p, v = home.copy(), np.zeros(2)
    xy = np.full((n, 2), np.nan)
    next_change, target, speed = 0, home, 0.0
    sp, pr = np.array([g[0] for g in GAITS]), np.array([g[1] for g in GAITS])
    for i in range(n):
        if not active[i]:
            continue
        if i >= next_change:
            if gk:
                target = np.array([home[0], np.clip(ball[i, 1], W / 2 - 3, W / 2 + 3)])
                speed = rng.choice([0.8, 2.0, 3.5], p=[.5, .35, .15])
            else:
                w = rng.uniform(0.4, 0.9)
                target = (1 - w) * home + w * ball[i] + rng.normal(0, 3.0, 2)
                speed = rng.choice(sp, p=pr)
            target = np.clip(target, [0.5, 0.5], [L - 0.5, W - 0.5])
            next_change = i + int(rng.uniform(1.5, 5.0) / dt)
        d = target - p
        dist = np.linalg.norm(d)
        desired = d / dist * min(speed, dist * 1.5) if dist > 1e-6 else np.zeros(2)
        dv = desired - v
        if np.linalg.norm(dv) > MAX_ACCEL * dt:
            dv *= MAX_ACCEL * dt / np.linalg.norm(dv)
        v = v + dv
        p = np.clip(p + v * dt, [0.3, 0.3], [L - 0.3, W - 0.3])
        xy[i] = p
    return xy


def synthetic_match(minutes: float = 40.0, fps: float = 10.0, seed: int = 1, ball: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    n = int(minutes * 60 * fps)
    half = n // 2
    bpath = _ball_path(n, dt, rng)
    # 역할 위치 (전반 기준: A가 왼쪽 골대를 지킴). 후반에는 180도 돌린다.
    roles = {"A": [(0.015, 0.5), (0.22, 0.25), (0.22, 0.75), (0.40, 0.5), (0.50, 0.35)],
             "B": [(0.985, 0.5), (0.78, 0.25), (0.78, 0.75), (0.60, 0.5), (0.50, 0.65)]}
    truth = []           # (team, name, xy)
    k = 0
    for team, slots in roles.items():
        # 교체: 각 반의 중간 10분 동안 슬롯 3(공격수)을 교체 선수가 뛴다
        sub_on = np.zeros(n, bool)
        for h0 in (0, half):
            sub_on[h0 + half // 4: h0 + 3 * half // 4] = True
        for s, (rx, ry) in enumerate(slots + [slots[3]]):
            name = NAMES[k]; k += 1
            if s == 3:
                active = ~sub_on
            elif s == 5:
                active = sub_on
            else:
                active = np.ones(n, bool)
            xy = np.full((n, 2), np.nan)
            for h0, h1, flip in ((0, half, False), (half, n, True)):
                home = np.array([rx * L, ry * W])
                if flip:
                    home = np.array([L, W]) - home
                xy[h0:h1] = _run(h1 - h0, dt, home, bpath[h0:h1], s == 0, rng, active[h0:h1])
            truth.append((team, name, xy))

    players, pid = [], 1
    for team, name, xy in truth:
        obs = xy.copy()
        noise = 0.12 + 0.18 * np.nan_to_num(xy[:, 1:2]) / W
        obs = obs + rng.normal(0, 1, obs.shape) * noise
        # 짧은 미검출 (가려짐): 경기당 약 40회, 0.5~8 s
        for _ in range(40):
            a = int(rng.integers(0, n)); obs[a:a + int(rng.uniform(5, 80))] = np.nan
        pieces = [(name, obs)]
        # ID 조각: 일부 선수의 40~120 s 구간이 새 ID로 잡힘
        if rng.random() < 0.35:
            frag = np.full_like(obs, np.nan)
            a = int(rng.integers(0, n - 1500)); b = a + int(rng.uniform(400, 1200))
            frag[a:b] = obs[a:b]; obs[a:b] = np.nan
            pieces.append((None, frag))
        # 잘못 이어 붙인 구간: 0.3 s 동안 다른 곳으로 순간이동
        if rng.random() < 0.3:
            a = int(rng.integers(100, n - 100))
            obs[a:a + 3] = obs[a:a + 3] + np.array([6.0, 4.0])
        for nm, arr in pieces:
            players.append({"id": pid, "team": team, "name": nm, "xy": arr}); pid += 1
    # 몇 초짜리 가짜 ID (팀 판별도 엉뚱할 수 있음)
    for _ in range(4):
        a = int(rng.integers(0, n - 200)); m = int(rng.uniform(20, 150))
        arr = np.full((n, 2), np.nan)
        arr[a:a + m] = rng.uniform([2, 2], [L - 2, W - 2]) + np.cumsum(rng.normal(0, 0.1, (m, 2)), 0)
        players.append({"id": pid, "team": str(rng.choice(["A", "B"])), "name": None, "xy": arr}); pid += 1

    def col(v):
        return [None if not np.isfinite(a) else round(float(a), 2) for a in v]

    d = {"version": 1, "court": {"length": L, "width": W}, "fps": fps, "start": 0.0, "n_frames": n,
         "players": [{"id": p["id"], "team": p["team"], "name": p["name"], "stats": {},
                      "x": col(p["xy"][:, 0]), "y": col(p["xy"][:, 1])} for p in players]}
    if ball:
        b = bpath + rng.normal(0, 0.2, bpath.shape)
        seen = np.ones(n, bool)
        for _ in range(int(n / 60)):
            a = int(rng.integers(0, n)); seen[a:a + int(rng.uniform(5, 60))] = False
        air = rng.random(n) < 0.01                                  # 떠 있는 공 → 엉뚱한 투영
        b[air] = b[air] + rng.normal(0, 6, (air.sum(), 2))
        b[~seen] = np.nan
        cam = np.where(seen, np.where(b[:, 0] < L / 2, 1, 2), 0)
        d["ball"] = {"x": col(b[:, 0]), "y": col(b[:, 1]), "cam": [int(c) for c in cam]}
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-ball", action="store_true")
    a = ap.parse_args(argv)
    d = synthetic_match(a.minutes, seed=a.seed, ball=not a.no_ball)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {a.out}: {len(d['players'])} IDs, {d['n_frames']} frames")


if __name__ == "__main__":
    main()
