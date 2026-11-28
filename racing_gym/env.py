"""The Gymnasium environment: a car, a track, lidar, and a decomposed reward."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np

from .car import CarState, car_corners, step_car, wrap_angle
from .config import EnvConfig, RewardConfig
from .track import Track, make_track

# 9 discrete actions for value-based agents (DQN): every (steer, throttle) combination.
DISCRETE_ACTIONS: list[tuple[float, float]] = [(s, t) for t in (1.0, 0.0, -1.0) for s in (-1.0, 0.0, 1.0)]

# Procedural training tracks use seeds >= this value, so "random:0".."random:999"
# are guaranteed to be tracks the agent never trained on.
TRAIN_SEED_OFFSET = 1000

REWARD_TERMS = ("progress", "time", "crash", "lap", "center", "steering")


@lru_cache(maxsize=128)
def _cached_track(spec: str, width: float | None) -> Track:
    return make_track(spec, width)


def observation_names(cfg: EnvConfig) -> list[str]:
    names = [f"lidar_{i}" for i in range(cfg.n_rays)]
    names += ["speed", "steer", "heading_err_sin", "heading_err_cos", "lateral_offset"]
    names += [f"curve_{int(d)}px" for d in cfg.lookahead]
    return names


class RaceEnv(gym.Env):
    """Drive laps around a closed track.

    Observation (all roughly in [-1, 1], see :func:`observation_names`):
        lidar distances to the track edge, speed, wheel angle, heading error
        relative to the track direction, lateral offset from the centreline,
        and how much the track turns at a few distances ahead. The observation
        is egocentric: it never contains absolute x/y, so a policy can drive
        tracks it has never seen.

    Action:
        continuous ``[steer, throttle]`` in [-1, 1]^2, or ``Discrete(9)``.

    Reward: weighted sum of the terms in :class:`RewardConfig`; each term is
    reported in ``info["reward_terms"]`` so you can see *why* an agent got a reward.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | dict | None = None, render_mode: str | None = None):
        super().__init__()
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            from .config import _build

            config = _build(EnvConfig, config, "env")
        self.config = config
        self.base_car = self.car_params = config.car_config()
        if not 0.0 <= config.car_randomize < 1.0:
            raise ValueError(f"car_randomize must be in [0, 1), got {config.car_randomize}")
        self.reward_cfg: RewardConfig = config.reward
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode
        if config.action_type not in ("continuous", "discrete"):
            raise ValueError("action_type must be 'continuous' or 'discrete'")
        if config.n_rays < 1:
            raise ValueError("n_rays must be >= 1")

        if config.action_type == "continuous":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(len(DISCRETE_ACTIONS))

        n_obs = config.n_rays + 5 + len(config.lookahead)
        low = -np.ones(n_obs, dtype=np.float32)
        high = np.ones(n_obs, dtype=np.float32)
        low[: config.n_rays] = 0.0
        low[config.n_rays + 4], high[config.n_rays + 4] = -2.0, 2.0  # lateral offset can exceed the edge
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.obs_names = observation_names(config)

        self.ray_offsets = (
            np.linspace(-math.pi / 2, math.pi / 2, config.n_rays) if config.n_rays > 1 else np.zeros(1)
        )
        self._ray_steps = np.arange(4.0, config.ray_length + 1e-9, 4.0)

        self.track: Track | None = None
        if config.track != "random":
            self.track = _cached_track(config.track, config.track_width)
        self.car: CarState | None = None
        self._renderer = None

    # ------------------------------------------------------------------ helpers
    def _lidar(self) -> np.ndarray:
        c = self.car
        ang = c.yaw + self.ray_offsets
        xs = c.x + np.outer(np.cos(ang), self._ray_steps)
        ys = c.y + np.outer(np.sin(ang), self._ray_steps)
        ix, iy = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
        size = self.track.size
        inside = (ix >= 0) & (iy >= 0) & (ix < size) & (iy < size)
        hit = ~inside
        hit[inside] = ~self.track.mask[iy[inside], ix[inside]]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), len(self._ray_steps))
        dist = np.where(first < len(self._ray_steps), self._ray_steps[np.minimum(first, len(self._ray_steps) - 1)],
                        self.config.ray_length)
        return dist

    def _project(self) -> tuple[float, float]:
        """(arc length s, signed lateral offset) of the car on the centreline."""
        t = self.track
        self._idx = t.nearest_index(self.car.x, self.car.y, hint=self._idx)
        dx, dy = self.car.x - t.centerline[self._idx, 0], self.car.y - t.centerline[self._idx, 1]
        s = t.cum_s[self._idx] + dx * t.tangents[self._idx, 0] + dy * t.tangents[self._idx, 1]
        lateral = dx * t.normals[self._idx, 0] + dy * t.normals[self._idx, 1]
        return float(s), float(lateral)

    def _crashed(self) -> bool:
        return any(not self.track.is_drivable(x, y) for x, y in car_corners(self.car, self.car_params))

    def _get_obs(self) -> np.ndarray:
        cfg, p, t = self.config, self.car_params, self.track
        heading_err = wrap_angle(self.car.yaw - t.heading_at(self._idx))
        curves = [wrap_angle(t.heading_at(t.index_ahead(self._idx, d)) - t.heading_at(self._idx)) / math.pi
                  for d in cfg.lookahead]
        obs = np.concatenate([
            self.lidar / cfg.ray_length,
            [self.car.speed / p.max_speed,
             self.car.steer / math.radians(p.max_steer_deg),
             math.sin(heading_err), math.cos(heading_err),
             self.lateral / (t.width / 2.0)],
            curves,
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "track" in options:
            self.track = _cached_track(options["track"], self.config.track_width)
        elif self.config.track == "random":
            track_seed = int(self.np_random.integers(TRAIN_SEED_OFFSET, 2**31 - 1))
            self.track = _cached_track(f"random:{track_seed}", self.config.track_width)
        n = len(self.track.centerline)
        start = int(self.np_random.integers(n)) if self.config.random_start else 0
        start = int(options.get("start_index", start)) % n

        r = self.config.car_randomize
        if r:  # drawn last, so the track and start point are the same as without randomisation
            b = self.base_car
            f = self.np_random.uniform(1.0 - r, 1.0 + r, size=3)
            self.car_params = replace(b, grip=b.grip * f[0], max_speed=b.max_speed * f[1],
                                      engine_accel=b.engine_accel * f[2])
        if self.car_params.max_speed >= self.track.wall_gap:
            # Collisions are checked once per step, so a car this fast could jump through the gap between
            # two parts of the track without ever touching the grass.
            raise ValueError(f"car max_speed {self.car_params.max_speed:g} px/step is not below the narrowest wall "
                             f"of track {self.track.name!r} ({self.track.wall_gap:.1f} px)")
        cx, cy = self.track.centerline[start]
        self.car = CarState(x=float(cx), y=float(cy), yaw=self.track.heading_at(start))
        self._idx = self.start_index = start
        self.s, self.lateral = self._project()
        self.lidar = self._lidar()

        self.steps = 0
        self.distance = 0.0  # signed distance along the track since reset
        self.best_distance = 0.0
        self.steps_since_best = 0
        self.laps = 0
        self.prev_steer_cmd = 0.0
        self.crashed = False
        self.episode_terms = dict.fromkeys(REWARD_TERMS, 0.0)
        self.last_terms = dict.fromkeys(REWARD_TERMS, 0.0)
        self.speed_sum = 0.0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def step(self, action):
        if self.car is None:
            raise RuntimeError("call reset() before step()")
        if self.config.action_type == "discrete":
            steer_cmd, throttle = DISCRETE_ACTIONS[int(action)]
        else:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            if a.shape != (2,):
                raise ValueError(f"expected action [steer, throttle], got shape {a.shape}")
            steer_cmd, throttle = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))

        step_car(self.car, steer_cmd, throttle, self.car_params)
        self.steps += 1
        self.speed_sum += self.car.speed

        old_s = self.s
        self.s, self.lateral = self._project()
        L = self.track.length
        ds = (self.s - old_s + L / 2.0) % L - L / 2.0  # wrap across the start line
        self.distance += ds
        new_laps = max(0, int(self.distance // L) - self.laps)
        self.laps += new_laps
        if self.distance > self.best_distance + 1e-6:
            self.best_distance = self.distance
            self.steps_since_best = 0
        else:
            self.steps_since_best += 1

        self.crashed = self._crashed()
        w = self.reward_cfg
        terms = {
            "progress": w.progress * ds / 10.0,
            "time": w.time,
            "crash": w.crash if self.crashed else 0.0,
            "lap": w.lap * new_laps,
            "center": w.center * (self.lateral / (self.track.width / 2.0)) ** 2,
            "steering": w.steering * abs(steer_cmd - self.prev_steer_cmd),
        }
        self.prev_steer_cmd = steer_cmd
        reward = float(sum(terms.values()))
        self.last_terms = terms
        for k, v in terms.items():
            self.episode_terms[k] += v

        terminated = self.crashed or (self.config.target_laps > 0 and self.laps >= self.config.target_laps)
        truncated = not terminated and (
            self.steps >= self.config.max_steps or self.steps_since_best >= self.config.stall_steps
        )
        self.lidar = self._lidar()
        obs = self._get_obs()
        info = self._info()
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _info(self) -> dict[str, Any]:
        return {
            "laps": self.laps,
            "lap_progress": (self.distance / self.track.length) % 1.0 if self.distance > 0 else 0.0,
            "speed": self.car.speed,
            "crashed": self.crashed,
            "track": self.track.name,
            "reward_terms": dict(self.last_terms),
        }

    def episode_stats(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "laps": self.laps,
            "distance": self.distance,
            "lap_fraction": self.distance / self.track.length,
            "avg_speed": self.speed_sum / max(1, self.steps),
            "crashed": self.crashed,
            "track": self.track.name,
            "reward_terms": dict(self.episode_terms),
        }

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import Renderer

            self._renderer = Renderer(human=self.render_mode == "human", fps=self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
