"""Typed experiment configuration loaded from TOML files.

Every knob a user is expected to touch lives here, with its default and unit.
Unknown keys raise an error instead of being silently ignored, so a typo in a
config file never produces a "successful" experiment with default values.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CarConfig:
    """Kinematic bicycle model with a grip limit. Units: px, steps (1 step = 1/30 s)."""

    max_speed: float = 8.0  # px/step
    max_reverse_speed: float = 3.0  # px/step
    engine_accel: float = 0.35  # px/step^2 at full throttle
    brake_accel: float = 0.6  # px/step^2 at full brake
    rolling_drag: float = 0.02  # px/step^2, constant
    aero_drag: float = 0.004  # px/step^2 per (px/step)^2
    wheel_base: float = 24.0  # px
    max_steer_deg: float = 32.0  # front wheel angle at full lock
    steer_rate_deg: float = 6.0  # max change of wheel angle per step
    grip: float = 0.9  # max lateral acceleration, px/step^2. Lower = understeer at speed
    length: float = 28.0  # collision box, px
    width: float = 14.0


CAR_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {},
    "grippy": {"max_speed": 6.0, "grip": 1.6, "engine_accel": 0.3},
    "rocket": {"max_speed": 11.0, "engine_accel": 0.5, "grip": 0.7},
}


@dataclass
class RewardConfig:
    """Weights of each reward term. The environment reports every term separately in ``info``."""

    progress: float = 1.0  # per 10 px driven along the centreline (negative when going backwards)
    time: float = -0.05  # every step
    crash: float = -20.0  # once, when the car leaves the track (episode ends)
    lap: float = 10.0  # each completed lap
    center: float = 0.0  # times (lateral_offset / half_width)^2, every step
    steering: float = 0.0  # times |change in steering command|, every step (smoothness)


@dataclass
class EnvConfig:
    track: str = "classic"  # preset name, "random:<seed>", or "random" (new track every episode)
    track_width: float | None = None  # None = the track's own width
    car: str | CarConfig = "balanced"  # preset name or explicit parameters
    reward: RewardConfig = field(default_factory=RewardConfig)
    action_type: str = "continuous"  # "continuous" [steer, throttle] or "discrete" (9 actions)
    n_rays: int = 9  # lidar rays spread over -90..+90 degrees
    ray_length: float = 250.0  # px
    lookahead: tuple[float, ...] = (60.0, 120.0, 240.0)  # px ahead to report track curvature
    random_start: bool = True  # start at a random point of the lap (else at the start line)
    max_steps: int = 2000
    stall_steps: int = 200  # truncate if the best progress has not improved for this many steps
    target_laps: int = 0  # end the episode after this many laps (0 = never)

    def car_config(self) -> CarConfig:
        if isinstance(self.car, CarConfig):
            return self.car
        if self.car not in CAR_PRESETS:
            raise ValueError(f"unknown car preset {self.car!r}; choose from {sorted(CAR_PRESETS)}")
        return CarConfig(**CAR_PRESETS[self.car])


@dataclass
class TrainConfig:
    total_timesteps: int = 300_000
    n_envs: int = 8
    seeds: tuple[int, ...] = (0,)
    eval_freq: int = 25_000  # timesteps between evaluations (summed over all envs)
    eval_episodes: int = 10
    eval_tracks: tuple[str, ...] = ()  # empty = the training track
    device: str = "cpu"  # small MLPs are faster on CPU


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    algo: str = "ppo"
    algo_kwargs: dict[str, Any] = field(default_factory=dict)  # passed straight to the SB3 constructor
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------- loading
def _build(cls, data: dict[str, Any], where: str):
    if not isinstance(data, dict):
        raise TypeError(f"[{where}] expected a table, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"[{where}] unknown key(s): {sorted(unknown)}. Valid: {sorted(fields)}")
    kwargs = {}
    for key, value in data.items():
        nested = {"reward": RewardConfig, "env": EnvConfig, "train": TrainConfig}.get(key)
        if key == "car" and isinstance(value, dict):
            kwargs[key] = _build(CarConfig, value, f"{where}.car")
        elif nested is not None and isinstance(value, dict):
            kwargs[key] = _build(nested, value, f"{where}.{key}" if where else key)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data, "")


def to_dict(cfg) -> dict[str, Any]:
    """Dataclass -> plain dict (tuples become lists), suitable for TOML/JSON."""

    def conv(v):
        if dataclasses.is_dataclass(v):
            return {k: conv(x) for k, x in dataclasses.asdict(v).items() if x is not None}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items() if x is not None}
        return v

    return conv(cfg)


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text  # bare word -> string


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (value parsed as TOML, bare words as strings)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.path=value")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {k!r} is not a table")
        node[keys[-1]] = _parse_value(raw.strip())
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    return config_from_dict(apply_overrides(data, overrides or []))


def dump_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for our own config dicts (scalars, lists, nested tables)."""

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return repr(v)

    lines: list[str] = []

    def table(prefix: str, d: dict[str, Any]):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            lines.append(f"\n[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {fmt(v)}")
        for k, v in tables.items():
            table(f"{prefix}.{k}" if prefix else k, v)

    table("", data)
    return "\n".join(lines).strip() + "\n"
