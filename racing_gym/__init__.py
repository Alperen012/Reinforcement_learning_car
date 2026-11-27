"""Racing Gym: a small, readable racing environment for learning reinforcement learning."""

import gymnasium as gym

from .config import CarConfig, EnvConfig, ExperimentConfig, RewardConfig, load_config
from .env import RaceEnv
from .track import PRESETS, Track, make_track, random_track

__version__ = "0.2.0"

if "RacingGym-v0" not in gym.registry:
    gym.register(id="RacingGym-v0", entry_point="racing_gym.env:RaceEnv")

__all__ = [
    "CarConfig",
    "EnvConfig",
    "ExperimentConfig",
    "PRESETS",
    "RaceEnv",
    "RewardConfig",
    "Track",
    "load_config",
    "make_track",
    "random_track",
]
