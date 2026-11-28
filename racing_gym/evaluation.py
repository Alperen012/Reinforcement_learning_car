"""Run policies for a fixed set of seeded episodes and summarise how well they drive.

Every policy (random, hand-written heuristic, untrained network, trained agent)
is evaluated on exactly the same tracks and start positions, so the numbers
are directly comparable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import EnvConfig, load_config
from .env import DISCRETE_ACTIONS, RaceEnv

Policy = Callable[[np.ndarray], Any]


# ----------------------------------------------------------------------------- policies
def random_policy(env: RaceEnv, seed: int = 0) -> Policy:
    space = env.action_space
    space.seed(seed)
    return lambda obs: space.sample()


def heuristic_policy(env: RaceEnv) -> Policy:
    """A hand-written controller: steer against heading error and lateral offset, slow for corners.

    Not reinforcement learning at all. It is here as a yardstick: an RL agent
    should at least match what three lines of classical control achieve.
    """
    n = env.config.n_rays

    def act(obs):
        heading_sin, lateral, curve = obs[n + 2], obs[n + 4], obs[n + 5]
        steer = float(np.clip(-2.0 * heading_sin - 0.8 * lateral + 1.5 * curve, -1, 1))
        throttle = 1.0 if obs[n] < 0.75 or abs(curve) < 0.15 else 0.2
        if env.config.action_type == "discrete":  # nearest of the 9 discrete actions
            return int(np.argmin([(s - steer) ** 2 + (t - throttle) ** 2 for s, t in DISCRETE_ACTIONS]))
        return np.array([steer, throttle], dtype=np.float32)

    return act


def model_policy(model, deterministic: bool = True) -> Policy:
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


# ----------------------------------------------------------------------------- running
# Episodes are deterministic given the start point (deterministic physics and policy), so validation and
# test are two disjoint sets of start points:
# * validation (checkpoint selection during training): the first N points of a fixed, well-spread
#   sequence around the track, identical for every checkpoint so checkpoints are compared on equal terms;
# * test (`racing-gym evaluate`, docs/results.md, challenges): random start points from TEST_SEED + i,
#   redrawn if one lands on any of the first MAX_VALIDATION_EPISODES validation points.
TEST_SEED = 50_000
VALIDATION_SEED = 20_000  # only matters with track="random", where the seed also picks the track
MAX_VALIDATION_EPISODES = 32


def _van_der_corput(k: int) -> float:
    """0.5, 0.25, 0.75, 0.125, ...: any prefix is spread evenly over [0, 1)."""
    x, denom = 0.0, 1.0
    while k:
        denom *= 2
        x += (k & 1) / denom
        k >>= 1
    return x


def validation_starts(n_points: int, episodes: int) -> list[int]:
    """Centreline indices of the validation start points (a prefix of one fixed sequence)."""
    if episodes > MAX_VALIDATION_EPISODES:
        raise ValueError(f"at most {MAX_VALIDATION_EPISODES} validation episodes, got {episodes}")
    return [int(_van_der_corput(i + 1) * n_points) for i in range(episodes)]


def run_episodes(policy: Policy, env_cfg: EnvConfig, tracks: list[str] | None = None, episodes: int = 5,
                 seed: int = TEST_SEED, render_mode: str | None = None, on_step=None,
                 validation: bool = False) -> list[dict[str, Any]]:
    """Run ``episodes`` per track.

    Test mode (default): episode ``i`` uses reset seed ``seed + i`` (a random start point).
    ``validation=True``: episode ``i`` starts at ``validation_starts(...)[i]`` instead.
    Always uses the nominal car: ``car_randomize`` is for training only.
    """
    tracks = list(tracks) if tracks else [env_cfg.track]
    results = []
    for track in tracks:
        env = RaceEnv(replace(env_cfg, track=track, car_randomize=0.0), render_mode=render_mode)
        try:
            for i in range(episodes):
                s = seed + i
                obs, _ = env.reset(seed=s)  # with track="random" this also picks the track
                n = len(env.track.centerline)
                if validation:  # same seed, so the same track; only the start point is overridden
                    obs, _ = env.reset(seed=s, options={"start_index": validation_starts(n, episodes)[i]})
                else:
                    redraw = 0
                    # never test on a validation point
                    while env_cfg.random_start and env.start_index in validation_starts(n, MAX_VALIDATION_EPISODES):
                        redraw += 1
                        s = seed + i + redraw * 1_000_003
                        obs, _ = env.reset(seed=s)
                        n = len(env.track.centerline)
                total, done = 0.0, False
                while not done:
                    obs, r, term, trunc, info = env.step(policy(obs))
                    total += r
                    done = term or trunc
                    if on_step is not None:
                        on_step(env)
                stats = info["episode_stats"]
                stats["return"] = total
                stats["seed"] = s  # the reset seed that reproduces this episode (with its start point)
                results.append(stats)
        finally:
            env.close()
    return results


def validate(policy: Policy, cfg) -> dict[str, float]:
    """The checkpoint-selection score used during training (and by scripts/reselect_best.py)."""
    tracks = list(cfg.train.eval_tracks) or [cfg.env.track]
    return summarize(run_episodes(policy, replace(cfg.env, random_start=True), tracks,
                                  episodes=cfg.train.eval_episodes, seed=VALIDATION_SEED, validation=True))


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    arr = lambda k: np.array([float(r[k]) for r in results])  # noqa: E731
    return {
        "episodes": len(results),
        "return_mean": float(arr("return").mean()),
        "return_std": float(arr("return").std()),
        "laps_mean": float(arr("laps").mean()),
        "lap_fraction_mean": float(arr("lap_fraction").mean()),
        "crash_rate": float(arr("crashed").mean()),
        "avg_speed": float(arr("avg_speed").mean()),
        "steps_mean": float(arr("steps").mean()),
    }


# ----------------------------------------------------------------------------- run dirs
def find_model(path: str | Path) -> tuple[Path, Any]:
    """Resolve a run directory (or a .zip inside one) to (model path, ExperimentConfig)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    run_dir = p if p.is_dir() else p.parent
    if run_dir.name == "checkpoints":  # runs/<name>/seed0/checkpoints/step_*.zip
        run_dir = run_dir.parent
    cfg_file = run_dir / "config.toml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"{cfg_file} not found. Pass a run directory created by `racing-gym train`.")
    cfg = load_config(cfg_file)
    if p.is_dir():
        for name in ("best_model.zip", "final_model.zip"):
            if (p / name).exists():
                return p / name, cfg
        raise FileNotFoundError(f"no best_model.zip or final_model.zip in {p}")
    return p, cfg


def make_policy(kind: str, env_cfg: EnvConfig, seed: int = 0, algo: str = "ppo",
                deterministic: bool = True) -> tuple[Policy, EnvConfig, str]:
    """Build a policy by name: ``random``, ``heuristic``, ``untrained`` or a run directory / model path.

    Returns (policy, env config the policy expects, label).
    """
    from .algos import get_spec, load_model, make_model

    if kind in ("random", "heuristic", "untrained"):
        if kind == "untrained":
            env_cfg = replace(env_cfg, action_type=get_spec(algo).action_type)
        env = RaceEnv(env_cfg)
        if kind == "random":
            return random_policy(env, seed), env_cfg, "random"
        if kind == "heuristic":
            return heuristic_policy(env), env_cfg, "heuristic"
        model = make_model(algo, env, seed=seed)
        return model_policy(model, deterministic), env_cfg, f"untrained {algo}"
    model_path, run_cfg = find_model(kind)
    model = load_model(str(model_path), run_cfg.algo)
    # Keep the caller's track/start settings, but the agent's own observation/action/car setup
    # (its nominal car: domain randomisation is for training only).
    agent_env = replace(run_cfg.env, track=env_cfg.track, random_start=env_cfg.random_start,
                        max_steps=env_cfg.max_steps, stall_steps=env_cfg.stall_steps, car_randomize=0.0)
    label = f"{run_cfg.name} ({run_cfg.algo})"
    if model_path.parent.name == "checkpoints":
        step = int(model_path.stem.split("_")[-1])
        label += " untrained" if step == 0 else f" after {step:,} steps"
    else:
        label += " trained"
    return model_policy(model, deterministic), agent_env, label


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
