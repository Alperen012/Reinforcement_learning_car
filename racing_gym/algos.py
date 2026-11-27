"""Which RL algorithms are supported, and their default hyper-parameters.

Algorithms come from Stable-Baselines3: well-tested implementations matter
more here than writing our own. Adding an algorithm = one entry in ``ALGOS``
(see docs/extending.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass(frozen=True)
class AlgoSpec:
    cls: type[BaseAlgorithm]
    action_type: str  # the env action_type this algorithm needs
    summary: str
    defaults: dict[str, Any] = field(default_factory=dict)


ALGOS: dict[str, AlgoSpec] = {
    "ppo": AlgoSpec(
        PPO,
        "continuous",
        "On-policy actor-critic with a clipped objective. Robust default; learns from fresh rollouts only.",
        {"n_steps": 512, "batch_size": 256, "n_epochs": 10, "learning_rate": 3e-4, "gamma": 0.99,
         "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [64, 64]}},
    ),
    "a2c": AlgoSpec(
        A2C,
        "continuous",
        "Synchronous advantage actor-critic: PPO's simpler ancestor (one gradient step per rollout).",
        {"n_steps": 16, "learning_rate": 7e-4, "gamma": 0.99, "gae_lambda": 0.95,
         "policy_kwargs": {"net_arch": [64, 64]}},
    ),
    "sac": AlgoSpec(
        SAC,
        "continuous",
        "Off-policy, maximum-entropy actor-critic with a replay buffer. Very sample-efficient, slower per step.",
        {"learning_rate": 3e-4, "buffer_size": 300_000, "learning_starts": 5_000, "batch_size": 256,
         "gamma": 0.99, "tau": 0.005, "train_freq": 1, "gradient_steps": 2,
         "policy_kwargs": {"net_arch": [64, 64]}},
    ),
    "td3": AlgoSpec(
        TD3,
        "continuous",
        "Off-policy deterministic actor-critic (twin critics, delayed actor updates, target smoothing).",
        {"learning_rate": 3e-4, "buffer_size": 300_000, "learning_starts": 5_000, "batch_size": 256,
         "gamma": 0.99, "train_freq": 1, "gradient_steps": 2, "policy_kwargs": {"net_arch": [64, 64]}},
    ),
    "dqn": AlgoSpec(
        DQN,
        "discrete",
        "Value-based: learns Q(s, a) for 9 fixed (steer, throttle) actions and picks the best one.",
        {"learning_rate": 5e-4, "buffer_size": 200_000, "learning_starts": 5_000, "batch_size": 128,
         "gamma": 0.99, "train_freq": 4, "gradient_steps": 2, "target_update_interval": 2_000,
         "exploration_fraction": 0.3, "exploration_final_eps": 0.03, "policy_kwargs": {"net_arch": [64, 64]}},
    ),
}


def get_spec(name: str) -> AlgoSpec:
    try:
        return ALGOS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(ALGOS)}") from None


def make_model(algo: str, env, seed: int, algo_kwargs: dict[str, Any] | None = None,
               tensorboard_log: str | None = None, device: str = "cpu", verbose: int = 0) -> BaseAlgorithm:
    spec = get_spec(algo)
    kwargs = {**spec.defaults, **(algo_kwargs or {})}
    if spec.cls is TD3 and "action_noise" not in kwargs:
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise

        n = env.action_space.shape[0]
        kwargs["action_noise"] = NormalActionNoise(np.zeros(n), 0.2 * np.ones(n))
    return spec.cls("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, device=device,
                    verbose=verbose, **kwargs)


def load_model(path: str, algo: str, device: str = "cpu") -> BaseAlgorithm:
    return get_spec(algo).cls.load(path, device=device)
