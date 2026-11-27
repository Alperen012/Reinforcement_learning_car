"""Training loop: one run directory per (experiment, seed) with everything needed to reproduce it.

runs/<name>/seed<k>/
    config.toml        the fully resolved config (re-run with `racing-gym train --config` on it)
    run.json           algo, seed, library versions, wall-clock time
    episodes.csv       one row per finished training episode (return, laps, crash, speed, reward terms)
    evals.csv          periodic deterministic evaluations on the eval tracks
    best_model.zip     model with the best mean eval return so far
    final_model.zip    model at the end of training
    tb/                TensorBoard logs
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .algos import get_spec, make_model
from .config import ExperimentConfig, dump_toml, to_dict
from .env import REWARD_TERMS, RaceEnv
from .evaluation import model_policy, run_episodes, summarize

EPISODE_FIELDS = ["timesteps", "return", "steps", "laps", "lap_fraction", "crashed", "avg_speed", "track",
                  *[f"r_{t}" for t in REWARD_TERMS]]
EVAL_FIELDS = ["timesteps", "wall_time", "return_mean", "return_std", "laps_mean", "lap_fraction_mean",
               "crash_rate", "avg_speed"]


class RacingCallback(BaseCallback):
    """Logs every finished episode and runs periodic evaluations (saving the best model)."""

    def __init__(self, cfg: ExperimentConfig, run_dir: Path, seed: int, verbose: int = 1):
        super().__init__(verbose)
        self.cfg, self.run_dir, self.seed = cfg, run_dir, seed
        self.eval_tracks = list(cfg.train.eval_tracks) or [cfg.env.track]
        self.best = float("-inf")
        self.next_eval = cfg.train.eval_freq
        self.t0 = time.time()
        self._ep_file = open(run_dir / "episodes.csv", "w", newline="")
        self._ep = csv.DictWriter(self._ep_file, EPISODE_FIELDS)
        self._ep.writeheader()
        self._ev_file = open(run_dir / "evals.csv", "w", newline="")
        self._ev = csv.DictWriter(self._ev_file, EVAL_FIELDS)
        self._ev.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            st = info.get("episode_stats")
            if st is None:
                continue
            ret = sum(st["reward_terms"].values())
            row = {"timesteps": self.num_timesteps, "return": round(ret, 3), "steps": st["steps"],
                   "laps": st["laps"], "lap_fraction": round(st["lap_fraction"], 4), "crashed": int(st["crashed"]),
                   "avg_speed": round(st["avg_speed"], 3), "track": st["track"]}
            row.update({f"r_{k}": round(v, 3) for k, v in st["reward_terms"].items()})
            self._ep.writerow(row)
            self.logger.record_mean("race/laps", st["laps"])
            self.logger.record_mean("race/crash_rate", float(st["crashed"]))
            self.logger.record_mean("race/avg_speed", st["avg_speed"])
            for k, v in st["reward_terms"].items():
                self.logger.record_mean(f"reward_terms/{k}", v)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.cfg.train.eval_freq
            self.evaluate()
        return True

    def evaluate(self) -> dict:
        env_cfg = replace(self.cfg.env, random_start=True)
        res = summarize(run_episodes(model_policy(self.model), env_cfg, self.eval_tracks,
                                     episodes=self.cfg.train.eval_episodes, seed=10_000))
        row = {"timesteps": self.num_timesteps, "wall_time": round(time.time() - self.t0, 1)}
        row.update({k: round(res[k], 4) for k in EVAL_FIELDS if k in res})
        self._ev.writerow(row)
        self._ev_file.flush()
        self._ep_file.flush()
        for k in ("return_mean", "laps_mean", "crash_rate", "avg_speed"):
            self.logger.record(f"eval/{k}", res[k])
        # Snapshots let you replay (and export to the browser) the agent at every stage of learning.
        self.model.save(self.run_dir / "checkpoints" / f"step_{self.num_timesteps:09d}.zip")
        if res["return_mean"] > self.best:
            self.best = res["return_mean"]
            self.model.save(self.run_dir / "best_model.zip")
        if self.verbose:
            print(f"[{self.cfg.name} seed{self.seed}] t={self.num_timesteps:>8} "
                  f"eval return {res['return_mean']:8.1f}  laps {res['laps_mean']:.2f}  "
                  f"crash {res['crash_rate']:.0%}  speed {res['avg_speed']:.2f}  ({row['wall_time']:.0f}s)")
        return res

    def _on_training_end(self) -> None:
        self.evaluate()
        self._ep_file.close()
        self._ev_file.close()


def train(cfg: ExperimentConfig, seed: int, out_dir: str | Path = "runs", verbose: int = 1) -> Path:
    spec = get_spec(cfg.algo)
    if cfg.env.action_type != spec.action_type:
        cfg = replace(cfg, env=replace(cfg.env, action_type=spec.action_type))
    run_dir = Path(out_dir) / cfg.name / f"seed{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    resolved = replace(cfg, train=replace(cfg.train, seeds=(seed,)))
    (run_dir / "config.toml").write_text(dump_toml(to_dict(resolved)))

    import gymnasium
    import stable_baselines3
    import torch

    meta = {"name": cfg.name, "algo": cfg.algo, "seed": seed, "python": platform.python_version(),
            "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__, "machine": platform.processor() or platform.machine(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    vec_env = make_vec_env(RaceEnv, n_envs=cfg.train.n_envs, seed=seed, env_kwargs={"config": cfg.env})
    model = make_model(cfg.algo, vec_env, seed=seed, algo_kwargs=cfg.algo_kwargs,
                       tensorboard_log=str(run_dir / "tb"), device=cfg.train.device)
    callback = RacingCallback(cfg, run_dir, seed, verbose=verbose)
    model.save(run_dir / "checkpoints" / f"step_{0:09d}.zip")  # the untrained policy, for comparison
    t0 = time.time()
    try:
        model.learn(total_timesteps=cfg.train.total_timesteps, callback=callback, tb_log_name="run")
    finally:
        model.save(run_dir / "final_model.zip")
        meta["train_seconds"] = round(time.time() - t0, 1)
        meta["timesteps"] = int(model.num_timesteps)
        meta["best_eval_return"] = callback.best
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
        vec_env.close()
    return run_dir
