"""Re-run checkpoint selection on finished runs with the current validation protocol.

    python scripts/reselect_best.py runs/ppo_classic runs/sac_classic ... [train.eval_episodes=10]

Training saves a checkpoint at every evaluation, and validation never touches the
training RNG (separate environment, deterministic actions), so re-validating the saved
checkpoints picks exactly the checkpoint a fresh run with the current code would pick.
Rewrites evals.csv (keeping wall_time), best_model.zip and the best_* fields of run.json.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reselect(job: tuple[str, list[str]]) -> str:
    import torch

    torch.set_num_threads(1)
    from racing_gym.algos import load_model
    from racing_gym.config import dump_toml, load_config, to_dict
    from racing_gym.evaluation import model_policy, validate
    from racing_gym.training import EVAL_FIELDS

    seed_dir, overrides = job
    d = Path(seed_dir)
    cfg = load_config(d / "config.toml", overrides)
    with open(d / "evals.csv", newline="") as fh:
        wall = {int(r["timesteps"]): r["wall_time"] for r in csv.DictReader(fh)}
    # step 0 (the untrained policy) was never part of selection
    ckpts = [c for c in sorted((d / "checkpoints").glob("step_*.zip")) if int(c.stem.split("_")[1]) in wall]
    if not ckpts:
        return f"{d}: SKIPPED, no checkpoints saved during training (retrain this run instead)"
    rows, best = [], float("-inf")
    for ckpt in ckpts:
        t = int(ckpt.stem.split("_")[1])
        res = validate(model_policy(load_model(str(ckpt), cfg.algo)), cfg)
        rows.append({"timesteps": t, "wall_time": wall[t], **{k: round(res[k], 4) for k in EVAL_FIELDS if k in res}})
        if res["return_mean"] > best:
            best, best_t, best_path = res["return_mean"], t, ckpt
    (d / "config.toml").write_text(dump_toml(to_dict(cfg)))  # the run now records the protocol it was selected with
    with open(d / "evals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, EVAL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    shutil.copyfile(best_path, d / "best_model.zip")
    meta = json.loads((d / "run.json").read_text())
    meta.update(best_eval_return=best, best_timesteps=best_t)
    (d / "run.json").write_text(json.dumps(meta, indent=2))
    return f"{d}: best checkpoint {best_t} (validation return {best:.1f})"


if __name__ == "__main__":
    # Arguments: run directories, plus optional config overrides such as train.eval_episodes=10
    overrides = [a for a in sys.argv[1:] if "=" in a]
    exps = [a for a in sys.argv[1:] if "=" not in a]
    jobs = [(str(s), overrides) for exp in exps for s in sorted(Path(exp).glob("seed*")) if (s / "run.json").exists()]
    with Pool() as pool:
        for line in pool.imap_unordered(reselect, jobs):
            print(line, flush=True)
