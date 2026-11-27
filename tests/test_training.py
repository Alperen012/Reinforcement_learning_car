"""End-to-end: train tiny agents, reload them, evaluate them, export them. A few seconds per algorithm."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from racing_gym.cli import main
from racing_gym.config import load_config
from racing_gym.evaluation import make_policy, run_episodes, summarize
from racing_gym.export import export_policy, forward, policy_layers
from racing_gym.training import train

ROOT = Path(__file__).resolve().parents[1]


def _tiny(algo, **extra):
    over = [f"algo={algo}", "name=t_" + algo, "env.track=oval", "train.total_timesteps=1500", "train.n_envs=2",
            "train.eval_freq=600", "train.eval_episodes=1", "env.max_steps=300"]
    if algo in ("sac", "td3", "dqn"):
        over += ["algo_kwargs.learning_starts=200", "algo_kwargs.buffer_size=5000"]
    if algo == "ppo":
        over += ["algo_kwargs.n_steps=128", "algo_kwargs.batch_size=64"]
    return load_config(None, over + [f"{k}={v}" for k, v in extra.items()])


@pytest.mark.parametrize("algo", ["ppo", "a2c", "sac", "td3", "dqn"])
def test_train_save_load_evaluate_export(algo, tmp_path):
    run = train(_tiny(algo), seed=0, out_dir=tmp_path, verbose=0)
    for f in ("config.toml", "run.json", "episodes.csv", "evals.csv", "best_model.zip", "final_model.zip"):
        assert (run / f).exists(), f
    with open(run / "evals.csv") as fh:
        assert len(list(csv.DictReader(fh))) >= 2
    meta = json.loads((run / "run.json").read_text())
    assert meta["timesteps"] >= 1500 and meta["algo"] == algo

    # Evaluation works from the run directory alone, independent of the training process.
    policy, env_cfg, label = make_policy(str(run), load_config(run / "config.toml").env)
    res = summarize(run_episodes(policy, env_cfg, episodes=2))
    assert res["episodes"] == 2 and algo in label
    assert env_cfg.action_type == ("discrete" if algo == "dqn" else "continuous")

    out = export_policy(run, tmp_path / "web" / f"{algo}.js", stages=3)
    text = out.read_text()
    assert "RG_MODELS" in text and (out.parent / "manifest.js").exists()
    payload = json.loads(text[text.index("= {") + 2: text.rindex("; })")])
    assert len(payload["stages"]) >= 2 and payload["stages"][0]["label"] == "untrained"


def test_exported_network_matches_stable_baselines(tmp_path):
    from racing_gym.algos import load_model
    from racing_gym.env import RaceEnv

    run = train(_tiny("ppo"), seed=1, out_dir=tmp_path, verbose=0)
    cfg = load_config(run / "config.toml")
    model = load_model(str(run / "final_model.zip"), "ppo")
    layers, output = policy_layers(model, "ppo")
    env = RaceEnv(cfg.env)
    obs, _ = env.reset(seed=3)
    for _ in range(50):
        np.testing.assert_allclose(forward(layers, output, obs), model.predict(obs, deterministic=True)[0], atol=1e-5)
        obs, *_ = env.step(env.action_space.sample())


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_runs_exported_policy_identically(tmp_path):
    run = train(_tiny("sac"), seed=0, out_dir=tmp_path, verbose=0)
    out = export_policy(run, tmp_path / "m.js", name="m")
    cfg = load_config(run / "config.toml")
    from racing_gym.env import RaceEnv

    env = RaceEnv(cfg.env)
    obs, _ = env.reset(seed=0)
    batch = [obs.tolist()]
    for _ in range(20):
        obs, *_ = env.step(env.action_space.sample())
        batch.append(obs.tolist())
    js = ("require(process.argv[1]); const RG = require(process.argv[2]); const m = globalThis.RG_MODELS.m;"
          "const s = m.stages[m.stages.length-1];"
          "const net = {layers: s.layers, output: m.output};"
          "console.log(JSON.stringify(JSON.parse(process.argv[3]).map(o => RG.mlpForward(net, o))));")
    res = subprocess.run(["node", "-e", js, str(out), str(ROOT / "web/js/sim.js"), json.dumps(batch)],
                         capture_output=True, text=True, check=True)
    from racing_gym.algos import load_model

    model = load_model(str(run / "best_model.zip"), "sac")
    expected = [model.predict(np.array(o, dtype=np.float32), deterministic=True)[0] for o in batch]
    np.testing.assert_allclose(json.loads(res.stdout), expected, atol=1e-4)


def test_cli_evaluate_baselines(capsys, tmp_path):
    rc = main(["evaluate", "heuristic", "random", "untrained", "--track", "oval", "--episodes", "2",
               "--json", str(tmp_path / "r.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| heuristic |" in out and "| random |" in out and "untrained ppo" in out
    rows = {r["policy"]: r for r in json.loads((tmp_path / "r.json").read_text())}
    assert rows["heuristic"]["laps_mean"] > rows["random"]["laps_mean"]


def test_cli_reports_bad_config(capsys):
    assert main(["evaluate", "heuristic", "--track", "nowhere"]) == 2
    assert "unknown track" in capsys.readouterr().err
