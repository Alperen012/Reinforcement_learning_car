"""Export a trained policy as plain weights so the browser playground can run it.

The policy networks used here are tiny MLPs, so the browser needs no ML
library: ``web/js/policy.js`` does the matrix multiplications itself. Export
checks that a NumPy re-implementation of the forward pass reproduces
``model.predict`` before writing anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

from .config import to_dict
from .evaluation import find_model

_ACTS = {nn.Tanh: "tanh", nn.ReLU: "relu"}


def _flatten(modules) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for m in modules:
        if isinstance(m, nn.Sequential):
            layers += _flatten(list(m))
        elif isinstance(m, nn.Linear):
            layers.append({"w": m.weight.detach().cpu().numpy().astype(float),
                           "b": m.bias.detach().cpu().numpy().astype(float), "act": "linear"})
        elif type(m) in _ACTS:
            if not layers:
                raise ValueError("activation before the first linear layer")
            layers[-1]["act"] = _ACTS[type(m)]
        elif isinstance(m, (nn.Flatten, nn.Identity)):
            continue
        else:
            raise ValueError(f"cannot export layer type {type(m).__name__}")
    return layers


def policy_layers(model, algo: str) -> tuple[list[dict[str, Any]], str]:
    """(layers, output transform) of the deterministic action path for each supported algorithm."""
    p = model.policy
    if algo in ("ppo", "a2c"):
        return _flatten([p.mlp_extractor.policy_net, p.action_net]), "clip"
    if algo == "sac":
        return _flatten([p.actor.latent_pi, p.actor.mu]), "tanh"
    if algo == "td3":
        return _flatten([p.actor.mu]), "identity"
    if algo == "dqn":
        return _flatten([p.q_net.q_net]), "argmax"
    raise ValueError(f"export not supported for {algo!r}")


def forward(layers, output: str, obs: np.ndarray):
    x = np.asarray(obs, dtype=np.float64)
    for layer in layers:
        x = layer["w"] @ x + layer["b"]
        if layer["act"] == "tanh":
            x = np.tanh(x)
        elif layer["act"] == "relu":
            x = np.maximum(x, 0.0)
    if output == "clip":
        return np.clip(x, -1, 1)
    if output == "tanh":
        return np.tanh(x)
    if output == "argmax":
        return int(np.argmax(x))
    return x


def _check(layers, output: str, model, obs) -> None:
    ours = forward(layers, output, obs)
    theirs = model.predict(obs, deterministic=True)[0]
    if not np.allclose(np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float), atol=1e-4):
        raise RuntimeError(f"exported forward pass disagrees with SB3: {ours} vs {theirs}")


def _verified_layers(model_path: Path, cfg) -> tuple[list[dict[str, Any]], str, int]:
    """Extract layers and check them against SB3 before trusting the export.

    Checked on 200 observations from real driving and 200 drawn uniformly from the observation
    space (states the agent rarely visits). Also returns the timestep the model was saved at.
    """
    from .algos import load_model
    from .env import RaceEnv

    model = load_model(str(model_path), cfg.algo)
    layers, output = policy_layers(model, cfg.algo)
    env = RaceEnv(cfg.env)
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        _check(layers, output, model, obs)
        obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
        if term or trunc:
            obs, _ = env.reset()
    env.observation_space.seed(0)
    for _ in range(200):
        _check(layers, output, model, env.observation_space.sample())
    rounded = [{"w": np.round(layer["w"], 6).tolist(), "b": np.round(layer["b"], 6).tolist(), "act": layer["act"]}
               for layer in layers]
    return rounded, output, int(model.num_timesteps)


def stage_label(path: Path, timesteps: int) -> str:
    """What a stage is, in words, from the file it came from and the step it was saved at."""
    if timesteps == 0:
        return "untrained"
    if path.name == "best_model.zip":
        return f"best checkpoint ({timesteps:,} steps)"
    if path.name == "final_model.zip":
        return f"end of training ({timesteps:,} steps)"
    return f"after {timesteps:,} steps"


def _eval_returns(run_dir: Path) -> dict[int, float]:
    import csv

    f = run_dir / "evals.csv"
    if not f.exists():
        return {}
    with open(f, newline="") as fh:
        return {int(r["timesteps"]): float(r["return_mean"]) for r in csv.DictReader(fh)}


def export_policy(run_or_model: str | Path, out: str | Path, name: str | None = None,
                  description: str = "", stages: int = 1) -> Path:
    """Write a policy as a script the playground can load.

    With ``stages > 1``, also includes that many earlier training checkpoints
    (always the untrained one), so the playground can replay the agent learning.
    """
    model_path, cfg = find_model(run_or_model)
    run_dir = model_path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent
    returns = _eval_returns(run_dir)
    final_layers, output, final_t = _verified_layers(model_path, cfg)
    picks: list[Path] = []
    if stages > 1:
        # Only checkpoints from before the exported model, so the stage slider always moves forward in time.
        ckpts = [c for c in sorted((run_dir / "checkpoints").glob("step_*.zip")) if int(c.stem.split("_")[1]) < final_t]
        if not ckpts:
            raise FileNotFoundError(f"no training checkpoints before step {final_t} in {run_dir / 'checkpoints'}")
        # Learning is fastest early on, so space the stages geometrically (in checkpoint index).
        idx = np.unique(np.round(np.geomspace(1, len(ckpts), stages - 1)).astype(int) - 1)
        picks = [ckpts[i] for i in idx]

    stage_list = []
    for path in picks:
        layers, _, t = _verified_layers(path, cfg)
        stage_list.append({"label": stage_label(path, t), "timesteps": t, "eval_return": returns.get(t),
                           "layers": layers})
    stage_list.append({"label": stage_label(model_path, final_t), "timesteps": final_t,
                       "eval_return": returns.get(final_t), "layers": final_layers})

    name = name or cfg.name
    payload = {"name": name, "algo": cfg.algo, "description": description, "env": to_dict(cfg.env),
               "output": output, "stages": stage_list}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"))
    out.write_text(
        "// Generated by `racing-gym export`. Do not edit by hand.\n"
        f"(function (g) {{ g.RG_MODELS = g.RG_MODELS || {{}}; g.RG_MODELS[{json.dumps(name)}] = {body}; }})"
        "(typeof window !== 'undefined' ? window : globalThis);\n"
    )
    _write_manifest(out.parent)
    return out


def _write_manifest(models_dir: Path) -> None:
    """``manifest.js`` lists the model scripts so the page can load them without a server (file:// works)."""
    files = sorted(p.name for p in models_dir.glob("*.js") if p.name != "manifest.js")
    (models_dir / "manifest.js").write_text(
        "// Generated by `racing-gym export`. Lists the model files in this folder.\n"
        f"window.RG_MODEL_FILES = {json.dumps(files)};\n"
    )
