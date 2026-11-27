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


def _verified_layers(model_path: Path, cfg) -> tuple[list[dict[str, Any]], str]:
    """Extract layers and check them against SB3 on real observations before trusting the export."""
    from .algos import load_model
    from .env import RaceEnv

    model = load_model(str(model_path), cfg.algo)
    layers, output = policy_layers(model, cfg.algo)
    env = RaceEnv(cfg.env)
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        ours = forward(layers, output, obs)
        theirs = model.predict(obs, deterministic=True)[0]
        if not np.allclose(np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float), atol=1e-4):
            raise RuntimeError(f"exported forward pass disagrees with SB3: {ours} vs {theirs}")
        obs, _, term, trunc, _ = env.step(theirs)
        if term or trunc:
            obs, _ = env.reset()
    rounded = [{"w": np.round(layer["w"], 6).tolist(), "b": np.round(layer["b"], 6).tolist(), "act": layer["act"]}
               for layer in layers]
    return rounded, output


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
    returns = _eval_returns(run_dir)
    picks: list[tuple[int | None, Path]] = []
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.zip"))
    if stages > 1:
        if len(ckpts) < 2:
            raise FileNotFoundError(f"no training checkpoints in {run_dir / 'checkpoints'}; retrain to export stages")
        # Learning is fastest early on, so space the stages geometrically (in checkpoint index).
        idx = np.unique(np.round(np.geomspace(1, len(ckpts), stages)).astype(int) - 1)
        picks = [(int(ckpts[i].stem.split("_")[1]), ckpts[i]) for i in idx[:-1]]
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    picks.append((meta.get("timesteps"), model_path))

    stage_list = []
    for t, path in picks:
        layers, output = _verified_layers(path, cfg)
        if path == model_path:
            label, ret = "fully trained", meta.get("best_eval_return")
        else:
            label, ret = ("untrained" if t == 0 else f"after {t:,} steps"), returns.get(t)
        stage_list.append({"label": label, "timesteps": t, "eval_return": ret, "layers": layers})

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
