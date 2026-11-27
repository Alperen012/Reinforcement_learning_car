"""Record policies driving side by side into an animated GIF (Pillow, no ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import EnvConfig
from .env import RaceEnv
from .evaluation import make_policy


def record_gif(policy_specs: list[str], env_cfg: EnvConfig, out: str | Path, steps: int = 600,
               algo: str = "ppo", every: int = 2, scale: float = 0.5, seed: int = 0,
               panel: bool | None = None) -> Path:
    """``panel``: show the reward/observation panel (default: only when recording a single policy)."""
    panel = len(policy_specs) == 1 if panel is None else panel
    racers = []
    for spec in policy_specs:
        policy, cfg, label = make_policy(spec, env_cfg, seed=seed, algo=algo)
        env = RaceEnv(cfg, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        env.render()
        env._renderer.overlay = [label]
        env._renderer.show_panel = panel
        racers.append({"env": env, "policy": policy, "obs": obs, "done": False, "frame": None})

    frames = []
    for t in range(steps):
        for r in racers:
            if not r["done"]:
                r["obs"], _, term, trunc, info = r["env"].step(r["policy"](r["obs"]))
                if term or trunc:
                    r["done"] = True
                    st = info["episode_stats"]
                    r["env"]._renderer.overlay.append("CRASHED" if st["crashed"] else "episode over")
                if t % every == 0 or r["done"]:
                    r["frame"] = r["env"].render()
        if t % every == 0:
            frames.append(np.concatenate([r["frame"] for r in racers], axis=1))
        if all(r["done"] for r in racers):
            frames += [frames[-1]] * 15
            break
    for r in racers:
        r["env"].close()

    imgs = [Image.fromarray(f) for f in frames]
    if scale != 1.0:
        w, h = imgs[0].size
        imgs = [im.resize((int(w * scale), int(h * scale)), Image.LANCZOS) for im in imgs]
    imgs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in imgs]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / 30 * every), loop=0,
                 optimize=True)
    return out
