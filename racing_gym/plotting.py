"""Learning-curve and track plots (matplotlib, which Stable-Baselines3 already depends on)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .track import PRESETS, make_track


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            continue
    return out


def plot_runs(exp_dirs: list[str], out: str | Path, metric: str = "return", title: str = "") -> Path:
    """Mean (line) and min-max over seeds (band) of the periodic *evaluation* metric per experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"return": "return_mean", "laps": "laps_mean", "crash_rate": "crash_rate", "avg_speed": "avg_speed"}[metric]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for exp in exp_dirs:
        seeds = sorted(Path(exp).glob("seed*/evals.csv"))
        curves = [d for d in (_read_csv(s) for s in seeds) if d]
        if not curves:
            print(f"warning: no evals.csv under {exp}")
            continue
        n = min(len(c["timesteps"]) for c in curves)
        x = curves[0]["timesteps"][:n]
        ys = np.stack([c[col][:n] for c in curves])
        label = f"{Path(exp).name} ({len(curves)} seed{'s' if len(curves) > 1 else ''})"
        line, = ax.plot(x, ys.mean(axis=0), label=label, lw=2)
        if len(curves) > 1:
            ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=line.get_color(), alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"evaluation {metric.replace('_', ' ')}")
    ax.set_title(title or f"Evaluation {metric} during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_tracks(out: str | Path, specs: list[str] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .render import track_image

    specs = specs or [*PRESETS, "random:0", "random:1", "random:2", "random:3"]
    cols = 4
    rows = (len(specs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, spec in zip(np.ravel(axes), specs):
        t = make_track(spec)
        ax.imshow(track_image(t))
        ax.plot(*t.centerline[0], "o", color="white", ms=5)
        ax.annotate("", xy=t.centerline[12], xytext=t.centerline[0],
                    arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5})
        ax.set_title(f"{spec}  ({t.length:.0f} px)", fontsize=10)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
