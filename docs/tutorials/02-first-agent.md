# 2 · Train your first agent (about 2 minutes on a laptop CPU)

## Train

```bash
racing-gym train -c configs/quickstart.toml
```

You will see a line per evaluation. Each line comes from 3 deterministic
test episodes, separate from training:

```
[quickstart seed0] t=   10000 eval return    ...  laps 0.00  crash 100%  speed ...
...
[quickstart seed0] t=  100000 eval return    ...  laps 7.00  crash 0%  speed 7.94
```

Early on every test episode crashes. A few tens of thousands of steps later
the car laps the track at nearly top speed. Your exact numbers will differ a
little, depending on your machine and library versions.

Everything about the run is in `runs/quickstart/seed0/`:

| file | what |
|---|---|
| `config.toml` | the exact config used: re-run it with `racing-gym train -c runs/quickstart/seed0/config.toml` |
| `run.json` | seed, library versions, training time |
| `episodes.csv` | every training episode: return, laps, crashed, speed, each reward term |
| `evals.csv` | the periodic test results printed above |
| `best_model.zip` | the best agent so far |
| `checkpoints/` | snapshots at every evaluation, for replay and export |

## Watch it

```bash
racing-gym watch runs/quickstart/seed0 --track oval
racing-gym watch runs/quickstart/seed0/checkpoints/step_000000000.zip --track oval   # before training
```

## Measure it, against baselines

A trained agent means nothing without something to compare it to:

```bash
racing-gym evaluate runs/quickstart/seed0 heuristic untrained random --track oval --episodes 10
```

This prints a table. Every policy drives the *same* 10 seeded episodes (same
start positions):

* `random`: random actions. The floor.
* `untrained`: the same network shape as the agent, before any learning.
* `heuristic`: a hand-written controller, no learning at all. The yardstick.
* your agent.

## Watch the learning curves

```bash
racing-gym plot runs/quickstart --out quickstart.png
tensorboard --logdir runs/quickstart        # race/*, reward_terms/*, eval/* and the algorithm's own losses
```

`reward_terms/*` in TensorBoard shows how each part of the reward evolves.
Early on `crash` dominates. Later, `progress` grows and `crash` vanishes.

## Make a GIF to share

```bash
racing-gym record runs/quickstart/seed0/checkpoints/step_000000000.zip runs/quickstart/seed0 --track oval --out before_after.gif
```

Next: [3 · Reward design (and how agents cheat)](03-reward-design.md)
