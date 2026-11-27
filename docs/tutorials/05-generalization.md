# 5 · Generalisation: does it understand driving, or did it memorise a track?

An agent trained on one track has seen one sequence of corners thousands of
times. Has it learned to *drive*, or just *this track*?

## Experiment

1. Train on a single track, then test on tracks it has never seen:

   ```bash
   racing-gym evaluate runs/ppo_classic/seed0 heuristic --tracks random:0 random:1 random:2 random:3 hairpin snake --episodes 3
   ```

2. Train on a **new random track every episode** (`track = "random"`). Training
   tracks use seeds ≥ 1000, so `random:0`…`random:999` are guaranteed unseen:

   ```bash
   racing-gym train -c configs/ppo_generalize.toml
   racing-gym evaluate runs/ppo_generalize/seed0 runs/ppo_classic/seed0 heuristic \
       --tracks random:0 random:1 random:2 random:3 hairpin snake --episodes 3
   ```

Our numbers are in [results.md](../results.md#generalisation).

## Why it can work at all

The observation is **egocentric**: lidar distances, speed, heading relative
to the track, curvature ahead. There are no map coordinates. A policy built on
those inputs *could* drive anywhere. Whether it *does* depends on how varied
its training experience was.

## Distribution shift, live

In the playground, open **Car physics** and lower the tyre grip, or raise the
top speed. The agent's network is unchanged, but the world now responds
differently from everything it trained on. Watch it understeer off the track.
This is *distribution shift*, the main reason RL agents fail when moved from
simulation to reality. The usual cure is *domain randomisation*: vary physics
during training, just as `track = "random"` varies the track. That is a good
first contribution (see the roadmap in the README).
