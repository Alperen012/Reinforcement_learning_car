# Extending Racing Gym

Run `pytest` after every change. `tests/test_web_parity.py` will catch any
difference between the Python and browser simulators.

## Add a track preset

1. In `racing_gym/track.py`, add a function returning
   `Track("name", control_points, width)` and register it in `PRESETS`.
   Control points are (x, y) in a 1000×1000 world. The track is a smooth
   closed spline through them, driven in point order. Keep the points ≥ ~100 px
   from the border.
2. Add the identical entry to `PRESETS` in `web/js/sim.js`.
3. Check it: `racing-gym tracks yourname --out t.png`, then `racing-gym play --track yourname`.
   `tests/test_track.py` asserts it is closed and does not overlap itself.

## Add a reward term

1. Add a field (default `0.0`, so existing experiments are unchanged) to
   `RewardConfig` in `racing_gym/config.py`, and to `REWARD_TERMS` in `env.py`.
2. Compute it in the `terms` dict in `RaceEnv.step`.
3. Mirror both in `web/js/sim.js` (`REWARD_DEFAULTS`, `REWARD_TERMS`, `terms`).
4. Add a test showing the term does what you claim, and cannot be gamed.

## Add an observation feature

Change `_get_obs` and `observation_names` in `env.py`, plus `observe()` and
`observationNames()` in `sim.js`. Old trained models expect the old size, so
retrain and re-export the models in `web/models/`.

## Add an algorithm

Any Stable-Baselines3-compatible class (including `sb3_contrib`, e.g. `TQC`,
`RecurrentPPO`) is one entry in `racing_gym/algos.py`:

```python
"tqc": AlgoSpec(TQC, "continuous", "SAC with distributional critics.", {"learning_starts": 5_000}),
```

To let it run in the browser, add its deterministic action path to
`policy_layers()` in `racing_gym/export.py`. `export` verifies the result
against `model.predict` before writing anything.

Not using SB3? The environment is plain Gymnasium
(`gymnasium.make("RacingGym-v0", config={...})`), so CleanRL-style scripts
work directly.

## Put an agent in the playground

```bash
racing-gym export runs/<name>/seed0 --out web/models/<name>.js --stages 6 --description "what is special about it"
```

`--stages` includes earlier checkpoints for the training-stage slider. The
model is added to `web/models/manifest.js` automatically.
