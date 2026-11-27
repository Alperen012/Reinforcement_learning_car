# Architecture

```
racing_gym/               Python package (pip install -e .)
  track.py                Track: spline centreline -> mask, arc length, tangents. Presets + procedural.
  car.py                  CarState + step_car(): kinematic bicycle with grip limit.
  env.py                  RaceEnv (gymnasium.Env): observation, reward terms, termination.
  config.py               Typed dataclasses <-> TOML, dotted CLI overrides.
  algos.py                Algorithm registry (SB3 class, action type, default hyper-parameters).
  training.py             train(): run dirs, CSV + TensorBoard logging, periodic eval, checkpoints.
  evaluation.py           Policies (random / heuristic / untrained / trained) and seeded evaluation.
  render.py               Pygame renderer (human window or rgb_array) with reward + observation panel.
  gif.py, plotting.py     GIF recording, learning curves, track previews.
  export.py               Trained policy -> web/models/*.js (verified against SB3).
  cli.py                  `racing-gym` command.
web/                      Static playground (open web/index.html)
  js/sim.js               JS mirror of track.py + car.py + env.py (kept identical by a test).
  js/app.js               UI: racers, training-stage slider, inspector, sliders.
  models/                 Exported agents + manifest.js.
configs/                  Experiment configs (TOML).
tests/                    pytest suite, incl. Python <-> JS parity.
docs/                     Tutorials, results, design notes.
```

## Data flow

```
config.toml ──► ExperimentConfig ──► train() ──► SB3 algorithm ◄──► VecEnv[RaceEnv × n_envs]
                                         │
                                         ├─► runs/<name>/seed<k>/{episodes,evals}.csv, tb/
                                         ├─► checkpoints/step_*.zip, best_model.zip
                                         ▼
          racing-gym evaluate / watch / record / plot          racing-gym export ──► web/models/*.js
                                                                                         │
                                                              web/index.html ◄───────────┘
```

## One environment step

1. **Action** → `[steer, throttle]` (or one of 9 discrete pairs).
2. **Car** (`step_car`): rate-limited steering, throttle/brake/drag, a yaw rate
   limited by grip, position update.
3. **Track projection**: nearest centreline point (local search around the last
   one) → arc length `s` and lateral offset.
4. **Progress**: `ds = wrap(s - s_prev)`; `distance += ds`; laps =
   `floor(distance / track_length)`.
5. **Crash**: any corner of the car's box outside the drivable mask.
6. **Reward**: weighted terms → `info["reward_terms"]`.
7. **Observation**: lidar ray-march on the mask (vectorised), plus car and
   track-relative features.

## Extension points

* New track preset: one function in `PRESETS` (`track.py`) *and* in `sim.js`.
  The parity test will tell you if they differ.
* New reward term: a field in `RewardConfig`, one line in `RaceEnv.step`, and
  the same in `sim.js`.
* New algorithm: one `AlgoSpec` in `algos.py` (+ `policy_layers` in
  `export.py` if it should run in the browser).

See [extending.md](extending.md) for step-by-step guides.
