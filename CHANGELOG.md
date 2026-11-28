# Changelog

## Unreleased: correctness audit, domain randomisation

### Added
- Domain randomisation (#4): `env.car_randomize = r` scales grip, max_speed and engine_accel by
  independent factors in [1 - r, 1 + r] at every training reset. Off by default (existing runs are
  unchanged); evaluation and checkpoint selection always use the nominal car. Example:
  `configs/ppo_generalize_dr.toml`.

### Fixed
- `racing-gym evaluate` (and the README/challenge commands) evaluated on reset seeds 10,000+, the same
  episodes training used to pick `best_model.zip`. Evaluation now uses a disjoint test set (D11).
- Exported playground stages reported the end-of-training step for the best checkpoint (e.g. PPO
  "fully trained, 303,104 steps" was really saved at 250,000). The step is now read from the model
  file, stages are strictly increasing, and labels say "best checkpoint (N steps)".
- Training logged the last evaluation twice when the budget ended exactly on an evaluation step.
- Preset tracks with a `track_width` override could overlap themselves (a shortcut the progress reward
  does not see) or leave the world; such tracks are now rejected. Negative or non-integer
  `random:<seed>` specs are rejected in Python and the browser (they produced different tracks).
- A car faster than the narrowest wall of a track could jump through it between collision checks;
  `RaceEnv` now refuses that combination, and the playground top-speed slider stops at 12 px/step.
- README: "every algorithm beats the hand-written controller" was false for DQN as published (4.87 laps,
  27% crashes). With the new validation DQN's best checkpoints lap 6.00 with 0% crashes, but its final
  weights crash in 73% of episodes; the README and results now show both.
- The generalist baseline's seeds 0-1 had been validated on its own test tracks (the config was fixed
  earlier, the runs were not). They are re-selected on disjoint tracks and a third seed is trained.
  The published generalisation numbers used final weights and were not affected by this.
- docs/results.md: best-checkpoint numbers are now shown next to final-weight numbers, which reveal
  that TD3 and DQN collapse late in training.

### Changed
- Checkpoint selection uses 10 evenly spread validation start points (was 5 random ones).
  `scripts/reselect_best.py` re-selects finished runs with the current protocol.
- `docs/results.md` adds the selected checkpoint's step, a validation-to-test gap table and notes on
  what the comparison can and cannot show.
- The playground ships all five algorithms (A2C, SAC and TD3 added), with honest stage labels.
- New regression tests: validation/test disjointness, lap bonus on re-crossing the line, track
  validity, physics limits for every car preset, export labels and timesteps, Python/JS parity for
  every car preset and every exported output transform.
- Rocket-on-hairpin baseline now uses 3 training seeds: 5.67 laps, 0% crashes (was 1 seed, 6.00).

## 0.2.0 (2026-11-27)

A rewrite into an educational RL racing playground. See [docs/audit.md](docs/audit.md)
for why each part changed and [docs/decisions.md](docs/decisions.md) for the design choices.

### Added
- MIT licence.
- `racing_gym` package with a `racing-gym` CLI: `train`, `evaluate`, `watch`, `play`, `record`, `plot`, `export`, `tracks`.
- Spline tracks: 4 presets (`oval`, `classic`, `hairpin`, `snake`) and seeded procedural tracks (`random:<seed>`).
- Egocentric observation (lidar, speed, wheel angle, heading error, lateral offset, curvature ahead).
- Decomposed reward (progress, time, crash, lap, centring, steering smoothness), each term logged separately.
- Five algorithms through one registry: PPO, A2C, SAC, TD3, DQN (with a 9-action discrete mode).
- TOML experiment configs with `-s key=value` overrides. Per-run directories with the resolved config, versions, CSV logs, TensorBoard and checkpoints.
- Baseline policies for every comparison: random, untrained, and a hand-written heuristic.
- Browser playground (`web/`): race exported agents, replay training stages, inspect observation, action and reward live, tweak physics.
- Pygame renderer with reward and observation panels. GIF recording. Learning-curve plots.
- Tutorials (`docs/tutorials/`), benchmark results with commands (`docs/results.md`), 44 tests and a CI workflow.

### Changed
- Progress reward is now signed distance along the centre line. The old clipped checkpoint reward could be farmed by rocking back and forth.
- Car dynamics: one kinematic bicycle model with a grip limit replaces the normal/pymunk modes.

### Removed
- `src/` scripts, the pymunk physics mode and the swarm preview.

### Fixed (on v0.1, before the rewrite)
- VecNormalize observation statistics were not saved, so evaluation saw unnormalised inputs.
- pymunk forces applied on only 1 of 5 substeps, and a px/s vs px/step unit mismatch.

## 0.1.0
- PPO on a single rectangular track (original version).
