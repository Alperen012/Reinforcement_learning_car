# Changelog

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
- Tutorials (`docs/tutorials/`), benchmark results with commands (`docs/results.md`), 43 tests and a CI workflow.

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
