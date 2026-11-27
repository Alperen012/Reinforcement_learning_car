"""Regression tests for the correctness audit: evaluation hygiene, lap accounting, track validity, physics."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from racing_gym import EnvConfig, RaceEnv
from racing_gym.config import CAR_PRESETS, CarConfig, load_config
from racing_gym.env import TRAIN_SEED_OFFSET
from racing_gym.evaluation import (
    MAX_VALIDATION_EPISODES,
    TEST_SEED,
    heuristic_policy,
    run_episodes,
    validation_starts,
)
from racing_gym.track import PRESETS, make_track

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- validation vs test
def test_validation_starts_are_spread_and_nested():
    n = 700
    ten = validation_starts(n, 10)
    assert len(set(ten)) == 10 and validation_starts(n, 5) == ten[:5]  # more episodes only adds points
    s = sorted(ten)
    gaps = np.diff(s + [s[0] + n])  # around the loop
    assert gaps.max() <= n / 4  # no large stretch of track left untested
    with pytest.raises(ValueError):
        validation_starts(n, MAX_VALIDATION_EPISODES + 1)


@pytest.mark.parametrize("track", ["classic", "random:3"])
def test_test_episodes_never_start_on_a_validation_point(track):
    cfg = EnvConfig(track=track)
    env = RaceEnv(cfg)
    reserved = set(validation_starts(len(env.track.centerline), MAX_VALIDATION_EPISODES))
    raw = set()
    for i in range(200):
        env.reset(seed=TEST_SEED + i)
        raw.add(env.start_index)
    assert reserved & raw, "without the redraw some test episodes would repeat validation episodes"
    seen = set()
    # reversing into the wall keeps episodes short
    res = run_episodes(lambda o: [0.0, -1.0], cfg, episodes=200, on_step=lambda e: seen.add(e.start_index))
    assert not reserved & seen and all(r["seed"] >= TEST_SEED for r in res)


def test_validation_mode_uses_validation_starts():
    cfg = EnvConfig(track="oval")
    seen = []
    run_episodes(lambda o: [0.0, -1.0], cfg, episodes=4, validation=True, on_step=lambda e: seen.append(e.start_index))
    n = len(RaceEnv(cfg).track.centerline)
    assert set(seen) == set(validation_starts(n, 4))


def test_recorded_seed_reproduces_the_episode_after_a_redraw():
    cfg = EnvConfig(track="classic")
    starts = []
    res = run_episodes(lambda o: [0.0, -1.0], cfg, episodes=200, on_step=lambda e: starts.append(e.start_index))
    assert any(r["seed"] > TEST_SEED + 1_000_000 for r in res), "expected at least one redraw"
    probe = RaceEnv(cfg)
    for r in res:
        probe.reset(seed=r["seed"])
        assert probe.start_index in starts


def test_random_track_mode_can_be_validated_and_tested(tmp_path):
    """track="random" has no track until reset(): evaluation and training must still work."""
    from racing_gym.evaluation import VALIDATION_SEED, validate
    from racing_gym.training import train

    cfg = EnvConfig(track="random")
    pilot = heuristic_policy(RaceEnv(EnvConfig(track="oval")))
    test = run_episodes(pilot, cfg, episodes=3)
    val = run_episodes(pilot, cfg, episodes=3, seed=VALIDATION_SEED, validation=True)
    assert {r["track"] for r in test}.isdisjoint({r["track"] for r in val})  # different procedural tracks
    exp = load_config(None, ["algo=ppo", "name=r", "env.track=random", "train.total_timesteps=600", "train.n_envs=2",
                             "train.eval_freq=300", "train.eval_episodes=2", "env.max_steps=200",
                             "algo_kwargs.n_steps=128", "algo_kwargs.batch_size=64"])
    assert validate(pilot, exp)["episodes"] == 2
    run = train(exp, seed=0, out_dir=tmp_path, verbose=0)
    assert (run / "best_model.zip").exists()


def test_procedural_training_tracks_are_disjoint_from_evaluation_tracks():
    gen = load_config(ROOT / "configs" / "ppo_generalize.toml")
    val = [int(t.split(":")[1]) for t in gen.train.eval_tracks if t.startswith("random:")]
    test = [0, 1, 2, 3, 4]  # the generalist challenge / docs/results.md
    assert max(val + test) < TRAIN_SEED_OFFSET and not set(val) & set(test)


# ----------------------------------------------------------------------------- reward accounting
def _teleport_run(env, indices):
    """Place the car on successive centreline points (no speed) and collect the lap reward."""
    lap_total = 0.0
    for k in indices:
        x, y = env.track.centerline[k % len(env.track.centerline)]
        env.car.x, env.car.y, env.car.speed = float(x), float(y), 0.0
        env.car.yaw = env.track.heading_at(k % len(env.track.centerline))
        _, _, term, trunc, info = env.step([0.0, 0.0])
        assert not term, "teleporting along the centre line must not crash"
        lap_total += info["reward_terms"]["lap"]
    return lap_total


def test_lap_bonus_is_paid_once_when_recrossing_the_line():
    env = RaceEnv(EnvConfig(track="oval", random_start=False, stall_steps=10_000))
    env.reset(seed=0)
    n = len(env.track.centerline)
    fwd = list(range(8, n + 40, 8))  # one lap and a bit
    back = list(range(n + 40, n - 60, -8))  # reverse back over the line
    again = list(range(n - 60, n + 80, 8))  # and forward over it again
    paid = _teleport_run(env, fwd) + _teleport_run(env, back) + _teleport_run(env, again)
    assert env.laps == 1
    assert paid == pytest.approx(env.reward_cfg.lap)


# ----------------------------------------------------------------------------- tracks
def test_overlapping_or_out_of_world_tracks_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        make_track("hairpin", 160)  # its two straights merge into one wide slab: a shortcut
    with pytest.raises(ValueError):
        make_track("oval", 700)
    for bad in ("random:-1", "random:1.5", f"random:{2**31}"):
        with pytest.raises(ValueError):
            make_track(bad)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_walls_are_thicker_than_any_car_moves_in_one_step(name):
    fastest = max(CarConfig(**v).max_speed for v in CAR_PRESETS.values())
    assert make_track(name).wall_gap > fastest


def test_random_track_walls_are_thicker_than_the_playground_speed_limit():
    app = (ROOT / "web" / "js" / "app.js").read_text(encoding="utf-8")
    slider_max = float(app.split('key: "max_speed"')[1].split("max:")[1].split(",")[0])
    # random_track accepts only clearance > 1.15 * width, so walls are > 0.15 * 80 = 12 px
    assert slider_max <= 0.15 * 80
    for s in range(40):
        assert make_track(f"random:{s}").wall_gap > slider_max


def test_car_too_fast_for_the_walls_is_refused():
    gap = make_track("random:0").wall_gap
    env = RaceEnv(EnvConfig(track="random:0", car=CarConfig(max_speed=math.ceil(gap) + 1)))
    with pytest.raises(ValueError, match="narrowest wall"):
        env.reset(seed=0)


# ----------------------------------------------------------------------------- physics
@pytest.mark.parametrize("car", sorted(CAR_PRESETS))
@pytest.mark.parametrize("track", ["classic", "random:5"])
def test_physics_stays_finite_and_within_limits(car, track):
    env = RaceEnv(EnvConfig(track=track, car=car))
    p = env.car_params
    rng = np.random.default_rng(0)
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        for _ in range(400):
            obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
            assert np.isfinite(obs).all() and math.isfinite(r)
            assert env.observation_space.contains(obs)
            assert -p.max_reverse_speed - 1e-9 <= env.car.speed <= p.max_speed + 1e-9
            assert abs(env.car.steer) <= math.radians(p.max_steer_deg) + 1e-9
            if term or trunc:
                break


def test_heuristic_drives_every_preset_track():
    for name in PRESETS:
        cfg = EnvConfig(track=name)
        res = run_episodes(heuristic_policy(RaceEnv(cfg)), cfg, episodes=2)
        assert all(r["laps"] >= 1 for r in res), name


# ----------------------------------------------------------------------------- re-selection script
def test_reselect_script_reproduces_training_time_selection(tmp_path):
    from racing_gym.training import train

    over = ["algo=ppo", "name=t", "env.track=oval", "train.total_timesteps=1200", "train.n_envs=2",
            "train.eval_freq=400", "train.eval_episodes=2", "env.max_steps=300",
            "algo_kwargs.n_steps=128", "algo_kwargs.batch_size=64"]
    run = train(load_config(None, over), seed=0, out_dir=tmp_path, verbose=0)
    before = (run / "evals.csv").read_text().splitlines()
    ts = [line.split(",")[0] for line in before[1:]]
    assert len(ts) == len(set(ts)), "the final evaluation must not be logged twice"
    meta = json.loads((run / "run.json").read_text())

    spec = importlib.util.spec_from_file_location("reselect_best", ROOT / "scripts" / "reselect_best.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.reselect((str(run), []))
    after = (run / "evals.csv").read_text().splitlines()
    assert after == before
    assert json.loads((run / "run.json").read_text())["best_timesteps"] == meta["best_timesteps"]
