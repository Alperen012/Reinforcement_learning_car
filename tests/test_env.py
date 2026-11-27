import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import racing_gym  # noqa: F401  (registers RacingGym-v0)
from racing_gym import EnvConfig, RaceEnv, RewardConfig
from racing_gym.evaluation import heuristic_policy, run_episodes


@pytest.mark.parametrize("action_type", ["continuous", "discrete"])
def test_gymnasium_api_compliance(action_type):
    check_env(RaceEnv(EnvConfig(action_type=action_type)), skip_render_check=True)


def test_registered_with_gymnasium():
    env = gym.make("RacingGym-v0", config={"track": "oval"})
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_same_seed_same_episode():
    def rollout():
        env = RaceEnv(EnvConfig(track="random"))
        obs, _ = env.reset(seed=123)
        env.action_space.seed(5)
        out = [obs]
        for _ in range(50):
            obs, *_ = env.step(env.action_space.sample())
            out.append(obs)
        return env.track.name, np.array(out)

    (t1, a), (t2, b) = rollout(), rollout()
    assert t1 == t2
    np.testing.assert_array_equal(a, b)


def _straight_env(**kw):
    env = RaceEnv(EnvConfig(track="oval", random_start=False, **kw))
    env.reset(seed=0)
    return env


def test_forward_progress_is_rewarded_and_reversing_is_penalised():
    env = _straight_env()
    fwd = sum(env.step([0.0, 1.0])[4]["reward_terms"]["progress"] for _ in range(20))
    assert fwd > 0
    env = _straight_env()
    back = sum(env.step([0.0, -1.0])[4]["reward_terms"]["progress"] for _ in range(20))
    assert back < 0


def test_forward_backward_oscillation_cannot_farm_reward():
    """The old project clipped backward progress penalties, so rocking back and forth paid off."""
    env = _straight_env()
    total = 0.0
    for i in range(400):
        throttle = 1.0 if (i // 25) % 2 == 0 else -1.0
        total += env.step([0.0, throttle])[4]["reward_terms"]["progress"]
    assert total == pytest.approx(env.distance / 10.0, abs=1e-6)  # progress reward == net distance only


def test_reward_is_sum_of_terms_and_obs_within_bounds():
    env = RaceEnv(EnvConfig(track="hairpin", reward=RewardConfig(center=-0.3, steering=-0.1)))
    obs, _ = env.reset(seed=1)
    env.action_space.seed(1)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert r == pytest.approx(sum(info["reward_terms"].values()))
        if term or trunc:
            obs, _ = env.reset()


def test_leaving_the_track_ends_the_episode_with_crash_penalty():
    env = _straight_env(reward=RewardConfig(crash=-77.0))
    for _ in range(200):
        _, r, term, trunc, info = env.step([1.0, 1.0])  # full lock: drives into the wall
        if term:
            break
    assert term and not trunc and info["crashed"]
    assert info["reward_terms"]["crash"] == -77.0
    assert info["episode_stats"]["crashed"]


def test_heuristic_driver_completes_laps_and_gets_lap_bonus():
    env_cfg = EnvConfig(track="oval", random_start=False, max_steps=1500, reward=RewardConfig(lap=33.0))
    env = RaceEnv(env_cfg)
    res = run_episodes(heuristic_policy(env), env_cfg, episodes=1)[0]
    assert res["laps"] >= 3 and not res["crashed"]
    assert res["reward_terms"]["lap"] == pytest.approx(33.0 * res["laps"])
    assert res["lap_fraction"] == pytest.approx(res["distance"] / env.track.length)


def test_standing_still_is_truncated_by_stall_limit():
    env = _straight_env(stall_steps=30)
    for _ in range(100):
        _, _, term, trunc, _ = env.step([0.0, 0.0])
        if trunc:
            break
    assert trunc and not term and env.steps == 30


def test_target_laps_terminates():
    env_cfg = EnvConfig(track="oval", random_start=False, target_laps=1, max_steps=3000)
    env = RaceEnv(env_cfg)
    res = run_episodes(heuristic_policy(env), env_cfg, episodes=1)[0]
    assert res["laps"] == 1 and not res["crashed"]


def test_car_config_changes_dynamics():
    def top_speed(car):
        env = _straight_env(car=car)
        for _ in range(60):
            env.step([0.0, 1.0])
        return env.car.speed

    assert top_speed("rocket") > top_speed("balanced") > top_speed("grippy")


def test_grip_limits_cornering():
    def turn(grip):
        from racing_gym.config import CarConfig

        env = _straight_env(car=CarConfig(grip=grip))
        for _ in range(25):
            env.step([0.0, 1.0])
        y0 = env.car.yaw
        for _ in range(10):
            env.step([1.0, 0.0])
        return abs(math.remainder(env.car.yaw - y0, 2 * math.pi))

    assert turn(2.0) > turn(0.4)


def test_random_training_tracks_never_use_evaluation_seeds():
    env = RaceEnv(EnvConfig(track="random"))
    for s in range(20):
        env.reset(seed=s)
        assert int(env.track.name.split(":")[1]) >= 1000


def test_rgb_render_produces_image():
    env = RaceEnv(replace(EnvConfig(), track="oval"), render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.std() > 10
