import argparse
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO

from envs.race_track_env import RaceTrackEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on RaceTrackEnv.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file (with or without .zip). Default: models/ppo_race_track_<mode>",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--dynamics-mode", choices=["normal", "physics"], default="normal", help="Environment dynamics mode.")
    parser.add_argument("--render-mode", choices=["human", "ansi", "none"], default="human", help="Rendering mode.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic policy.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay per step for slower playback.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Hard step cap per episode for quick tests.")
    return parser.parse_args()


def resolve_model_path(args):
    project_root = Path(__file__).resolve().parents[1]

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
    else:
        model_path = project_root / "models" / f"ppo_race_track_{args.dynamics_mode}"

    if model_path.exists() or model_path.with_suffix(".zip").exists():
        return model_path

    raise FileNotFoundError(
        f"Model not found at '{model_path}' or '{model_path.with_suffix('.zip')}'. "
        "Run training first or pass --model-path."
    )


def main():
    args = parse_args()
    model_path = resolve_model_path(args)

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = RaceTrackEnv(render_mode=render_mode, dynamics_mode=args.dynamics_mode)
    model = PPO.load(str(model_path))

    deterministic = not args.stochastic
    rewards = []
    laps = []
    running = True

    try:
        for ep in range(1, args.episodes + 1):
            if not running:
                break

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            episode_done = False
            last_info = {}

            while running and not episode_done:
                if args.render_mode == "human":
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                if not running:
                    break

                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += float(reward)
                ep_steps += 1
                last_info = info

                if args.render_mode != "none":
                    env.render()

                if args.sleep > 0:
                    time.sleep(args.sleep)

                if ep_steps >= args.max_steps:
                    truncated = True

                episode_done = terminated or truncated

            if not running:
                break

            ep_laps = int(last_info.get("laps", 0))
            rewards.append(ep_reward)
            laps.append(ep_laps)
            print(f"Episode {ep}/{args.episodes} | reward={ep_reward:.2f} | steps={ep_steps} | laps={ep_laps}")

    finally:
        env.close()

    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        avg_laps = sum(laps) / len(laps)
        print(
            f"[eval] Summary | episodes={len(rewards)} | "
            f"avg_reward={avg_reward:.2f} | best_reward={max(rewards):.2f} | avg_laps={avg_laps:.2f}"
        )
    else:
        print("[eval] No completed episodes.")


if __name__ == "__main__":
    main()
