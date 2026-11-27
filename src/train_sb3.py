import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.race_track_env import RaceTrackEnv


class RenderDuringTrainingCallback(BaseCallback):
    """Render periodic training previews (swarm or policy style)."""

    def __init__(
        self,
        dynamics_mode: str,
        render_freq: int = 50_000,
        render_steps: int = 300,
        preview_style: str = "swarm",
        swarm_size: int = 140,
        swarm_command_steps: int = 12,
        swarm_seed: int = 42,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.dynamics_mode = dynamics_mode
        self.render_freq = max(1, int(render_freq))
        self.render_steps = max(1, int(render_steps))
        self.preview_style = str(preview_style).lower()

        self.swarm_size = max(10, int(swarm_size))
        self.swarm_command_steps = max(1, int(swarm_command_steps))
        self._rng = np.random.default_rng(int(swarm_seed))

        self._last_render_timestep = 0
        self._render_enabled = True
        self._render_env = None
        self._preview_scores = deque(maxlen=20)
        self._best_preview_score = float("-inf")

        self._swarm_command_bank = [
            (-1.0, 1.0),
            (-0.7, 1.0),
            (-0.4, 1.0),
            (0.0, 1.0),
            (0.4, 1.0),
            (0.7, 1.0),
            (1.0, 1.0),
            (-1.0, 0.3),
            (1.0, 0.3),
            (-0.5, -0.8),
            (0.5, -0.8),
            (0.0, -1.0),
            (0.0, 0.4),
        ]
        self._swarm_colors = [
            (255, 120, 120),
            (255, 180, 120),
            (255, 220, 120),
            (180, 255, 140),
            (120, 220, 255),
            (180, 160, 255),
        ]

    def _ensure_render_env(self):
        if self._render_env is None:
            self._render_env = RaceTrackEnv(render_mode="human", dynamics_mode=self.dynamics_mode)

    @staticmethod
    def _sparkline(values):
        if not values:
            return "-"

        blocks = "▁▂▃▄▅▆▇█"
        v_min = min(values)
        v_max = max(values)
        if abs(v_max - v_min) < 1e-6:
            return blocks[0] * len(values)

        chars = []
        for v in values:
            norm = (v - v_min) / (v_max - v_min)
            idx = int(round(norm * (len(blocks) - 1)))
            idx = max(0, min(idx, len(blocks) - 1))
            chars.append(blocks[idx])
        return "".join(chars)

    def _handle_quit_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._render_enabled = False
                if self._render_env is not None:
                    self._render_env.set_overlay_lines(None)
                    self._render_env.set_swarm_agents(None)
                    self._render_env.close()
                    self._render_env = None
                if self.verbose > 0:
                    print("[train] Render preview window closed. Training continues without preview.")
                return False
        return True

    def _sample_swarm_command(self):
        idx = int(self._rng.integers(0, len(self._swarm_command_bank)))
        return self._swarm_command_bank[idx]

    def _init_swarm_agents(self):
        env = self._render_env
        start_x, start_y, start_angle = env.start_x, env.start_y, env.start_angle

        forward = np.array([math.cos(start_angle), math.sin(start_angle)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)

        columns = max(10, min(24, int(math.sqrt(self.swarm_size) * 2)))
        spacing_side = 3.2
        spacing_back = 6.0

        agents = []
        for i in range(self.swarm_size):
            row = i // columns
            col = i % columns
            side_offset = (col - (columns - 1) / 2.0) * spacing_side
            back_offset = row * spacing_back

            jitter_x = float(self._rng.uniform(-0.8, 0.8))
            jitter_y = float(self._rng.uniform(-0.8, 0.8))
            angle_jitter = float(self._rng.uniform(-0.2, 0.2))

            x = start_x + right[0] * side_offset - forward[0] * back_offset + jitter_x
            y = start_y + right[1] * side_offset - forward[1] * back_offset + jitter_y
            color = self._swarm_colors[i % len(self._swarm_colors)]

            cmd = self._sample_swarm_command()
            cmd_steps_left = int(self._rng.integers(1, self.swarm_command_steps + 1))

            agent = {
                "x": float(x),
                "y": float(y),
                "angle": float(env._normalize_angle(start_angle + angle_jitter)),
                "speed": 0.0,
                "steering": 0.0,
                "command": cmd,
                "command_steps_left": cmd_steps_left,
                "alive": True,
                "color": color,
                "radius": 3,
            }

            if env._is_wall(agent["x"], agent["y"]):
                agent["alive"] = False

            agents.append(agent)

        return agents

    def _step_swarm_agent(self, agent):
        env = self._render_env
        if not agent["alive"]:
            return False

        if agent["command_steps_left"] <= 0:
            agent["command"] = self._sample_swarm_command()
            agent["command_steps_left"] = self.swarm_command_steps

        steering_cmd, acceleration_cmd = agent["command"]
        agent["command_steps_left"] -= 1

        target_steer = float(np.clip(steering_cmd, -1.0, 1.0) * env.max_steer)
        steer_delta = float(np.clip(target_steer - agent["steering"], -env.max_steer_rate, env.max_steer_rate))
        agent["steering"] += steer_delta

        if acceleration_cmd >= 0.0:
            accel_term = float(acceleration_cmd * env.engine_accel)
        else:
            accel_term = float(acceleration_cmd * env.brake_accel)

        drag = env.rolling_drag * np.sign(agent["speed"]) + env.aero_drag * agent["speed"] * abs(agent["speed"])
        if abs(agent["speed"]) < 1e-4:
            drag = 0.0

        agent["speed"] += accel_term - float(drag)
        agent["speed"] = float(np.clip(agent["speed"], -env.max_reverse_speed, env.max_speed))

        if abs(agent["speed"]) < 0.02 and abs(acceleration_cmd) < 0.1:
            agent["speed"] = 0.0

        if abs(agent["speed"]) > 1e-4:
            beta = math.atan((env.rear_wheel_base / env.wheel_base) * math.tan(agent["steering"]))
            yaw_rate = (agent["speed"] / env.wheel_base) * math.sin(beta)
            agent["angle"] = float(env._normalize_angle(agent["angle"] + yaw_rate))
            agent["x"] += float(agent["speed"] * math.cos(agent["angle"] + beta))
            agent["y"] += float(agent["speed"] * math.sin(agent["angle"] + beta))

        # Simple footprint collision checks
        cos_a = math.cos(agent["angle"])
        sin_a = math.sin(agent["angle"])
        nose_x = agent["x"] + cos_a * (env.car_length * 0.45)
        nose_y = agent["y"] + sin_a * (env.car_length * 0.45)
        tail_x = agent["x"] - cos_a * (env.car_length * 0.35)
        tail_y = agent["y"] - sin_a * (env.car_length * 0.35)

        if env._is_wall(agent["x"], agent["y"]) or env._is_wall(nose_x, nose_y) or env._is_wall(tail_x, tail_y):
            agent["alive"] = False
            return False

        return True

    def _run_swarm_preview(self):
        self._render_env.reset()
        agents = self._init_swarm_agents()
        total_crashed = 0
        alive_history = []

        for _ in range(self.render_steps):
            if not self._handle_quit_event():
                return None

            alive_count = 0
            for agent in agents:
                was_alive = agent["alive"]
                is_alive = self._step_swarm_agent(agent)
                if is_alive:
                    alive_count += 1
                elif was_alive:
                    total_crashed += 1

            alive_history.append(alive_count)

            alive_ratio = alive_count / max(1, self.swarm_size)
            self._best_preview_score = max(self._best_preview_score, alive_ratio)

            trend = self._sparkline(alive_history[-20:])
            overlay_lines = [
                f"TIMESTEP : {self.num_timesteps}",
                "STYLE    : SWARM / RANDOM COMMANDS",
                f"ALIVE    : {alive_count}/{self.swarm_size}",
                f"CRASHED  : {total_crashed}",
                f"BEST ALV : {self._best_preview_score * 100:5.1f}%",
                f"TREND    : {trend}",
            ]

            self._render_env.set_overlay_lines(overlay_lines)
            self._render_env.set_swarm_agents([a for a in agents if a["alive"]])
            self._render_env.render()

            if alive_count <= 0:
                break

        self._render_env.set_swarm_agents(None)

        if not alive_history:
            return {"alive_ratio": 0.0, "last_alive": 0, "crashed": total_crashed}

        mean_alive_ratio = float(np.mean(np.array(alive_history, dtype=np.float32) / max(1, self.swarm_size)))
        return {
            "alive_ratio": mean_alive_ratio,
            "last_alive": int(alive_history[-1]),
            "crashed": int(total_crashed),
        }

    def _run_policy_preview(self):
        obs, _ = self._render_env.reset()
        self._render_env.set_swarm_agents(None)

        preview_reward = 0.0
        preview_laps = 0

        for _ in range(self.render_steps):
            if not self._handle_quit_event():
                return None

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self._render_env.step(action)
            preview_reward += float(reward)
            preview_laps = max(preview_laps, int(info.get("laps", 0)))

            trend_values = list(self._preview_scores) + [preview_reward]
            self._best_preview_score = max(self._best_preview_score, preview_reward)
            overlay_lines = [
                f"TIMESTEP : {self.num_timesteps}",
                "STYLE    : POLICY PREVIEW",
                f"REWARD   : {preview_reward:+.2f}",
                f"BEST RW  : {self._best_preview_score:+.2f}",
                f"LAPS     : {preview_laps}",
                f"TREND    : {self._sparkline(trend_values)}",
            ]
            self._render_env.set_overlay_lines(overlay_lines)
            self._render_env.render()

            if terminated or truncated:
                obs, _ = self._render_env.reset()

        return {"reward": preview_reward, "laps": preview_laps}

    def _on_step(self) -> bool:
        if not self._render_enabled:
            return True

        if (self.num_timesteps - self._last_render_timestep) < self.render_freq:
            return True

        self._last_render_timestep = self.num_timesteps
        self._ensure_render_env()

        if self.preview_style == "policy":
            metrics = self._run_policy_preview()
            if metrics is None:
                return True

            score = float(metrics["reward"])
            self._preview_scores.append(score)
            self.logger.record("preview/reward", score)
            self.logger.record("preview/laps", int(metrics["laps"]))
            self.logger.record("preview/best_reward", self._best_preview_score)

            if self.verbose > 0:
                print(
                    f"[train] Policy preview t={self.num_timesteps}: "
                    f"reward={score:.2f}, laps={int(metrics['laps'])}"
                )
        else:
            metrics = self._run_swarm_preview()
            if metrics is None:
                return True

            score = float(metrics["alive_ratio"])
            self._preview_scores.append(score)
            self.logger.record("preview/swarm_alive_ratio", score)
            self.logger.record("preview/swarm_last_alive", int(metrics["last_alive"]))
            self.logger.record("preview/swarm_crashed", int(metrics["crashed"]))
            self.logger.record("preview/swarm_best_alive_ratio", self._best_preview_score)

            if self.verbose > 0:
                print(
                    f"[train] Swarm preview t={self.num_timesteps}: "
                    f"alive_ratio={score:.3f}, alive={int(metrics['last_alive'])}/{self.swarm_size}, "
                    f"crashed={int(metrics['crashed'])}"
                )

        return True

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.set_overlay_lines(None)
            self._render_env.set_swarm_agents(None)
            self._render_env.close()
            self._render_env = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for RaceTrackEnv.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps.")
    parser.add_argument("--dynamics-mode", choices=["normal", "physics"], default="normal", help="Environment dynamics mode.")
    parser.add_argument("--save-name", type=str, default="ppo_race_track", help="Base name for final model file.")
    parser.add_argument("--log-name", type=str, default="ppo_race_track", help="TensorBoard log subdirectory name.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="Save checkpoint every N timesteps.")

    # Optional live preview while learning
    parser.add_argument("--render-training", action="store_true", help="Enable periodic visual preview during training.")
    parser.add_argument("--render-freq", type=int, default=50_000, help="Preview every N timesteps.")
    parser.add_argument("--render-steps", type=int, default=300, help="Number of steps shown during each preview.")
    parser.add_argument(
        "--preview-style",
        choices=["swarm", "policy"],
        default="policy",
        help="Training preview style. 'swarm' shows many random-command cars, 'policy' shows single-agent behavior.",
    )
    parser.add_argument("--swarm-size", type=int, default=140, help="Number of cars in swarm preview.")
    parser.add_argument("--swarm-command-steps", type=int, default=12, help="How many frames a swarm car keeps the same command.")
    parser.add_argument("--swarm-seed", type=int, default=42, help="Random seed for swarm preview behavior.")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    logs_dir = project_root / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = RaceTrackEnv(render_mode=None, dynamics_mode=args.dynamics_mode)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    # Obs are already scaled to ~[0, 1] by the env; normalizing them here would require
    # shipping VecNormalize stats to evaluate.py and the preview callback. Only rewards are normalized.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    # Create Evaluation Env (To save the best model)
    eval_env = RaceTrackEnv(render_mode=None, dynamics_mode=args.dynamics_mode)
    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=False, norm_reward=False, training=False)

    # Hyperparameter-optimized PPO Model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(logs_dir / args.log_name),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01, # Entropy coefficient to increase exploration
        clip_range=0.2,
        seed=args.seed,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir / "eval_logs"),
        eval_freq=max(1, args.checkpoint_freq // 5),
        deterministic=True,
        render=False,
    )
    
    callbacks = [
        eval_callback,
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq),
            save_path=str(models_dir),
            name_prefix=f"{args.save_name}_{args.dynamics_mode}",
        )
    ]

    if args.render_training:
        callbacks.append(
            RenderDuringTrainingCallback(
                dynamics_mode=args.dynamics_mode,
                render_freq=args.render_freq,
                render_steps=args.render_steps,
                preview_style=args.preview_style,
                swarm_size=args.swarm_size,
                swarm_command_steps=args.swarm_command_steps,
                swarm_seed=args.swarm_seed,
                verbose=1,
            )
        )

    callback = CallbackList(callbacks)

    print(f"[train] Starting training: timesteps={args.timesteps}, mode={args.dynamics_mode}")
    if args.render_training:
        print(
            f"[train] Preview enabled: style={args.preview_style}, "
            f"freq={args.render_freq}, steps={args.render_steps}, swarm_size={args.swarm_size}"
        )
    model.learn(total_timesteps=args.timesteps, callback=callback)

    final_model_path = models_dir / f"{args.save_name}_{args.dynamics_mode}"
    model.save(str(final_model_path))
    env.close()

    print(f"[train] Training complete. Model saved to: {final_model_path}.zip")


if __name__ == "__main__":
    main()