import importlib
import math

import gymnasium as gym
import numpy as np
import pygame


class RaceTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode=None, dynamics_mode="normal", physics_substeps=5):
        super().__init__()
        self.render_mode = render_mode

        self.dynamics_mode = str(dynamics_mode).strip().lower()
        if self.dynamics_mode not in {"normal", "physics"}:
            raise ValueError("dynamics_mode must be either 'normal' or 'physics'.")

        self.physics_substeps = max(1, int(physics_substeps))
        self.window_size = 1000
        self.window = None
        self.clock = None
        self.extra_overlay_lines = []
        self.swarm_agents = []

        # Optional physics backend (lazy-loaded only for physics mode)
        self._pymunk = None
        if self.dynamics_mode == "physics":
            try:
                self._pymunk = importlib.import_module("pymunk")
            except ModuleNotFoundError as exc:
                raise ImportError("Physics mode requires 'pymunk'. Install it with: pip install pymunk") from exc

        self.physics_space = None
        self.physics_body = None
        self.physics_shape = None

        # Action: [steering, acceleration] in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Speed limits
        self.max_speed = 8.0
        self.max_reverse_speed = self.max_speed * 0.5

        # Lidar config
        self.max_lidar_distance = 260.0
        self.ray_angles = np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], dtype=np.float32)
        self.lidar_distances = np.zeros(5, dtype=np.float32)

        # Car geometry for collision checks
        self.car_length = 30.0
        self.car_width = 16.0

        # Improved normal model (kinematic bicycle style)
        self.wheel_base = 24.0
        self.rear_wheel_base = self.wheel_base * 0.5
        self.max_steer = np.deg2rad(35.0)
        self.max_steer_rate = np.deg2rad(8.0)
        self.steering_angle = 0.0
        self.engine_accel = 0.55
        self.brake_accel = 0.75
        self.rolling_drag = 0.03
        self.aero_drag = 0.005

        # Physics mode parameters
        # pymunk works in px/s, the rest of the env in px/step. Forces are derived from the
        # normal-mode accel/drag params (px/step^2) so both modes share the same longitudinal tuning.
        self.physics_fps = float(self.metadata["render_fps"])
        self.physics_dt = 1.0 / self.physics_fps
        self.physics_mass = 1200.0
        self.physics_lateral_grip = 0.22
        self.physics_angular_damping = 0.88

        # Observation: [x, y, angle, speed, ray1..ray5]
        # (NORMALIZED to roughly [-1.0, 1.0] ranges for PPO)
        low = np.array(
            [0.0, 0.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = 3000
        self.current_step = 0

        # Simple Oval / Rectangular Track (For faster and stable PPO learning)
        self.checkpoints = [
            (200, 500),  # Start
            (200, 200),  # Top-Left corner start
            (500, 200),  # Top-Mid straight
            (800, 200),  # Top-Right corner
            (800, 500),  # Right-Mid straight
            (800, 800),  # Bottom-Right corner
            (500, 800),  # Bottom-Mid straight
            (200, 800),  # Bottom-Left corner
        ]
        self.checkpoint_radius = 80
        self.laps_completed = 0

        self._init_track()

    def _init_track(self):
        pygame.init()
        self.track_surface = pygame.Surface((self.window_size, self.window_size))
        self.track_surface.fill((0, 0, 0)) # Wall (Black)

        # Draw the track path (White - Simple oval with wide thickness)
        pygame.draw.lines(self.track_surface, (255, 255, 255), True, self.checkpoints, 120)
        # Draw circles at the corners to round the turns
        for p in self.checkpoints:
            pygame.draw.circle(self.track_surface, (255, 255, 255), p, 60)

        # Fast memory mapping to read track collisions
        self.track_data = pygame.surfarray.pixels2d(self.track_surface)

        # Starting position (slightly behind the first checkpoint)
        self.start_x = 200.0
        self.start_y = 600.0
        # Facing upwards (-pi/2)
        self.start_angle = -np.pi / 2.0

    def _init_physics_world(self):
        if self._pymunk is None:
            raise ImportError("Physics mode requires 'pymunk'. Install it with: pip install pymunk")

        pymunk = self._pymunk
        self.physics_space = pymunk.Space()
        self.physics_space.gravity = (0.0, 0.0)
        self.physics_space.damping = 0.93

        moment = pymunk.moment_for_box(self.physics_mass, (self.car_length, self.car_width))
        self.physics_body = pymunk.Body(self.physics_mass, moment)
        self.physics_body.position = (self.start_x, self.start_y)
        self.physics_body.angle = self.start_angle

        self.physics_shape = pymunk.Poly.create_box(self.physics_body, (self.car_length, self.car_width))
        self.physics_shape.friction = 1.0
        self.physics_shape.elasticity = 0.05
        self.physics_space.add(self.physics_body, self.physics_shape)

    @staticmethod
    def _normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _to_safe_point(x, y):
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(round(float(x))), int(round(float(y))))

    def set_overlay_lines(self, lines):
        if lines is None:
            self.extra_overlay_lines = []
            return
        self.extra_overlay_lines = [str(line) for line in lines]

    def set_swarm_agents(self, agents):
        if agents is None:
            self.swarm_agents = []
            return
        self.swarm_agents = list(agents)

    def _is_wall(self, x, y):
        if x < 0 or x >= self.window_size or y < 0 or y >= self.window_size:
            return True
        return self.track_data[int(x), int(y)] == 0

    def _get_car_corners(self, x=None, y=None, angle=None):
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        heading = self.angle if angle is None else angle

        half_l = self.car_length * 0.5
        half_w = self.car_width * 0.5

        local_corners = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )

        cos_a = math.cos(heading)
        sin_a = math.sin(heading)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        world_corners = local_corners @ rotation.T
        world_corners[:, 0] += cx
        world_corners[:, 1] += cy
        return world_corners

    def _car_hits_wall(self):
        corners = self._get_car_corners()
        sample_points = [*corners]

        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            sample_points.append((p1 + p2) * 0.5)

        sample_points.append(np.array([self.x, self.y], dtype=np.float32))

        for px, py in sample_points:
            if self._is_wall(px, py):
                return True
        return False

    def _get_lidar_distances(self):
        if not (np.isfinite(self.x) and np.isfinite(self.y) and np.isfinite(self.angle)):
            return np.full(5, self.max_lidar_distance, dtype=np.float32)

        distances = []
        step_size = 4.0

        for offset in self.ray_angles:
            ray_angle = self.angle + float(offset)
            dx = math.cos(ray_angle) * step_size
            dy = math.sin(ray_angle) * step_size

            rx, ry = self.x, self.y
            dist = 0.0

            while dist < self.max_lidar_distance:
                rx += dx
                ry += dy
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    dist = self.max_lidar_distance
                    break
                dist += step_size
                if self._is_wall(rx, ry):
                    break

            distances.append(dist)

        return np.array(distances, dtype=np.float32)

    def _step_normal_dynamics(self, steering, acceleration):
        target_steer = np.clip(steering, -1.0, 1.0) * self.max_steer
        steer_delta = np.clip(target_steer - self.steering_angle, -self.max_steer_rate, self.max_steer_rate)
        self.steering_angle += steer_delta

        if acceleration >= 0.0:
            throttle_accel = acceleration * self.engine_accel
        else:
            throttle_accel = acceleration * self.brake_accel

        drag = self.rolling_drag * np.sign(self.speed) + self.aero_drag * self.speed * abs(self.speed)
        if abs(self.speed) < 1e-4:
            drag = 0.0

        self.speed += throttle_accel - drag
        self.speed = float(np.clip(self.speed, -self.max_reverse_speed, self.max_speed))

        if abs(self.speed) < 0.02 and abs(acceleration) < 0.1:
            self.speed = 0.0

        if abs(self.speed) > 1e-4:
            beta = math.atan((self.rear_wheel_base / self.wheel_base) * math.tan(self.steering_angle))
            yaw_rate = (self.speed / self.wheel_base) * math.sin(beta)
            self.angle = self._normalize_angle(self.angle + yaw_rate)
            self.x += self.speed * math.cos(self.angle + beta)
            self.y += self.speed * math.sin(self.angle + beta)

    def _step_physics_dynamics(self, steering, acceleration):
        if self.physics_space is None or self.physics_body is None:
            self._init_physics_world()

        target_steer = np.clip(steering, -1.0, 1.0) * self.max_steer
        steer_delta = np.clip(target_steer - self.steering_angle, -self.max_steer_rate * 0.6, self.max_steer_rate * 0.6)
        self.steering_angle += steer_delta

        heading = float(self.physics_body.angle)
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)

        velocity = np.array(self.physics_body.velocity, dtype=np.float32)
        forward_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, right))

        fps = self.physics_fps
        # px/step^2 -> px/s^2 is a factor of fps^2
        accel_per_step = acceleration * (self.engine_accel if acceleration >= 0.0 else self.brake_accel)
        engine_force = tuple(forward * (self.physics_mass * accel_per_step * fps * fps))

        lateral_impulse = -lateral_speed * self.physics_mass * self.physics_lateral_grip
        self.physics_body.apply_impulse_at_world_point(tuple(right * lateral_impulse), tuple(self.physics_body.position))

        desired_yaw_rate = (forward_speed / max(self.wheel_base, 1e-4)) * math.tan(self.steering_angle)
        self.physics_body.angular_velocity += (desired_yaw_rate - self.physics_body.angular_velocity) * 0.35
        self.physics_body.angular_velocity *= self.physics_angular_damping

        # pymunk resets body.force after every space.step(), so forces must be re-applied each substep.
        sub_dt = self.physics_dt / self.physics_substeps
        for _ in range(self.physics_substeps):
            pos = tuple(self.physics_body.position)
            self.physics_body.apply_force_at_world_point(engine_force, pos)
            velocity = np.array(self.physics_body.velocity, dtype=np.float32)
            speed_px_s = float(np.linalg.norm(velocity))
            if speed_px_s > 1e-3:
                v_step = speed_px_s / fps
                drag_per_step = self.rolling_drag + self.aero_drag * v_step * v_step
                drag_force = -(velocity / speed_px_s) * (self.physics_mass * drag_per_step * fps * fps)
                self.physics_body.apply_force_at_world_point(tuple(drag_force), pos)
            self.physics_space.step(sub_dt)

        self.x = float(self.physics_body.position.x)
        self.y = float(self.physics_body.position.y)
        self.angle = self._normalize_angle(float(self.physics_body.angle))

        velocity = np.array(self.physics_body.velocity, dtype=np.float32)
        forward = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)
        forward_speed = float(np.dot(velocity, forward))

        clipped_speed = float(np.clip(forward_speed, -self.max_reverse_speed * fps, self.max_speed * fps))
        if clipped_speed != forward_speed:
            velocity += (clipped_speed - forward_speed) * forward
            self.physics_body.velocity = tuple(velocity)

        self.speed = clipped_speed / fps  # px/step, same unit as normal mode

    def _get_obs(self):
        return np.array([
            self.x / self.window_size,
            self.y / self.window_size,
            self.angle / np.pi,
            self.speed / self.max_speed,
            *(self.lidar_distances / self.max_lidar_distance)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.set_overlay_lines(None)
        self.set_swarm_agents(None)
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0.0
        self.steering_angle = 0.0
        self.current_step = 0

        if self.dynamics_mode == "physics":
            self._init_physics_world()

        self.current_checkpoint_idx = 1
        target_cp = self.checkpoints[self.current_checkpoint_idx]
        self.prev_dist_to_cp = math.hypot(self.x - target_cp[0], self.y - target_cp[1])

        self.laps_completed = 0
        self.trail = []

        self.lidar_distances = self._get_lidar_distances()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError("Action must contain [steering, acceleration].")

        steering = float(np.clip(action[0], -1.0, 1.0))
        acceleration = float(np.clip(action[1], -1.0, 1.0))

        if self.dynamics_mode == "physics":
            self._step_physics_dynamics(steering, acceleration)
        else:
            self._step_normal_dynamics(steering, acceleration)

        self.trail.append((self.x, self.y))
        if len(self.trail) > 30:
            self.trail.pop(0)

        self.current_step += 1

        terminated = False
        truncated = False
        reward = 0.0

        if self._car_hits_wall():
            terminated = True
            reward = -100.0
        else:
            reward -= 0.1

            target_cp = self.checkpoints[self.current_checkpoint_idx]
            dist_to_cp = math.hypot(self.x - target_cp[0], self.y - target_cp[1])

            progress = self.prev_dist_to_cp - dist_to_cp
            progress = max(-2.0, min(10.0, progress)) # Capping to prevent huge penalties from backward movements
            reward += progress * 0.5
            self.prev_dist_to_cp = dist_to_cp

            if dist_to_cp < self.checkpoint_radius:
                reward += 50.0
                self.current_checkpoint_idx += 1

                if self.current_checkpoint_idx >= len(self.checkpoints):
                    self.current_checkpoint_idx = 0
                    self.laps_completed += 1
                    reward += 500.0

                next_cp = self.checkpoints[self.current_checkpoint_idx]
                self.prev_dist_to_cp = math.hypot(self.x - next_cp[0], self.y - next_cp[1])

        if self.current_step >= self.max_steps:
            truncated = True

        self.lidar_distances = self._get_lidar_distances()
        obs = self._get_obs()
        info = {
            "laps": self.laps_completed,
            "dynamics_mode": self.dynamics_mode,
            "steering_angle": float(self.steering_angle),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            print(
                f"[{self.dynamics_mode}] Pos ({self.x:.0f}, {self.y:.0f}), "
                f"Angle {self.angle:.2f}, Laps: {self.laps_completed}"
            )
        elif self.render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("2D RL Glowing Track")

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.fill((10, 12, 20))

            # 1) Track (Mavi hat - geniş)
            pygame.draw.lines(self.window, (30, 40, 100), True, self.checkpoints, 120)
            for p in self.checkpoints:
                pygame.draw.circle(self.window, (30, 40, 100), p, 47)

            pygame.draw.lines(self.window, (25, 27, 36), True, self.checkpoints, 80)
            for p in self.checkpoints:
                pygame.draw.circle(self.window, (25, 27, 36), p, 40)

            fl_x, fl_y = self.checkpoints[0]
            pygame.draw.circle(self.window, (200, 180, 50), (fl_x, fl_y), self.checkpoint_radius, 2)
            cp_x, cp_y = self.checkpoints[self.current_checkpoint_idx]
            pygame.draw.circle(self.window, (50, 150, 200), (cp_x, cp_y), self.checkpoint_radius, 2)

            swarm_mode = len(self.swarm_agents) > 0

            if swarm_mode:
                for agent in self.swarm_agents:
                    ax = agent.get("x")
                    ay = agent.get("y")
                    angle = float(agent.get("angle", 0.0))
                    color = tuple(agent.get("color", (220, 120, 120)))
                    radius = int(agent.get("radius", 3))

                    agent_pos = self._to_safe_point(ax, ay)
                    if agent_pos is None:
                        continue

                    pygame.draw.circle(self.window, color, agent_pos, max(1, radius))

                    nose_x = float(ax) + 8.0 * math.cos(angle)
                    nose_y = float(ay) + 8.0 * math.sin(angle)
                    nose_pos = self._to_safe_point(nose_x, nose_y)
                    if nose_pos is not None:
                        pygame.draw.line(self.window, color, agent_pos, nose_pos, 1)
            else:
                # 2) Trail
                if len(self.trail) > 1:
                    trail_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
                    for i, (tx, ty) in enumerate(self.trail):
                        alpha = int(255 * (i / len(self.trail)))
                        color = (255, 100, 0, alpha)
                        radius = max(2, int(4 * (i / len(self.trail))))
                        pygame.draw.circle(trail_surface, color, (int(tx), int(ty)), radius)
                    self.window.blit(trail_surface, (0, 0))

                # 3) Lidar rays (near: yellow, far: green)
                ray_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
                start_pos = self._to_safe_point(self.x, self.y)
                for dist, offset in zip(self.lidar_distances, self.ray_angles):
                    ray_angle = self.angle + float(offset)
                    ray_end_x = self.x + dist * math.cos(ray_angle)
                    ray_end_y = self.y + dist * math.sin(ray_angle)

                    end_pos = self._to_safe_point(ray_end_x, ray_end_y)
                    if start_pos is None or end_pos is None:
                        continue

                    ratio = float(np.clip(dist / self.max_lidar_distance, 0.0, 1.0))
                    ray_r = int(255 - 205 * ratio)
                    ray_g = 255
                    ray_b = int(90 - 40 * ratio)
                    ray_a = int(90 + 120 * ratio)

                    pygame.draw.line(
                        ray_surface,
                        (ray_r, ray_g, ray_b, ray_a),
                        start_pos,
                        end_pos,
                        2,
                    )
                    pygame.draw.circle(
                        ray_surface,
                        (ray_r, ray_g, ray_b, min(255, ray_a + 40)),
                        end_pos,
                        3,
                    )
                self.window.blit(ray_surface, (0, 0))

                # 4) Car shape
                car_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
                p1 = (30, 20)
                p2 = (10, 10)
                p3 = (10, 30)
                pygame.draw.polygon(car_surface, (255, 80, 50), [p1, p2, p3])
                pygame.draw.polygon(car_surface, (255, 200, 100), [p1, p2, p3], 2)

                rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
                car_rect = rotated_car.get_rect(center=(int(self.x), int(self.y)))
                self.window.blit(rotated_car, car_rect.topleft)

                # Collision footprint visualization
                car_corners = self._get_car_corners()
                pygame.draw.polygon(
                    self.window,
                    (255, 220, 140),
                    [(int(px), int(py)) for px, py in car_corners],
                    1,
                )

            # 5) HUD
            font = pygame.font.SysFont("consolas", 24)
            hud_info = [
                f"M O D E   : {self.dynamics_mode.upper()}",
                f"S P E E D : {self.speed:+.2f}/{self.max_speed:.1f}",
                f"L A P S   : {self.laps_completed}",
                f"S T E P S : {self.current_step}/{self.max_steps}",
                f"C H E C K : {self.current_checkpoint_idx + 1}/{len(self.checkpoints)}",
            ]

            if swarm_mode:
                alive_count = sum(1 for a in self.swarm_agents if a.get("alive", True))
                hud_info[1] = f"S W A R M : {alive_count}/{len(self.swarm_agents)}"

            y_offset = 20
            for text in hud_info:
                t_surface = font.render(text, True, (0, 255, 255))
                self.window.blit(t_surface, (20, y_offset))
                y_offset += 30

            if self.extra_overlay_lines:
                overlay_title = font.render("T R A I N I N G  P R E V I E W", True, (255, 220, 120))
                self.window.blit(overlay_title, (20, y_offset + 8))
                y_offset += 40

                for line in self.extra_overlay_lines:
                    o_surface = font.render(str(line), True, (255, 190, 110))
                    self.window.blit(o_surface, (20, y_offset))
                    y_offset += 28

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
            self.clock = None

        if pygame.get_init():
            pygame.quit()

        self.physics_space = None
        self.physics_body = None
        self.physics_shape = None
