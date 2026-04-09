import gymnasium as gym
import numpy as np
import pygame
import math

class RaceTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window_size = 1000  # Increased size for Nürburgring
        self.window = None
        self.clock = None
        
        # Continuous action space matching the user's NeuralCar 
        # [steering, acceleration] mapped between -1 and 1
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Max limits
        self.max_speed = 8.0
        
        # OBSERVATION: [x, y, angle, speed, ray1, ray2, ray3, ray4, ray5]
        low = np.array([0.0, 0.0, -np.inf, -self.max_speed, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1000.0, 1000.0, np.inf, self.max_speed, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.ray_angles = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        self.lidar_distances = np.zeros(5, dtype=np.float32)
        
        self.max_steps = 3000
        self.current_step = 0
        
        # Nürburgring Nordschleife Approximated Checkpoints
        self.checkpoints = [
            (250, 750), # Start/Finish (Antoniusbuche/Tiergarten)
            (150, 700), # Hatzenbach
            (100, 550), # Hocheichen
            (150, 400), # Flugplatz
            (250, 350), # Schwedenkreuz
            (150, 250), # Fuchsrohre
            (250, 150), # Adenauer Forst
            (400, 150), # Metzgesfeld
            (450, 250), # Kallenhard
            (350, 350), # Wehrseifen
            (450, 450), # Ex-Muhle / Bergwerk
            (650, 350), # Kesselchen
            (800, 250), # Steilstrecke
            (900, 150), # Karussell
            (950, 300), # Hohe Acht
            (850, 450), # Wippermann
            (950, 550), # Brunnchen
            (850, 650), # Pflanzgarten
            (900, 800), # Schwalbenschwanz
            (600, 850), # Dottinger Hohe 1
            (400, 850)  # Dottinger Hohe 2
        ]
        self.checkpoint_radius = 50 # Slightly tighter
        self.laps_completed = 0
        
        self._init_track()

    def _init_track(self):
        # We generate a hidden pygame surface internally for masking and collisions
        import pygame
        pygame.init()
        self.track_surface = pygame.Surface((self.window_size, self.window_size))
        self.track_surface.fill((0, 0, 0)) # 0 means wall (black)
        
        # Draw the main road with thick lines (Tighter lines because the track is longer/more complex)
        pygame.draw.lines(self.track_surface, (255, 255, 255), True, self.checkpoints, 80)
        # Add smooth corners by dropping big circles at joints
        for p in self.checkpoints:
            pygame.draw.circle(self.track_surface, (255, 255, 255), p, 40)
            
        # Fast memory mapping to read track collisions
        self.track_data = pygame.surfarray.pixels2d(self.track_surface)
        
        # Start coordinates setup
        self.start_x = 250.0
        self.start_y = 750.0
        self.start_angle = -2.6 # Pointing roughly towards Hatzenbach (-100, -50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0.0
        self.current_step = 0
        
        self.current_checkpoint_idx = 1
        target_cp = self.checkpoints[self.current_checkpoint_idx]
        self.prev_dist_to_cp = math.hypot(self.x - target_cp[0], self.y - target_cp[1])
        
        self.laps_completed = 0
        self.trail = []  # For the visual trail behind the car
        
        self.lidar_distances = self._get_lidar_distances()
        obs = np.array([self.x, self.y, self.angle, self.speed, *self.lidar_distances], dtype=np.float32)
        return obs, {}

    def _is_wall(self, x, y):
        # Out of bounds counts as wall
        if x < 0 or x >= self.window_size or y < 0 or y >= self.window_size:
            return True
        # 0 = black (background), which is "wall"
        return self.track_data[int(x), int(y)] == 0

    def _get_lidar_distances(self):
        distances = []
        max_dist = 250.0  # Limit radar distance
        step_size = 4.0   # Jump forward along the ray to check for walls
        
        for offset in self.ray_angles:
            ray_angle = self.angle + offset
            dx = np.cos(ray_angle) * step_size
            dy = np.sin(ray_angle) * step_size
            
            rx, ry = self.x, self.y
            dist = 0.0
            
            while dist < max_dist:
                rx += dx
                ry += dy
                dist += step_size
                if self._is_wall(rx, ry):
                    break
                    
            distances.append(dist)
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        # Action mappings [-1, 1]
        steering = float(action[0])
        acceleration = float(action[1])

        # Speed update dynamics
        speed_delta = 0.5 * acceleration
        self.speed = np.clip(self.speed + speed_delta, -self.max_speed / 2, self.max_speed)
        self.speed *= 0.95 # Simulates friction/drag
        
        # Turn dynamics (only steer if moving)
        if abs(self.speed) > 0.1:
            direction = 1 if self.speed > 0 else -1 # Turn is reversed in reverse
            self.angle += steering * 0.15 * direction

        # Update position
        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)
        
        # Add position to trail for glowing trail effect
        if not hasattr(self, 'trail'): self.trail = []
        self.trail.append((self.x, self.y))
        if len(self.trail) > 30: # Max tail length 
            self.trail.pop(0)

        self.current_step += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        
        if self._is_wall(self.x, self.y):
            terminated = True
            reward = -100.0  # Heavy crash penalty
        else:
            # 1. Small time penalty each step to encourage faster completion
            reward -= 0.1 
            
            target_cp = self.checkpoints[self.current_checkpoint_idx]
            dist_to_cp = math.hypot(self.x - target_cp[0], self.y - target_cp[1])
            
            # 2. Progress reward (Penalty if driving away from checkpoint, reward if getting closer)
            progress = self.prev_dist_to_cp - dist_to_cp
            reward += progress * 0.5  # Multiplier to scale the distance progress
            self.prev_dist_to_cp = dist_to_cp
            
            # 3. Checkpoint rewards for moving correctly through the circuit
            if dist_to_cp < self.checkpoint_radius:
                reward += 50.0 # Reward for hitting checkpoints
                self.current_checkpoint_idx += 1
                
                # Reached finish line (completed a lap)
                if self.current_checkpoint_idx >= len(self.checkpoints):
                    self.current_checkpoint_idx = 0
                    self.laps_completed += 1
                    reward += 500.0 # Bonus for completing lap!
                    
                # Update next checkpoint distance reference
                next_cp = self.checkpoints[self.current_checkpoint_idx]
                self.prev_dist_to_cp = math.hypot(self.x - next_cp[0], self.y - next_cp[1])

        if self.current_step >= self.max_steps:
            truncated = True

        self.lidar_distances = self._get_lidar_distances()
        obs = np.array([self.x, self.y, self.angle, self.speed, *self.lidar_distances], dtype=np.float32)
        info = {"laps": self.laps_completed}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            print(f"Pos ({self.x:.0f}, {self.y:.0f}), Angle {self.angle:.2f}, Laps: {self.laps_completed}")
        elif self.render_mode == "human":            
            import pygame
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("2D RL Glowing Track")
            
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Deep dark background like space/midnight
            self.window.fill((10, 12, 20))
            
            # --- 1. Draw track ---
            # Outer neon glow boundary (thicker glow scaled down)
            pygame.draw.lines(self.window, (30, 40, 100), True, self.checkpoints, 94)
            for p in self.checkpoints: pygame.draw.circle(self.window, (30, 40, 100), p, 47)
            # Track surface
            pygame.draw.lines(self.window, (25, 27, 36), True, self.checkpoints, 80)
            for p in self.checkpoints: pygame.draw.circle(self.window, (25, 27, 36), p, 40)
            
            # Finish Line and Checkpoints
            fl_x, fl_y = self.checkpoints[0]
            pygame.draw.circle(self.window, (200, 180, 50), (fl_x, fl_y), self.checkpoint_radius, 2)
            cp_x, cp_y = self.checkpoints[self.current_checkpoint_idx]
            pygame.draw.circle(self.window, (50, 150, 200), (cp_x, cp_y), self.checkpoint_radius, 2)

            # --- 2. Draw Trails (Glowing dots) ---
            if hasattr(self, 'trail') and len(self.trail) > 1:
                # We draw the trail using alpha transparency blending
                trail_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
                for i, (tx, ty) in enumerate(self.trail):
                    alpha = int(255 * (i / len(self.trail))) # Fade out 
                    color = (255, 100, 0, alpha)            # Neon orange
                    radius = max(2, int(4 * (i / len(self.trail))))
                    pygame.draw.circle(trail_surface, color, (int(tx), int(ty)), radius)
                self.window.blit(trail_surface, (0,0))

            # --- 3. Draw Lidar Rays (Laser green) ---
            ray_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            for dist, offset in zip(self.lidar_distances, self.ray_angles):
                ray_angle = self.angle + offset
                ray_end_x = self.x + dist * np.cos(ray_angle)
                ray_end_y = self.y + dist * np.sin(ray_angle)
                pygame.draw.line(ray_surface, (50, 255, 50, 100), (self.x, self.y), (ray_end_x, ray_end_y), 2)
                pygame.draw.circle(ray_surface, (100, 255, 100, 150), (int(ray_end_x), int(ray_end_y)), 3)
            self.window.blit(ray_surface, (0,0))

            # --- 4. Draw Car (Neon sci-fi shape) ---
            car_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
            
            # Car vertices relative to center
            # It's an arrowhead/triangle shape pointing right
            p1 = (30, 20)  # nose
            p2 = (10, 10)  # back top
            p3 = (10, 30)  # back bottom
            
            pygame.draw.polygon(car_surface, (255, 80, 50), [p1, p2, p3])
            pygame.draw.polygon(car_surface, (255, 200, 100), [p1, p2, p3], 2) # outline
            
            # Rotate car graphic
            # Pygame rotation is counter-clockwise and in degrees
            rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
            car_rect = rotated_car.get_rect(center=(int(self.x), int(self.y)))
            self.window.blit(rotated_car, car_rect.topleft)

            # --- 5. Stats Text HUD ---
            font = pygame.font.SysFont("consolas", 24)
            hud_info = [
                f"S P E E D : {abs(self.speed):.1f}/{self.max_speed}",
                f"L A P S   : {self.laps_completed}",
                f"S T E P S : {self.current_step}/{self.max_steps}",
                f"C . P O I N T : {self.current_checkpoint_idx}/8"
            ]
            
            y_offset = 20
            for text in hud_info:
                t_surface = font.render(text, True, (0, 255, 255))
                self.window.blit(t_surface, (20, y_offset))
                y_offset += 30

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            self.window = None
            self.clock = None