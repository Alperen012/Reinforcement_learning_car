import gymnasium as gym
import pygame
from envs.race_track_env import RaceTrackEnv

def main():
    # Trigger Pygame screen by setting render_mode to "human"
    env = RaceTrackEnv(render_mode="human")
    obs, info = env.reset()

    # Initial render to initialize pygame's video system
    env.render()

    running = True
    while running:
        action = [0.0, 0.0]  # [steering, acceleration]
        # Listen for Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Get pressed keyboard keys
        keys = pygame.key.get_pressed()
        
        # Determine acceleration (action[1])
        if keys[pygame.K_UP]:
            action[1] = 1.0
        elif keys[pygame.K_DOWN]:
            action[1] = -1.0
            
        # Determine steering (action[0])
        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        elif keys[pygame.K_RIGHT]:
            action[0] = 1.0
            
        # Move the car based on continuous input
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished! Laps: {info.get('laps', 0)} Reward: {reward:.2f}")
            obs, info = env.reset() # Start a new episode

        # Render the environment
        env.render()

    env.close()

if __name__ == "__main__":
    main()
