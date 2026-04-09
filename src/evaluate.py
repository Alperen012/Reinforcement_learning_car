import time
import pygame
from stable_baselines3 import PPO
from envs.race_track_env import RaceTrackEnv

def main():
    # TODO 1: Initialize the environment FOR VISUALIZATION
    # Create an instance of RaceTrackEnv with render_mode="human"
    env = RaceTrackEnv(render_mode="human")
    
    
    # TODO 2: Load the trained model
    # Use PPO.load() and point to your saved model file in the 'models' folder.
    model = PPO.load("models/ppo_race_track")
    
    # TODO 3: Reset the environment to start the episode
    # Call env.reset() and get the initial observation.
    obs, info = env.reset()
    
    # TODO 4: Create an evaluation loop
    # Create a while loop that runs continuously.
    # Inside the loop:
    #   a. Use `model.predict(obs)` to get the best action from the trained agent.
    #      (Remember that predict returns a tuple: action, state. You only need action)
    #   b. Use `env.step(action)` to execute the chosen action.
    #   c. Render the frame.
    #   d. If the episode is terminated or truncated, print the score and reset the environment.
    #   e. Add a small delay (like `time.sleep(0.01)`) or use pygame clock so it doesn't run too fast to see.
    #   f. Handle Pygame QUIT events so the window can be closed properly.
    running = True
    while running:
        # Get the action from the model
        action, _ = model.predict(obs)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Check for episode termination or truncation
        if terminated or truncated:
            print(f"Episode finished! Reward: {reward}")
            obs, info = env.reset()  # Start a new episode
        
        # Handle Pygame events to allow window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Add a small delay to control the speed of the loop
        # time.sleep(0.01)
    
    # TODO 5: Close the environment when the loop breaks
    env.close()

if __name__ == "__main__":
    main()
