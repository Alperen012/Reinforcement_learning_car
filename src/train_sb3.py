import os
from stable_baselines3 import PPO
from envs.race_track_env import RaceTrackEnv
import tensorboard

def main():

    # Create directories for logs and models if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialize the environment
    env = RaceTrackEnv()

    # Create the PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/ppo_race_track"
    )
    
    # Train the model
    model.learn(total_timesteps=1000000)

    
    # Save the trained model
    model.save("models/ppo_race_track")
    
if __name__ == "__main__":
    main()