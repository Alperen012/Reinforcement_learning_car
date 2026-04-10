# PPO Race Track - Reinforcement Learning 🏎️🧠

This project is a professional reference repository demonstrating how to optimally train a Proximal Policy Optimization (PPO) algorithm using Stable-Baselines3 in a 2D racing track environment (`RaceTrackEnv`).

## 🚀 Highlights and Performance Results
Even after a short training process (e.g., 20,000 steps), our model successfully completes the track without crashing (Lap = 1).

**Example Evaluation Result:**
```text
Episode 1/10 | reward=1708.89 | steps=280 | laps=1
Episode 2/10 | reward=5615.78 | steps=869| laps=3
```
*(Note: The identical outputs in each lap are not a bug. They result from the PPO model's **deterministic** evaluation which applies optimal and consistent rules. Since the environment and the starting position of the car are always exactly the same, the model flawlessly repeats its best actions.)*

## 🏗️ Project Architecture (Optimized for PPO)
1. **Normalized Observations and Rewards (`VecNormalize`):** Automatically normalizes sensor data and rewards to resolve imbalances and provide stable gradients for the AI during learning.
2. **`EvalCallback` (Evaluation System):** Tests the model periodically during training. The model achieving the highest score is automatically saved as a `.zip` in the `models/best_model/` directory.
3. **Early Stopping & Logical Reward Philosophy:** The car receives dense rewards for moving forward. To prevent "suicidal" behavior (intentionally crashing rapidly to avoid cumulative penalties), penalties for reversing are capped.
4. **Stable Environment:** Complex track geometries have been replaced with a simple oval track, allowing the model to quickly grasp driving logic.
5. **Visual Tracking (`RenderDuringTrainingCallback`):** Watch the car's actions live (policy mode) on the screen to observe its improvement during training without interrupting it.

## 💻 Setup and Environment
First, install all the required libraries:
```bash
pip install -r requirements.txt
```

## 🏎️ Usage Guide

### 1. Start Training (Training Process)
The following command trains the model for `2,000,000` steps. It opens a window every `50,000` steps to show a live visual preview for `500` frames:
```bash
python src/train_sb3.py --timesteps 2000000 --render-training --render-freq 50000 --render-steps 500 --dynamics-mode normal
```
*(Tip: If you are impatient and cancel the training midway using `CTRL+C` in the terminal, don't worry! `EvalCallback` always keeps the best-performing version safely saved as "best_model.zip".)*

### 2. Evaluating the Trained Model
To load your best model and watch it navigate the track for 10 episodes, use the following commands:

**Deterministic (Precise & Flawless) Driving:**
The model takes no risks and flawlessly executes the exact optimal actions like a robot.
```bash
python src/evaluate.py --model-path ./models/best_model/best_model.zip --episodes 10 --dynamics-mode normal
```

**Stochastic (Dynamic / Probabilistic) Driving:**
If you want the model to try different lanes and make more human-like deviations/mistakes, simply add the `--stochastic` flag:
```bash
python src/evaluate.py --model-path ./models/best_model/best_model.zip --episodes 10 --dynamics-mode normal --stochastic
```

### 3. Manual Driving (To test the environment)
You can drive the car using arrow keys to test the environment dynamics yourself:
```bash
python src/test_env.py
```
