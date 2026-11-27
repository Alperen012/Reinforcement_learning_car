# 4 · Comparing algorithms

Five algorithms ship with the project, each chosen because it differs from the
others in one instructive way:

| | learns from | actor | action space | in one sentence |
|---|---|---|---|---|
| **PPO** | fresh rollouts only (on-policy) | stochastic | continuous | Collect a batch, take several careful (clipped) gradient steps, throw the batch away. |
| **A2C** | fresh rollouts only | stochastic | continuous | PPO's ancestor: one gradient step per short rollout, no clipping. |
| **SAC** | replay buffer (off-policy) | stochastic, max-entropy | continuous | Reuse every transition many times; reward randomness itself so it keeps exploring. |
| **TD3** | replay buffer | deterministic + noise | continuous | Like SAC's critic machinery, but explores by adding Gaussian noise to actions. |
| **DQN** | replay buffer | none (argmax of Q) | **9 discrete actions** | Learn the value of each of 9 (steer, throttle) choices, pick the best. |

## Run the comparison

```bash
racing-gym train -c configs/ppo_classic.toml
racing-gym train -c configs/a2c_classic.toml
racing-gym train -c configs/sac_classic.toml
racing-gym train -c configs/td3_classic.toml
racing-gym train -c configs/dqn_classic.toml
racing-gym plot runs/ppo_classic runs/a2c_classic runs/sac_classic runs/td3_classic runs/dqn_classic --out algos.png
```

Each config trains 3 seeds. Always compare several seeds: one lucky seed is an
anecdote. The plot shows the mean, with a band from the worst to the best seed.

Our measured results, with hardware and wall-clock time, are in
[results.md](../results.md).

## Things to notice

* **Sample efficiency vs wall-clock.** Off-policy methods (SAC, TD3) reuse data
  and often need fewer environment steps. But every step includes gradient
  updates, so they can be slower in seconds. Compare the x-axis of the plot
  with `train_seconds` in each `run.json`.
* **Discrete actions** (DQN) can't steer *a little*. Watch it with
  `racing-gym watch runs/dqn_classic/seed0` and look for bang-bang steering.
* **Seed variance.** Some algorithm/seed pairs collapse or plateau. That is
  normal in RL and exactly why benchmarks report several seeds.

## Adding your own algorithm

One entry in `racing_gym/algos.py`. See [extending.md](../extending.md).

Next: [5 · Generalisation](05-generalization.md)
