# 1 · Reinforcement learning, explained with a race car

**Goal:** understand the five words every RL paper uses, by pointing at
things on screen. No training yet.

Open the playground (`web/index.html`), or in a terminal:

```bash
racing-gym play --track oval      # drive yourself with the arrow keys
```

## Agent, environment, step

The **agent** is the driver. The **environment** is everything else: the
track, the car's physics, the rules. They take turns 30 times per simulated
second:

```
          observation, reward
   ┌──────────────────────────────┐
   ▼                              │
 AGENT ──── action ────►  ENVIRONMENT
```

In code (`racing_gym/env.py`) that is one call:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

## Observation: what the agent sees

Not the picture you see. The agent gets **17 numbers** (the *What it sees*
panel in the playground, or the right-hand panel of `racing-gym play`):

| numbers | meaning |
|---|---|
| `lidar_0..8` | distance to the track edge along 9 rays, from hard left to hard right (1 = far) |
| `speed` | forward speed ÷ top speed |
| `steer` | current wheel angle ÷ full lock |
| `heading_err_sin/cos` | angle between the car and the track direction |
| `lateral_offset` | distance from the centre line ÷ half the track width |
| `curve_60/120/240px` | how much the track turns 60, 120 and 240 px ahead |

Notice what is *missing*: no x/y position. The agent does not know where on
the map it is, only what is around it. This is why one agent can drive tracks
it has never seen (tutorial 5).

## Action: what the agent does

Two numbers in [-1, 1]: **steer** (−1 = full left) and **throttle** (−1 =
brake / reverse). Some algorithms (DQN) can only choose from a menu of 9 fixed
combinations. That is the `discrete` action mode.

## Reward: the only feedback

After every step the environment hands out a number. Here it is a sum of terms
you can see in the *Why it gets rewarded* panel:

| term | default | when |
|---|---:|---|
| progress | +1 per 10 px | driving forward along the track (negative when going backwards) |
| time | −0.05 | every step: dawdling costs |
| crash | −20 | leaving the track (the episode ends) |
| lap | +10 | completing a lap |

The agent never sees these rules. It only sees the numbers that come out of
them. **Learning = finding behaviour that makes the total large.**

## Episode and return

An **episode** runs from `reset()` until the car crashes (*terminated*) or
time runs out (*truncated*). The **return** is the sum of all rewards in an
episode: the single score RL tries to maximise. The *Episode returns* chart in
the playground shows one bar per episode.

## Policy: the agent's brain

A **policy** maps observation → action. Ours is a small neural network
(17 → 64 → 64 → 2). Training adjusts its weights so that actions which led to
high returns become more likely.

Try this in the playground: select the agent and drag the **training stage**
slider. *Untrained* (same network, random weights) barely moves: its outputs
hover around zero throttle. After 25,000 steps it drives off, but into the
first wall. *Fully trained*, it races the line at top speed. Everything in
between is what RL did.

## Exploration

How does it ever find out that driving forward is good, if it starts out
random? By **exploring**: during training the actions are noisy (PPO samples
from a Gaussian, SAC maximises entropy, DQN takes random actions some of the
time). A lucky push forward earns progress reward, that behaviour gets
reinforced, and the cycle repeats.

## Check yourself

1. Add the *Hand-written controller* racer. It drives well and never trained.
   Which observation numbers does it use? (Answer: `racing_gym/evaluation.py`,
   `heuristic_policy`.)
2. In *Reward weights*, set `crash` to 0. Watch the return of a crashing
   random racer. Why would an agent trained with this reward be reckless?

Next: [2 · Train your first agent](02-first-agent.md)
