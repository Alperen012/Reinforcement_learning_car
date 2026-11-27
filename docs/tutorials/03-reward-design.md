# 3 · Reward design (and how agents cheat)

The reward is the *only* thing that tells the agent what you want. It will
optimise exactly what you wrote, not what you meant.

Every term is a config value you can override from the command line, so each
experiment below is one command. Train, then compare with
`racing-gym evaluate` and `racing-gym plot`.

## Experiment A: no crash penalty

```bash
racing-gym train -c configs/ppo_classic.toml -s name=no_crash -s env.reward.crash=0 --seeds 0
```

Question: does it still learn to avoid crashing? (Crashing still ends the
episode, so the agent loses all *future* progress reward. That is an implicit
penalty.) Compare `crash_rate` with the default run.

## Experiment B: no time penalty, no lap bonus

```bash
racing-gym train -c configs/ppo_classic.toml -s name=lazy -s env.reward.time=0 -s env.reward.lap=0 --seeds 0
```

Look at `avg_speed` in the evaluation. Progress reward alone already pays for
speed: going faster means more progress per step.

## Experiment C: stay in the middle

```bash
racing-gym train -c configs/ppo_classic.toml -s name=centred -s env.reward.center=-0.5 --seeds 0
racing-gym record runs/ppo_classic/seed0 runs/centred/seed0 --out lines.gif
```

The default agent cuts corners (the racing line). The centred one gives up
some speed to follow the middle. Neither is "right". The reward decides.

## Case study: a real reward hack in this project's history

Version 0.1 rewarded the *decrease in distance to the next checkpoint*, clipped
to [-2, +10] "to prevent huge penalties". The clip made moving backwards
cheaper than moving forwards paid. So rocking forward and back earned reward
without going anywhere. A clever enough agent would have found it.

The fix (v0.2) measures progress as **signed distance along the centre
line**, with no clip. Going back exactly cancels going forward. A test pins
this down: `tests/test_env.py::test_forward_backward_oscillation_cannot_farm_reward`.

**Checklist for any reward you write:**

1. Can the agent earn it without doing the task? (oscillation, standing still, circling)
2. Is "end the episode early" ever better than continuing? (an agent will crash on purpose if staying alive costs more than crashing)
3. Is each term's scale comparable? Look at `reward_terms/*` in TensorBoard.

## Seeing reward live

In the playground, the *Why it gets rewarded* panel shows every term for the
current step and the episode total. The *Reward weights* sliders change what
is shown, which is a quick way to check what an episode *would* have scored
under a different reward before you spend time retraining.

Next: [4 · Comparing algorithms](04-algorithms.md)
