# Decision log

Short records of the choices that shape the project. Add a new entry when you
make a decision someone might later want to revisit.

### D1. Python + Gymnasium + Stable-Baselines3 for training
Most RL learners already know this stack. SB3 is tested and documented, and
writing our own algorithms would add bugs without adding insight. The env is
plain Gymnasium, so CleanRL, RLlib and others work too.

### D2. A static browser playground, not a web app
The playground is `web/index.html`: no build step, no server, works from
`file://` and GitHub Pages. Trained policies are exported as weight arrays
(`racing-gym export`) and evaluated by a 20-line MLP in JS. Training stays in
Python, where it is reproducible. *Rejected:* a FastAPI/WebSocket backend that
trains live, because it means two processes, state syncing, hosting cost, and
nothing a learner gains from it.

### D3. The simulator exists twice (Python and JS), and a test keeps them identical
`web/js/sim.js` mirrors `track.py`/`car.py`/`env.py` line by line.
`tests/test_web_parity.py` runs both through Node and compares tracks, poses,
observations and rewards step by step, so agents behave the same in the browser
as in training. Procedural tracks use a tiny shared LCG instead of NumPy's RNG,
so the same seed gives the same track in both languages.

### D4. Kinematic bicycle model with a grip limit instead of a physics engine
About 30 readable lines. It is fast (~7–10k env steps/s in pure Python) and
trivial to port. The grip limit gives understeer, so speed management matters.
*Rejected:* pymunk/Box2D (see the audit for the bugs this caused before).

### D5. Egocentric, low-dimensional observation
Lidar + speed + wheel angle + heading error + lateral offset + curvature
ahead, 17 numbers. A policy trains in minutes on a CPU, each input can be
shown and explained, and it can transfer to unseen tracks. *Rejected:* pixels,
which are slow and opaque.

### D6. Progress reward = signed arc length along the centreline
It is dense and cannot be gamed by oscillating: net reward equals net distance
(tested). Other terms (time, crash, lap, centring, steering smoothness) are
separate weights, and each is reported separately in `info["reward_terms"]` so
the playground and logs can show *why* an episode scored what it did.

### D7. TOML configs, stdlib only
`tomllib` is in Python 3.11's standard library, so no PyYAML dependency. Typed
dataclasses reject unknown keys, so a typo fails loudly. `-s key.path=value`
overrides any field from the command line.

### D8. Five algorithms, chosen for contrast
PPO (default, robust), A2C (PPO's simpler ancestor), SAC (off-policy, entropy),
TD3 (off-policy, deterministic actor), DQN (value-based; uses the 9-action
discrete mode). Each teaches something the others do not. Adding one is a
single registry entry.

### D9. Baselines in every comparison
Every evaluation table includes `random`, `untrained` and a hand-written
`heuristic` controller next to the trained agents, on identical seeded
episodes. The heuristic shows what three lines of classical control achieve,
which keeps claims about RL honest.

### D10. Licence left unchanged
The repository had no licence when this work started. Picking one is the
owner's decision, so none was added.
