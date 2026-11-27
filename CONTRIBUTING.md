# Contributing

Thanks for wanting to help! Good first contributions:

* **A new track preset**: [docs/extending.md](docs/extending.md#add-a-track-preset)
* **A trained agent for the playground** (a new algorithm, car or reward): train it, `racing-gym export`, open a PR with the model file and the command you used
* **A tutorial fix**: anything that confused you is worth a PR
* An item from the roadmap in the README

## Set up

```bash
git clone https://github.com/Alperen012/Reinforcement_learning_car
cd Reinforcement_learning_car
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest          # ~1 minute; the parity tests need Node.js (skipped otherwise)
```

## Ground rules

* **Keep Python and `web/js/sim.js` in sync.** If you change the track, car, observation or
  reward in Python, mirror it in `sim.js`. `tests/test_web_parity.py` checks they match.
* **Numbers in docs come from commands.** If you add a result to `docs/results.md`, include the
  exact command, the number of seeds, and your hardware. Never report a single seed as a result.
* **Defaults are a contract.** Changing a default in `config.py` changes every existing experiment.
  Add new options with defaults that keep current behaviour.
* **New dependencies need a reason.** The core needs only Gymnasium, SB3, NumPy, pygame,
  matplotlib and Pillow, and we would like to keep it that way.
* Style: `ruff check .` and type hints on public functions.

## Pull requests

Describe what changed and why, and how you tested it. Link an issue if there is
one. Small PRs are reviewed faster than big ones.

By contributing, you agree that your contribution is released under the project's
[MIT licence](LICENSE).
