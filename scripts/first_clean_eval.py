"""Score for the "Fastest learner" challenge (CHALLENGES.md).

    python scripts/first_clean_eval.py runs/<experiment>

For each seed, prints the first evaluation step with >= 6 laps and 0% crashes on
the training-time evaluations (evals.csv), then the mean over seeds.
"""

import csv
import sys
from pathlib import Path


def first_clean(evals: Path, min_laps: float = 6.0) -> int | None:
    with open(evals, newline="") as fh:
        for row in csv.DictReader(fh):
            if float(row["laps_mean"]) >= min_laps and float(row["crash_rate"]) == 0.0:
                return int(row["timesteps"])
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    exp = Path(sys.argv[1])
    scores = []
    for f in sorted(exp.glob("seed*/evals.csv")):
        s = first_clean(f)
        print(f"{f.parent.name}: {'never' if s is None else f'{s:,} steps'}")
        if s is not None:
            scores.append(s)
    if not scores:
        print("no seed reached the bar")
        return 1
    print(f"score: {sum(scores) / len(scores):,.0f} steps (mean of {len(scores)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
