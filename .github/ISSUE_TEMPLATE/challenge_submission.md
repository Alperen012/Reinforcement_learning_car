---
name: Challenge submission
about: Submit a score for one of the challenges in CHALLENGES.md (no git needed)
title: "[submission] <challenge>: <your score>"
labels: challenge
---

**Challenge:** <!-- rocket-hairpin / fastest-learner / generalist -->

**Method:** <!-- algorithm and what you changed, e.g. "SAC, crash penalty -50, 400k steps" -->

**Number of training seeds:** <!-- at least 3 for a ranked entry -->

**Score output:** paste the output of the scoring command from CHALLENGES.md, unchanged.

```
paste here
```

**Config:** paste your config (or the `load_config([...])` overrides you used in Colab).

```toml
paste here
```

**Agent file:** rename `my_agent.js` to `my_agent.txt` (GitHub does not accept `.js` attachments)
and drag it into this box. The playground loads the `.txt` file as it is.

**Anything you learned?** <!-- optional: what worked, what did not. This is the most useful part for others. -->
