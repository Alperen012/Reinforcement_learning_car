"""The browser playground (web/js/sim.js) must simulate exactly what the Python environment simulates.

Runs a scripted episode in both implementations through Node.js and compares
tracks, car states, observations and rewards step by step. Skipped when
Node.js is not installed.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from racing_gym import EnvConfig, RaceEnv, make_track

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

JS = r"""
const RG = require(process.argv[1]);
const spec = JSON.parse(process.argv[2]);
const out = {};
for (const t of spec.tracks) {
  const tr = RG.makeTrack(t);
  out[t] = {length: tr.length, n: tr.centerline.length, mask_sum: tr.mask.reduce((a, b) => a + b, 0),
            c0: tr.centerline[0], c100: tr.centerline[100]};
}
const env = new RG.RaceEnv({track: spec.episode_track, car: spec.car, random_start: false});
let obs = env.reset({startIndex: spec.start});
const traj = [];
for (const a of spec.actions) {
  const r = env.step(a);
  traj.push({x: env.car.x, y: env.car.y, yaw: env.car.yaw, speed: env.car.speed, obs: r.obs, reward: r.reward,
             done: r.terminated || r.truncated});
  if (r.terminated || r.truncated) break;
}
out.traj = traj;
console.log(JSON.stringify(out));
"""


def _actions(track="classic", car="balanced", n=700):
    """Heuristic driving plus a growing wiggle: laps the track for a while, then crashes."""
    from racing_gym.evaluation import heuristic_policy

    env = RaceEnv(EnvConfig(track=track, car=car, random_start=False))
    obs, _ = env.reset(seed=0, options={"start_index": 37})
    pilot, actions = heuristic_policy(env), []
    for i in range(n):
        steer, throttle = pilot(obs)
        a = [float(steer) + math.sin(i / 7.0) * i / 900.0, float(throttle) if i % 120 < 100 else -0.6]
        actions.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return actions


# Every car preset, on a preset track and on a procedural one.
@pytest.mark.parametrize("track,car", [("classic", "balanced"), ("random:7", "grippy"), ("oval", "rocket")])
def test_python_and_browser_simulations_match(track, car):
    tracks = ["oval", "classic", "hairpin", "snake", "random:0", "random:7", "random:123"]
    spec = {"tracks": tracks, "episode_track": track, "car": car, "start": 37, "actions": _actions(track, car)}
    res = subprocess.run([NODE, "-e", JS, str(ROOT / "web" / "js" / "sim.js"), json.dumps(spec)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)

    for t in tracks:
        py = make_track(t)
        assert js[t]["n"] == len(py.centerline), t
        assert js[t]["length"] == pytest.approx(py.length, abs=1e-6), t
        assert js[t]["mask_sum"] == int(py.mask.sum()), t
        np.testing.assert_allclose(js[t]["c100"], py.centerline[100], atol=1e-6)

    env = RaceEnv(EnvConfig(track=track, car=car, random_start=False))
    env.reset(seed=0, options={"start_index": 37})
    for i, (a, j) in enumerate(zip(spec["actions"], js["traj"])):
        obs, r, term, trunc, _ = env.step(a)
        assert (env.car.x, env.car.y) == (pytest.approx(j["x"], abs=1e-6), pytest.approx(j["y"], abs=1e-6)), i
        assert env.car.speed == pytest.approx(j["speed"], abs=1e-9), i
        np.testing.assert_allclose(obs, j["obs"], atol=1e-5, err_msg=f"step {i}")
        assert r == pytest.approx(j["reward"], abs=1e-6), i
        assert (term or trunc) == j["done"], i
        if term or trunc:
            break
    assert len(js["traj"]) == i + 1
    assert len(js["traj"]) > 100, "scripted episode too short to be a meaningful comparison"
