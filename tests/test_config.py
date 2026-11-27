from pathlib import Path

import pytest

from racing_gym.algos import ALGOS
from racing_gym.config import config_from_dict, dump_toml, load_config, to_dict

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.algo in ALGOS
    cfg.env.car_config()


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="crsh"):
        config_from_dict({"env": {"reward": {"crsh": -1}}})
    with pytest.raises(ValueError, match="unknown car preset"):
        config_from_dict({"env": {"car": "tractor"}}).env.car_config()


def test_overrides_are_parsed_as_toml():
    cfg = load_config(None, ["env.reward.crash=-50", "env.track=hairpin", "train.seeds=[3, 4]",
                             "env.car.grip=1.2", "algo_kwargs.learning_rate=1e-4"])
    assert cfg.env.reward.crash == -50.0
    assert cfg.env.track == "hairpin"
    assert cfg.train.seeds == (3, 4)
    assert cfg.env.car_config().grip == 1.2
    assert cfg.algo_kwargs["learning_rate"] == 1e-4


def test_dump_and_reload_round_trip(tmp_path):
    cfg = load_config(CONFIGS[0], ["env.car.grip=1.1", "algo_kwargs.policy_kwargs.net_arch=[32, 32]"])
    f = tmp_path / "c.toml"
    f.write_text(dump_toml(to_dict(cfg)))
    assert load_config(f) == cfg
