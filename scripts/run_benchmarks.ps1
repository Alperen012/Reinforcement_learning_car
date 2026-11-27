# Reproduces every number in docs/results.md. Takes ~1-2 h on an 8-core laptop CPU.
# Linux/macOS: the same commands work in bash (replace the backtick line breaks).
$ErrorActionPreference = "Stop"
foreach ($cfg in "quickstart", "ppo_classic", "a2c_classic", "dqn_classic", "sac_classic", "td3_classic", "ppo_generalize") {
    racing-gym train -c "configs/$cfg.toml"
}
