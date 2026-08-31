# Configuration

- `sim/`: scenes, robots, sensors, controllers, and randomization.
- `data/`: recording, validation, and export configuration.
- `policy/`: SmolVLA training and inference configuration.
- `rl/`: standalone PPO and later bounded residual PPO configuration. The v1 standalone profile is
  retained as the initial reward-diagnostic record; v2 is the active reviewed profile and writes
  to a separate external run root.
- `evaluation/`: task suites, seeds, and robustness sweeps.
- `omniverse/`: Omniverse global path configuration used by the explicit project shell.
