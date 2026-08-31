# Scripts

- `bootstrap/`: explicit shell setup and environment lifecycle helpers.
- `preflight/`: read-only machine and dependency checks.
- `sim/`: simulation and demonstration entry points.
- `data/`: validation and conversion entry points.
- `train/`: VLA, standalone PPO, and residual-policy training entry points. The standalone PPO
  entry supports fixed/distribution geometry, periodic checkpoints, and exact checkpoint resume.
- `evaluate/`: policy and robustness evaluation entry points. The PPO evaluator compares zero,
  random, and checkpoint controllers on the frozen Stage 8 scenario manifest.
