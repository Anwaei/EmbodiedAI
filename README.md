# EmbodiedAI

Simulation-based language-conditioned robotic manipulation using Isaac Sim / Isaac Lab,
LeRobot / SmolVLA, residual reinforcement learning, and a ROS 2 deployment boundary.

The repository contains code, configuration, documentation, tests, and reproducible lock
files only. Large environments, caches, datasets, models, checkpoints, simulator assets,
runs, and temporary files live under `/root/autodl-tmp/EmbodiedAI`.

## Environment setup

```bash
cd /root/projects/EmbodiedAI
source scripts/bootstrap/project_env.sh
embodiedai_uv_lock dev
embodiedai_uv_lock isaac
embodiedai_uv_lock vla
```

See `docs/ENVIRONMENT.md` and `env/README.md` before syncing or installing an environment.
Stages 4 and later require separate approval.
