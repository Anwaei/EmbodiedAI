# External storage policy

No large state is stored in this Git repository. The canonical data root is:

```text
/root/autodl-tmp/EmbodiedAI/
├── artifacts/
├── assets/
├── caches/
├── checkpoints/
├── datasets/
├── envs/
├── models/
├── python/
├── runs/
├── tmp/
├── tools/
└── vendor/
```

`scripts/bootstrap/project_env.sh` exports the corresponding variables only when explicitly
sourced. It does not modify `.bashrc`, Miniconda base, CUDA/cuDNN library paths, the NVIDIA
driver, or ROS 2 global state.
