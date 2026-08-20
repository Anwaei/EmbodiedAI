# Isolated uv environments

Each child directory is an independent uv project with its own `pyproject.toml`, Python
minor version, indexes, and `uv.lock`. These directories are intentionally not a uv workspace.

Runtime environments live outside Git:

| Project | Definition | Runtime path |
|---|---|---|
| dev | `env/dev` | `/root/autodl-tmp/EmbodiedAI/envs/dev` |
| isaac | `env/isaac` | `/root/autodl-tmp/EmbodiedAI/envs/isaac` |
| vla | `env/vla` | `/root/autodl-tmp/EmbodiedAI/envs/vla` |

Load path and cache policy explicitly in each new shell:

```bash
cd /root/projects/EmbodiedAI
source scripts/bootstrap/project_env.sh
```

Lock-only operations (Stage 3):

```bash
embodiedai_uv_lock dev
embodiedai_uv_lock isaac
embodiedai_uv_lock vla
embodiedai_uv_check dev
embodiedai_uv_check isaac
embodiedai_uv_check vla
```

Do not run `uv sync` until the corresponding installation stage is approved. When approved,
`embodiedai_uv_sync NAME` directs the environment to the data disk instead of creating a
repository-local `.venv`.
