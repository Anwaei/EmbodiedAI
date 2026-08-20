#!/usr/bin/env bash

# Source this file from a project shell. It intentionally does not modify global shell files,
# Miniconda base, CUDA/cuDNN library paths, the NVIDIA driver, or ROS 2 state.

export EMBODIEDAI_REPO="/root/projects/EmbodiedAI"
export EMBODIEDAI_DATA="/root/autodl-tmp/EmbodiedAI"
export EMBODIEDAI_ENVS="$EMBODIEDAI_DATA/envs"
export EMBODIEDAI_DATASETS="$EMBODIEDAI_DATA/datasets"
export EMBODIEDAI_CHECKPOINTS="$EMBODIEDAI_DATA/checkpoints"
export EMBODIEDAI_MODELS="$EMBODIEDAI_DATA/models"
export EMBODIEDAI_RUNS="$EMBODIEDAI_DATA/runs"
export EMBODIEDAI_ASSETS="$EMBODIEDAI_DATA/assets"
export EMBODIEDAI_ARTIFACTS="$EMBODIEDAI_DATA/artifacts"

# Standalone tools and uv-managed Python installations.
export PATH="$EMBODIEDAI_DATA/tools/bin:$PATH"
export UV_CACHE_DIR="$EMBODIEDAI_DATA/caches/uv"
export UV_PYTHON_INSTALL_DIR="$EMBODIEDAI_DATA/python"
export UV_TOOL_DIR="$EMBODIEDAI_DATA/tools"
export UV_TOOL_BIN_DIR="$EMBODIEDAI_DATA/tools/bin"

# Package, model, dataset, compilation, and application caches.
export PIP_CACHE_DIR="$EMBODIEDAI_DATA/caches/pip"
export XDG_CACHE_HOME="$EMBODIEDAI_DATA/caches/xdg"
export XDG_DATA_HOME="$EMBODIEDAI_DATA/caches/xdg-data"
export HF_HOME="$EMBODIEDAI_DATA/caches/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$EMBODIEDAI_DATA/caches/torch"
export TORCH_EXTENSIONS_DIR="$EMBODIEDAI_DATA/caches/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$EMBODIEDAI_DATA/caches/torch_inductor"
export TRITON_CACHE_DIR="$EMBODIEDAI_DATA/caches/triton"
export CUDA_CACHE_PATH="$EMBODIEDAI_DATA/caches/cuda"
export NUMBA_CACHE_DIR="$EMBODIEDAI_DATA/caches/numba"
export CUPY_CACHE_DIR="$EMBODIEDAI_DATA/caches/cupy"
export MPLCONFIGDIR="$EMBODIEDAI_DATA/caches/matplotlib"
export PYTHONPYCACHEPREFIX="$EMBODIEDAI_DATA/caches/pycache"
export WANDB_CACHE_DIR="$EMBODIEDAI_DATA/caches/wandb"
export WANDB_DATA_DIR="$EMBODIEDAI_DATA/caches/wandb-data"
export WANDB_DIR="$EMBODIEDAI_RUNS/wandb"
export WANDB_ARTIFACT_DIR="$EMBODIEDAI_ARTIFACTS/wandb"

# Omniverse reads paths from the committed config in this directory.
export OMNI_CONFIG_PATH="$EMBODIEDAI_REPO/configs/omniverse"

# Temporary state must not fill the 30 GiB root filesystem.
export TMPDIR="$EMBODIEDAI_DATA/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# The audited GPU-mode quota is 25 CPUs. No-GPU mode may be smaller, so default
# conservatively and allow an explicit positive override.
_EMBODIEDAI_THREAD_COUNT="${EMBODIEDAI_CPU_THREADS:-8}"
case "$_EMBODIEDAI_THREAD_COUNT" in
    ''|*[!0-9]*|0)
        echo "EMBODIEDAI_CPU_THREADS must be a positive integer" >&2
        return 2 2>/dev/null || exit 2
        ;;
    *)
        export OMP_NUM_THREADS="$_EMBODIEDAI_THREAD_COUNT"
        export MKL_NUM_THREADS="$_EMBODIEDAI_THREAD_COUNT"
        ;;
esac
unset _EMBODIEDAI_THREAD_COUNT

_embodiedai_env_project() {
    case "$1" in
        dev|isaac|vla) printf '%s/env/%s\n' "$EMBODIEDAI_REPO" "$1" ;;
        *) echo "environment must be one of: dev, isaac, vla" >&2; return 2 ;;
    esac
}

_embodiedai_env_runtime() {
    case "$1" in
        dev|isaac|vla) printf '%s/%s\n' "$EMBODIEDAI_ENVS" "$1" ;;
        *) echo "environment must be one of: dev, isaac, vla" >&2; return 2 ;;
    esac
}

embodiedai_uv_lock() {
    local project
    project="$(_embodiedai_env_project "$1")" || return
    uv lock --project "$project"
}

embodiedai_uv_check() {
    local project
    project="$(_embodiedai_env_project "$1")" || return
    uv lock --check --project "$project"
}

embodiedai_uv_sync() {
    local project runtime
    project="$(_embodiedai_env_project "$1")" || return
    runtime="$(_embodiedai_env_runtime "$1")" || return
    UV_PROJECT_ENVIRONMENT="$runtime" uv sync --project "$project" --locked
}

embodiedai_activate() {
    local runtime
    runtime="$(_embodiedai_env_runtime "$1")" || return
    # shellcheck disable=SC1090
    source "$runtime/bin/activate"
}
