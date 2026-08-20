#!/usr/bin/env bash

# Install the locked Isaac Lab extensions in the editable layout required by
# their Omniverse extension metadata. Dependency resolution remains owned by
# env/isaac/uv.lock; this script only replaces the four locked Git wheels with
# editable installs from the matching, immutable source checkout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

readonly ISAACLAB_COMMIT="37ddf626871758333d6ed89cf64ad702aef127d0"
readonly ISAACLAB_SOURCE="$EMBODIEDAI_DATA/vendor/IsaacLab"
readonly ISAAC_PYTHON="$EMBODIEDAI_ENVS/isaac/bin/python"
export ISAACLAB_COMMIT

if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "Isaac environment is missing: $ISAAC_PYTHON" >&2
    echo "Run embodiedai_uv_sync isaac before this script." >&2
    exit 2
fi

if [[ ! -d "$ISAACLAB_SOURCE/.git" ]]; then
    echo "Isaac Lab checkout is missing: $ISAACLAB_SOURCE" >&2
    exit 2
fi

actual_commit="$(git -C "$ISAACLAB_SOURCE" rev-parse HEAD)"
if [[ "$actual_commit" != "$ISAACLAB_COMMIT" ]]; then
    echo "Isaac Lab checkout mismatch: expected $ISAACLAB_COMMIT, got $actual_commit" >&2
    exit 2
fi

if [[ -n "$(git -C "$ISAACLAB_SOURCE" status --porcelain --untracked-files=all)" ]]; then
    echo "Isaac Lab checkout must be clean before installation: $ISAACLAB_SOURCE" >&2
    exit 2
fi

embodiedai_uv_check isaac

uv pip install \
    --python "$ISAAC_PYTHON" \
    --no-deps \
    --editable "$ISAACLAB_SOURCE/source/isaaclab" \
    --editable "$ISAACLAB_SOURCE/source/isaaclab_assets" \
    --editable "$ISAACLAB_SOURCE/source/isaaclab_tasks" \
    --editable "$ISAACLAB_SOURCE/source/isaaclab_rl"

"$ISAAC_PYTHON" - <<'PY'
import os
from importlib import metadata
from pathlib import Path

expected = {
    "isaaclab": "0.54.2",
    "isaaclab-assets": "0.2.4",
    "isaaclab-tasks": "0.11.12",
    "isaaclab-rl": "0.4.7",
}
source_root = Path("/root/autodl-tmp/EmbodiedAI/vendor/IsaacLab/source").resolve()

for distribution, version in expected.items():
    installed = metadata.distribution(distribution)
    if installed.version != version:
        raise RuntimeError(f"{distribution}: expected {version}, got {installed.version}")

    direct_url = installed.read_text("direct_url.json") or ""
    if '"editable":true' not in direct_url.replace(" ", ""):
        raise RuntimeError(f"{distribution}: install is not editable")
    if source_root.as_uri() not in direct_url:
        raise RuntimeError(f"{distribution}: editable source is outside {source_root}")

print("ISAACLAB_EDITABLE_OK", f"commit={os.environ['ISAACLAB_COMMIT']}")
PY
