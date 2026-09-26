# Stage 3 locks and Stage 4/5 runtime report

Date: 2026-08-21 (Asia/Shanghai)

Scope: initial Stage 3 lock resolution, the explicitly authorized pre-Stage-4 Isaac Lab lock
amendment, and the approved Stage 4 Isaac and Stage 5 VLA installation/runtime validations.

## Resolved environments

| Project | Resolver Python | Reviewed direct result | Locked packages | Index policy |
|---|---:|---|---:|---|
| `env/dev` | 3.12.14 | mypy 1.20.2, pytest 9.1.1, pytest-cov 7.1.0, pytest-timeout 2.4.0, ruff 0.16.3 | 15 | Default PyPI; no CUDA or simulator dependencies |
| `env/isaac` | 3.11.16 | Isaac Sim 5.1.0.0 (`all`, `extscache`); Isaac Lab v2.3.2 core 0.54.2, assets 0.2.4, tasks 0.11.12, RL 0.4.7 (`rsl-rl`); PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, torchaudio 2.7.0+cu128 | 220 | Isaac Sim metapackage explicitly bound to NVIDIA's index; Isaac Lab packages bound to one immutable Git commit/subdirectories; Torch family explicitly bound to the PyTorch cu128 index |
| `env/vla` | 3.12.14 | LeRobot 0.6.0 (`peft`, `smolvla`, `training`), NumPy 2.2.6, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128, TorchCodec 0.7.0+cu128 | 107 | Torch family and TorchCodec explicitly bound to the PyTorch cu128 index |

The LeRobot extras resolve the requested training stack, including Accelerate, Datasets,
PyArrow, video support, Weights & Biases, PEFT, and the SmolVLA-specific dependencies. Each
environment is an independent uv project rather than a member of a shared workspace.

The Isaac Lab source is pinned to tag `v2.3.2`, commit
`37ddf626871758333d6ed89cf64ad702aef127d0`. The same detached source checkout is stored at
`/root/autodl-tmp/EmbodiedAI/vendor/IsaacLab` for official scripts and examples. The lock resolves
the four required packages and all dependencies from that exact Git commit. At runtime those four
packages are reinstalled from the matching checkout with `--no-deps --editable`, because their
upstream Omniverse extension metadata requires the source-directory layout. `isaaclab_mimic` and
`isaaclab_contrib` are intentionally deferred.

## Resolver conflicts and dispositions

The first Linux resolution for Isaac Sim failed because `isaacsim-core==5.1.0.0` declares
`pywin32==306` without a Linux-excluding marker. `pywin32` publishes no Linux x86_64 wheel,
so uv correctly reported the graph as unsatisfiable.

`env/isaac/pyproject.toml` contains this platform-scoped metadata correction:

```toml
[tool.uv]
override-dependencies = [
    "pywin32==306; sys_platform == 'win32'",
    "starlette==0.45.3",
]
```

This preserves the upstream requirement on Windows and removes only the impossible Windows
runtime from the Linux target. It does not change any requested Isaac Sim, Python, PyTorch,
CUDA, or torchvision pin.

Adding Isaac Lab exposed two more upstream metadata defects:

- `flatdict==4.0.1` imports `pkg_resources` while preparing its wheel but does not declare
  setuptools as a build dependency. The uv definition supplies `setuptools<82` only to that
  isolated build through `[tool.uv.extra-build-dependencies]`; it is not a runtime override.
- Isaac Lab core pins `starlette==0.49.1`, while Isaac Sim kernel pins FastAPI 0.115.7, which
  requires Starlette `>=0.40,<0.46`. No Starlette import was found under the pinned Isaac Lab
  core source. The combined lock therefore selects `starlette==0.45.3`, satisfying Isaac Sim's
  FastAPI. Stage 4 explicitly covered an ASGI request, Isaac/Lab execution, and private WebRTC
  livestream startup; the override is now runtime-validated.

After these reviewed corrections, all three locks resolve and validate. `uv pip check` in the
installed Isaac environment still reports the overridden Starlette declaration as a metadata
incompatibility; it is intentional and covered by the Stage 4 runtime tests rather than an
unreviewed resolver conflict.

## Index and ABI review

- The two CUDA stacks are isolated: Isaac resolves the `2.7.0/0.22.0/2.7.0` cu128 family;
  VLA resolves the `2.8.0/0.23.0/2.8.0` cu128 family plus TorchCodec `0.7.0+cu128`.
- The Isaac Lab constraints resolve Isaac NumPy to 1.26.0 and packaging to 23.0, consistent
  with the pinned Isaac Sim/Lab graph. These versions do not affect the independent VLA lock.
- The NVIDIA Isaac Sim index is explicit, so unrelated packages cannot be selected from it.
  Isaac component packages published on the default Python index remain transitive
  dependencies of the NVIDIA metapackage.
- LeRobot's NumPy constraint resolves to 2.2.6, below the required 2.3 boundary.
- No ROS 2 package is part of any uv lock; the planned Humble/system-Python boundary remains
  intact.
- No system CUDA or cuDNN path is exported. The lock files select CUDA wheels by their
  explicit package index.

## Validation

The following validation completed successfully for `dev`, `isaac`, and `vla`:

```bash
uv lock --check --project env/NAME
uv tree --locked --project env/NAME
```

Lock SHA-256 values at review time:

```text
8b4ed92145199d5160c6914940b2b3619be089b0926d3242a3ada943f4b09165  env/dev/uv.lock
997119551e29b5a014feb2fdb872e2d1b1bb2a3420f039dcee79a2c351f61bf1  env/isaac/uv.lock
136754fe057383cfff64a59dc7bb96668f83ad94004a09760ec70a496847d873  env/vla/uv.lock
```

The `dev` runtime directory remains an unsynchronized seed environment. The approved Stage 4 sync
installed 220 packages into `/root/autodl-tmp/EmbodiedAI/envs/isaac`; its final inventory is
`/root/autodl-tmp/EmbodiedAI/runs/bootstrap/stage4-packages-final.txt`. The approved Stage 5 sync
installed 107 packages into `/root/autodl-tmp/EmbodiedAI/envs/vla`; its final inventory is
`/root/autodl-tmp/EmbodiedAI/runs/bootstrap/stage5-packages-final.txt`.

## Stage 4 installation-layout correction and validation

The ordinary Git-subdirectory wheels place each Python package directly in `site-packages`, while
Isaac Lab v2.3.2 computes its extension directory as the package's parent and reads
`config/extension.toml` from there. Consequently, the locked wheel form cannot import the full
Isaac Lab stack. This is an upstream packaging/layout behavior, not a version-resolution failure;
Isaac Lab's official `isaaclab.sh --install` also installs extensions editable.

`scripts/bootstrap/install_isaaclab_editable.sh` implements the constrained project form. It
requires a clean checkout at commit `37ddf626871758333d6ed89cf64ad702aef127d0`, runs
`uv lock --check`, and uses `uv pip install --no-deps --editable` for only `isaaclab`,
`isaaclab-assets`, `isaaclab-tasks`, and `isaaclab-rl`. It then verifies exact versions and
editable provenance. After any future `embodiedai_uv_sync isaac`, run:

```bash
embodiedai_isaaclab_install
```

Stage 4 runtime coverage passed for the official compatibility checker, PyTorch CUDA 12.8 on
`sm_120`, Vulkan, PhysX, offscreen RTX rendering/camera capture, a four-environment Franka lift
task, one bounded RSL-RL training iteration, the FastAPI/Starlette override, and private WebRTC
service startup. Logs are under `/root/autodl-tmp/EmbodiedAI/runs/bootstrap/`.

## Stage 5 VLA installation and validation

`embodiedai_uv_sync vla` synchronized the unchanged reviewed lock into the external Python 3.12.14
environment. `uv pip check` passes without exceptions. The runtime inventory confirms LeRobot
0.6.0, torch/torchvision/torchaudio 2.8.0/0.23.0/2.8.0 cu128, TorchCodec 0.7.0+cu128,
NumPy 2.2.6, Transformers 5.5.4, PEFT 0.20.0, Accelerate 1.14.0, Datasets 4.8.5,
PyArrow 25.0.1, PyAV 15.1.0, and OpenCV headless 4.13.0.92.

PyTorch reports its wheel-bundled CUDA 12.8 and cuDNN 9.10.2, sees the RTX 5090 as capability
`(12, 0)`, includes `sm_120`, and completes a CUDA matrix operation. LeRobot and SmolVLA imports
also pass. The Python lock was not changed during installation.

TorchCodec loaded only after installing Ubuntu's user-space `ffmpeg`
`7:4.4.2-0ubuntu0.22.04.1`, which supplies the previously missing `libavdevice.so.58` and related
FFmpeg 4 libraries. This is an OS media-runtime prerequisite, not an ad-hoc Python dependency or
CUDA/driver change. A generated four-frame 64x64 H.264 fixture decodes on CPU; hardware decoding
is not claimed by this gate.

The reviewed model artifacts are external to Git:

| Artifact | Immutable revision | Local content | Integrity |
|---|---|---|---|
| `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` | Full 906,712,520-byte checkpoint plus policy/config metadata | `model.safetensors` SHA-256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | Configuration and tokenizer only | Revision-pinned Hub snapshot; separate 2 GB VLM weights are unnecessary because the SmolVLA checkpoint is complete |

Forced-offline local loading passed a finite SmolVLA CUDA inference. A bounded LeRobot PEFT LoRA
step then updated all 74 adapter tensors (92,832 trainable of 450,139,008 total parameters) with
finite loss and gradients. External `nvidia-smi` sampling peaked at 1,861 MiB for inference and
1,963 MiB for training, well within the 32,607 MiB allocation. A three-frame/one-episode LeRobot
dataset also passed write, `finalize()`, reload, and value checks. Reproduction scripts are
`scripts/train/smolvla_stage5_smoke.py` and `scripts/data/lerobot_dataset_smoke.py`; complete logs
remain under `/root/autodl-tmp/EmbodiedAI/runs/bootstrap/`.

## Network note

Direct package endpoints were intermittently slow while resolving and syncing. Seven exact
lock-addressed wheels were downloaded through the Tsinghua PyPI mirror, verified against lock
sizes/SHA-256 hashes, and preinstalled from an external wheelhouse before a normal
`uv sync --locked` converged. NVIDIA redirected package hosts were routed directly where needed.
`/etc/network_turbo` was used temporarily for GitHub and initial Isaac asset population. During
Stage 5, a direct VLA sync stalled after 758 seconds; the same unchanged lock synchronized through
the temporary proxy in 2,105 seconds, and the reviewed Hugging Face files were then downloaded by
immutable revision. All proxy variables were unset afterward. No mirror or proxy setting was
written into a lock or global shell configuration.
