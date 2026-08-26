# Environment and Bootstrap Plan

Status: **Stages 1-5 completed; Stage 6 steps 1-6 completed; Stages 7-8 not approved**
Audit date: 2026-08-17 (hardware), updated 2026-08-25 for Stage 6 step 6
Target host: `embodied-5090` / `/root/projects/EmbodiedAI`

## 1. Constraints and decision summary

- The project targets language-conditioned manipulation in Isaac Sim / Isaac Lab, demonstration generation, SmolVLA fine-tuning, optional residual RL, and a ROS 2 deployment interface.
- Simulation, VLA training, and ROS 2 must remain isolated because their supported Python and package stacks differ.
- Do not mutate Miniconda `base`, install system-wide Python packages, change the NVIDIA driver, or start a heavy-environment installation without approval for its later stage.
- Recommended stable baseline:
  - Isaac Sim `5.1.0` + Isaac Lab `v2.3.2`, Python `3.11`, PyTorch `2.7.0+cu128`.
  - LeRobot `v0.6.0` with the `smolvla` feature set, Python `3.12`, PyTorch `2.8.0+cu128`, TorchCodec `0.7.x`, NumPy `2.2.x`.
  - ROS 2 Humble on Ubuntu 22.04 using its native Python `3.10`, communicating with Isaac Sim through DDS and the Isaac Sim ROS 2 bridge.
- The restarted container now passes the static hardware/resource preflight: one usable RTX 5090, a 25-CPU cgroup quota, 90 GiB RAM, 45 GiB shared memory, and 550 GiB on the data disk. The 30 GiB root filesystem remains intentionally code-only.
- Stage 0 was approved. Repository/bootstrap and initial lock-only work in Stages 2-3 completed on 2026-08-20. Stages 4 and 5 were subsequently approved and completed in GPU-mode allocations. Stage 6 steps 1-6 established the contracts, task skeleton, deterministic reset/evaluation, immutable episode path, and instruction-bearing state-machine expert generation.
- Stage 2/3 execution occurred while the leased server was in no-GPU mode; this was acceptable for lock-only work and no GPU test was attempted. The GPU-mode hardware audit above remains the acceptance baseline for the later runtime stages.

## 2. Remote-machine audit

### 2.1 Observed state

| Area | Observation | Assessment |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy), x86_64; kernel `5.15.0-119-generic`; glibc `2.35` | Suitable for the proposed Isaac Sim 5.1 / ROS 2 Humble baseline. |
| Container | Overlay root filesystem, AutoDL container hostname, no Docker/Podman/Apptainer CLI | Treat the leased instance itself as the outer isolation boundary; use Python environments inside it. |
| CPU | Host topology exposes 208 CPUs; cgroup `cpu.max=2500000 100000` | Effective quota is **25 CPUs**. `nproc` alone is misleading because the cpuset still exposes CPUs 0-207. Resource gate passes. |
| Memory | Host reports 754 GiB; cgroup `memory.max=96636764160`; no swap | Effective limit is **90 GiB**, with about 370 MiB in use during audit. Resource gate passes. |
| Shared memory | `/dev/shm` is a 45 GiB tmpfs | Adequate starting point for data-loading and simulator IPC; monitor during vectorized workloads. |
| GPU | One NVIDIA GeForce RTX 5090, current allocation UUID `GPU-50163d07-93c3-b029-7a85-5404818875ad`, compute capability 12.0, 32,607 MiB VRAM | GPU gate passes. It is currently physical `/dev/nvidia5` but logical CUDA/NVML device 0; allocation restarts can change the UUID/device node, so code must select logical devices rather than hard-code either value. |
| NVIDIA driver | `580.105.08`; `nvidia-smi` reports maximum supported CUDA `13.0` | Meets the Isaac Sim 5.1 recommendation (`>=580.65.06`) and LeRobot cu128 floor (`>=570.86`). No driver change is proposed. |
| `nvidia-smi` | Executable and healthy; idle GPU, 0 MiB allocated at audit, 575 W power limit | Identity, VRAM, driver, and runtime access are validated. |
| CUDA toolkit | `/usr/local/cuda -> /usr/local/cuda-12.8`; NVCC `12.8.93`; CUDA packages `12.8.1` | Appropriate for cu128 builds. NVCC works by absolute path but `/usr/local/cuda/bin` is not on `PATH`. Do not set a global CUDA path yet. |
| cuDNN | System packages `9.8.0.87` for CUDA 12; base PyTorch reports cuDNN `91002` from its wheel/runtime | Avoid mixing system cuDNN with wheel-bundled runtimes via global `LD_LIBRARY_PATH`. |
| Python | Miniconda base only: Python `3.12.3`, pip `24.0`; no `/usr/bin/python3` | Base must remain untouched. Dedicated Python 3.11 and 3.12 environments are required; ROS 2 later needs Ubuntu's system Python 3.10 packages. |
| Existing ML packages | Base has `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `numpy 2.3.2`; CUDA device count 1 | A CUDA tensor operation succeeds, capability `(12, 0)` is detected, and the wheel contains `sm_120`. Still do not reuse base: NumPy is outside LeRobot's `<2.3` bound and Python is wrong for Isaac Sim 5.x. |
| ROS 2 | No `/opt/ros`, no `ros2`, no ROS environment variables/packages | ROS 2 is not installed. |
| Graphics/media | No `DISPLAY`, Wayland, or Xorg; Stage 4 added `libvulkan1`, `vulkan-tools`, `libglu1-mesa`, `libegl1`, and `libxt6` user-space prerequisites plus diagnostic `strace`; Stage 5 added Ubuntu's user-space `ffmpeg` package and its runtime libraries | Vulkan sees only the logical RTX 5090. Headless RTX rendering and short-lived private WebRTC service startup pass. TorchCodec 0.7.0 successfully decodes a generated H.264 file on CPU through FFmpeg 4.4.2. |
| System disk | Overlay `/`: 30 GiB total, 29 GiB free after Stage 5 | Heavy state remained off the root filesystem. |
| Data disk | `/root/autodl-tmp`: 550 GiB total, 490 GiB free after Stage 5 | The VLA environment is 7.2 GiB, the shared uv cache is 32 GiB, and reviewed VLA model files are 870 MiB. Continue using this disk for all large state. |
| Thread environment | `OMP_NUM_THREADS=0` and `MKL_NUM_THREADS=0` | Invalid for libgomp (confirmed warning). Project launch wrappers must unset them or set a positive value derived from the 25-CPU quota before running Python workloads. |
| Repository | Stage 2 began from `main` at `98a7d8759b47`; `docs/PROJECT_CONTEXT.md` and this file were committed | Stage 2-5 changes remain an uncommitted review set. Environment locks, runtime documentation, reproducible smoke scripts, and the repository skeleton are present. |

### 2.2 Preflight gate status

| Gate | Required | Re-audit result | Status |
|---|---|---|---|
| GPU allocation | One RTX GPU with control/UVM device access | RTX 5090, logical device 0; `/dev/nvidia3`, `nvidiactl`, and UVM nodes present | Pass |
| Driver/runtime | Healthy `nvidia-smi`; compatible driver | 580.105.08; CUDA access and a PyTorch CUDA tensor operation succeed | Pass |
| CPU | >=8 effective CPUs | 25-CPU cgroup quota | Pass |
| RAM | >=32 GiB; 64 GiB preferred | 90 GiB cgroup limit | Pass |
| Shared memory | Sufficient for headless simulator/data workers | 45 GiB `/dev/shm` | Pass |
| Data disk | >=350 GiB free | About 490 GiB free at `/root/autodl-tmp` after Stage 5 | Pass |
| System disk policy | Keep heavy state off 30 GiB root | Environments, tools, caches, datasets, models, checkpoints, artifacts, runs, and temporary state route to `/root/autodl-tmp/EmbodiedAI` | Pass (Stage 2) |
| Thread variables | Positive/valid BLAS/OpenMP settings | Explicit project wrapper assigns positive defaults (`8`) and rejects invalid overrides | Pass (Stage 2) |
| Isaac compatibility check | Official checker plus headless smoke test | Official checker passed; Vulkan, PhysX, RTX camera, Isaac Lab Franka/vectorization, RSL-RL, and WebRTC startup tests passed | Pass (Stage 4) |
| VLA runtime check | Locked install plus imports, CUDA, media, dataset, inference, and PEFT smoke tests | PyTorch cu128 runs on `sm_120`; TorchCodec CPU decode, LeRobot dataset round trip, offline SmolVLA inference, and one LoRA optimizer step passed | Pass (Stage 5) |

The former no-GPU/low-resource blocker was resolved by the GPU-mode audit. Stages 4 and 5 then completed in GPU allocations. Stage 6 steps 1-6 were later completed under explicit user direction. LeRobot conversion and VLA training remain outside the completed Stage 6 scope.

## 3. Proposed repository structure

```text
EmbodiedAI/
├── AGENTS.md
├── README.md
├── pyproject.toml                 # lightweight shared code/dev tooling only
├── .gitignore
├── configs/
│   ├── sim/                       # scenes, robots, sensors, randomization
│   ├── data/                      # demonstration/export schemas
│   ├── policy/                    # SmolVLA train/eval configs
│   ├── rl/                        # residual PPO/SAC configs
│   └── evaluation/                # task suites and robustness sweeps
├── docs/
│   ├── PROJECT_CONTEXT.md         # existing project goals and scope
│   ├── ARCHITECTURE.md
│   ├── ENVIRONMENT.md             # this document and future change log
│   ├── ROADMAP.md
│   └── DATA_FORMAT.md
├── env/
│   ├── README.md                  # activation rules and cache locations
│   ├── isaac/                     # independent pyproject/uv.lock, Python 3.11
│   ├── vla/                       # independent pyproject/uv.lock, Python 3.12
│   └── dev/                       # CPU-only lint/test environment
├── src/embodied_ai/
│   ├── contracts/                 # observations/actions/episode schemas
│   ├── sim/                       # Isaac Lab tasks and scene adapters
│   ├── data/                      # recording, conversion, validation
│   ├── policies/                  # LeRobot/SmolVLA adapters
│   ├── rl/                        # residual policy and reward components
│   ├── evaluation/                # metrics and robustness harness
│   └── utils/
├── ros2_ws/src/
│   ├── embodied_ai_interfaces/    # add custom interfaces only if needed
│   ├── embodied_ai_policy/        # external inference/control node
│   └── embodied_ai_bringup/       # launch/config package
├── scripts/
│   ├── preflight/
│   ├── bootstrap/
│   ├── sim/
│   ├── data/
│   ├── train/
│   └── evaluate/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
└── storage/README.md              # documents external paths; no large data
```

Design rules:

- `contracts/` contains dependency-light, versioned schemas shared at process/file boundaries; it must not import Isaac Sim, LeRobot, or ROS 2.
- Isaac-specific code stays in `sim/`; LeRobot-specific code stays behind adapters in `policies/`.
- Demonstrations are exported to an explicit, validated LeRobot-compatible dataset contract instead of importing LeRobot into the Isaac environment.
- Configuration is committed; datasets, checkpoints, simulator assets, caches, videos, and runs are not.
- Large state lives under `/root/autodl-tmp/EmbodiedAI/{assets,caches,datasets,checkpoints,runs,envs}`. Repository paths may point there through documented environment variables, not committed absolute symlinks.

## 4. Environment isolation strategy

### 4.1 Isolation boundaries

| Environment | Location (proposed) | Python | Owns | Must not contain |
|---|---|---:|---|---|
| `dev` | `/root/autodl-tmp/EmbodiedAI/envs/dev` | 3.12 | lint, formatting, unit tests for dependency-light code | Isaac Sim, CUDA PyTorch, ROS 2 |
| `isaac` | `/root/autodl-tmp/EmbodiedAI/envs/isaac` | 3.11 | Isaac Sim/Lab, simulator task code, demo generation, optional residual RL | LeRobot training stack, system ROS Python packages |
| `vla` | `/root/autodl-tmp/EmbodiedAI/envs/vla` | 3.12 | LeRobot, SmolVLA, dataset tooling, fine-tuning/evaluation | Isaac Sim/Lab, `rclpy` from system ROS |
| ROS 2 Humble | `/opt/ros/humble` + `ros2_ws/install` | system 3.10 | deployment nodes, messages, launch, DDS | Conda/uv Python runtimes and their shared libraries |

Stage 3 installed uv `0.12.5` at `/root/autodl-tmp/EmbodiedAI/tools/bin/uv` and uv-managed CPython `3.11.16` and `3.12.14` under the data disk. Each Python environment now has an independent project and lock file. `UV_CACHE_DIR`, Hugging Face, Torch/Triton/compiler, CUDA shader, Omniverse, W&B, Python bytecode, and temporary paths are redirected to `/root/autodl-tmp/EmbodiedAI` by the explicitly sourced project wrapper.

Do not activate Miniconda `base` when running project commands. Do not source ROS 2 in `.bashrc`; use explicit launch wrappers so ROS library paths cannot contaminate Isaac or VLA shells. Do not export system CUDA/cuDNN library directories globally. PyTorch wheels carry their CUDA runtime; `/usr/local/cuda-12.8` is used only for extensions that explicitly require NVCC.

### 4.2 Cross-environment communication

- **Simulation to training:** immutable episode directories plus a manifest and schema version; export/validate to LeRobot dataset format in a separate step.
- **Training to simulation:** model checkpoints plus a small policy metadata file defining observation normalization, action space, control rate, and model revision.
- **Online evaluation:** initially a narrow process boundary (local RPC or ROS 2 topics/services) rather than importing both stacks into one interpreter.
- **ROS 2:** use standard messages for the MVP (`sensor_msgs`, `geometry_msgs`, `trajectory_msgs`, etc.) so Isaac Sim can use its bundled Python 3.11 bridge while external Humble nodes use Python 3.10. If custom messages become necessary, build the interfaces for both ABIs as documented by Isaac Sim.

### 4.3 Reproducibility policy

- Pin released upstream tags and exact CUDA wheel index; never track `main`/`develop` for the MVP.
- Commit `pyproject.toml` and `uv.lock` per environment plus a machine-readable preflight report.
- Record `git rev-parse HEAD`, Python/package lock hash, driver, GPU UUID, and run config in every experiment manifest.
- Use `pip check`/`uv lock --check` and import smoke tests at each stage.
- Prefer rebuildable environments; do not patch packages in `site-packages`.

## 5. Compatibility matrix

### 5.1 Proposed stable baseline

| Layer | Proposed pin | Python | CUDA/PyTorch relationship | Host fit and notes |
|---|---|---:|---|---|
| Isaac Sim | `5.1.0.0` pip distribution | 3.11 | Uses the Isaac 5.x Python ABI; pair with cu128 PyTorch below | Ubuntu 22.04 and glibc 2.35 fit. Run headless first. |
| Isaac Lab | tag `v2.3.2` | 3.11 | Built for Isaac Sim 5.1; pin source tag/commit | Stable 2.x baseline. Prefer 2.3.2 over 2.3.1 because 2.3.2 is the current patch release. |
| PyTorch for Isaac | `torch 2.7.0+cu128`, `torchvision 0.22.0+cu128`, `torchaudio 2.7.0+cu128` | 3.11 | Isaac Lab 2.3.x's x86_64 pin; CUDA 12.8 adds Blackwell support | Driver 580.105.08 is above the Isaac 5.1 recommendation. Base PyTorch 2.8 has already verified `sm_120`; repeat with the pinned Isaac wheel during Stage 4. |
| LeRobot / SmolVLA | LeRobot tag `v0.6.0`, `smolvla` extra | >=3.12 (choose 3.12) | LeRobot allows PyTorch `>=2.7,<2.12` and NumPy `>=2.0,<2.3` | Keep separate from Isaac. Pin the released tag, not current main (`0.6.1` metadata). |
| PyTorch for VLA | `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `torchaudio 2.8.0+cu128`; TorchCodec `0.7.x`; NumPy `2.2.x` | 3.12 | Official cu128 wheels; TorchCodec 0.7 matches PyTorch 2.8 | Driver 580.105.08 exceeds LeRobot's documented cu128 floor 570.86. Validate NVDEC/FFmpeg separately; CPU decoding is acceptable initially. |
| ROS 2 | Humble binary packages for Ubuntu 22.04 | system 3.10 | No PyTorch/CUDA dependency in the ROS environment | Native distro match. External nodes communicate over DDS; Isaac uses its bundled Humble bridge libraries under Python 3.11. |

The host CUDA toolkit and a wheel's CUDA runtime need not be identical at the patch level. The critical constraints are a sufficiently new NVIDIA driver, a wheel that supports RTX 5090/Blackwell, and ABI-compatible versions within each isolated environment.

### 5.2 Deliberately deferred alternative

Isaac Sim 6.0/6.0.1 and Isaac Lab 3.0 beta/develop are not selected for the MVP. They move to Python 3.12 and newer PyTorch, but current Isaac Sim 6.0 requirements list Linux driver `595.58.03`, while this host has `580.105.08`; Isaac Lab 3.0 also carries active-development limitations and breaking API changes. Revisit after the 3.0 stable release and only with an approved driver/platform change.

### 5.3 Known incompatibilities to prevent

- Do not install Isaac Sim 5.x into the current Python 3.12 base; it requires Python 3.11.
- Do not install current LeRobot into the Isaac Python 3.11 environment; LeRobot v0.6 requires Python 3.12 or newer.
- Do not reuse base NumPy 2.3.2 for LeRobot; LeRobot requires NumPy `<2.3`.
- Do not source Ubuntu ROS 2 Humble's Python 3.10 environment before launching Isaac Sim's Python 3.11 runtime.
- Do not upgrade PyTorch independently inside the Isaac environment; Isaac Sim/Lab pins must be treated as a unit.
- Do not select the default PyPI CUDA variant implicitly. Record the `cu128` index explicitly in each lock.

## 6. Staged bootstrap plan

Stage 1's read-only hardware audit was completed after the user restarted the server in GPU mode. Stage 0 was subsequently approved, and Stages 2-3 completed on 2026-08-20. Stages 4 and 5 were then separately approved and completed. Stage 6 steps 1-6 are complete. Stages 7-8 remain unauthorized.

### GPU-mode execution policy for Stages 4 and 5

The earlier A/B split is cancelled. Although package download and extraction do not inherently
require a CUDA device, the observed no-GPU allocation exposed only about 0.5 CPU and 2 GiB RAM.
That is not a reliable operating envelope for these large environments. Installation and
validation will therefore run together in GPU mode so that the audited 25-CPU, 90-GiB RAM,
45-GiB shared-memory, RTX 5090 allocation remains stable throughout each stage.

Rules for both stages:

- Begin from a fresh GPU-mode shell and repeat GPU identity, driver, device-node, cgroup,
  shared-memory, and data-disk checks before any package synchronization.
- Verify the reviewed lock hash before installation, install with `--locked`, then record the
  resulting package inventory before runtime tests.
- Keep environments, source checkouts, caches, assets, models, datasets, logs, and temporary
  state below `/root/autodl-tmp/EmbodiedAI`.
- Do not modify Miniconda base, the NVIDIA driver, system CUDA/cuDNN, global library paths, or
  global ROS 2 state. A failed check returns to plan review instead of authorizing such changes.
- No-GPU mode remains suitable for documentation and lock review only; do not install or run
  the Isaac or VLA stacks there.

### Stage 0 — Review and freeze the plan (approved)

- Review version choices, repository layout, storage paths, and isolation boundaries.
- Decide whether the stable Isaac 5.1/Lab 2.3.2 path is accepted.
- Record approval in this document's change log.

Exit gate: explicit approval to begin provisioning/bootstrap.

Result: approved by the user before the 2026-08-20 Stage 2/3 execution.

### Stage 1 — Provision and re-audit hardware (static preflight complete)

- GPU-enabled allocation now provides one RTX 5090, 25 effective CPUs, 90 GiB RAM, and 45 GiB `/dev/shm`.
- `nvidia-smi`, device nodes, cgroup limits, disk, and a PyTorch `sm_120` CUDA tensor operation pass.
- Run the Isaac Sim Compatibility Checker during Stage 4, after the approved simulator tooling is installed.
- Do **not** change the NVIDIA driver unless a separate reviewed plan requires it.

Exit gate: static hardware preflight passes. Compatibility-checker and rendering gates remain in Stage 4.

### Stage 2 — Create repository skeleton and storage policy (completed 2026-08-20)

- Create the approved directories and add the missing architecture/roadmap/data-format documents. `PROJECT_CONTEXT.md` is already under `docs/`.
- Add `.gitignore` rules for environments, assets, datasets, caches, runs, checkpoints, and generated ROS build trees.
- Create `/root/autodl-tmp/EmbodiedAI/*` storage directories and document environment variables.
- Add launch wrappers that unset or assign positive values to `OMP_NUM_THREADS` and `MKL_NUM_THREADS` based on the 25-CPU quota.
- Add dependency-light schemas and a minimal test harness only.

Exit gate: clean repository, documented paths, no large files in Git.

Result: repository skeleton, missing core documents, ignore policy, dependency-light
contracts/tests, Omniverse path configuration, external storage tree, and explicit project
environment wrapper were created. Large runtime state is outside Git.

### Stage 3 — Bootstrap tooling and resolve locks (completed 2026-08-20)

- Install `uv` in user/data-disk scope, not system-wide.
- Create independent `dev`, `isaac`, and `vla` definitions with exact Python versions and indexes.
- Resolve and review lock files before installing the heavy environments.
- Produce a dependency report showing Python, PyTorch, CUDA wheel, NumPy, and known ABI pins.

Exit gate: lock review succeeds; no resolver conflicts.

Result: independent `dev`, `isaac`, and `vla` projects and lock files validate with
`uv lock --check`. The Isaac Linux metadata's unmarked Windows-only `pywin32==306`
dependency required a platform-scoped uv override; requested version pins were unchanged.
Before Stage 4, an explicitly authorized lock amendment added Isaac Lab v2.3.2 core, assets,
tasks, and RSL-RL packages from immutable commit
`37ddf626871758333d6ed89cf64ad702aef127d0`. Two additional upstream metadata defects required
reviewed build/resolution constraints: `flatdict==4.0.1` needs `setuptools<82` to provide its
undeclared `pkg_resources` build import, and Isaac Lab's exact `starlette==0.49.1` pin conflicts
with Isaac Sim's FastAPI requirement. No Starlette import was found under the pinned core package,
so the combined lock selects FastAPI-compatible `starlette==0.45.3`; this override must receive
explicit import and livestream coverage in Stage 4.
See `docs/DEPENDENCY_LOCKS.md` for the resolved versions, index review, hashes, and deferred
runtime validation.

### Stage 4 — Install and validate Isaac Sim and Isaac Lab in GPU mode (completed 2026-08-20)

- Pass the GPU-mode resource audit and confirm at least 200 GiB remains free on the data disk.
- Revalidate the reviewed `env/isaac/uv.lock`. It now includes Isaac Lab tag `v2.3.2` at
  immutable commit `37ddf626871758333d6ed89cf64ad702aef127d0`, represented as separate
  `isaaclab`, `isaaclab-assets`, `isaaclab-tasks`, and `isaaclab-rl[rsl-rl]` Git-subdirectory
  packages. `isaaclab_mimic` and `isaaclab_contrib` are deliberately deferred until a concrete
  Stage 6 requirement justifies their additional dependencies.
- Keep a matching source checkout at `/root/autodl-tmp/EmbodiedAI/vendor/IsaacLab` for official
  scripts/examples, but install the four packages from the locked commit rather than running an
  unconstrained `isaaclab.sh --install` that could drift the environment.
- Synchronize the complete Isaac Sim 5.1.0, Isaac Lab 2.3.2, PyTorch 2.7 cu128 stack to
  `/root/autodl-tmp/EmbodiedAI/envs/isaac` with `--locked`; then run package consistency and
  inventory checks before importing native components.
- Run the Isaac Sim Compatibility Checker, Isaac Sim/Lab imports, PyTorch CUDA availability,
  RTX 5090 capability/kernel execution, and a minimal headless `SimulationApp` lifecycle.
- Test a small PhysX stage, camera/RTX rendering, a Franka scene, a small vectorized Isaac Lab
  environment, and one bounded RSL-RL training smoke run. WebRTC GUI streaming is optional and
  comes only after headless rendering passes.
- Record lock and source hashes, package inventory, logs, timings, peak RAM/VRAM,
  driver/runtime versions, cache locations, and disk use.

Exit gate: the locked environment is installed and deterministic Isaac Sim/Isaac Lab/CUDA
smoke tests pass from a fresh GPU-mode shell.

Result:

- `env/isaac/uv.lock` revalidated at SHA-256
  `997119551e29b5a014feb2fdb872e2d1b1bb2a3420f039dcee79a2c351f61bf1`, and the
  220-package Python 3.11.16 environment was synchronized to
  `/root/autodl-tmp/EmbodiedAI/envs/isaac`. Final key versions are Isaac Sim 5.1.0.0,
  Isaac Lab v2.3.2, PyTorch 2.7.0+cu128, torchvision/torchaudio 0.22.0/2.7.0+cu128,
  NumPy 1.26.0, and RSL-RL 3.1.2.
- Isaac Lab's four Git-subdirectory wheels preserve the dependency lock but do not preserve the
  Omniverse extension layout expected by upstream `__init__.py` files. The official installer uses
  editable installs for this reason. `scripts/bootstrap/install_isaaclab_editable.sh` now verifies
  the clean source checkout at commit `37ddf626871758333d6ed89cf64ad702aef127d0`, revalidates the
  lock, and replaces only those four packages with `--no-deps` editable installs. Run
  `embodiedai_isaaclab_install` after every `embodiedai_uv_sync isaac`; dependency versions remain
  owned by the lock.
- The official compatibility checker passed with driver 580.105.08, Ubuntu 22.04.5, 32 GiB-class
  RTX 5090 VRAM, and no display. PyTorch reported CUDA 12.8, cuDNN 9.7.1, capability `(12, 0)`,
  `sm_120`, and completed a CUDA matrix operation.
- Bounded tests passed: PhysX plus a 64x64 offscreen RTX camera in 29 seconds (peak 3,215 MiB
  VRAM); `Isaac-Lift-Cube-Franka-v0` with four CUDA environments and 16 steps in 13-14 seconds
  (peak process RSS 2,974,196 KiB and 1,817 MiB VRAM); and one RSL-RL Cartpole iteration with
  16 environments/256 timesteps in 18 seconds (peak 2,807 MiB VRAM).
- FastAPI 0.115.7 plus Starlette 0.45.3 completed an ASGI request, and private WebRTC extensions
  started their streaming server before a two-environment Cartpole test passed. `uv pip check`
  still reports Isaac Lab's incompatible `starlette==0.49.1` metadata declaration; this is the
  single intentional lock override described above, now runtime-covered rather than unresolved.
- Direct PyPI downloads were unreliable. Seven exact lock-addressed wheels were fetched through
  the Tsinghua mirror into an external wheelhouse and SHA-256 verified before local installation;
  a normal `uv sync --locked` then converged without lock changes. `/etc/network_turbo` was used
  temporarily to populate Franka assets from CloudFront and was unset afterward. No proxy setting
  was persisted.
- Final state: 506 GiB remains free on the data disk; the Isaac environment is about 19 GiB and
  the shared uv cache about 25 GiB. Miniconda base history retained SHA-256
  `f02020941438832c53f99164fba644e32d36356f517a872aa887aa29759fb65f`; driver 580.105.08 is
  unchanged; `CUDA_HOME`, `LD_LIBRARY_PATH`, and `ROS_DISTRO` remain unset in the project shell.
  Detailed logs and the final package inventory are under
  `/root/autodl-tmp/EmbodiedAI/runs/bootstrap/`.

### Stage 5 — Install and validate the VLA environment in GPU mode (completed 2026-08-21)

- Repeat the GPU-mode resource audit, confirm at least 100 GiB remains free on the data disk,
  and verify all uv, Hugging Face, Torch, compiler, W&B, model, dataset, and temporary paths
  point to `/root/autodl-tmp/EmbodiedAI`.
- Revalidate `env/vla/uv.lock`, then synchronize the pinned LeRobot 0.6.0 SmolVLA,
  training/PEFT, PyTorch 2.8 cu128, torchvision/torchaudio, TorchCodec 0.7.x, and NumPy 2.2.x
  stack to `/root/autodl-tmp/EmbodiedAI/envs/vla` with `--locked`.
- Run package consistency and inventory checks, then validate LeRobot/SmolVLA imports, PyTorch
  CUDA and RTX 5090 kernel execution, TorchCodec/FFmpeg behavior with a documented CPU fallback,
  and one tiny dataset round trip.
- Download only the reviewed SmolVLA checkpoint into the external model/cache paths, then run
  one reproducible inference and one bounded LoRA/PEFT training step.
- Measure the 32 GiB VRAM fit before proposing longer fine-tuning; do not begin a full training
  run as part of this validation stage.

Exit gate: the locked environment is installed and one reproducible GPU inference, media-path
check, dataset round trip, and bounded training smoke test pass from a fresh GPU-mode shell.

Result:

- `env/vla/uv.lock` revalidated at SHA-256
  `136754fe057383cfff64a59dc7bb96668f83ad94004a09760ec70a496847d873`, and the
  107-package Python 3.12.14 environment was synchronized to
  `/root/autodl-tmp/EmbodiedAI/envs/vla`. `uv pip check` reports no incompatibilities. Key
  versions are LeRobot 0.6.0, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio
  2.8.0+cu128, TorchCodec 0.7.0+cu128, NumPy 2.2.6, Transformers 5.5.4, PEFT 0.20.0,
  Accelerate 1.14.0, and Datasets 4.8.5.
- CUDA imports and a matrix operation passed on the RTX 5090. PyTorch reports CUDA 12.8,
  cuDNN 9.10.2, capability `(12, 0)`, and a wheel architecture list containing `sm_120`.
- TorchCodec initially exposed a missing host `libavdevice.so.58`; Ubuntu's `ffmpeg`
  `7:4.4.2-0ubuntu0.22.04.1` package was installed without modifying any Python environment,
  CUDA component, or NVIDIA driver. TorchCodec then decoded a four-frame 64x64 H.264 fixture
  on CPU. CPU decode is the accepted Stage 5 media baseline; hardware video decode is deferred.
- A three-frame, one-episode synthetic LeRobot dataset was written, explicitly finalized,
  reopened, and value-checked below `/root/autodl-tmp/EmbodiedAI/datasets/stage5`. LeRobot 0.6.0
  buffers metadata and Parquet writers, so `finalize()` is required before an immediate reload.
- The reviewed `lerobot/smolvla_base` checkpoint is pinned to immutable revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`. Its 906,712,520-byte safetensors file matches
  SHA-256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
  The tokenizer/config dependency `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` is pinned to
  revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; only its configuration/tokenizer files are
  needed because the complete SmolVLA checkpoint supplies the model weights.
- Forced-offline checkpoint loading produced a finite `(1, 6)` CUDA inference result. The
  policy's internal peak allocation/reservation was 1,209.67/1,252 MiB; external `nvidia-smi`
  sampling observed 1,861 MiB maximum total GPU use.
- One official LeRobot `wrap_with_peft` LoRA step completed with rank 2. It trained 92,832 of
  450,139,008 parameters (about 0.0206%); all 74 trainable tensors had finite gradients and were
  updated. Internal peak allocation/reservation was 1,286.53/1,344 MiB, while external sampling
  observed 1,963 MiB maximum total GPU use. The small adapter artifact lives under the external
  checkpoint directory, not Git.
- Reproducible checks are recorded in `scripts/data/lerobot_dataset_smoke.py` and
  `scripts/train/smolvla_stage5_smoke.py`. Detailed inventories, statuses, GPU samples, and logs
  are under `/root/autodl-tmp/EmbodiedAI/runs/bootstrap/`.
- Direct downloads stalled; `/etc/network_turbo` was enabled temporarily for the locked sync and
  reviewed model downloads, then unset. Final state retains about 490 GiB free on the data disk;
  the VLA environment is 7.2 GiB, uv cache 32 GiB, and model directory 870 MiB. Miniconda base,
  NVIDIA driver 580.105.08, global CUDA/cuDNN paths, and global ROS state remain unchanged.

### Stage 6 — Integrate demonstrations and policy evaluation

- Steps 1-5 completed the versioned observation/action/episode contracts, Franka task skeleton,
  deterministic reset/evaluation interface, dependency boundary, and immutable NPY structural
  episode recorder.
- Step 6 adds an Isaac-side deterministic state-machine expert plus an expert-rollout
  collector. The initial task is `franka-pick-place` and its canonical instruction is
  `Pick up the cube and place it in the goal.`
- Expert episodes extend the additive v1 metadata with the exact instruction, instruction
  variant/language, and structured expert provenance while retaining the existing stable `task`
  identifier, payload checksums, and task/config/environment provenance.
- The expert protocol allows state-machine, learned RL-policy, and teleoperation action
  sources to emit the same normalized action schema. Instruction paraphrases remain separate
  from task definitions; a task with different goal/evaluation semantics receives a new stable
  task identifier.
- Collection remains in the Isaac Python 3.11 environment and writes one immutable episode per
  environment. It must not import LeRobot. LeRobot conversion, VLA dataset validation, training,
  and policy replay remain later reviewed work in the Python 3.12 VLA environment.
- The state-machine rollout was GPU-validated on 2026-08-25: one immutable 108-step success
  episode, including RGB observations and instruction/expert provenance, passed the file-level
  validator under the external dataset root. No package was changed and no VLA training ran.

Exit gate: end-to-end pick-and-place round trip passes without cross-installing environments.

### Stage 7 — Add ROS 2 deployment boundary

- Install ROS 2 Humble using the Ubuntu 22.04 binary path and build the overlay workspace.
- Start with standard messages and separate external policy/control nodes.
- Validate Isaac's internal Humble bridge, DDS discovery, timestamps, QoS, action rate, and emergency-stop/timeout behavior.
- Add custom interfaces only if standard messages are insufficient; then build/test both Python ABIs explicitly.

Exit gate: simulated sensor/state/language input reaches the policy node and safe actions reach the simulated controller.

### Stage 8 — Reproducibility and robustness gate

- Recreate each environment from locks in a clean location.
- Run unit, integration, GPU smoke, dataset, ROS, and domain-randomization tests.
- Capture versions, hardware, timings, VRAM/RAM peaks, and known limitations.

Exit gate: a documented clean-room rebuild and MVP evaluation report succeed.

## 7. Approved decisions and next review gate

The Stage 0 approval accepted:

1. Stable baseline: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 instead of the Isaac 6 / Lab 3 beta line.
2. Three runtime boundaries: Isaac Python 3.11, VLA Python 3.12, ROS 2 Humble system Python 3.10.
3. `uv` for Python locking, with all heavy environments/caches on `/root/autodl-tmp`.
4. Accept the now-passing resource gate: one RTX 5090, 25 CPU quota, 90 GiB RAM, 45 GiB shared memory, and about 550 GiB data-disk free.
5. Standard ROS messages for the MVP, deferring dual-ABI custom interface builds.

Next review gate: define and review the Isaac-to-LeRobot conversion and VLA evaluation stage.
LeRobot conversion and VLA training remain separate later gates.

## 8. References

- [Isaac Lab local installation and 5.1 requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)
- [Isaac Lab v2.3.2 source tag](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2)
- [Isaac Lab v2.3.1/v2.3.0 release notes (Isaac Sim 5.1 baseline)](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/refs/release_notes.html)
- [Isaac Sim 5.1 Python environment](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Sim 5.1 ROS 2 compatibility](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [Current Isaac Sim system requirements (used to assess the deferred 6.x path)](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [LeRobot v0.6.0 release](https://github.com/huggingface/lerobot/releases/tag/v0.6.0)
- [LeRobot dependency metadata](https://github.com/huggingface/lerobot/blob/main/pyproject.toml)
- [LeRobot installation and CUDA-wheel guidance](https://github.com/huggingface/lerobot/blob/main/docs/source/installation.mdx)
- [TorchCodec/PyTorch compatibility table](https://github.com/pytorch/torchcodec#compatibility)
- [PyTorch official previous-version wheels](https://pytorch.org/get-started/previous-versions/)
- [ROS 2 Humble supported platforms](https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html)

## 9. Change log

- 2026-08-17: Initial proposal recorded after read-only audit. No packages installed or modified. Awaiting review.
- 2026-08-17: Re-audited after restart in GPU mode. Confirmed one usable RTX 5090, driver 580.105.08, CUDA capability 12.0, 25-CPU quota, 90 GiB RAM, and 45 GiB shared memory. Removed the former hardware blocker and recorded invalid thread-count variables. No packages installed or modified; plan still awaits review.
- 2026-08-20: Recorded explicit Stage 0 approval and authorization for Stage 2 and Stage 3 only.
- 2026-08-20: Completed Stage 2 repository/storage bootstrap and initial Stage 3 uv definitions, lock resolution, and lock review. Stages 4-8 remain unapproved.
- 2026-08-20: Considered an A/B split for no-GPU installation and GPU validation, then cancelled it because the no-GPU allocation cannot reliably meet CPU/RAM installation needs. Stage 4 and Stage 5 now each install and validate entirely in GPU mode.
- 2026-08-20: Added Isaac Lab v2.3.2 commit `37ddf626871758333d6ed89cf64ad702aef127d0` core/assets/tasks/RSL-RL source packages to the Isaac uv definition and lock before Stage 4. No environment was synchronized or installed.
- 2026-08-20: Completed approved Stage 4 in GPU mode. Installed the locked Isaac Sim/Lab stack,
  added the exact-source editable Isaac Lab bootstrap required by upstream extension layout, and
  passed compatibility, CUDA, Vulkan/RTX camera, Franka/vectorization, RSL-RL, FastAPI/Starlette,
  and private WebRTC startup tests. Stage 5 remains unapproved and unmodified.
- 2026-08-21: Completed approved Stage 5 in GPU mode. Installed the locked VLA stack, added the
  Ubuntu FFmpeg runtime required by TorchCodec, pinned and checksum-verified the reviewed SmolVLA
  model, and passed CUDA/import, CPU media decode, dataset round-trip, offline inference, and
  bounded LoRA/PEFT optimizer-step tests.
- 2026-08-23 to 2026-08-24: Completed Stage 6 steps 1-5: dependency-light contracts, the Franka
  task skeleton, deterministic reset/evaluation, the Isaac/VLA dependency boundary, and immutable
  NPY structural episode publication.
- 2026-08-25: Added the documentation-only Stage 6 step 6 plan for instruction-bearing expert
  demonstrations, a deterministic state-machine expert, and an extensible collection interface.
  No expert code, episode generation, package change, LeRobot conversion, or VLA training was run.
- 2026-08-25: Implemented Stage 6 step 6. Added additive instruction/expert metadata, the
  vectorized state-machine expert, a reusable one-episode collector, committed controller
  configuration, and a GPU-validated collection entry point. The accepted 108-step episode
  passed immutable NPY validation. No dependency, environment, LeRobot, or training change was
  made.
