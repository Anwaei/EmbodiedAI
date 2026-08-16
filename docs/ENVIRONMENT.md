# Environment and Bootstrap Plan

Status: **revised proposal only — hardware preflight passes; installation remains blocked pending review**
Audit date: 2026-08-17 (Asia/Shanghai), re-audited after GPU-mode restart
Target host: `embodied-5090` / `/root/projects/EmbodiedAI`

## 1. Constraints and decision summary

- The project targets language-conditioned manipulation in Isaac Sim / Isaac Lab, demonstration generation, SmolVLA fine-tuning, optional residual RL, and a ROS 2 deployment interface.
- Simulation, VLA training, and ROS 2 must remain isolated because their supported Python and package stacks differ.
- Do not mutate Miniconda `base`, install system-wide Python packages, change the NVIDIA driver, or start dependency installation until this document is reviewed.
- Recommended stable baseline:
  - Isaac Sim `5.1.0` + Isaac Lab `v2.3.2`, Python `3.11`, PyTorch `2.7.0+cu128`.
  - LeRobot `v0.6.0` with the `smolvla` feature set, Python `3.12`, PyTorch `2.8.0+cu128`, TorchCodec `0.7.x`, NumPy `2.2.x`.
  - ROS 2 Humble on Ubuntu 22.04 using its native Python `3.10`, communicating with Isaac Sim through DDS and the Isaac Sim ROS 2 bridge.
- The restarted container now passes the static hardware/resource preflight: one usable RTX 5090, a 25-CPU cgroup quota, 90 GiB RAM, 45 GiB shared memory, and 550 GiB on the data disk. The 30 GiB root filesystem remains intentionally code-only.
- Installation is still prohibited until this revised document is reviewed. The Isaac Sim Compatibility Checker and simulator smoke tests remain pending because Isaac Sim has not been installed.

## 2. Remote-machine audit

### 2.1 Observed state

| Area | Observation | Assessment |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy), x86_64; kernel `5.15.0-119-generic`; glibc `2.35` | Suitable for the proposed Isaac Sim 5.1 / ROS 2 Humble baseline. |
| Container | Overlay root filesystem, AutoDL container hostname, no Docker/Podman/Apptainer CLI | Treat the leased instance itself as the outer isolation boundary; use Python environments inside it. |
| CPU | Host topology exposes 208 CPUs; cgroup `cpu.max=2500000 100000` | Effective quota is **25 CPUs**. `nproc` alone is misleading because the cpuset still exposes CPUs 0-207. Resource gate passes. |
| Memory | Host reports 754 GiB; cgroup `memory.max=96636764160`; no swap | Effective limit is **90 GiB**, with about 370 MiB in use during audit. Resource gate passes. |
| Shared memory | `/dev/shm` is a 45 GiB tmpfs | Adequate starting point for data-loading and simulator IPC; monitor during vectorized workloads. |
| GPU | One NVIDIA GeForce RTX 5090, UUID `GPU-11df53b9-8a6e-3ea4-81df-ec6ed8b5cc54`, compute capability 12.0, 32,607 MiB VRAM | GPU gate passes. It is physical `/dev/nvidia3` but logical CUDA/NVML device 0; code must select logical devices rather than hard-code a physical device node. |
| NVIDIA driver | `580.105.08`; `nvidia-smi` reports maximum supported CUDA `13.0` | Meets the Isaac Sim 5.1 recommendation (`>=580.65.06`) and LeRobot cu128 floor (`>=570.86`). No driver change is proposed. |
| `nvidia-smi` | Executable and healthy; idle GPU, 0 MiB allocated at audit, 575 W power limit | Identity, VRAM, driver, and runtime access are validated. |
| CUDA toolkit | `/usr/local/cuda -> /usr/local/cuda-12.8`; NVCC `12.8.93`; CUDA packages `12.8.1` | Appropriate for cu128 builds. NVCC works by absolute path but `/usr/local/cuda/bin` is not on `PATH`. Do not set a global CUDA path yet. |
| cuDNN | System packages `9.8.0.87` for CUDA 12; base PyTorch reports cuDNN `91002` from its wheel/runtime | Avoid mixing system cuDNN with wheel-bundled runtimes via global `LD_LIBRARY_PATH`. |
| Python | Miniconda base only: Python `3.12.3`, pip `24.0`; no `/usr/bin/python3` | Base must remain untouched. Dedicated Python 3.11 and 3.12 environments are required; ROS 2 later needs Ubuntu's system Python 3.10 packages. |
| Existing ML packages | Base has `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `numpy 2.3.2`; CUDA device count 1 | A CUDA tensor operation succeeds, capability `(12, 0)` is detected, and the wheel contains `sm_120`. Still do not reuse base: NumPy is outside LeRobot's `<2.3` bound and Python is wrong for Isaac Sim 5.x. |
| ROS 2 | No `/opt/ros`, no `ros2`, no ROS environment variables/packages | ROS 2 is not installed. |
| Graphics | No `DISPLAY`, Wayland, Vulkan tools, GLX tools, Xorg, or FFmpeg | Start with headless mode. Verify Vulkan/RTX rendering and install only reviewed FFmpeg dependencies when their stage is approved. |
| System disk | Overlay `/`: 30 GiB total, about 30 GiB free | Too small for Isaac Sim, environments, caches, and artifacts together. Keep it code-only. |
| Data disk | `/root/autodl-tmp`: 550 GiB total, essentially empty | Use for environments, Isaac binaries/assets, caches, datasets, checkpoints, and experiment runs. |
| Thread environment | `OMP_NUM_THREADS=0` and `MKL_NUM_THREADS=0` | Invalid for libgomp (confirmed warning). Project launch wrappers must unset them or set a positive value derived from the 25-CPU quota before running Python workloads. |
| Repository | Clean `main` at `c902e3f`; `docs/PROJECT_CONTEXT.md` and this file are committed | `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` referenced by `AGENTS.md` do not yet exist; create them only in the approved repository-bootstrap stage. |

### 2.2 Preflight gate status

| Gate | Required | Re-audit result | Status |
|---|---|---|---|
| GPU allocation | One RTX GPU with control/UVM device access | RTX 5090, logical device 0; `/dev/nvidia3`, `nvidiactl`, and UVM nodes present | Pass |
| Driver/runtime | Healthy `nvidia-smi`; compatible driver | 580.105.08; CUDA access and a PyTorch CUDA tensor operation succeed | Pass |
| CPU | >=8 effective CPUs | 25-CPU cgroup quota | Pass |
| RAM | >=32 GiB; 64 GiB preferred | 90 GiB cgroup limit | Pass |
| Shared memory | Sufficient for headless simulator/data workers | 45 GiB `/dev/shm` | Pass |
| Data disk | >=350 GiB free | About 550 GiB free at `/root/autodl-tmp` | Pass |
| System disk policy | Keep heavy state off 30 GiB root | Storage strategy defined; not yet implemented | Pending Stage 2 |
| Thread variables | Positive/valid BLAS/OpenMP settings | Both are currently `0` and produce a warning | Pending launch-wrapper fix in Stage 2 |
| Isaac compatibility check | Official checker plus headless smoke test | Tool is not installed, by instruction | Pending approved Stage 4 |

The former no-GPU/low-resource blocker is resolved. The only authorization blocker is review of this plan; the remaining technical checks are deliberately placed in their later approved stages.

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

Use `uv` with one independent project and lock file per Python environment after approval. `uv` is not currently installed. Install it only in user/data-disk scope; place `UV_CACHE_DIR`, Hugging Face cache, Torch cache, Omniverse cache, and shader/cache directories on `/root/autodl-tmp`.

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

Stage 1's read-only hardware audit was completed after the user restarted the server in GPU mode. No repository-bootstrap or installation-bearing stage (Stages 2-8) is authorized by this proposal.

### Stage 0 — Review and freeze the plan (current stage)

- Review version choices, repository layout, storage paths, and isolation boundaries.
- Decide whether the stable Isaac 5.1/Lab 2.3.2 path is accepted.
- Record approval in this document's change log.

Exit gate: explicit approval to begin provisioning/bootstrap.

### Stage 1 — Provision and re-audit hardware (static preflight complete)

- GPU-enabled allocation now provides one RTX 5090, 25 effective CPUs, 90 GiB RAM, and 45 GiB `/dev/shm`.
- `nvidia-smi`, device nodes, cgroup limits, disk, and a PyTorch `sm_120` CUDA tensor operation pass.
- Run the Isaac Sim Compatibility Checker at the start of Stage 4, once the approved simulator tooling exists.
- Do **not** change the NVIDIA driver unless a separate reviewed plan requires it.

Exit gate: static hardware preflight passes. Compatibility-checker and rendering gates remain in Stage 4.

### Stage 2 — Create repository skeleton and storage policy

- Create the approved directories and add the missing architecture/roadmap/data-format documents. `PROJECT_CONTEXT.md` is already under `docs/`.
- Add `.gitignore` rules for environments, assets, datasets, caches, runs, checkpoints, and generated ROS build trees.
- Create `/root/autodl-tmp/EmbodiedAI/*` storage directories and document environment variables.
- Add launch wrappers that unset or assign positive values to `OMP_NUM_THREADS` and `MKL_NUM_THREADS` based on the 25-CPU quota.
- Add dependency-light schemas and a minimal test harness only.

Exit gate: clean repository, documented paths, no large files in Git.

### Stage 3 — Bootstrap tooling and resolve locks

- Install `uv` in user/data-disk scope, not system-wide.
- Create independent `dev`, `isaac`, and `vla` definitions with exact Python versions and indexes.
- Resolve and review lock files before installing the heavy environments.
- Produce a dependency report showing Python, PyTorch, CUDA wheel, NumPy, and known ABI pins.

Exit gate: lock review succeeds; no resolver conflicts.

### Stage 4 — Install and validate the Isaac environment

- Install the pinned Isaac Sim/Lab/PyTorch stack in the data-disk environment.
- Test imports, `pip check`, CUDA availability, RTX 5090 kernel execution, and a minimal headless stage.
- Test a camera/render sample, Franka scene, a small vectorized environment, and one short training smoke run.
- Record resource use and cache locations.

Exit gate: deterministic smoke tests pass from a fresh shell.

### Stage 5 — Install and validate the VLA environment

- Install pinned LeRobot/SmolVLA, PyTorch/TorchCodec, and only the required extras.
- Validate imports, CUDA execution, video decode fallback, a tiny dataset round trip, and SmolVLA checkpoint inference.
- Estimate 32 GiB VRAM fit before fine-tuning; start with LoRA/PEFT if full fine-tuning is not justified.

Exit gate: one reproducible inference and one bounded training smoke test pass.

### Stage 6 — Integrate demonstrations and policy evaluation

- Implement the versioned observation/action/episode contract.
- Generate a tiny deterministic Isaac demonstration set, export it to LeRobot format, validate it in the VLA environment, and replay policy actions in Isaac.
- Add dataset checksums, provenance, and task/config metadata.

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

## 7. Review decisions requested

Before installation, approve or amend:

1. Stable baseline: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 instead of the Isaac 6 / Lab 3 beta line.
2. Three runtime boundaries: Isaac Python 3.11, VLA Python 3.12, ROS 2 Humble system Python 3.10.
3. `uv` for Python locking, with all heavy environments/caches on `/root/autodl-tmp`.
4. Accept the now-passing resource gate: one RTX 5090, 25 CPU quota, 90 GiB RAM, 45 GiB shared memory, and about 550 GiB data-disk free.
5. Standard ROS messages for the MVP, deferring dual-ABI custom interface builds.

## 8. References

- [Isaac Lab local installation and 5.1 requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)
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
