# Stage 3 dependency lock report

Date: 2026-08-20 (Asia/Shanghai)

Scope: lock resolution and review only. No heavy environment was synchronized, no simulator
or VLA package was installed, and no GPU smoke test was run. The server was allowed to be in
no-GPU mode for this stage.

## Resolved environments

| Project | Resolver Python | Reviewed direct result | Locked packages | Index policy |
|---|---:|---|---:|---|
| `env/dev` | 3.12.14 | mypy 1.20.2, pytest 9.1.1, pytest-cov 7.1.0, pytest-timeout 2.4.0, ruff 0.16.3 | 15 | Default PyPI; no CUDA or simulator dependencies |
| `env/isaac` | 3.11.16 | Isaac Sim 5.1.0.0 (`all`, `extscache`), PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, torchaudio 2.7.0+cu128 | 141 | Isaac Sim metapackage explicitly bound to NVIDIA's index; Torch family explicitly bound to the PyTorch cu128 index |
| `env/vla` | 3.12.14 | LeRobot 0.6.0 (`peft`, `smolvla`, `training`), NumPy 2.2.6, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128, TorchCodec 0.7.0+cu128 | 107 | Torch family and TorchCodec explicitly bound to the PyTorch cu128 index |

The LeRobot extras resolve the requested training stack, including Accelerate, Datasets,
PyArrow, video support, Weights & Biases, PEFT, and the SmolVLA-specific dependencies. Each
environment is an independent uv project rather than a member of a shared workspace.

## Resolver conflict and disposition

The first Linux resolution for Isaac Sim failed because `isaacsim-core==5.1.0.0` declares
`pywin32==306` without a Linux-excluding marker. `pywin32` publishes no Linux x86_64 wheel,
so uv correctly reported the graph as unsatisfiable.

`env/isaac/pyproject.toml` contains this narrowly scoped metadata correction:

```toml
[tool.uv]
override-dependencies = ["pywin32==306; sys_platform == 'win32'"]
```

This preserves the upstream requirement on Windows and removes only the impossible Windows
runtime from the Linux target. It does not change any requested Isaac Sim, Python, PyTorch,
CUDA, or torchvision pin. After the correction, all three locks resolve and validate. No
other dependency conflict remains at lock-review time.

## Index and ABI review

- The two CUDA stacks are isolated: Isaac resolves the `2.7.0/0.22.0/2.7.0` cu128 family;
  VLA resolves the `2.8.0/0.23.0/2.8.0` cu128 family plus TorchCodec `0.7.0+cu128`.
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
2aac632da5d1a574492db95e5788ae632c329c498c68f82355dd51cfaf111ca5  env/isaac/uv.lock
136754fe057383cfff64a59dc7bb96668f83ad94004a09760ec70a496847d873  env/vla/uv.lock
```

The pre-created runtime directories remain unsynchronized seed environments. They contain
only pip in `dev` and `vla`, and pip plus packaging/setuptools/wheel in `isaac`. Installation
and GPU/runtime validation remain Stage 4 and Stage 5 work and require separate approval.

## Network note

Direct package endpoints were intermittently slow while resolving. The temporary shell used
`/etc/network_turbo`, while the NVIDIA and PyTorch package hosts were routed directly with
`no_proxy`. These proxy variables were not written to global shell configuration.
