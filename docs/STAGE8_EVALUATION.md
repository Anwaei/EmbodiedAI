# Stage 8 Closed-loop Policy Evaluation

Status: **approved 2026-08-31; implementation pending**

Stage 8 evaluates the existing small-corpus SmolVLA base and Step 6A adapter in closed-loop Isaac
rollouts. It does not wait for Step 6B, train a policy, introduce ROS 2, or combine Isaac and
LeRobot in one interpreter.

## 1. Runtime boundary

Stage 8 uses two independent processes on the same host:

```text
Isaac / Isaac Lab Python 3.11
  Robot Client
    -> contract observation + instruction
    -> local HTTP/JSON request
VLA Python 3.12
  Policy Server
    -> Stage 7 processor + pinned base or PEFT adapter
    -> canonical normalized 50 x 7 action chunk
    -> local HTTP/JSON response
Robot Client
  -> validate, safety-clip, schedule, env.step, evaluate, record
```

The server binds only to `127.0.0.1`. The first implementation uses Python standard-library
HTTP/1.1 and JSON; RGB bytes use base64. ROS 2, gRPC, ZeroMQ, UDP, WebRTC, shared memory, public
ports, authentication, and remote clients are outside this stage.

## 2. Wire protocol

The versioned dependency-light protocol exposes:

- `GET /health`: policy/model/processor/runtime identity and readiness;
- `POST /v1/reset`: explicit episode boundary and `policy.reset()`;
- `POST /v1/action-chunk`: one live observation/instruction to one action chunk.

Every inference request carries a schema version, request ID, episode ID, monotonic step index,
instruction, 9D joint position, `3 x 224 x 224` uint8 front RGB, and deterministic noise seed.
Every response echoes the identities and returns status, a `50 x 7` float action chunk, inference
latency, and postprocessing diagnostics. Requests and responses have bounded byte sizes. The
Robot Client rejects identity mismatches, stale/out-of-order replies, missing fields, unexpected
shapes/dtypes, and non-finite values.

## 3. Action scheduling and safety

The accepted default is receding-horizon control:

- policy prediction horizon: 50 actions;
- `execute_horizon = 5`;
- control rate: 20 Hz;
- execute the first five actions, discard the remaining 45, observe again, and replan;
- reset both the policy and client action queue at every episode boundary.

The first implementation is synchronous. Simulation time advances only through `env.step()`, so
the client waits for inference without applying stale actions. Client round-trip latency and server
model latency are recorded separately.

Online safety is explicit and measurable:

- invalid shape/dtype or any NaN/Inf terminates the rollout;
- raw absolute action above the reviewed hard limit terminates the rollout;
- finite output outside `[-1, 1]` but within the hard limit is clipped before Isaac execution;
- raw actions, executed actions, violation count, and clipping count are all recorded;
- timeout or connection loss terminates the rollout without executing a fallback/stale action.

## 4. Stage 8 steps

### Step 8.1 — RPC, policy identity, scenario, and rollout contracts

Add dependency-light schemas for health/reset/inference messages and evaluation rollout metadata.
The evaluation metadata names the policy rather than misusing Stage 6 `ExpertMetadata`. Freeze a
versioned Stage 8 run configuration and a five-scenario manifest derived from the existing
20-episode validation split.

Exit gate: invalid messages and incompatible schema/model/processor identities fail closed in
dependency-light unit tests.

### Step 8.2 — SmolVLA Policy Server

Add a VLA-only online adapter and local HTTP server. It loads exactly one reviewed base or PEFT
policy at startup, verifies the pinned model and processor artifacts, converts live contract keys
to the Stage 7 feature profile, calls `predict_action_chunk`, and returns postprocessed canonical
actions without importing simulator modules.

Exit gate: health/reset/inference work for one real dataset-shaped observation and the response is
finite, reproducible, and `50 x 7`.

### Step 8.3 — Isaac Robot Client and receding-horizon scheduler

Add a one-environment Robot Client. It creates the reviewed Franka task, extracts pre-action joint
state/front RGB, sends policy requests, validates and clips responses, executes five actions before
replanning, invokes the public task evaluator, and terminates on success, failure, policy error, or
the configured step limit. It must not import LeRobot or SmolVLA.

Exit gate: scheduler and action safety behavior pass unit tests and an Isaac client can consume a
fake deterministic server.

### Step 8.4 — Protocol, fake-server, failure, and dependency tests

Cover round trips, payload bounds, monotonic IDs, reset semantics, malformed responses, timeout,
connection loss, NaN/Inf, clipping, hard-limit termination, queue exhaustion, and dependency
boundaries. Policy code must not import Isaac; simulation client code must not import LeRobot;
Stage 8 code must not import ROS 2.

Exit gate: all dependency-light and fake-server tests pass without a GPU.

### Step 8.5 — Single-episode two-process smoke

Start the Policy Server in the VLA environment and the Robot Client in the Isaac environment on
the same RTX 5090. Run one bounded headless base-policy episode with `execute_horizon = 5`, save a
rollout manifest/metrics/video, and verify both processes exit or remain healthy as designed.
Task success is not required for the engineering smoke.

Exit gate: one real closed-loop rollout finishes without cross-environment imports, invalid
actions, stale replies, or private partial output.

### Step 8.6 — Batch rollout collection and metrics

Evaluate the five held-out small-corpus validation scenarios. Use three fixed policy-noise seeds
per learned policy. Record task outcome, final/minimum goal error, steps/simulation time, raw and
executed actions, clipping/saturation/smoothness, chunk refresh/discard counts, server inference
latency, round-trip latency, errors, runtime identity, and optional video.

Launch one Isaac process per scenario and run that scenario's three seeds in the process. This
keeps the reviewed scenario/seed matrix unchanged while avoiding unsupported repeated scene
construction in one Isaac Kit lifecycle. A hard-limit violation executes one safe zero action,
publishes an auditable `error` rollout, and continues with the remaining matrix.

Exit gate: all configured rollouts atomically publish under `$EMBODIEDAI_RUNS/stage8`; evaluation
outputs never enter `$EMBODIEDAI_DATASETS`.

### Step 8.7 — Paired baseline comparison

Compare, on the same scenario/seed matrix:

1. deterministic state-machine expert as a privileged scripted reference;
2. unfine-tuned SmolVLA base with the reviewed small-corpus processor;
3. the Step 6A rank-2 LoRA adapter with the same processor.

Use five scenarios by three seeds, yielding 15 rollouts per policy and 45 total. Report paired
results; do not describe the privileged expert as a like-for-like visual policy.

Exit gate: one versioned summary contains per-policy/per-scenario outcomes and comparable metrics.

### Step 8.8 — Reproducibility report and handoff

Publish the final JSON report and human-readable summary with Git/config/model/adapter/processor/
lock/runtime identities, exact commands, known limitations, and failure examples. Closed-loop
engineering completion does not require a minimum learned-policy success rate, but Stage 9 must
not use a degenerate or unstable VLA baseline without a separate review.

Exit gate: bounded closed-loop rollouts and the baseline comparison can be reproduced from reviewed
configuration without code or artifact ambiguity.

## 5. Initial evaluation matrix

The scenario manifest reuses validation episode indices `0, 6, 7, 18, 9`, covering five exact
instructions, all four goals, and four cube reset positions. The initial policy matrix is:

| Policy | Rollouts | Notes |
|---|---:|---|
| state-machine expert | 15 | privileged scripted reference |
| SmolVLA base | 15 | pinned base plus small-corpus train-only processor |
| Step 6A LoRA adapter | 15 | feasibility adapter plus the same processor |

Step 6B may later add another Policy Server launch target without changing the protocol, client,
scheduler, recorder, scenario manifest, or metric definitions.

## 6. Initial configuration

```toml
schema_version = "embodied-ai.stage8-run-config/v1"
control_hz = 20
prediction_horizon = 50
execute_horizon = 5
max_episode_steps = 200
request_timeout_s = 2.0
hard_action_limit = 1.5
server_host = "127.0.0.1"
server_port = 8765
record_video = true
```

The implementation must validate these values and refuse public bind addresses or output paths
outside the external run/artifact roots.

## 7. Artifact boundary

```text
$EMBODIEDAI_RUNS/stage8/<run-id>/
  logs/
  <policy-kind>/rollouts/<rollout-id>/
    manifest.json
    joint_position.npy
    camera_front_rgb.npy
    raw_action.npy
    executed_action.npy
    cube_position_env_m.npy
    goal_error_m.npy
    cube_speed_m_s.npy
    gripper_open.npy
    timestamp_ns.npy
    inference_requests.jsonl
    camera_front.mp4      # optional derived preview
  reports/
    comparison.json
    comparison.md
```

Rollout publication uses a private sibling partial directory and same-filesystem rename. Metrics
and manifests identify the observation/action/RPC schemas, scenario, instruction, task/reset
parameters, policy/checkpoint/processor, noise seed, scheduler, clipping policy, simulator/VLA
runtime, and payload hashes.

## 8. Known risks

- Offline model latency is about 142 ms, longer than a 50 ms control period. Synchronous paused
  simulation plus five-action receding-horizon scheduling is the accepted initial interpretation;
  real-time asynchronous control is deferred.
- Base offline outputs already show some bound violations. Stage 8 records raw values and applies
  reviewed client-side clipping rather than hiding them.
- Isaac and SmolVLA share one GPU in separate processes. The first smoke remeasures latency and
  memory under simultaneous load.
- Closed-loop distribution shift may produce zero task success. That is an evaluation result, not
  permission to weaken task success semantics or expose privileged state to the learned policy.
