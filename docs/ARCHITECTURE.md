# Architecture

## System boundary

The MVP is split into processes and file contracts rather than a single Python runtime:

```text
Isaac Sim / Isaac Lab (Python 3.11)
  -> versioned episode data
LeRobot / SmolVLA (Python 3.12)
  -> checkpoint + policy metadata
Evaluation adapter (process boundary)
  -> observations/actions
ROS 2 Humble (system Python 3.10)
  -> deployment topics, services, and control safety
```

This boundary prevents Isaac, VLA, and ROS dependencies from contaminating each other.

## Roadmap pipeline ownership

The post-environment roadmap follows the runtime and artifact boundaries above:

| Stage | Pipeline owner | Primary output |
|---|---|---|
| 6 | Isaac Demonstration Pipeline | validated immutable expert episodes and derived previews |
| 7 | LeRobot + VLA Training Pipeline | validated `LeRobotDataset` plus SmolVLA base/PEFT artifacts |
| 8 | Closed-loop Policy Evaluation | Isaac rollout metrics and baseline comparisons |
| 9 | RL Policy Refinement | privileged standalone PPO and bounded FT-VLA residual results |
| 10 | ROS 2 deployment boundary | deployment messages, nodes, and safety-checked control path |
| 11 | Reproducibility and robustness gate | clean rebuild and perturbation/robustness report |

Stage 6 ends at contract-compliant raw demonstration data. Stage 7 owns Contract → LeRobot
conversion and offline VLA training. Stage 8 owns online policy/Isaac interaction. Stage 9 is
gated on the Stage 8 imitation baseline and may refine it without changing the public action
contract. ROS 2 deployment and the final robustness gate remain separate Stages 10 and 11.

## Repository modules

- `src/embodied_ai/contracts`: dependency-light data and policy interface schemas.
- `src/embodied_ai/sim`: Isaac Lab tasks, scenes, sensors, experts, demonstration collection,
  recording, and randomization.
- `src/embodied_ai/data`: dataset validation and LeRobot conversion.
- `src/embodied_ai/policies`: policy metadata and LeRobot/SmolVLA adapters.
- `src/embodied_ai/rl`: standalone PPO adapters, shared reward/state components, nominal-policy
  providers, and bounded residual composition.
- `src/embodied_ai/evaluation`: task metrics and robustness sweeps.
- `ros2_ws/src`: ROS 2 packages added during the approved deployment stage.

## Storage and communication

- Simulator outputs immutable episodes described by `docs/DATA_FORMAT.md`.
- The Isaac-side NPY recorder writes contract-keyed tensors and a manifest to a private
  partial directory, then atomically publishes the complete episode. It imports neither
  LeRobot nor the VLA training stack.
- Training emits checkpoints plus metadata defining observations, normalization, action
  dimensions, control frequency, and source revision.
- Online evaluation uses a narrow local RPC or ROS 2 boundary; it does not import Isaac and
  LeRobot into the same interpreter.
- Standard ROS messages are preferred for the MVP. Custom interfaces require explicit
  Python 3.10 and Python 3.11 build validation.

## Closed-loop evaluation boundary

Stage 8 implements online evaluation as a Robot Client + Policy Server pair on one host:

```text
Isaac Python 3.11 Robot Client
  -> versioned observation/instruction HTTP/JSON on 127.0.0.1
VLA Python 3.12 Policy Server
  -> postprocessed canonical 50 x 7 action chunk
Robot Client
  -> validate + safety clip + execute first 5 + replan
```

The transport uses only Python standard-library HTTP/JSON and base64 RGB for the first version.
The Policy Server owns model/adapter loading, Stage 7 preprocessing/postprocessing, deterministic
noise, inference, and model diagnostics; it must not import Isaac. The Robot Client owns live
contract extraction, receding-horizon scheduling with `execute_horizon = 5`, action safety,
timeouts, `env.step`, task evaluation, and rollout recording; it must not import LeRobot. Policy
inference is synchronous and simulation time pauses while waiting. ROS 2 remains Stage 10.

Stage 8 writes evaluation-specific rollout manifests under the external run root rather than
reusing demonstration `ExpertMetadata` or publishing evaluation behavior as training data.

## RL refinement boundary

Stage 9 keeps two PPO modes behind one canonical 7D task action and one public evaluator:

```text
Standalone mode (Isaac Python 3.11)
  privileged low-dimensional task state
    -> RSL-RL PPO 7D action
    -> Isaac action adapter

Residual mode
  VLA Python 3.12 batched Policy Server
    RGB + joint state + instruction -> frozen 7D nominal chunks
  Isaac Python 3.11 Robot Client + RSL-RL
    compact state + current nominal action -> bounded 6D residual
    -> clip(clip(nominal arm) + scaled residual)
    -> nominal gripper pass-through
    -> Isaac action adapter
```

The standalone PPO actor is intentionally privileged and is reported as a state-based simulation
baseline, not a deployable visual policy. The first residual actor receives no RGB or language;
multimodal task interpretation remains inside the frozen VLA. The VLA process and Isaac/RSL-RL
process remain isolated, and RL gradients never cross the process boundary.

The implemented standalone task is registered as
`EmbodiedAI-Franka-PickPlace-State-PPO-v0`. It subclasses the existing Franka scene/action
foundation but owns a separate state-only observation group, randomized cube/goal reset event,
movable non-colliding goal marker, left/right cube-filtered finger contact sensors, staged reward,
and termination diagnostics. The original RGB demonstration and Stage 8 task remains registered
separately and is not mutated by the RL configuration.

The existing Stage 8 single-observation protocol remains valid for evaluation. Residual training
requires a separately versioned batched request and independent per-environment action queues so
on-policy vectorization cannot mix episodes or wait on one request per environment. A synchronous
one-environment path is retained for integration debugging.

The residual composer validates finiteness, clips the raw VLA action, clips and scales the 6D PPO
correction with separate translation/rotation factors, clips the combined arm action again, and
leaves the binary gripper decision with the VLA. Raw nominal, bounded nominal, raw/scaled residual,
final action, and clip masks are all recorded. Because a 6D residual cannot repair gripper timing,
usable nominal gripper behavior and reviewed FT-VLA saturation are formal training gates.

Stage 9 evaluates the scripted expert, base SmolVLA, standalone PPO, FT-VLA, and FT-VLA+PPO on the
same frozen scenarios/seeds. The VLA-only and residual candidates use the same versioned Stage 9
execution-safety profile; the original Stage 8 fail-closed evidence is retained rather than
rewritten. Detailed design and gates are in `docs/STAGE9_RL.md`.

## Demonstration generation boundary

Stage 6 keeps task semantics, language, controlled scene parameters, control, and storage as
separate concerns:

```text
task definition + instruction selection
  -> versioned per-episode cube/goal parameters
  -> Isaac-side expert
  -> canonical ActionSchema action
  -> Isaac Lab step
  -> one immutable episode per environment
  -> later LeRobot conversion in the VLA environment
```

- `task` is a stable machine-readable task identifier whose reset and success semantics are
  owned by the task definition.
- `instruction` is the exact natural-language command attached to the episode. Multiple
  instructions may describe one task without duplicating the task implementation.
- `expert` describes the source of the actions and its revision. State machines, learned RL
  policies, and teleoperation must implement the same Isaac-side action-producing interface.
- Experts may read privileged simulator state, but they emit only the public normalized action
  contract. Expert-only state is not silently added to the eventual VLA policy input.
- The collector owns rollout lifecycle and recording; an expert does not write files. With
  vectorized environments, the collector maintains one recorder and one terminal outcome per
  environment rather than storing an environment batch as one episode.
- `FrankaPickPlaceEpisodeParameters` is the single runtime source for the cube reset position and
  goal position. Before environment creation it configures the cube default state, visual goal
  marker, success termination, expert context, and manifest task/reset parameters.
- The first batch implementation consumes a dependency-light, versioned TOML plan and launches one
  fresh Isaac process per row. This preserves the accepted one-reset lifecycle. It writes one
  immutable directory per episode and a separate atomic collection summary; it never combines an
  environment batch dimension into one episode.

## Offline dataset conversion boundary

Stage 7 keeps mapping decisions separate from LeRobot I/O:

```text
immutable Contract episode
  -> NPY/checksum/schema eligibility validation
  -> versioned Contract-to-LeRobot mapping
  -> serial mmap-backed LeRobot writer
  -> finalize + reopen in a private directory
  -> atomic dataset publication + conversion provenance
  -> independent source/dataset/statistics/reload validation
  -> atomic derived validation report
```

The mapping module depends only on the shared contracts. The converter runs only in the VLA
environment and may import NumPy and LeRobot; simulator code continues to reject those imports.
The initial policy sees joint position and front RGB. Joint velocity is an explicit future mapping
decision, while cube position is privileged simulator state and is never silently exposed to the
VLA. Original episode directories remain immutable and traceable through manifest hashes in the
conversion sidecar. The validator independently reopens both the immutable sources and the
published `LeRobotDataset`: it compares the complete state/action/timestamp/index/task table,
recomputes policy normalization inputs, decodes both boundaries of every episode through two
fresh LeRobot instances, and writes its report below the external run root. Validation never
changes the source episodes or converted dataset.

## Offline SmolVLA processing and training boundary

Stage 7 steps 4-6 run entirely in the isolated VLA environment:

```text
validated LeRobotDataset
  -> project feature binding + explicit statistics bundle
  -> preprocessing (temporal batch, language, image, normalization, device)
  -> pinned SmolVLA base + optional PEFT adapter
  -> postprocessing (unnormalization, CPU transfer, canonical action validation)
  -> offline predictions, metrics, checkpoints, and provenance
```

The base checkpoint's saved feature/processor metadata is not the project interface: it describes
6D state, 6D action, and three cameras, while the first project dataset contains 9D joint state,
canonical 7D action, and one front camera. A project-owned policy profile must therefore rebind
input/output features and supply project statistics before loading compatible pretrained weights.
It must never invent absent cameras, expose privileged cube position, or silently reuse the base
checkpoint's incompatible normalization arrays.

The implemented `embodied_ai.policies.smolvla` package owns the reviewed profile, split/config
parsing, preprocessing/postprocessing, local model loading, offline metrics, and bounded PEFT
mechanics. The package may depend on LeRobot, SmolVLA, PyTorch, and the
dependency-light contracts, but not on `embodied_ai.sim`, Isaac Sim, or Isaac Lab. Its output stays
in the normalized end-effector-delta/gripper action contract. Offline postprocessing reports bound
violations rather than hiding them with clipping. Stage 8 later reuses this package but separately
owns process transport, model action-queue scheduling, timeouts, online safety clipping, and live
Isaac interaction.

Fine-tuning statistics are computed from training episodes only. Split manifests use whole
episodes or scenarios, never individual frames across train/validation/test. Model, tokenizer,
dataset, mapping, statistics, processor, split, configuration, lock, seed, and Git identities are
recorded with every inference/training artifact. The detailed sequence and gates are in
`docs/SMOLVLA_PIPELINE.md`.

## Safety and scope

- The first task is Franka pick-and-place.
- Residual RL follows a working imitation-learning baseline.
- Domain randomization evaluation is described as Sim2Real-oriented; no physical-robot
  transfer claim is made.
- World-model work remains a non-blocking stretch goal.
