# Stage 6 Contracts

This document records the shared contract decisions introduced in Stage 6 step 1, the
task-specific Franka contract instance added in step 2, the Stage 6 steps 3-5 simulator
adapter and immutable episode publication decisions, and the additive metadata implemented
for step 6 expert demonstrations.

## Boundary and dependency policy

`src/embodied_ai/contracts` is the only Python object model shared by the development,
Isaac, VLA, and future ROS 2 runtimes. It uses the Python standard library only. A unit test
parses every module's imports and rejects dependencies outside the approved standard-library
set or sibling contract modules.

The package deliberately contains no arrays, tensors, simulator handles, policies, or ROS
messages. Runtime-specific adapters convert those values to contract metadata and immutable
episode payloads. In particular, neither Isaac nor LeRobot is imported across this boundary.

Contract objects are frozen, slotted dataclasses and require tuples for ordered collections.
This prevents accidental mutation after an episode is finalized. Every top-level schema has
`to_dict()` and `from_dict()` methods whose output is directly JSON serializable.

## Versioning policy

The three initial identifiers are:

- `embodied-ai.observation/v1`
- `embodied-ai.action/v1`
- `embodied-ai.episode/v1`

Readers reject an unknown identifier. A v1 reader ignores unknown object fields, so writers
may add optional metadata without changing the identifier. Removing a field, changing its
meaning, or changing validation rules incompatibly requires a new major identifier.

## Observation schema

An `ObservationSchema` is an ordered, unique collection of `ObservationField` objects. The
order is stable and may be used by later adapters, while the key remains the primary identity.
Keys and semantic names use lowercase dotted or hyphenated identifiers.

Stage 6 v1 supports two field kinds:

- `state`: a rank-1 vector on the `component` axis. Each scalar has a unique name and an
  explicit unit; spatial components may also name a coordinate frame.
- `rgb_image`: a three-channel `uint8` tensor in CHW order with axes `channel`, `height`, and
  `width`. The optical camera frame is mandatory.

Portable storage dtypes are represented as strings rather than NumPy objects. The initial
set is `bool`, `float32`, `float64`, `int64`, and `uint8`. State producers must declare one
component description per scalar, which makes ordering and units reviewable without loading
payloads.

The Franka task-specific v1 instance defines 9D joint position, 9D joint velocity, 3D cube
position, and `camera.front.rgb` at 224 x 224. The Isaac adapter converts renderer output to
the canonical three-channel CHW `uint8` representation. Exact ordering, units, frames, and
Isaac term bindings are recorded in `TASKS.md`.

## Action schema

An `ActionSchema` declares the ordered components, physical bounds, units, storage dtype,
control frequency, normalization state, and optional coordinate frame. Its dimension is
derived from the component list; a serialized `dimension` is validated rather than trusted.

Stage 6 v1 recognizes `joint_position`, `joint_velocity`, and
`end_effector_delta_pose`. End-effector delta actions require an explicit frame. Action data
must use `float32` or `float64`. When `normalized` is true, every component uses unit `1` and
bounds `[-1, 1]`; physical scaling must be recorded by a later adapter rather than hidden in
the contract.

The Franka task-specific v1 instance uses a normalized seven-dimensional
`end_effector_delta_pose` boundary: six relative IK values in the robot base frame followed
by one binary gripper value. Translation and rotation scales are task configuration rather
than hidden contract units and are recorded in `TASKS.md`. The deterministic expert emits
this same interface.

## Episode metadata schema

`EpisodeMetadata` is the self-describing `manifest.json` model for a finalized episode. It
contains:

- episode, task, robot, and scene identifiers plus the reset seed;
- the complete observation and action schemas;
- positive step count and an inclusive timestamp range;
- a `simulation` time base with integer nanosecond timestamps;
- a terminal outcome of `success`, `failure`, or `truncated`;
- a required reason for failed or truncated episodes;
- unique relative POSIX payload paths, media types, byte sizes, and lowercase SHA-256 hashes;
- simulator, repository, configuration, and environment-lock provenance.

Absolute paths and parent-directory traversal are forbidden so an episode directory remains
portable. Payload checksums cover data files, while the manifest itself remains outside its
own checksum list. `EpisodeManifest` remains an alias of `EpisodeMetadata` for compatibility
with the earlier repository placeholder.

The Isaac-side NPY recorder writes strictly increasing, synchronized observation/action
timestamp arrays and atomically publishes an episode directory only after all payload hashes
are known. Its companion validator checks checksums, file sizes, array shapes/dtypes,
timestamp monotonicity, and agreement with the manifest. These runtime responsibilities stay
outside the dependency-light metadata objects.

The Stage 6 smoke records pre-action pairs: observation at time `t`, followed by the action
issued at the same control boundary. The episode produced by the bounded zero-action smoke is
classified as `truncated` with reason `smoke-test-step-limit`; it is structural test data, not
a successful demonstration.

## Expert-demonstration metadata

Stage 6 step 6 extends `EpisodeMetadata` additively while retaining
`embodied-ai.episode/v1`. Existing structural episodes remain readable, but an episode offered
as an expert demonstration must provide all of the following:

- `task`: the existing stable machine-readable task identifier, initially
  `franka-pick-place`. It identifies reset, goal, and evaluation semantics and must not be
  replaced by free-form language.
- `instruction`: the exact non-empty natural-language command used for the episode, initially
  `Pick up the cube and place it in the goal.` It is constant for the episode and is the text
  later mapped to the LeRobot task/instruction field.
- `instruction_id`: a stable identifier for a wording variant, allowing several paraphrases to
  map to one task while preserving the exact wording used for collection.
- `instruction_language`: the language tag for the instruction, initially `en`.
- `expert`: structured `ExpertMetadata` containing `kind`, `identifier`, `revision`, and an
  expert-specific `configuration_revision`.

The initial expert kinds are `state_machine`, `rl_policy`, and `teleoperation`. Their common
fields have source-specific interpretations: a state machine records its controller/config
revision, an RL expert records an immutable policy/checkpoint revision, and teleoperation
records the control mapping and collection-session revision without requiring personal
operator identity in the portable manifest.

These fields are optional for structural smoke episodes but required together for training
demonstrations. Expert metadata describes provenance and must not be consumed as a policy
observation. `task`, instruction wording, and expert identity are orthogonal: changing a
paraphrase does not create a new task, and changing the action source does not change task
semantics.

The contracts layer validates non-empty normalized strings and supported expert kinds. The
four demonstration fields are optional as a group for backward compatibility, but supplying
only part of the group is rejected. This is an additive v1 change because existing v1 readers
already ignore unknown fields; any future change that alters the meaning of `task` or the
action/observation boundary requires a new major schema.

## Stage 7 Contract to LeRobot mapping

Stage 7 step 1 adds the separate mapping identifier
`embodied-ai.lerobot-mapping/v1`. The initial profile is
`franka-pick-place-smolvla-v1`:

| Contract source | LeRobot target | Policy role |
|---|---|---|
| `robot.joint_position` | `observation.state` | 9D proprioceptive input |
| `camera.front.rgb` | `observation.images.front` | 3 x 224 x 224 visual input |
| canonical action payload | `action` | normalized 7D output target |
| episode `instruction` | per-frame `task` | exact language input |
| `robot.joint_velocity` | excluded | deferred from the initial proprioceptive baseline |
| `object.cube.position` | excluded | privileged simulator state, never a VLA input |

Every observation must be classified exactly once as state input, visual input, or an explicitly
excluded field. The mapping validates the complete task, robot, scene, observation schema, and
action schema before conversion, so future schema drift cannot be silently dropped. It remains a
dependency-light description; only the Stage 7 converter imports NumPy and LeRobot.

## Deferred decisions

The following remain deferred after step 6:

- production camera codec and finalized camera calibration metadata;
- streaming/chunked recording for long episodes;
- Stage 7 dataset normalization statistics and any later proprioceptive-state expansion;
- Stage 7 VLA training and Stage 8 learned-policy inference.

These decisions must build on the v1 contracts and may not introduce LeRobot imports into
the Isaac environment.
