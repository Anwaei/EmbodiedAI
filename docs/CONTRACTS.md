# Stage 6 Contracts

This document records the shared contract decisions introduced in Stage 6 step 1 and the
task-specific Franka contract instance added in step 2. The deterministic controller,
recorder, and end-to-end episode smoke test remain later steps.

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
than hidden contract units and are recorded in `TASKS.md`. The later deterministic expert
must emit this same interface.

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

The recorder is responsible for writing monotonically increasing observation/action
timestamp arrays and for atomically publishing an episode directory only after all payload
hashes are known. Payload contents and monotonicity are not checked by these dependency-light
metadata objects; a later data validator will perform those file-level checks.

## Deferred decisions

The following are explicitly outside step 1:

- camera payload codec and finalized camera calibration metadata;
- deterministic reset, goal placement, expert control, and evaluation thresholds;
- NumPy/video payload codecs and atomic recorder implementation;
- LeRobot feature mapping and normalization statistics;
- Isaac Lab launch/reset/step integration and GPU validation.

These decisions must build on the v1 contracts and may not introduce LeRobot imports into
the Isaac environment.
