# Episode Data Contract

The simulator and training environments communicate through immutable episode directories.
Large episode data lives under `/root/autodl-tmp/EmbodiedAI/datasets`, not in Git.
The dependency-light Python object model and its design rationale are documented in
`CONTRACTS.md`.

## Versioning

The initial schema identifiers are `embodied-ai.observation/v1`,
`embodied-ai.action/v1`, and `embodied-ai.episode/v1`. Readers reject unknown identifiers and
accept additive fields that retain a known identifier. Incompatible field or semantic changes
require a new major identifier.

## Layout

```text
episode-000001/
├── manifest.json
├── observations/
│   ├── robot_joint_position.npy
│   ├── robot_joint_velocity.npy
│   ├── object_cube_position.npy
│   ├── camera_front_rgb.npy
│   └── timestamps_ns.npy
└── actions/
    ├── data.npy
    └── timestamps_ns.npy
```

Stage 6 uses one NPY payload per observation contract key. Dots and hyphens in a key are
normalized to underscores in its filename; the complete schema in `manifest.json` remains
the semantic source of truth. Every observation and action is sampled as a synchronized
pre-action pair at the 20 Hz control boundary, so both timestamp arrays are identical in this
initial recorder. Arrays have a leading `step` dimension followed by the per-step shape in
the corresponding schema.

The initial recorder intentionally buffers only a bounded short rollout in memory. It is
appropriate for the Stage 6 smoke and short demonstrations, but long-running production
collection must add streaming/chunking. Video encoding is also deferred; a future compatible
recorder may replace the camera NPY payload with MP4 while retaining the declared observation
semantics and timestamps.

`manifest.json` is the serialized `EpisodeMetadata` contract and records:

- schema version and episode identifier;
- task, robot, scene, and random seed;
- observation keys, kinds, shapes, dtypes, axes, components, units, and frames;
- action representation, dimension, units, and control frequency;
- simulation time base, nanosecond timestamp range, step count, and terminal outcome;
- simulator, repository, configuration, and environment-lock revisions;
- relative payload paths, media types, byte sizes, and SHA-256 checksums.

There is no separate `metadata.json`; keeping one manifest avoids duplicated sources of
truth. NumPy remains an Isaac-side recorder dependency and is not imported by the contracts
package.

## Invariants

- Observation/action timestamp arrays are integer nanoseconds on the simulation clock and are
  monotonically increasing.
- Required observation keys and action dimensions are fixed for an episode.
- Units and coordinate frames are explicit.
- Failed and truncated episodes remain distinguishable from successful episodes.
- Payload paths are normalized relative POSIX paths and every finalized payload is hashed.
- An episode is published atomically only after all payload files and the manifest are final.
- An existing finalized episode directory is never overwritten. A repeated episode identifier
  is an error.
- Conversion to LeRobot format is a validated, separate step in the VLA environment.
- Checkpoints are accompanied by policy metadata that names the compatible schema and
  normalization statistics.
