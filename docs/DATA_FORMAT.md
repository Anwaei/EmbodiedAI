# Episode Data Contract

The simulator and training environments communicate through immutable episode directories.
Large episode data lives under `/root/autodl-tmp/EmbodiedAI/datasets`, not in Git.

## Versioning

The initial schema identifier is `embodied-ai.episode/v1`. Readers must reject unknown major
versions and may accept additive fields within the same major version.

## Layout

```text
episode-000001/
├── manifest.json
├── observations/
│   ├── robot_state.npy
│   └── camera_front/              # encoded video or frame chunks
├── actions.npy
└── metadata.json
```

`manifest.json` records:

- schema version and episode identifier;
- task, robot, scene, and random seed;
- observation keys, shapes, dtypes, and timestamps;
- action representation, dimension, units, and control frequency;
- termination/success state;
- simulator, repository, configuration, and environment-lock revisions;
- checksums for payload files.

## Invariants

- Observation/action timestamps are monotonic.
- Required observation keys and action dimensions are fixed for an episode.
- Units and coordinate frames are explicit.
- Failed and truncated episodes remain distinguishable from successful episodes.
- Conversion to LeRobot format is a validated, separate step in the VLA environment.
- Checkpoints are accompanied by policy metadata that names the compatible schema and
  normalization statistics.
