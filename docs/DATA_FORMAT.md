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
collection must add streaming/chunking. The camera NPY remains authoritative raw episode data.
`scripts/data/npy_episode_to_video.py` can validate one RGB payload and derive an H.264 MP4
preview without mutating the episode. Production camera codecs inside the episode remain
deferred until their timestamp and validation semantics are reviewed.

`manifest.json` is the serialized `EpisodeMetadata` contract and records:

- schema version and episode identifier;
- task, robot, scene, and random seed;
- for expert demonstrations, the exact instruction, its stable variant identifier and
  language, plus structured expert kind/identifier/revision provenance;
- observation keys, kinds, shapes, dtypes, axes, components, units, and frames;
- action representation, dimension, units, and control frequency;
- simulation time base, nanosecond timestamp range, step count, and terminal outcome;
- simulator, repository, configuration, and environment-lock revisions;
- relative payload paths, media types, byte sizes, and SHA-256 checksums.

There is no separate `metadata.json`; keeping one manifest avoids duplicated sources of
truth. NumPy remains an Isaac-side recorder dependency and is not imported by the contracts
package.

## Expert episode fields

Stage 6 step 6 adds the following additive `embodied-ai.episode/v1` manifest fields:

```json
{
  "task": "franka-pick-place",
  "instruction": "Pick up the cube and place it in the goal.",
  "instruction_id": "pick-place-cube-goal-en-001",
  "instruction_language": "en",
  "expert": {
    "kind": "state_machine",
    "identifier": "franka-pick-place-state-machine",
    "revision": "v1",
    "configuration_revision": "<lowercase-sha256>"
  }
}
```

`task` remains the machine-readable task definition. `instruction` is episode-invariant
language and may vary across episodes with the same task. The VLA-side LeRobot converter will
map the exact instruction text to LeRobot's task/instruction representation and retain the
stable task and expert provenance in conversion metadata. Expert provenance is never a model
input.

Raw collection may retain successful, failed, and truncated expert attempts, but only
successful episodes are eligible for the initial imitation-learning training split by
default. Dataset selection policy belongs to the later converter and must not mutate the raw
immutable episode directories.

## Derived camera previews

Run the standalone converter from a configured project shell:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/isaac/bin/python" \
  scripts/data/npy_episode_to_video.py \
  "$EMBODIEDAI_DATASETS/stage6-expert/episode-stage6-expert-000001" \
  "$EMBODIEDAI_ARTIFACTS/stage6/expert-videos/episode-stage6-expert-000001.mp4"
```

The converter verifies the manifest entry, byte size, SHA-256, array shape, and dtype before
streaming CHW RGB frames to FFmpeg. It uses the action schema control frequency unless `--fps`
is supplied, refuses to overwrite an existing output, writes through a private partial file,
and rejects outputs inside the immutable episode directory. The MP4 is a reproducible artifact,
not a manifest payload and not an additional training observation.

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
- Every training demonstration has one task, one exact instruction, and one expert provenance
  record for the complete episode.
