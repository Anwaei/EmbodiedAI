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
language and may vary across episodes with the same task. The Stage 7 LeRobot converter will
map the exact instruction text to LeRobot's task/instruction representation and retain the
stable task and expert provenance in conversion metadata. Expert provenance is never a model
input.

Step 7 records controlled spatial variation additively in the same manifest:

```json
{
  "task_parameters": {
    "goal_position_env_m": [0.62, -0.18, 0.03]
  },
  "reset_parameters": {
    "cube_position_env_m": [0.46, -0.05, 0.03]
  }
}
```

The goal parameter drives the visible marker, expert target, and success termination. The reset
parameter drives the per-environment cube default state. The original constants remain default
values for existing fixed-reset tests.

Raw collection may retain successful, failed, and truncated expert attempts, but only
successful episodes are eligible for the initial imitation-learning training split by
default. Dataset selection policy belongs to the later converter and must not mutate the raw
immutable episode directories.

## Expert collection layout

```text
stage6-expert-batch-v1-20260827/
├── collection_summary.json
├── episode-stage6-batch-v1-000001/
├── episode-stage6-batch-v1-000002/
└── ...
```

The versioned TOML collection plan is committed under
`configs/sim/franka_pick_place/expert_collection_v1.toml`. The batch launcher validates the whole
plan before execution, starts a fresh Isaac process for each row, reopens each published episode
through `validate_npy_episode()`, and atomically rewrites `collection_summary.json` after every
result. The summary records the plan hash, requested/processed/success counts, per-episode
parameters, outcome, frame count, manifest hash, and external log path. It is a collection index,
not an episode payload.

Run the reviewed first matrix from an Isaac project shell:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/isaac/bin/python" \
  scripts/sim/collect_franka_pick_place_expert_batch.py \
  --output_root "$EMBODIEDAI_DATASETS/stage6-expert-batch-v1-20260827" \
  --device cuda:0
```

On 2026-08-27 all 20 planned episodes succeeded. The corpus contains 2,138 frames, five exact
instruction variants, five cube reset positions, four goal positions, and 20 unique manifest
hashes. Individual episodes span 97-114 steps. The 308 MiB corpus has no private partial
directories; `collection_summary.json` has SHA-256
`a2705d03e8e15418100fd666ffa6b81b749cf404cdf8de27ccec4bc8131a2642`.

## Stage 7 LeRobot conversion

`src/embodied_ai/data/lerobot_mapping.py` owns the versioned feature/task mapping, while
`src/embodied_ai/data/lerobot_converter.py` owns VLA-environment I/O. The first mapping profile
uses 9D joint position as `observation.state`, front RGB as `observation.images.front`, the
normalized 7D contract action as `action`, and the exact episode instruction as LeRobot `task`.
Joint velocity and privileged cube position are intentionally absent from the initial policy
input. The immutable Contract episode remains the source of truth.

The converter accepts successful instruction-bearing expert episodes only. Before writing, it
validates every source payload/checksum, exact task/schema identity, the regular 20 Hz timestamp
grid, finite selected state/action values, and action bounds. Source arrays are memory-mapped and
processed one frame at a time. One source directory becomes one destination episode.

The destination is first written below a private sibling `.partial-*` directory. LeRobot metadata
is finalized and reopened before a same-filesystem rename publishes the requested root. The
converter refuses to overwrite an existing destination. It adds
`meta/embodied_ai_conversion.json`, identified by `embodied-ai.lerobot-conversion/v1`, containing
the mapping, source manifest hashes, exact instruction IDs/text, task/reset parameters, expert
provenance, source timestamps, and source-to-destination episode indices. These fields are
provenance, not policy inputs.

Run the compact video-backed form from the isolated VLA environment:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$EMBODIEDAI_ENVS/vla/bin/python" \
  scripts/data/contract_episodes_to_lerobot.py \
  "$EMBODIEDAI_DATASETS/<raw-collection>"/<episode-id-pattern> \
  --output_root "$EMBODIEDAI_DATASETS/<converted-dataset>" \
  --repo_id embodiedai/<dataset-id> --storage videos
```

`--storage images` remains useful for small debugging fixtures. On 2026-08-27 a three-frame
image-backed unit round trip finalized and reloaded successfully in no-GPU mode. The first real
attempt could not fit the temporary 2 GiB allocation and left no published partial dataset. After
returning to the 25-CPU/90-GiB GPU allocation, the accepted 108-frame Stage 6 expert episode was
successfully published as
`/root/autodl-tmp/EmbodiedAI/datasets/stage7-franka-pick-place-v1`. LeRobot 0.6.0 reopened it as
one 20 Hz episode; the video-backed front camera is a 224 x 224 AV1 stream with 108 readable frames
and a 5.4-second duration. The conversion sidecar retains the exact instruction, expert/source
provenance, and source manifest SHA-256.

The completed 20-episode corpus was converted on 2026-08-28 to
`/root/autodl-tmp/EmbodiedAI/datasets/stage7-franka-pick-place-batch-v1` with repo ID
`embodiedai/franka-pick-place-stage7-batch-v1`. It preserves all 20 episode boundaries, 2,138
frames, five exact instruction tasks, and the source task/reset/expert provenance. Its single
chunked front-camera AV1 file is 224 x 224 at 20 Hz and contains 2,138 readable frames over
106.9 seconds. The complete dataset is 8.3 MiB and has no private partial directory.

## Stage 7 LeRobot validation

`src/embodied_ai/data/lerobot_validation.py` implements the independent validation gate and
`scripts/data/validate_lerobot_dataset.py` provides the VLA-environment entry point. The validator:

1. revalidates every immutable source payload and manifest hash;
2. checks the conversion sidecar against exact source order, counts, instruction IDs/text,
   task/reset parameters, expert provenance, and mapping revision;
3. compares every LeRobot state/action value with the mapped source NPY values and checks all
   episode/frame/global indices, 20 Hz timestamps, and task indices;
4. recomputes finite 9D state and 7D action normalization inputs and compares min/max/mean/std/count
   with LeRobot's stored statistics; and
5. decodes the first and last front-camera frame of every episode through two independent
   `LeRobotDataset` instances and requires exact reload equality.

Run it without modifying either dataset:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/vla/bin/python" \
  scripts/data/validate_lerobot_dataset.py \
  --dataset_root "$EMBODIEDAI_DATASETS/stage7-franka-pick-place-batch-v1" \
  --source_root "$EMBODIEDAI_DATASETS/stage6-expert-batch-v1-20260827" \
  --repo_id embodiedai/franka-pick-place-stage7-batch-v1
```

The derived report is atomically written to
`$EMBODIEDAI_RUNS/stage7-validation/stage7-franka-pick-place-batch-v1.json`; its schema is
`embodied-ai.lerobot-validation/v1`. The accepted report passed for 20 episodes, 2,138 frames,
five tasks, and 40 decoded episode-boundary image samples. No state/action dimension was constant.
Its SHA-256 is `1758b137b4a90c62bd239eeaee8cf74dfdf3ca6d1b7626c414e81a44166f45ef`.

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
- Conversion to LeRobot format is a validated, separate Stage 7 step in the VLA environment.
- Checkpoints are accompanied by policy metadata that names the compatible schema and
  normalization statistics.
- Every training demonstration has one task, one exact instruction, and one expert provenance
  record for the complete episode.
