# Data loaders

## Whisper (synthetic)

| Package | `--data` flag | Description |
|---------|----------------|-------------|
| [`synthetic_whisper/`](./synthetic_whisper/) | `synthetic_whisper` | **Disk-backed:** generate ~5.5k samples, cache as `.pt`, load for training. |
| [`synthetic_whisper_milabench/`](./synthetic_whisper_milabench/) | `synthetic_whisper_milabench` | **In-memory Milabench pattern:** `n` unique samples × `repeat` length; no training-time disk reads. |

Full comparison and commands: [`docs/WHISPER_DATA_LOADING.md`](../../docs/WHISPER_DATA_LOADING.md).
