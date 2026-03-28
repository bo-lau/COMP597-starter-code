# Whisper experiment scripts

Synthetic Whisper training uses **one of two data loaders** (see [`docs/WHISPER_DATA_LOADING.md`](../../docs/WHISPER_DATA_LOADING.md)).

## 5-minute resource runs (Slurm via `srun.sh`)

| Script | Data mode | Logs / plots |
|--------|-----------|--------------|
| [`synthetic_disk_5min.sh`](./synthetic_disk_5min.sh) | `synthetic_whisper` — cached `.pt` on disk | `logs/resource_util*.csv` → run `plot_resources.py` on `logs/` |
| [`milabench_5min.sh`](./milabench_5min.sh) | `synthetic_whisper_milabench` — in-memory, Milabench pattern | `logs/milabench_whisper/` + `milabench_whisper/` (script runs plotting) |

## Compatibility wrappers (repo root `scripts/`)

- `../start-whisper-resource-util.sh` → `synthetic_disk_5min.sh`
- `../start-whisper-milabench-data.sh` → `milabench_5min.sh`

## Multi-run (3 runs per batch size, 5 min each)

| Script | Data | Output |
|--------|------|--------|
| [`../run_experiments_disk.sh`](../run_experiments_disk.sh) | Disk `synthetic_whisper` | `logs/experiments/batch_N/run_R/` |
| [`../run_experiments_milabench.sh`](../run_experiments_milabench.sh) | Milabench `synthetic_whisper_milabench` | `logs/experiments_milabench/batch_N/run_R/` |
| [`../run_experiments.sh`](../run_experiments.sh) | Same as `run_experiments_disk.sh` | (backward-compatible alias) |

## Other

| Script | Purpose |
|--------|---------|
| [`../run-whisper-synthetic.sh`](../run-whisper-synthetic.sh) | Local `launch.py`, disk data, `noop` stats (quick test) |
