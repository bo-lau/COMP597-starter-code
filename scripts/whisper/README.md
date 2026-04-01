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

## Full sweep (disk + Milabench, all batch sizes × all workers)

| Script | Purpose |
|--------|---------|
| [`../run_all_experiments.sh`](../run_all_experiments.sh) | Runs `run_experiments_disk.sh` then `run_experiments_milabench.sh` with the same `WORKERS` and batch sizes (default `128 64 32`; `8 4 2` left commented in scripts). Disk default `WORKERS="0 4"`. Flags: `--disk-only`, `--milabench-only`, `--dry-run`. |

## Multi-run (3 runs per batch size, 5 min each)

| Script | Data | Output |
|--------|------|--------|
| [`../run_experiments_disk.sh`](../run_experiments_disk.sh) | Disk `synthetic_whisper` | `logs/experiments_disk/workers_W/batch_N/run_R/` (`WORKERS` env, default `0 4`) |
| [`../run_experiments_milabench.sh`](../run_experiments_milabench.sh) | Milabench | `logs/experiments_milabench/workers_W/batch_N/run_R/` |
| [`../run_experiments.sh`](../run_experiments.sh) | Same as `run_experiments_disk.sh` | (backward-compatible alias) |

## Other

| Script | Purpose |
|--------|---------|
| [`../run-whisper-synthetic.sh`](../run-whisper-synthetic.sh) | Local `launch.py`, disk data, `noop` stats (quick test) |
| [`test_one_random_run.sh`](./test_one_random_run.sh) | **One** random smoke config (`resource_util_csv`), then `plot_resources.py` → `logs/test_runs/.../plots/`. **Uses Slurm** (`srun.sh`) by default; `USE_LOCAL=1` for local `launch.py`. |
