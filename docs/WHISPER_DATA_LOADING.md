# Whisper synthetic data: two loading modes

There are two ways to feed synthetic Whisper classification data into training. Pick one per experiment; they measure different things.

| | **Disk-backed (`synthetic_whisper`)** | **Milabench-style (`synthetic_whisper_milabench`)** |
|---|----------------------------------------|------------------------------------------------------|
| **Config name** | `synthetic_whisper` | `synthetic_whisper_milabench` |
| **Python module** | `src/data/synthetic_whisper/` | `src/data/synthetic_whisper_milabench/` |
| **How data is built** | Generate **N_SAMPLES** (5500) waveforms, extract features, save **`.pt`** cache on disk; load from cache on later runs | **n** unique samples in RAM; **`repeat`** scales `__len__` to n×repeat (cycles `i % n`) |
| **I/O during training** | Read from disk (unless fully cached in OS page cache) | None after init (pure in-memory) |
| **Matches Milabench HF bench** | No (same idea, different packaging) | Yes (generator dict + `gen()`, 10k audio samples, same extractor call pattern) |
| **Typical use** | `run_experiments_disk.sh` | `run_experiments_milabench.sh` |

## Quick commands

See [`scripts/whisper/README.md`](../scripts/whisper/README.md) for runnable scripts.

| Goal | Entry point |
|------|-------------|
| 5 min, disk data + CSV + plots | `./scripts/whisper/synthetic_disk_5min.sh` |
| 5 min, Milabench data + CSV + plots | `./scripts/whisper/milabench_5min.sh` |
| 3× batch sizes × 3 runs, **disk** | `./scripts/run_experiments_disk.sh 8 4 2` (alias: `./scripts/run_experiments.sh`) |
| 3× batch sizes × 3 runs, **Milabench** | `./scripts/run_experiments_milabench.sh 8 4 2` → `logs/experiments_milabench/` |

Wrappers `./scripts/start-whisper-resource-util.sh` and `./scripts/start-whisper-milabench-data.sh` still work and call the same scripts under `scripts/whisper/`.

## CLI flags

**Disk-backed**

- `--data synthetic_whisper`
- `--data_configs.synthetic_whisper.data_path` — cache file (default `synthetic_whisper_data.pt`)
- `--data_configs.synthetic_whisper.force_regenerate` — force rebuild

**Milabench-style**

- `--data synthetic_whisper_milabench`
- `--data_configs.synthetic_whisper_milabench.n` — unique samples in memory
- `--data_configs.synthetic_whisper_milabench.repeat` — dataset length multiplier
- `--num_workers 0` — often used with Milabench runs to match single-threaded loading

## Multi-run experiments

Use `./scripts/run_experiments_disk.sh` (disk) or `./scripts/run_experiments_milabench.sh` (Milabench). For Milabench, optional env: `DATA_N`, `DATA_REPEAT` (defaults 500 and 200).

Aggregate with `aggregate_and_plot.py`; for Milabench output pass `--experiments-dir logs/experiments_milabench`.
