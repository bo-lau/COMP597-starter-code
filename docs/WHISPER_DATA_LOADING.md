# Whisper synthetic data: two loading modes

There are two ways to feed synthetic Whisper classification data into training. Pick one per experiment; they measure different things.

| | **Disk-backed (`synthetic_whisper`)** | **Milabench-style (`synthetic_whisper_milabench`)** |
|---|----------------------------------------|------------------------------------------------------|
| **Config name** | `synthetic_whisper` | `synthetic_whisper_milabench` |
| **Python module** | `src/data/synthetic_whisper/` | `src/data/synthetic_whisper_milabench/` |
| **How data is built** | Generate **N_SAMPLES** (5500) waveforms, extract features, save **`.pt`** cache on disk; load from cache on later runs | **`batch_size`** unique samples in RAM (same rule as sham-bolic `synthetic_whisper` memory mode); **`repeat`** scales `__len__` to batch_size×repeat (cycles `i % n`) |
| **I/O during training** | Read from disk (unless fully cached in OS page cache) | None after init (pure in-memory) |
| **Matches Milabench HF bench** | No (same idea, different packaging) | Mostly (generator dict + `gen()`); audio length **16k** @ 16 kHz (1 s), aligned with sham-bolic memory samples |
| **Typical use** | `run_experiments_disk.sh` | `run_experiments_milabench.sh` |

## Quick commands

See [`scripts/whisper/README.md`](../scripts/whisper/README.md) for runnable scripts.

| Goal | Entry point |
|------|-------------|
| 5 min, disk data + CSV + plots | `./scripts/whisper/synthetic_disk_5min.sh` |
| 5 min, Milabench data + CSV + plots | `./scripts/whisper/milabench_5min.sh` |
| 3× batch sizes × 3 runs, **disk** | `./scripts/run_experiments_disk.sh` (default batch sizes `128 64 32`; alias: `./scripts/run_experiments.sh`) |
| 3× batch sizes × 3 runs, **Milabench** | `./scripts/run_experiments_milabench.sh` → `logs/experiments_milabench/workers_<W>/` |

Wrappers `./scripts/start-whisper-resource-util.sh` and `./scripts/start-whisper-milabench-data.sh` still work and call the same scripts under `scripts/whisper/`.

## CLI flags

**Disk-backed**

- `--data synthetic_whisper`
- `--data_configs.synthetic_whisper.data_path` — cache file (default `synthetic_whisper_data.pt`)
- `--data_configs.synthetic_whisper.force_regenerate` — force rebuild

**Milabench-style**

- `--data synthetic_whisper_milabench`
- `--batch_size B` — number of **unique** tensors in memory (**`n = B`**, same idea as sham-bolic `memory_only`)
- `--data_configs.synthetic_whisper_milabench.repeat` — dataset length multiplier
- `--num_workers N` — multi-run scripts set this per sweep; compare with `WORKERS="0 4 8" ./scripts/run_experiments_milabench.sh ...`

## Multi-run experiments

Use `./scripts/run_experiments_disk.sh` (disk) or `./scripts/run_experiments_milabench.sh` (Milabench). **`WORKERS`** — space-separated `num_workers` values, each under `workers_<W>/`. For Milabench sweeps, **`MILABENCH_TOTAL_SAMPLES`** (default `16000`) fixes **`len(dataset) = batch_size × repeat`** across batch sizes by setting **`repeat = MILABENCH_TOTAL_SAMPLES / batch_size`** per run.

**Memory (Milabench):** RAM is dominated by **`batch_size`** (unique cached tensors = **`n`**). **`repeat`** sets epoch length (`len = n × repeat`). **`num_workers > 0`** copies the dataset into worker processes — use **`WORKERS=0`** on small nodes or smaller batch sizes. Scripts default to **`WORKERS=0`** for Slurm.

### vs sham-bolic `synthetic_whisper` (memory / `memory_only`)

After aligning **audio length (16k)** and **`n = batch_size`**, the remaining **functional** differences are:

| | **sham-bolic memory** | **This repo `synthetic_whisper_milabench`** |
|---|------------------------|---------------------------------------------|
| **CLI** | `--data synthetic_whisper` + `memory_only=1` / `data_type=memory` | `--data synthetic_whisper_milabench` only |
| **Sample construction** | `_one_sample` in a loop (`_sample_list`) | Milabench-style **generator dict** (`igen` / `ogen`, `gen_one()`, list comprehension) — picklable for `spawn` workers |
| **`repeat` default** | `1` in upstream config | `10` in `synthetic_whisper_milabench` config (override with `--data_configs...repeat`) |
| **Disk modes** | Same codebase also supports chunks / shards / memmap | In-memory only (no `data_path` cache) |

Aggregate with `aggregate_and_plot.py --experiments-dir logs/experiments_disk/workers_0` (choose one `workers_*` per plot batch).
