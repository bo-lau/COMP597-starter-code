# Experiment Guide (COMP597 Final Report)

This guide covers the experiment structure for your project report.

**Whisper data loading (disk vs Milabench-style):** see [WHISPER_DATA_LOADING.md](./WHISPER_DATA_LOADING.md) and [`scripts/whisper/README.md`](../scripts/whisper/README.md).

## Terminology

- **Step**: Single parameter update = Forward pass + Backward pass + Optimizer update
- **Phases**: Forward, Backward, Optimizer (the three components of a step)
- **Epoch**: One complete pass through the dataset

## CPU utilization in `resource_util_csv` (for your report)

`psutil` uses the **same name** for two different measures:

| API | Meaning |
|-----|--------|
| **`psutil.cpu_percent()`** (no `Process`) | **System-wide average** over **all logical cores** on the machine (0–100%). Core count is **hardware** (e.g. 192 on some nodes), not your Slurm `--cpus-per-task`. |
| **`psutil.Process().cpu_percent()`** | **This process only**, as a **sum** across cores — values **can exceed 100%**. |

This repo’s **`cpu_util` column** uses **`psutil.Process().cpu_percent()`** (the second). Values are **per-process** and **summed across cores** (e.g. **190** ≈ 190% total core usage, not 1.9%). DataLoader worker processes are **not** included in this number. Roughly, `psutil.cpu_percent() * psutil.cpu_count()` ≈ `Process().cpu_percent()` when only your Python process uses the CPU.

Plots label this as **“Process CPU (sum %, all cores)”**. **State in your report** which API you used and how figures are scaled. Use `--cpu-cores N` on overlap plots if you want to divide by **N** for a **per-core average** (0–100) on the same axis as GPU utilization.

## Data: two Whisper modes (summary)

| Mode | Script (5 min) |
|------|------------------|
| **Disk** (`synthetic_whisper`; multi-run: `run_experiments_disk.sh`) | `./scripts/whisper/synthetic_disk_5min.sh` (same as `start-whisper-resource-util.sh`) |
| **Milabench** (`synthetic_whisper_milabench`; multi-run: `run_experiments_milabench.sh`) | `./scripts/whisper/milabench_5min.sh` (same as `start-whisper-milabench-data.sh`) |

Details: [WHISPER_DATA_LOADING.md](./WHISPER_DATA_LOADING.md).

## Quick Reference

| Task | Command |
|------|---------|
| 5-min run, disk data | `./scripts/whisper/synthetic_disk_5min.sh` |
| 5-min run, Milabench data | `./scripts/whisper/milabench_5min.sh` |
| Multi-run (3× per batch), **disk** | `./scripts/run_experiments_disk.sh` (default batch: `128 64 32`) or `./scripts/run_experiments.sh` |
| Multi-run (3× per batch), **Milabench** | `./scripts/run_experiments_milabench.sh` (same default batch sizes) |
| Vary **DataLoader workers** (separate dirs per worker count) | `WORKERS="0 4 8" ./scripts/run_experiments_disk.sh` (same for `_milabench`) |
| **Full sweep** (disk + Milabench, all batches × workers) | `./scripts/run_all_experiments.sh` (see `--disk-only` / `--milabench-only` / `--dry-run`) |
| Aggregate & plot | `python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/workers_0` (pick `workers_N`) |
| Aggregate & plot (Milabench) | `python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/workers_0` |
| **Plot all workers + batches** (disk + Milabench) | `./scripts/plotting/plot_all_experiments.sh` |
| Overhead check | `./scripts/check_overhead.sh` or `./scripts/check_overhead.sh --slurm` |
| Energy (500ms sampling) | Use `--trainer_stats codecarbon` with `--trainer_stats_configs.codecarbon.measure_power_secs 0.5` |

## 1. Five-Minute Runs

Training stops after 5 minutes via `--max_time_minutes 5`. Use the scripts in the table above, or:

```bash
./scripts/srun.sh --model whisper --trainer simple --data synthetic_whisper \
  --batch_size 8 --max_time_minutes 5 --trainer_stats resource_util_csv ...
```

## 2. Phase Bar Charts

The `resource_util_csv` tracker writes `phase_times.csv` (forward/backward/optimizer ms per step). Plot with:

```bash
python scripts/plotting/plot_resources.py
# Produces phase_time_bars.png (mean ± std per phase)
```

## 3. Three Batch Sizes

Find your max batch size (power of 2), then run max, max/2, max/4. Example for max=8:

```bash
./scripts/run_experiments_disk.sh
# Milabench data (defaults: batch sizes 128 64 32):
./scripts/run_experiments_milabench.sh
# Include 8 4 2, e.g.:
./scripts/run_experiments_disk.sh 128 64 32 8 4 2
```

Output layout: `logs/experiments_disk/workers_<W>/batch_<B>/run_<R>` and `logs/experiments_milabench/workers_<W>/batch_<B>/run_<R>`. Default `WORKERS="0 4"` (set `WORKERS=0` for a single worker sweep).

## 4. Three Runs and Averaging

`run_experiments_disk.sh` / `run_experiments_milabench.sh` run each batch size 3 times. Then:

```bash
python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/workers_0
python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/workers_4
```

This averages CSVs across the 3 runs and writes to `batch_N/averaged/`, then plots under `plots/batch_N/` (under the `--experiments-dir` you pass—one worker folder at a time).

## 5. Energy Sampling (500ms)

CodeCarbon is configured to measure power every 0.5 seconds:

```bash
./scripts/srun.sh ... --trainer_stats codecarbon \
  --trainer_stats_configs.codecarbon.measure_power_secs 0.5 \
  --trainer_stats_configs.codecarbon.output_dir logs \
  ...
```

Default is already 0.5 in config.

## 6. Overhead Check

Ensure metrics collection adds &lt; 5% to total time:

```bash
./scripts/check_overhead.sh        # Local
./scripts/check_overhead.sh --slurm # On Slurm
```

Runs 5-min baseline (noop) and 5-min with resource_util_csv, reports overhead %.

## Required Experiments Checklist

- [ ] End-to-end time (no metrics) – use `--trainer_stats noop`
- [ ] End-to-end energy (CodeCarbon only) – use `--trainer_stats codecarbon`
- [ ] Timelines (GPU/CPU/memory) – use `resource_util_csv`, plot with `plot_resources.py`
- [ ] Phase time bars – `phase_time_bars.png` from `plot_resources.py`
- [ ] 3 batch sizes × 3 runs, averaged – `run_experiments_disk.sh` or `run_experiments_milabench.sh` + `aggregate_and_plot.py`
- [ ] Overhead &lt; 5% – `check_overhead.sh`
