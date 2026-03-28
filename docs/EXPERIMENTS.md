# Experiment Guide (COMP597 Final Report)

This guide covers the experiment structure for your project report.

**Whisper data loading (disk vs Milabench-style):** see [WHISPER_DATA_LOADING.md](./WHISPER_DATA_LOADING.md) and [`scripts/whisper/README.md`](../scripts/whisper/README.md).

## Terminology

- **Step**: Single parameter update = Forward pass + Backward pass + Optimizer update
- **Phases**: Forward, Backward, Optimizer (the three components of a step)
- **Epoch**: One complete pass through the dataset

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
| Multi-run (3× per batch), **disk** | `./scripts/run_experiments_disk.sh 8 4 2` (or `./scripts/run_experiments.sh` — same) |
| Multi-run (3× per batch), **Milabench** | `./scripts/run_experiments_milabench.sh 8 4 2` |
| Aggregate & plot (disk, default dir) | `python scripts/plotting/aggregate_and_plot.py` |
| Aggregate & plot (Milabench runs) | `python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench` |
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
./scripts/run_experiments_disk.sh 8 4 2
# Milabench data:
./scripts/run_experiments_milabench.sh 8 4 2
```

Output: `logs/experiments/batch_*` (disk) or `logs/experiments_milabench/batch_*` (Milabench), each with `run_1/` … `run_3/`.

## 4. Three Runs and Averaging

`run_experiments_disk.sh` / `run_experiments_milabench.sh` run each batch size 3 times. Then:

```bash
python scripts/plotting/aggregate_and_plot.py
# Milabench output:
python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench
```

This averages CSVs across the 3 runs and writes to `batch_N/averaged/`, then plots under `plots/batch_N/` (under the experiments dir you pass).

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
