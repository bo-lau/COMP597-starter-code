# Experiment Guide (COMP597 Final Report)

This guide covers the experiment structure for your project report.

**Whisper data loading (disk vs Milabench-style):** see [WHISPER_DATA_LOADING.md](./WHISPER_DATA_LOADING.md) and [`scripts/whisper/README.md`](../scripts/whisper/README.md).

## Terminology

- **Step**: Single parameter update = Forward pass + Backward pass + Optimizer update
- **Phases**: Forward, Backward, Optimizer (the three components of a step)
- **Epoch**: One complete pass through the dataset

## CPU utilization in `resource_util` / `resource_util_steps.csv` (for your report)

`psutil` uses the **same name** for two different measures:

| API | Meaning |
|-----|--------|
| **`psutil.cpu_percent()`** (no `Process`) | **System-wide average** over **all logical cores** on the machine (0–100%). Core count is **hardware** (e.g. 192 on some nodes), not your Slurm `--cpus-per-task`. |
| **`psutil.Process().cpu_percent()`** | **This process only**, as a **sum** across cores — values **can exceed 100%**. |

This repo’s **`cpu_util_pct` / mapped `cpu_util`** uses **`psutil.Process().cpu_percent()`** (the second). Values are **per-process** and **summed across cores** (e.g. **190** ≈ 190% total core usage, not 1.9%). DataLoader worker processes are **not** included in this number. Roughly, `psutil.cpu_percent() * psutil.cpu_count()` ≈ `Process().cpu_percent()` when only your Python process uses the CPU.

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
| Aggregate & plot | `python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/resource_util/workers_0` (pick `resource_util` / `workers_N`) |
| Aggregate & plot (Milabench) | `python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/resource_util/workers_0` |
| **Plot all workers + batches** (disk + Milabench) | `./scripts/plotting/plot_all_experiments.sh` |
| Overhead check | `./scripts/check_overhead.sh` or `./scripts/check_overhead.sh --slurm` |
| Energy (500ms sampling) | Use `--trainer_stats codecarbon` with `--trainer_stats_configs.codecarbon.measure_power_secs 0.5` |
| Energy (minimal polling / ~one coarse sample on short runs) | `--trainer_stats codecarbon_e2e` (default `measure_power_secs` 86400s; override if your job runs longer). Same CodeCarbon implementation as `codecarbon`; compare wall time vs `codecarbon` to see overhead of frequent sampling. |
| **Phase times only** (`phase_times.csv` → `phase_time_bars.png`) | `--trainer_stats phase_times` with `--trainer_stats_configs.phase_times.output_dir …` (separate from `resource_util`; run again if you need both metrics and phase bars). |

## 1. Five-Minute Runs

Training stops after 5 minutes via `--max_time_minutes 5`. Use the scripts in the table above, or:

```bash
./scripts/srun.sh --model whisper --trainer simple --data synthetic_whisper \
  --batch_size 32 --max_time_minutes 5 --trainer_stats resource_util \
  --trainer_stats_configs.resource_util.output_dir logs ...
```

## 2. Phase Bar Charts

**`resource_util`** writes **`resource_util_steps.csv`** (and a summary txt), not phase timings. For **`phase_times.csv`** and **`phase_time_bars.png`**, use **`--trainer_stats phase_times`** (writes only phase timings; same `aggregate_and_plot` layout). **`plot_resources.py`** can still build **`phase_time_bars.png`** when **`phase_times.csv`** sits next to your main CSV. **`resource_util_phases.png`** needs a substep/phase CSV; not produced by `phase_times` alone.

```bash
python scripts/plotting/plot_resources.py
# Produces phase_time_bars.png when phase_times.csv exists beside the input CSV
```

## 3. Three Batch Sizes

Find your max batch size (power of 2), then run max, max/2, max/4. Default sweeps use **128, 64, 32** (all ≥ 32).

```bash
./scripts/run_experiments_disk.sh
# Milabench data (defaults: batch sizes 128 64 32):
./scripts/run_experiments_milabench.sh
```

Output layout: `logs/experiments_{disk|milabench}/<trainer_stats>/workers_<W>/batch_<B>/run_<R>`. Set **`TRAINER_STATS`** (space-separated) to choose which stats to run; default includes `resource_util`, `resource_util_max`, `phase_times`, `noop`, `simple`, `codecarbon`, `codecarbon_e2e` (omit CodeCarbon in `TRAINER_STATS` for faster sweeps). Legacy trees without the `<trainer_stats>/` segment still work for plotting if you pass that path to `aggregate_and_plot.py`. Default `WORKERS="0 4"` (set `WORKERS=0` for a single worker sweep).

## 4. Three Runs and Averaging

`run_experiments_disk.sh` / `run_experiments_milabench.sh` run each batch size 3 times. Then:

```bash
python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/resource_util/workers_0
python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/resource_util/workers_4
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

Runs 5-min baseline (noop) and 5-min with resource_util, reports overhead %.

## Required Experiments Checklist

- [ ] End-to-end time (no metrics) – use `--trainer_stats noop`
- [ ] End-to-end energy (CodeCarbon only) – use `--trainer_stats codecarbon`
- [ ] Timelines (GPU/CPU/memory) – use `resource_util`, plot with `plot_resources.py` (reads `resource_util_steps.csv`)
- [ ] Phase time bars – optional; `phase_time_bars.png` only if `phase_times.csv` is present beside a sham-bolic-style run
- [ ] 3 batch sizes × 3 runs, averaged – `run_experiments_disk.sh` or `run_experiments_milabench.sh` + `aggregate_and_plot.py`
- [ ] Overhead &lt; 5% – `check_overhead.sh`
