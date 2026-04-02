# Plotting scripts (same as sham-bolic/COMP597-starter-code)

## plot_all_experiments.sh

Runs `aggregate_and_plot.py` for **every** `logs/experiments_disk/workers_*` and `logs/experiments_milabench/workers_*` directory. Any `batch_*` name works (`batch_128`, `batch_64`, `batch_32`, etc.); batches are processed in **numeric** order (not string order).

```bash
./scripts/plotting/plot_all_experiments.sh
./scripts/plotting/plot_all_experiments.sh --disk-only
SMOOTH=3 ./scripts/plotting/plot_all_experiments.sh
```

## plot_resource_util_steps.py

Thin wrapper for **`resource_util_steps.csv`** from **`--trainer_stats resource_util`**. Prefer **`plot_resources.py`**, which loads the same file via `load_resource_plot_df` and produces the same overview and GPU/CPU overlay.

```bash
python scripts/plotting/plot_resource_util_steps.py \
  --input logs/whisper_resource_util_milabench/resource_util_steps.csv \
  --output-dir logs/whisper_resource_util_milabench/plots
```

## plot_resources.py

Produces the same three plots as the [sham-bolic](https://github.com/sham-bolic/COMP597-starter-code) repo:

1. **resource_util.png** – 2×4 overview (GPU/CPU util, GPU/CPU memory, RAM, I/O over step)
2. **resource_util_gpu_cpu.png** – GPU vs CPU utilization overlay (or 3 panels by phase)
3. **resource_util_phases.png** – Violin plots by phase (forward/backward/optimizer), when phase data exists

### Workflow: generate data then plot

1. **Run Whisper with tracking** (writes `logs/resource_util_steps.csv`):

   ```bash
   ./scripts/whisper/synthetic_disk_5min.sh
   ```

   (Same as `./scripts/start-whisper-resource-util.sh`.) For Milabench-style in-memory data, `./scripts/whisper/milabench_5min.sh` writes to `logs/milabench_whisper/` and runs plotting. See [`docs/WHISPER_DATA_LOADING.md`](../../docs/WHISPER_DATA_LOADING.md).

   This uses `--trainer_stats resource_util` and writes into `logs/` for the disk-backed run.

2. **Plot** (default input is `logs/resource_util_steps.csv`):

   ```bash
   python scripts/plotting/plot_resources.py
   ```

   You can also pass sham-bolic **`resource_util.csv`** (same plots).

### Usage (custom paths)

```bash
# Default: read logs/resource_util_steps.csv, write PNGs to this directory
python scripts/plotting/plot_resources.py

# Custom input/output
python scripts/plotting/plot_resources.py --input path/to/resource_util_steps.csv --output-dir ./out

# Zoom and smoothing
python scripts/plotting/plot_resources.py --zoom 10 --smooth 20
```

### Expected CSV format

- **``resource_util_steps.csv``** (from **`resource_util`**): columns include `step`, `gpu_util_pct`, `cpu_util_pct`, etc. — auto-mapped to plot columns.
- **``resource_util.csv``** (sham-bolic): columns `step`, `gpu_util`, `cpu_util`, `gpu_mem_gb`, `cpu_mem_gb`, `ram_gb`, `io_read_gb`, `io_write_gb` (optional: `elapsed_s`).
- **Phase CSV** (optional, for phase plots): same columns plus `phase` (`forward` / `backward` / `optimizer`). Default path: `resource_util_substeps.csv` next to the main CSV when the input is named `resource_util.csv`.

Use **`--trainer_stats resource_util`** with **`--trainer_stats_configs.resource_util.output_dir logs`**.

Or run `./scripts/whisper/synthetic_disk_5min.sh` (wrapper: `start-whisper-resource-util.sh`).
