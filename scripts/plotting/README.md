# Plotting scripts (same as sham-bolic/COMP597-starter-code)

## plot_resources.py

Produces the same three plots as the [sham-bolic](https://github.com/sham-bolic/COMP597-starter-code) repo:

1. **resource_util.png** – 2×4 overview (GPU/CPU util, GPU/CPU memory, RAM, I/O over step)
2. **resource_util_gpu_cpu.png** – GPU vs CPU utilization overlay (or 3 panels by phase)
3. **resource_util_phases.png** – Violin plots by phase (forward/backward/optimizer), when phase data exists

### Workflow: generate data then plot

1. **Run Whisper with tracking** (writes `logs/resource_util.csv` and `logs/resource_util_substeps.csv`):

   ```bash
   ./scripts/whisper/synthetic_disk_5min.sh
   ```

   (Same as `./scripts/start-whisper-resource-util.sh`.) For Milabench-style in-memory data, `./scripts/whisper/milabench_5min.sh` writes to `logs/milabench_whisper/` and runs plotting. See [`docs/WHISPER_DATA_LOADING.md`](../../docs/WHISPER_DATA_LOADING.md).

   This uses `--trainer_stats resource_util_csv` and writes into `logs/` for the disk-backed run.

2. **Plot** (default input is `logs/resource_util.csv`):

   ```bash
   python scripts/plotting/plot_resources.py
   ```

### Usage (custom paths)

```bash
# Default: read logs/resource_util.csv, write PNGs to this directory
python scripts/plotting/plot_resources.py

# Custom input/output
python scripts/plotting/plot_resources.py --input path/to/resource_util.csv --output-dir ./out

# Zoom and smoothing
python scripts/plotting/plot_resources.py --zoom 10 --smooth 20
```

### Expected CSV format

- **Main CSV** (`resource_util.csv`): columns `step`, `gpu_util`, `cpu_util`, `gpu_mem_gb`, `cpu_mem_gb`, `ram_gb`, `io_read_gb`, `io_write_gb` (optional: `elapsed_s`).
- **Phase CSV** (optional, for phase plots): same columns plus `phase` (`forward` / `backward` / `optimizer`). Default path: `resource_util_substeps.csv` next to the main CSV.

The tracker in this repo that produces this format is **`resource_util_csv`**. Use it with:

- `--trainer_stats resource_util_csv`
- `--trainer_stats_configs.resource_util_csv.output_dir logs`
- (optional) `--trainer_stats_configs.resource_util_csv.output_file resource_util.csv`

Or run `./scripts/whisper/synthetic_disk_5min.sh` (wrapper: `start-whisper-resource-util.sh`).
