#!/usr/bin/env python3
"""
Aggregate metrics across 3 runs (averaged) and produce final plots.

Reads from <experiments-dir>/batch_{N}/run_{1,2,3}/ (any N, e.g. batch_128, batch_64, batch_32) and writes averaged
CSVs to batch_{N}/averaged/, then calls plot_resources.

Usage:
    python scripts/plotting/aggregate_and_plot.py [--experiments-dir PATH]
    python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/workers_0
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add plotting dir for imports
PLOTTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOTTING_DIR.parent.parent
sys.path.insert(0, str(PLOTTING_DIR))

from plot_resources import (
    plot_gpu_cpu_overlap,
    plot_overview,
    plot_phase_bars,
    plot_phases_boxplot,
)


def _numeric_suffix_key(path: Path) -> tuple:
    """Sort batch_N / run_N by N (not lexicographic: batch_128 before batch_32)."""
    try:
        n = int(path.name.split("_", 1)[1])
        return (0, n)
    except (IndexError, ValueError):
        return (1, path.name)


def aggregate_csvs(paths: list) -> Optional[pd.DataFrame]:
    """Load CSVs, align by step, average across runs. Returns None if empty."""
    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    # Align by step (min length)
    min_len = min(len(d) for d in dfs)
    dfs = [d.iloc[:min_len].copy() for d in dfs]
    # Average numeric columns
    combined = dfs[0].copy()
    for col in combined.select_dtypes(include=[np.number]).columns:
        stacked = np.stack([d[col].values for d in dfs])
        combined[col] = np.mean(stacked, axis=0)
        if len(dfs) > 1:
            std_col = f"{col}_std"
            combined[std_col] = np.std(stacked, axis=0)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment runs and plot")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "experiments_disk" / "workers_0",
        help="Base directory with batch_N/run_R/ (e.g. logs/experiments_disk/workers_0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: experiments-dir/plots)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling window for timeline smoothing",
    )
    args = parser.parse_args()

    exp_dir = args.experiments_dir
    if not exp_dir.exists():
        print(f"Experiments dir not found: {exp_dir}")
        print("Run ./scripts/run_experiments_disk.sh or ./scripts/run_experiments_milabench.sh first.")
        sys.exit(1)

    batch_dirs = sorted(exp_dir.glob("batch_*"), key=_numeric_suffix_key)
    if not batch_dirs:
        print(f"No batch_* dirs in {exp_dir}")
        sys.exit(1)

    out_dir = args.output_dir or (exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch_dir in batch_dirs:
        batch_name = batch_dir.name  # e.g. batch_128
        run_dirs = sorted(batch_dir.glob("run_*"), key=_numeric_suffix_key)
        if len(run_dirs) < 1:
            print(f"Skipping {batch_name}: no runs")
            continue

        avg_dir = batch_dir / "averaged"
        avg_dir.mkdir(exist_ok=True)
        df_sub_avg = None

        # Aggregate resource_util.csv
        main_csvs = [d / "resource_util.csv" for d in run_dirs]
        df_main = aggregate_csvs(main_csvs)
        if df_main is not None:
            out_csv = avg_dir / "resource_util.csv"
            df_main[[c for c in df_main.columns if not c.endswith("_std")]].to_csv(out_csv, index=False)
            print(f"  {batch_name}: aggregated {len(run_dirs)} runs -> {out_csv}")

        # Aggregate substeps (phase data)
        sub_csvs = [d / "resource_util_substeps.csv" for d in run_dirs]
        dfs_sub = [pd.read_csv(p) for p in sub_csvs if p.exists()]
        if dfs_sub:
            df_sub = pd.concat(dfs_sub, ignore_index=True)
            df_sub_avg = df_sub.groupby(["step", "phase"], as_index=False).mean(numeric_only=True)
            sub_out = avg_dir / "resource_util_substeps.csv"
            df_sub_avg.to_csv(sub_out, index=False)

        # Aggregate phase_times.csv
        phase_csvs = [d / "phase_times.csv" for d in run_dirs]
        df_phase = aggregate_csvs(phase_csvs)
        if df_phase is not None:
            phase_out = avg_dir / "phase_times.csv"
            df_phase[[c for c in df_phase.columns if not c.endswith("_std")]].to_csv(phase_out, index=False)

        # Plot for this batch
        if df_main is not None:
            batch_out = out_dir / batch_name
            batch_out.mkdir(exist_ok=True)
            avg_dir = batch_dir / "averaged"
            plot_overview(df_main, batch_out / "resource_util.png", smooth=args.smooth)
            plot_gpu_cpu_overlap(df_main, batch_out / "resource_util_gpu_cpu.png", smooth=args.smooth)
            if df_sub_avg is not None and "phase" in df_sub_avg.columns:
                plot_phases_boxplot(df_sub_avg, batch_out / "resource_util_phases.png")
            phase_path = avg_dir / "phase_times.csv"
            if phase_path.exists():
                plot_phase_bars(phase_path, batch_out / "phase_time_bars.png")
            print(f"  Plots -> {batch_out}")

    print(f"\nDone. Plots in {out_dir}")


if __name__ == "__main__":
    main()
