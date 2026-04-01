#!/usr/bin/env python3
"""
Plot outputs from ``--trainer_stats resource_util`` (``resource_util_steps.csv``).

That CSV uses different column names than ``resource_util_csv``'s ``resource_util.csv``.
This script maps them and reuses ``plot_resources.plot_overview`` and
``plot_gpu_cpu_overlap`` (same figures as the CSV pipeline, minus phase/substep plots).

Usage:
  python scripts/plotting/plot_resource_util_steps.py \\
    --input logs/whisper_resource_util_milabench/resource_util_steps.csv \\
    --output-dir logs/whisper_resource_util_milabench/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/plotting/plot_resource_util_steps.py` from repo root
_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from plot_resources import plot_gpu_cpu_overlap, plot_overview


def resource_util_steps_to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``resource_util_steps.csv`` columns to names expected by ``plot_resources``."""
    need = {"step", "gpu_util_pct", "cpu_util_pct"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(need)}, got {list(df.columns)}. "
            "Is this a resource_util_steps.csv from --trainer_stats resource_util?"
        )
    out = pd.DataFrame()
    out["step"] = df["step"]
    out["gpu_util"] = df["gpu_util_pct"]
    out["cpu_util"] = df["cpu_util_pct"]
    if "gpu_mem_alloc_mb" in df.columns:
        out["gpu_mem_gb"] = df["gpu_mem_alloc_mb"] / 1024.0
    if "gpu_mem_util_pct" in df.columns:
        out["gpu_mem_pct"] = df["gpu_mem_util_pct"]
    if "cpu_mem_mb" in df.columns:
        out["cpu_mem_gb"] = df["cpu_mem_mb"] / 1024.0
    if "disk_read_mb" in df.columns:
        out["io_read_gb"] = df["disk_read_mb"] / 1024.0
    if "disk_write_mb" in df.columns:
        out["io_write_gb"] = df["disk_write_mb"] / 1024.0
    # System RAM is not recorded by resource_util.py — overview panel stays empty.
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot resource_util_steps.csv (trainer_stats resource_util)")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to resource_util_steps.csv",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for PNGs (default: same dir as CSV, subdir plots/)",
    )
    parser.add_argument("--smooth", "-s", type=int, default=1, help="Rolling mean window (default: 1)")
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=0,
        metavar="N",
        help="If N>0, divide cpu_util by N in GPU/CPU overlap plot.",
    )
    parser.add_argument("--no-normalize-cpu", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    raw = pd.read_csv(args.input)
    df = resource_util_steps_to_plot_df(raw)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = args.input.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_cores = None if args.no_normalize_cpu else (args.cpu_cores if args.cpu_cores > 0 else None)

    plot_overview(df, out_dir / "resource_util.png", smooth=args.smooth)
    plot_gpu_cpu_overlap(df, out_dir / "resource_util_gpu_cpu.png", cpu_cores=cpu_cores, smooth=args.smooth)
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
