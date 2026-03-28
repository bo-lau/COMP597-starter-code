#!/usr/bin/env python3
"""
Aggregate and plot resource utilization.

Two modes:
  1) Whisper summary mode (default): reads ../whisper-results/whisper-results-*.txt
     and produces resource_util_summary_table.csv and resource_util_summary.png.

  2) CSV mode (--input PATH): reads resource_util.csv (and optional phase substep CSV)
     and produces: resource_util.png, resource_util_gpu_cpu.png, resource_util_phases.png
     (and optional zoomed views with --zoom N).

Usage:
    python scripts/plot_resource_util.py
    python scripts/plot_resource_util.py --input logs/resource_util.csv [--output-dir DIR]
    python scripts/plot_resource_util.py --input logs/resource_util.csv --zoom 10 --smooth 20
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(REPO_ROOT), "whisper-results")

# --- CSV-mode metrics (same as reference script) ---
METRICS_OVERVIEW = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "CPU Util (%)", "CPU Utilization"),
    (("gpu_mem_gb", "gpu_mem_pct"), ("GB", "%"), "GPU Memory"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
    ("ram_gb", "GB", "System RAM"),
    ("io_read_gb", "GB", "I/O Read"),
    ("io_write_gb", "GB", "I/O Write"),
]

METRICS_BOXPLOT = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "CPU Util (%)", "CPU Utilization"),
    (("gpu_mem_pct", "gpu_mem_gb"), ("%", "GB"), "GPU Memory"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
]


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def _smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def plot_overview(df: pd.DataFrame, output_path: Path, smooth: int = 1) -> None:
    _setup_style()
    has_phase = "phase" in df.columns
    if has_phase:
        df_plot = df.groupby("step", as_index=False).mean(numeric_only=True)
    else:
        df_plot = df.copy()
    x_col = "elapsed_s" if "elapsed_s" in df_plot.columns else "step"
    xlabel = "Time (s)" if x_col == "elapsed_s" else "Step"
    metrics = METRICS_OVERVIEW
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if isinstance(metric[0], tuple):
            col = next((c for c in metric[0] if c in df_plot.columns), None)
            ylabel = metric[1][0] if col == metric[0][0] else metric[1][1]
            title = metric[2]
        else:
            col, ylabel, title = metric
        if col is None or col not in df_plot.columns:
            ax.text(0.5, 0.5, "Column not found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        y = _smooth_series(df_plot[col], smooth)
        ax.plot(df_plot[x_col], y, linewidth=1.2, color="#2980b9")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ymax = df_plot[col].max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)
        if col == "io_write_gb" and df_plot[col].max() < 1e-3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.0f}"))
            ax.set_ylabel("KB", fontsize=9)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")
    fig.supxlabel(xlabel, fontsize=10, y=-0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview to {output_path}")


def plot_gpu_cpu_overlap(
    df: pd.DataFrame,
    output_path: Path,
    cpu_cores: Optional[int] = None,
    smooth: int = 1,
) -> None:
    _setup_style()
    if "gpu_util" not in df.columns or "cpu_util" not in df.columns:
        print("Skipping GPU/CPU overlap plot: gpu_util or cpu_util not in data")
        return
    phase_order = ["forward", "backward", "optimizer"]
    has_phase = "phase" in df.columns and df["phase"].isin(phase_order).any()

    def _plot_overlay(ax: plt.Axes, df_plot: pd.DataFrame, title: str, x_col: str = "step") -> None:
        gpu_smooth = _smooth_series(df_plot["gpu_util"], smooth)
        cpu_util = df_plot["cpu_util"].copy()
        if cpu_cores is not None and cpu_cores > 0:
            cpu_util = cpu_util / cpu_cores
        cpu_smooth = _smooth_series(cpu_util, smooth)
        ax.plot(df_plot[x_col], gpu_smooth, linewidth=1.5, color="#2980b9", label="GPU Util (%)")
        ax.plot(
            df_plot[x_col], cpu_smooth, linewidth=1.5, color="#e74c3c",
            label="CPU Util (%)" + (f" (norm/{cpu_cores})" if cpu_cores else ""),
        )
        ax.set_ylabel("Utilization (%)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.legend(loc="upper right", fontsize=8)
        ymax = max(gpu_smooth.max(), cpu_smooth.max())
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)

    x_col = "elapsed_s" if "elapsed_s" in df.columns else "step"
    xlabel = "Time (s)" if x_col == "elapsed_s" else "Step"
    if has_phase:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, phase in zip(axes, phase_order):
            df_phase = df[df["phase"] == phase].copy()
            if df_phase.empty:
                ax.text(0.5, 0.5, f"No data for {phase}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(phase.capitalize(), fontsize=10, fontweight="medium")
                continue
            df_phase = df_phase.sort_values(x_col)
            _plot_overlay(ax, df_phase, phase.capitalize(), x_col)
        axes[-1].set_xlabel(xlabel, fontsize=10)
    else:
        df_plot = (
            df if "elapsed_s" in df.columns
            else (df.groupby("step", as_index=False).mean(numeric_only=True) if "step" in df.columns else df)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_overlay(ax, df_plot, "GPU vs CPU Utilization", x_col)
        ax.set_xlabel(xlabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved GPU/CPU overlap to {output_path}")


def plot_phases_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    _setup_style()
    phase_order = ["forward", "backward", "optimizer"]
    phase_colors = {"forward": "#27ae60", "backward": "#e74c3c", "optimizer": "#3498db"}
    df_plot = df[df["phase"].isin(phase_order)].copy()
    metrics = METRICS_BOXPLOT
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        if isinstance(metric[0], tuple):
            col = next((c for c in metric[0] if c in df_plot.columns), None)
            ylabel = metric[1][0] if col == metric[0][0] else metric[1][1]
            title = metric[2]
        else:
            col, ylabel, title = metric
        if col is None or col not in df_plot.columns:
            ax.text(0.5, 0.5, "Column not found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        data = [df_plot.loc[df_plot["phase"] == p, col].dropna().values for p in phase_order]
        parts = ax.violinplot(data, positions=[0, 1, 2], widths=0.7, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(phase_colors[phase_order[i]])
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(phase_order, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.set_ylabel(ylabel, fontsize=9)
        ymax = max((np.max(d) for d in data if len(d) > 0), default=1)
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.supxlabel("Phase", fontsize=10, y=-0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phase plot to {output_path}")


def parse_summary_file(path: str) -> Dict:
    """Parse a single resource_utilization_summary.txt-style file."""
    metrics: Dict[str, float] = {}
    current_section = None

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Detect section headers
            if line.startswith("###############"):
                if "GPU COMPUTE UTILIZATION" in line:
                    current_section = "gpu_util_mean"
                elif "GPU MEMORY UTILIZATION" in line:
                    current_section = "gpu_mem_util_mean"
                elif "GPU POWER CONSUMPTION" in line:
                    current_section = "gpu_power_mean"
                elif "CPU MEMORY (MB)" in line:
                    current_section = "cpu_mem_mean_mb"
                elif "DISK READ (MB)" in line:
                    current_section = "disk_read_mean_mb"
                elif "DISK WRITE (MB)" in line:
                    current_section = "disk_write_mean_mb"
                else:
                    current_section = None
                continue

            # Parse mean line inside a section
            if current_section and line.lower().startswith("mean"):
                # Format: "mean   : 74.74%" or "mean   : 203.04"
                try:
                    _, value_part = line.split(":", 1)
                    value_str = value_part.strip().split()[0].rstrip("%")
                    metrics[current_section] = float(value_str)
                except Exception:
                    pass

            # Total training steps
            if line.startswith("Total training steps:"):
                try:
                    _, value_part = line.split(":", 1)
                    metrics["steps"] = int(value_part.strip())
                except Exception:
                    pass

    return metrics


def collect_all_results() -> pd.DataFrame:
    pattern = os.path.join(RESULTS_DIR, "whisper-results-*.txt")
    files: List[str] = sorted(glob.glob(pattern))

    if not files:
        raise SystemExit(f"No result files found matching {pattern}")

    rows = []
    for path in files:
        metrics = parse_summary_file(path)
        # Use timestamp (or full filename) as run label
        basename = os.path.basename(path)
        run_label = os.path.splitext(basename)[0].replace("whisper-results-", "")
        metrics["run"] = run_label
        rows.append(metrics)

    df = pd.DataFrame(rows)
    # Sort by run label just to keep plots ordered in time
    df = df.sort_values("run")
    return df


def plot_summary(df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save raw table for inspection
    table_path = os.path.join(RESULTS_DIR, "resource_util_summary_table.csv")
    df.to_csv(table_path, index=False)

    metrics_to_plot = [
        ("gpu_util_mean", "Average GPU Compute Utilization (%)"),
        ("gpu_mem_util_mean", "Average GPU Memory Utilization (%)"),
        ("gpu_power_mean", "Average GPU Power (W)"),
        ("cpu_mem_mean_mb", "Average CPU Memory (MB)"),
        ("disk_read_mean_mb", "Average Disk Read (MB)"),
        ("disk_write_mean_mb", "Average Disk Write (MB)"),
    ]

    num_metrics = len(metrics_to_plot)
    ncols = 2
    nrows = (num_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    x = range(len(df))
    labels = df["run"].tolist()

    for idx, (col, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue

        # Plot as a line chart over runs instead of bars
        ax.plot(x, df[col], marker="o")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="both", linestyle="--", alpha=0.3)

    # Hide any unused subplots
    for j in range(len(metrics_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "resource_util_summary.png")
    plt.savefig(out_path, dpi=150)
    print(f"Wrote table to {table_path}")
    print(f"Wrote plots to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot resource utilization (Whisper summary or CSV)")
    parser.add_argument("--input", "-i", type=Path, default=None,
        help="Input CSV (resource_util.csv). If omitted, use Whisper summary mode.")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
        help="Output directory for CSV-mode plots (default: script dir).")
    parser.add_argument("--substep-input", type=Path, default=None,
        help="Substep (phase) CSV. Default: {input_dir}/resource_util_substeps.csv when input is resource_util.csv.")
    parser.add_argument("--zoom", "-z", type=int, default=None, metavar="N",
        help="CSV mode: also produce zoomed overview of N steps.")
    parser.add_argument("--zoom-start", type=int, default=1, metavar="STEP",
        help="First step for zoom window (default: 1).")
    parser.add_argument("--cpu-cores", type=int, default=4, metavar="N",
        help="CSV mode: normalize CPU util by N cores in GPU/CPU overlap (default: 4).")
    parser.add_argument("--no-normalize-cpu", action="store_true",
        help="Disable CPU normalization in overlap plot.")
    parser.add_argument("--smooth", "-s", type=int, default=1, metavar="N",
        help="CSV mode: rolling window for line smoothing (default: 1, disabled).")
    args = parser.parse_args()

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df = pd.read_csv(args.input)
        has_phase = "phase" in df.columns
        if not has_phase:
            substep_path = args.substep_input
            if substep_path is None and args.input.name == "resource_util.csv":
                substep_path = args.input.parent / "resource_util_substeps.csv"
            if substep_path is not None and substep_path.exists():
                df_substep = pd.read_csv(substep_path)
                if "phase" in df_substep.columns:
                    df = df_substep
                    has_phase = True
        output_dir = args.output_dir or Path(__file__).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cpu_cores = None if args.no_normalize_cpu else args.cpu_cores
        plot_overview(df, output_dir / "resource_util.png", smooth=args.smooth)
        plot_gpu_cpu_overlap(df, output_dir / "resource_util_gpu_cpu.png", cpu_cores=cpu_cores, smooth=args.smooth)
        if has_phase:
            plot_phases_boxplot(df, output_dir / "resource_util_phases.png")
        if args.zoom is not None:
            step_end = args.zoom_start + args.zoom - 1
            mask = (df["step"] >= args.zoom_start) & (df["step"] <= step_end)
            df_zoom = df[mask]
            if df_zoom.empty:
                raise ValueError(f"No data in step range {args.zoom_start}-{step_end}")
            plot_overview(df_zoom, output_dir / f"resource_util_zoom_steps_{args.zoom_start}-{step_end}.png", smooth=args.smooth)
            plot_gpu_cpu_overlap(df_zoom, output_dir / f"resource_util_gpu_cpu_zoom_steps_{args.zoom_start}-{step_end}.png", cpu_cores=cpu_cores, smooth=args.smooth)
    else:
        df = collect_all_results()
        plot_summary(df)


if __name__ == "__main__":
    main()

