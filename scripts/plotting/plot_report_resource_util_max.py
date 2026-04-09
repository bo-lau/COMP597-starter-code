#!/usr/bin/env python3
"""
Generate report-equivalent figures using resource_util_max data.

Produces the same figure types as the report but from the resource_util_max
trainer stats (no CUDA sync, no per-phase breakdowns). Figures that require
phase data (phase_time_bars, resource_phases violins) are skipped.

Output: report/figures_resource_util_max/
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = REPO_ROOT / "logs"
OUT_DIR = REPO_ROOT / "report" / "figures_resource_util_max"


def setup_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            pass


def smooth(series: pd.Series, window: int = 1) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def load_avg_csv(data_source: str, workers: int, batch: int) -> Optional[pd.DataFrame]:
    path = LOGS / data_source / "resource_util_max" / f"workers_{workers}" / f"batch_{batch}" / "averaged" / "resource_util.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Figure 3 equivalent: GPU utilization across batch sizes ─────────
def plot_gpu_util_comparison(data_source: str, workers: int, out_path: Path):
    setup_style()
    batch_sizes = [32, 64, 128]
    colors = {"32": "#2980b9", "64": "#e74c3c", "128": "#27ae60"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for bs in batch_sizes:
        df = load_avg_csv(data_source, workers, bs)
        if df is None or "gpu_util" not in df.columns:
            continue
        ax.plot(df["step"], df["gpu_util"], linewidth=1.5,
                color=colors[str(bs)], label=f"B={bs}")

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("GPU Utilization (%)", fontsize=10)
    ds_label = "Disk" if "disk" in data_source else "Milabench"
    ax.set_title(f"GPU Utilization Across Batch Sizes ({ds_label}, w={workers}) — resource_util_max",
                 fontsize=11, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 4 equivalent: GPU + CPU utilization overlay ──────────────
def plot_gpu_cpu_overlay(data_source: str, workers: int, batch: int, out_path: Path):
    setup_style()
    df = load_avg_csv(data_source, workers, batch)
    if df is None:
        print(f"  Skipped (no data): {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["gpu_util"], linewidth=1.5, color="#2980b9",
            label="GPU Util (%)")
    ax.plot(df["step"], df["cpu_util"], linewidth=1.5, color="#e74c3c",
            label="Process CPU (sum %, all cores)")
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Utilization (%)", fontsize=10)
    ds_label = "Disk" if "disk" in data_source else "Milabench"
    ax.set_title(f"GPU vs CPU Utilization ({ds_label}, w={workers}, B={batch}) — resource_util_max",
                 fontsize=11, fontweight="medium")
    ymax = max(df["gpu_util"].max(), df["cpu_util"].max())
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 5 equivalent: full resource overview (2x4 grid) ─────────
METRICS_OVERVIEW = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "Process CPU (sum %)", "Process CPU (sum %, all cores)"),
    (("gpu_mem_gb", "gpu_mem_pct"), ("GB", "%"), "GPU Memory"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
    ("ram_gb", "GB", "System RAM"),
    ("io_read_gb", "GB", "I/O Read"),
    ("io_write_gb", "GB", "I/O Write"),
]


def plot_resource_overview(data_source: str, workers: int, batch: int, out_path: Path):
    setup_style()
    df = load_avg_csv(data_source, workers, batch)
    if df is None:
        print(f"  Skipped (no data): {out_path.name}")
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(METRICS_OVERVIEW):
        ax = axes_flat[idx]
        if isinstance(metric[0], tuple):
            col = next((c for c in metric[0] if c in df.columns), None)
            ylabel = metric[1][0] if col == metric[0][0] else metric[1][1]
            title = metric[2]
        else:
            col, ylabel, title = metric

        if col is None or col not in df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        ax.plot(df["step"], df[col], linewidth=1.2, color="#2980b9")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ymax = df[col].max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)
        if col == "io_write_gb" and df[col].max() < 1e-3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.0f}"))
            ax.set_ylabel("KB", fontsize=9)

    for idx in range(len(METRICS_OVERVIEW), len(axes_flat)):
        axes_flat[idx].axis("off")

    ds_label = "Disk" if "disk" in data_source else "Milabench"
    fig.suptitle(f"Resource Timeline ({ds_label}, w={workers}, B={batch}) — resource_util_max",
                 fontsize=12, fontweight="medium")
    fig.supxlabel("Step", fontsize=10, y=-0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Overhead comparison (from trainer_overhead_data.csv) ────────────
def plot_overhead_vs_noop(workers: int, out_path: Path):
    setup_style()
    csv_path = LOGS / "comparison_plots" / "trainer_overhead_data.csv"
    if not csv_path.exists():
        print(f"  Skipped (no overhead data): {out_path.name}")
        return

    df = pd.read_csv(csv_path)
    df = df[(df["workers"] == workers) & (df["trainer"] != "noop")]
    if df.empty:
        return

    trainers_order = ["simple", "phase_times", "resource_util", "resource_util_max",
                      "codecarbon", "codecarbon_e2e"]
    colors_map = {
        "simple": "#3498db", "phase_times": "#9b59b6", "resource_util": "#2ecc71",
        "resource_util_max": "#e67e22", "codecarbon": "#e74c3c", "codecarbon_e2e": "#1abc9c",
    }

    data_sources = sorted(df["data_source"].unique())
    batch_sizes = sorted(df["batch_size"].unique())
    ncols = len(batch_sizes)
    nrows = len(data_sources)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for ri, ds in enumerate(data_sources):
        for ci, bs in enumerate(batch_sizes):
            ax = axes[ri][ci]
            subset = df[(df["data_source"] == ds) & (df["batch_size"] == bs)]
            subset = subset.set_index("trainer").reindex(
                [t for t in trainers_order if t in subset["trainer"].values]
            ).reset_index()
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            bar_colors = [colors_map.get(t, "gray") for t in subset["trainer"]]
            highlight = ["resource_util_max" == t for t in subset["trainer"]]
            edge_colors = ["#c0392b" if h else "black" for h in highlight]
            edge_widths = [2.0 if h else 0.5 for h in highlight]

            bars = ax.barh(range(len(subset)), subset["overhead_pct"],
                           color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths)
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels(subset["trainer"], fontsize=9)
            ax.set_xlabel("Overhead vs noop (%)", fontsize=9)
            ax.set_title(f"{ds}, B={bs}", fontsize=10, fontweight="medium")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(5, color="red", linewidth=0.8, linestyle="--", alpha=0.4)

            for bar, pct in zip(bars, subset["overhead_pct"]):
                offset = 0.3 if bar.get_width() >= 0 else -0.3
                ax.text(bar.get_width() + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", ha="left" if bar.get_width() >= 0 else "right",
                        va="center", fontsize=8)

    fig.suptitle(f"Instrumentation Overhead vs noop (w={workers}) — resource_util_max highlighted",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    # ── Report Figure 3 equivalents: GPU util across batch sizes ────
    print("=== GPU Utilization Across Batch Sizes ===")
    plot_gpu_util_comparison("experiments_disk", 0, OUT_DIR / "gpu_util_disk_w0.png")
    plot_gpu_util_comparison("experiments_disk", 4, OUT_DIR / "gpu_util_disk_w4.png")
    plot_gpu_util_comparison("experiments_milabench", 0, OUT_DIR / "gpu_util_mila_w0.png")
    plot_gpu_util_comparison("experiments_milabench", 4, OUT_DIR / "gpu_util_mila_w4.png")

    # ── Report Figure 4 equivalents: GPU + CPU overlay ──────────────
    print("\n=== GPU vs CPU Utilization ===")
    for ds in ["experiments_disk", "experiments_milabench"]:
        for w in [0, 4]:
            for b in [32, 64, 128]:
                ds_short = "disk" if "disk" in ds else "mila"
                plot_gpu_cpu_overlay(ds, w, b,
                    OUT_DIR / f"gpu_cpu_util_{ds_short}_w{w}_b{b}.png")

    # ── Report Figure 5 equivalents: full resource timeline ─────────
    print("\n=== Full Resource Timeline ===")
    for ds in ["experiments_disk", "experiments_milabench"]:
        for w in [0, 4]:
            for b in [32, 64, 128]:
                ds_short = "disk" if "disk" in ds else "mila"
                plot_resource_overview(ds, w, b,
                    OUT_DIR / f"resources_{ds_short}_w{w}_b{b}.png")

    # ── Report Figures 6-7 equivalents: overhead with highlight ─────
    print("\n=== Overhead vs noop (resource_util_max highlighted) ===")
    plot_overhead_vs_noop(0, OUT_DIR / "overhead_vs_noop_w0.png")
    plot_overhead_vs_noop(4, OUT_DIR / "overhead_vs_noop_w4.png")

    print(f"\nDone. All plots saved to {OUT_DIR}")
    print("\nNote: Figures 1 (phase_time_bars) and 2 (resource_phases violins)")
    print("cannot be reproduced — resource_util_max has no per-phase data.")


if __name__ == "__main__":
    main()
