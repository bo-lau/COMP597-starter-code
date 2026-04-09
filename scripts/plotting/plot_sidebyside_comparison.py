#!/usr/bin/env python3
"""
Side-by-side PDF comparison: resource_util vs resource_util_max.

Each page shows the same plot type with resource_util on the left and
resource_util_max on the right, making it easy to see how the two
trainer stats modules differ in recorded metrics.

Output: report/resource_util_vs_max_comparison.pdf
"""
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = REPO_ROOT / "logs"
OUT_PDF = REPO_ROOT / "report" / "resource_util_vs_max_comparison.pdf"

BATCH_COLORS = {"32": "#2980b9", "64": "#e74c3c", "128": "#27ae60"}

METRICS_OVERVIEW = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
    (("gpu_mem_gb", "gpu_mem_pct"), ("GB", "%"), "GPU Memory"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
    ("ram_gb", "GB", "System RAM"),
    ("io_read_gb", "GB", "I/O Read"),
    ("io_write_gb", "GB", "I/O Write"),
]


def setup_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            pass


def load_csv(trainer: str, data_source: str, workers: int, batch: int) -> Optional[pd.DataFrame]:
    path = (LOGS / data_source / trainer / f"workers_{workers}"
            / f"batch_{batch}" / "averaged" / "resource_util.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def ds_label(data_source: str) -> str:
    return "Disk" if "disk" in data_source else "Milabench"


# ────────────────────────────────────────────────────────────────────
# Page type 1: GPU utilization across batch sizes
# ────────────────────────────────────────────────────────────────────
def page_gpu_util(pdf: PdfPages, data_source: str, workers: int):
    setup_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, trainer, label in [
        (ax_l, "resource_util", "resource_util"),
        (ax_r, "resource_util_max", "resource_util_max"),
    ]:
        for bs in [32, 64, 128]:
            df = load_csv(trainer, data_source, workers, bs)
            if df is None or "gpu_util" not in df.columns:
                continue
            ax.plot(df["step"], df["gpu_util"], linewidth=1.4,
                    color=BATCH_COLORS[str(bs)], label=f"B={bs}")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("GPU Utilization (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GPU Utilization Across Batch Sizes — {ds_label(data_source)}, w={workers}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 2: GPU vs CPU utilization overlay
# ────────────────────────────────────────────────────────────────────
def page_gpu_cpu(pdf: PdfPages, data_source: str, workers: int, batch: int):
    setup_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, trainer, label in [
        (ax_l, "resource_util", "resource_util"),
        (ax_r, "resource_util_max", "resource_util_max"),
    ]:
        df = load_csv(trainer, data_source, workers, batch)
        if df is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(label, fontsize=11, fontweight="bold")
            continue
        ax.plot(df["step"], df["gpu_util"], linewidth=1.4, color="#2980b9",
                label="GPU Util (%)")
        ax.plot(df["step"], df["cpu_util"], linewidth=1.4, color="#e74c3c",
                label="Process CPU (sum %)")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Utilization (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ymax = max(df["gpu_util"].max(), df["cpu_util"].max())
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GPU vs CPU Utilization — {ds_label(data_source)}, w={workers}, B={batch}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 3: full resource overview (2x4 grids side by side → 2 rows of panels)
# ────────────────────────────────────────────────────────────────────
def _plot_overview_on_axes(axes_flat, df: Optional[pd.DataFrame], trainer_label: str):
    """Fill a 2x4 = 8 axes array with overview line plots."""
    for idx, metric in enumerate(METRICS_OVERVIEW):
        ax = axes_flat[idx]
        if isinstance(metric[0], tuple):
            col = next((c for c in metric[0] if df is not None and c in df.columns), None)
            ylabel = metric[1][0] if col == metric[0][0] else metric[1][1]
            title = metric[2]
        else:
            col, ylabel, title = metric

        if df is None or col is None or col not in df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(title, fontsize=9, fontweight="medium")
            continue

        ax.plot(df["step"], df[col], linewidth=1.1, color="#2980b9")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="medium")
        ymax = df[col].max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == "io_write_gb" and df[col].max() < 1e-3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.0f}"))
            ax.set_ylabel("KB", fontsize=8)

    for idx in range(len(METRICS_OVERVIEW), len(axes_flat)):
        axes_flat[idx].axis("off")


def page_resource_overview(pdf: PdfPages, data_source: str, workers: int, batch: int):
    setup_style()
    fig = plt.figure(figsize=(16, 12))

    # Top half: resource_util (rows 0-1, 4 cols)
    axes_top = [fig.add_subplot(4, 4, i + 1) for i in range(8)]
    # Bottom half: resource_util_max (rows 2-3, 4 cols)
    axes_bot = [fig.add_subplot(4, 4, i + 9) for i in range(8)]

    df_ru = load_csv("resource_util", data_source, workers, batch)
    df_rum = load_csv("resource_util_max", data_source, workers, batch)

    _plot_overview_on_axes(axes_top, df_ru, "resource_util")
    _plot_overview_on_axes(axes_bot, df_rum, "resource_util_max")

    # Section labels
    fig.text(0.5, 0.97, "resource_util", ha="center", fontsize=12,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5e8d4", alpha=0.8))
    fig.text(0.5, 0.49, "resource_util_max", ha="center", fontsize=12,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#dae8fc", alpha=0.8))

    fig.suptitle(
        f"Resource Timeline — {ds_label(data_source)}, w={workers}, B={batch}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.55)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 4: per-metric comparison (both trainers overlaid)
# ────────────────────────────────────────────────────────────────────
def page_overlay_comparison(pdf: PdfPages, data_source: str, workers: int, batch: int):
    """Single page with shared metrics overlaid: blue=resource_util, orange=resource_util_max."""
    setup_style()
    df_ru = load_csv("resource_util", data_source, workers, batch)
    df_rum = load_csv("resource_util_max", data_source, workers, batch)
    if df_ru is None and df_rum is None:
        return

    shared_cols = [
        ("gpu_util", "GPU Util (%)", "GPU Utilization"),
        ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
        ("cpu_mem_gb", "GB", "CPU Memory"),
        ("io_read_gb", "GB", "I/O Read"),
        ("io_write_gb", "GB", "I/O Write"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(shared_cols):
        ax = axes_flat[idx]
        plotted = False
        if df_ru is not None and col in df_ru.columns:
            ax.plot(df_ru["step"], df_ru[col], linewidth=1.3, color="#2980b9",
                    alpha=0.85, label="resource_util")
            plotted = True
        if df_rum is not None and col in df_rum.columns:
            ax.plot(df_rum["step"], df_rum[col], linewidth=1.3, color="#e67e22",
                    alpha=0.85, label="resource_util_max")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.set_xlabel("Step", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Last subplot: GPU memory (different column names)
    ax = axes_flat[5]
    plotted = False
    if df_ru is not None and "gpu_mem_gb" in df_ru.columns:
        ax.plot(df_ru["step"], df_ru["gpu_mem_gb"], linewidth=1.3, color="#2980b9",
                alpha=0.85, label="resource_util")
        plotted = True
    if df_rum is not None and "gpu_mem_gb" in df_rum.columns:
        ax.plot(df_rum["step"], df_rum["gpu_mem_gb"], linewidth=1.3, color="#e67e22",
                alpha=0.85, label="resource_util_max")
        plotted = True
    if plotted:
        ax.set_ylabel("GB", fontsize=9)
        ax.set_title("GPU Memory", fontsize=10, fontweight="medium")
        ax.set_xlabel("Step", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    fig.suptitle(
        f"Metric Overlay — {ds_label(data_source)}, w={workers}, B={batch}\n"
        r"$\bf{blue}$ = resource_util (with CUDA sync)   "
        r"$\bf{orange}$ = resource_util_max (no sync)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page: summary statistics table
# ────────────────────────────────────────────────────────────────────
def page_summary_table(pdf: PdfPages):
    setup_style()
    rows = []
    for ds in ["experiments_disk", "experiments_milabench"]:
        for w in [0, 4]:
            for b in [32, 64, 128]:
                for trainer in ["resource_util", "resource_util_max"]:
                    df = load_csv(trainer, ds, w, b)
                    if df is None:
                        continue
                    row = {
                        "Data": ds_label(ds),
                        "W": w,
                        "B": b,
                        "Trainer": trainer,
                        "Steps": len(df),
                        "GPU Util Mean": f"{df['gpu_util'].mean():.1f}%",
                        "GPU Util Std": f"{df['gpu_util'].std():.1f}",
                        "CPU Util Mean": f"{df['cpu_util'].mean():.1f}%",
                    }
                    if "gpu_mem_gb" in df.columns:
                        row["GPU Mem (GB)"] = f"{df['gpu_mem_gb'].mean():.2f}"
                    rows.append(row)

    if not rows:
        return

    tbl_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 0.4 * len(tbl_df) + 2))
    ax.axis("off")
    ax.set_title("Summary: resource_util vs resource_util_max",
                 fontsize=14, fontweight="bold", pad=20)

    col_labels = list(tbl_df.columns)
    cell_text = tbl_df.values.tolist()

    colors = []
    for row in rows:
        if row["Trainer"] == "resource_util":
            colors.append(["#d5e8d4"] * len(col_labels))
        else:
            colors.append(["#dae8fc"] * len(col_labels))

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    data_sources = ["experiments_disk", "experiments_milabench"]
    workers_list = [0, 4]
    batch_sizes = [32, 64, 128]

    with PdfPages(str(OUT_PDF)) as pdf:
        # Title page
        fig = plt.figure(figsize=(16, 10))
        fig.text(0.5, 0.55, "resource_util  vs  resource_util_max", ha="center",
                 fontsize=28, fontweight="bold", color="#2c3e50")
        fig.text(0.5, 0.45, "Side-by-Side Comparison", ha="center",
                 fontsize=18, color="#7f8c8d")
        fig.text(0.5, 0.35, "COMP 597 — Energy Profiling of Whisper Fine-Tuning",
                 ha="center", fontsize=14, color="#95a5a6")
        fig.text(0.5, 0.22,
                 "resource_util: per-step GPU/CPU metrics with CUDA synchronization\n"
                 "resource_util_max: per-step GPU/CPU metrics without CUDA sync (async max readings)",
                 ha="center", fontsize=11, color="#7f8c8d", linespacing=1.6)
        fig.text(0.5, 0.10,
                 "Green = resource_util  |  Blue = resource_util_max",
                 ha="center", fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1"))
        pdf.savefig(fig)
        plt.close(fig)

        # Summary table
        print("Page: Summary table")
        page_summary_table(pdf)

        # GPU utilization across batch sizes (4 pages)
        for ds in data_sources:
            for w in workers_list:
                print(f"Page: GPU util — {ds_label(ds)} w={w}")
                page_gpu_util(pdf, ds, w)

        # GPU vs CPU overlay (key configs matching report figures)
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print(f"Page: GPU vs CPU — {ds_label(ds)} w={w} B={b}")
                    page_gpu_cpu(pdf, ds, w, b)

        # Metric overlay (both trainers on same axes)
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print(f"Page: Overlay — {ds_label(ds)} w={w} B={b}")
                    page_overlay_comparison(pdf, ds, w, b)

        # Full resource overview (stacked: top=resource_util, bottom=resource_util_max)
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print(f"Page: Overview — {ds_label(ds)} w={w} B={b}")
                    page_resource_overview(pdf, ds, w, b)

    print(f"\nDone. PDF saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
