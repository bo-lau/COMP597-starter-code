#!/usr/bin/env python3
"""
Side-by-side PDF comparison: resource_util vs resource_util_max.

Each page shows the same plot type with resource_util on the left and
resource_util_max on the right, making it easy to see how the two
trainer stats modules differ in recorded metrics.

IMPORTANT: the two trainers record `gpu_mem_gb` differently:
  - resource_util:     torch.cuda.memory_allocated() (tensor memory only, ~0.1 GB)
  - resource_util_max: pynvml.nvmlDeviceGetMemoryInfo().used (total GPU mem, ~6-24 GB)
This script labels them distinctly and never overlays them on the same axis.

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

# Shared metrics that are comparable between both trainers
SHARED_METRICS = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
    ("cpu_mem_gb", "GB", "CPU Memory (RSS)"),
    ("io_read_gb", "GB", "I/O Read (cumulative)"),
    ("io_write_gb", "GB", "I/O Write (cumulative)"),
]

# Per-trainer overview metrics (labels clarify the different gpu_mem_gb semantics)
METRICS_RU = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
    ("gpu_mem_gb", "GB", "GPU Mem (torch alloc)"),
    ("gpu_mem_pct", "%", "GPU Mem Util (NVML %)"),
    ("cpu_mem_gb", "GB", "CPU Memory (RSS)"),
    ("io_read_gb", "GB", "I/O Read"),
    ("io_write_gb", "GB", "I/O Write"),
]

METRICS_RUM = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
    ("gpu_mem_gb", "GB", "GPU Mem (NVML used)"),
    ("cpu_mem_gb", "GB", "CPU Memory (RSS)"),
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


def load_csv(trainer, data_source, workers, batch):
    # type: (str, str, int, int) -> Optional[pd.DataFrame]
    path = (LOGS / data_source / trainer / "workers_{}".format(workers)
            / "batch_{}".format(batch) / "averaged" / "resource_util.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def ds_label(data_source):
    # type: (str) -> str
    return "Disk" if "disk" in data_source else "Milabench"


# ────────────────────────────────────────────────────────────────────
# Page type 1: GPU utilization across batch sizes
# ────────────────────────────────────────────────────────────────────
def page_gpu_util(pdf, data_source, workers):
    setup_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, trainer, label in [
        (ax_l, "resource_util", "resource_util (with CUDA sync)"),
        (ax_r, "resource_util_max", "resource_util_max (no sync)"),
    ]:
        for bs in [32, 64, 128]:
            df = load_csv(trainer, data_source, workers, bs)
            if df is None or "gpu_util" not in df.columns:
                continue
            ax.plot(df["step"], df["gpu_util"], linewidth=1.4,
                    color=BATCH_COLORS[str(bs)], label="B={}".format(bs))
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("GPU Utilization (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_l.set_ylim(0, 105)

    fig.suptitle(
        "GPU Utilization Across Batch Sizes \u2014 {}, w={}".format(ds_label(data_source), workers),
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 2: GPU vs CPU utilization overlay
# ────────────────────────────────────────────────────────────────────
def page_gpu_cpu(pdf, data_source, workers, batch):
    setup_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Compute shared ylim from both panels
    global_ymax = 0
    for trainer in ["resource_util", "resource_util_max"]:
        df = load_csv(trainer, data_source, workers, batch)
        if df is not None:
            for col in ["gpu_util", "cpu_util"]:
                if col in df.columns:
                    global_ymax = max(global_ymax, df[col].max())

    for ax, trainer, label in [
        (ax_l, "resource_util", "resource_util (with CUDA sync)"),
        (ax_r, "resource_util_max", "resource_util_max (no sync)"),
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
        ax.set_ylim(0, global_ymax * 1.15 if global_ymax > 0 else 105)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "GPU vs CPU Utilization \u2014 {}, w={}, B={}".format(ds_label(data_source), workers, batch),
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 3: full resource overview (stacked with SHARED Y-axes)
# ────────────────────────────────────────────────────────────────────
def _get_col_for_metric(df, metric_spec):
    """Return (col_name, ylabel, title) for a metric spec, or None if missing."""
    if isinstance(metric_spec[0], tuple):
        col = next((c for c in metric_spec[0] if df is not None and c in df.columns), None)
        ylabel = metric_spec[1][0] if col == metric_spec[0][0] else metric_spec[1][1]
        return col, ylabel, metric_spec[2]
    col, ylabel, title = metric_spec
    if df is not None and col in df.columns:
        return col, ylabel, title
    return None, metric_spec[1], metric_spec[2]


def page_resource_overview(pdf, data_source, workers, batch):
    setup_style()
    df_ru = load_csv("resource_util", data_source, workers, batch)
    df_rum = load_csv("resource_util_max", data_source, workers, batch)

    # Use per-trainer metric lists so labels are accurate
    fig = plt.figure(figsize=(16, 13))

    n_metrics = max(len(METRICS_RU), len(METRICS_RUM))
    ncols = 4
    nrows_half = (n_metrics + ncols - 1) // ncols
    total_rows = nrows_half * 2

    axes_top = []
    axes_bot = []
    for i in range(len(METRICS_RU)):
        r, c = divmod(i, ncols)
        axes_top.append(fig.add_subplot(total_rows, ncols, r * ncols + c + 1))
    for i in range(len(METRICS_RUM)):
        r, c = divmod(i, ncols)
        axes_bot.append(fig.add_subplot(total_rows, ncols, (r + nrows_half) * ncols + c + 1))

    # Plot resource_util (top)
    for idx, (col, ylabel, title) in enumerate(METRICS_RU):
        ax = axes_top[idx]
        if df_ru is not None and col in df_ru.columns:
            ax.plot(df_ru["step"], df_ru[col], linewidth=1.1, color="#2980b9")
            ymax = df_ru[col].max()
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="medium")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Plot resource_util_max (bottom)
    for idx, (col, ylabel, title) in enumerate(METRICS_RUM):
        ax = axes_bot[idx]
        if df_rum is not None and col in df_rum.columns:
            ax.plot(df_rum["step"], df_rum[col], linewidth=1.1, color="#e67e22")
            ymax = df_rum[col].max()
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="medium")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Shared Y-axes for comparable metrics (gpu_util, cpu_util, cpu_mem_gb, io_read_gb, io_write_gb)
    shared_pairs = [
        ("gpu_util", 0, 0),
        ("cpu_util", 1, 1),
        ("cpu_mem_gb", 4, 3),
        ("io_read_gb", 5, 5),
        ("io_write_gb", 6, 6),
    ]
    for col, top_idx, bot_idx in shared_pairs:
        if top_idx < len(axes_top) and bot_idx < len(axes_bot):
            ymax = 0
            if df_ru is not None and col in df_ru.columns:
                ymax = max(ymax, df_ru[col].max())
            if df_rum is not None and col in df_rum.columns:
                ymax = max(ymax, df_rum[col].max())
            if ymax > 0:
                ylim = ymax * 1.15
                axes_top[top_idx].set_ylim(0, ylim)
                axes_bot[bot_idx].set_ylim(0, ylim)

    fig.text(0.5, 0.97, "resource_util (with CUDA sync)", ha="center", fontsize=11,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5e8d4", alpha=0.8))
    fig.text(0.5, 0.50, "resource_util_max (no sync)", ha="center", fontsize=11,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#dae8fc", alpha=0.8))

    fig.suptitle(
        "Resource Timeline \u2014 {}, w={}, B={}".format(ds_label(data_source), workers, batch),
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.65)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page type 4: per-metric overlay (only COMPARABLE metrics)
# ────────────────────────────────────────────────────────────────────
def page_overlay_comparison(pdf, data_source, workers, batch):
    """Overlay both trainers on same axes for shared, comparable metrics only.
    GPU memory is shown in separate subplots because the two trainers
    measure fundamentally different quantities."""
    setup_style()
    df_ru = load_csv("resource_util", data_source, workers, batch)
    df_rum = load_csv("resource_util_max", data_source, workers, batch)
    if df_ru is None and df_rum is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    # Shared comparable metrics (overlay is meaningful)
    for idx, (col, ylabel, title) in enumerate(SHARED_METRICS):
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

    # GPU Memory: separate side-by-side bars showing the DIFFERENT metrics
    ax = axes_flat[5]
    ru_mem_val = None
    rum_mem_val = None
    if df_ru is not None and "gpu_mem_gb" in df_ru.columns:
        ru_mem_val = df_ru["gpu_mem_gb"].mean()
    if df_rum is not None and "gpu_mem_gb" in df_rum.columns:
        rum_mem_val = df_rum["gpu_mem_gb"].mean()

    if ru_mem_val is not None or rum_mem_val is not None:
        labels = []
        vals = []
        colors = []
        if ru_mem_val is not None:
            labels.append("resource_util\n(torch alloc)")
            vals.append(ru_mem_val)
            colors.append("#2980b9")
        if rum_mem_val is not None:
            labels.append("resource_util_max\n(NVML used)")
            vals.append(rum_mem_val)
            colors.append("#e67e22")
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("GB", fontsize=9)
        ax.set_title("GPU Memory (DIFFERENT metrics!)", fontsize=10,
                      fontweight="bold", color="#c0392b")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    "{:.2f} GB".format(v), ha="center", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")

    fig.suptitle(
        "Metric Overlay \u2014 {}, w={}, B={}\n"
        "blue = resource_util (CUDA sync)   "
        "orange = resource_util_max (no sync)".format(ds_label(data_source), workers, batch),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Page: summary statistics table
# ────────────────────────────────────────────────────────────────────
def page_summary_table(pdf):
    setup_style()
    rows = []
    for ds in ["experiments_disk", "experiments_milabench"]:
        for w in [0, 4]:
            for b in [32, 64, 128]:
                for trainer in ["resource_util", "resource_util_max"]:
                    df = load_csv(trainer, ds, w, b)
                    if df is None:
                        continue
                    ss = df[df["step"] > 1]  # skip warmup step
                    gpu_mem_label = "torch alloc" if trainer == "resource_util" else "NVML used"
                    row = {
                        "Data": ds_label(ds),
                        "W": w,
                        "B": b,
                        "Trainer": trainer.replace("resource_util", "res_util"),
                        "Steps": len(df),
                        "GPU Util\nMean": "{:.1f}%".format(ss["gpu_util"].mean()),
                        "GPU Util\nStd": "{:.1f}".format(ss["gpu_util"].std()),
                        "CPU Util\nMean": "{:.1f}%".format(ss["cpu_util"].mean()),
                    }
                    if "gpu_mem_gb" in df.columns:
                        row["GPU Mem\n({})".format(gpu_mem_label)] = "{:.2f} GB".format(ss["gpu_mem_gb"].mean())
                    rows.append(row)

    if not rows:
        return

    tbl_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 0.4 * len(tbl_df) + 2.5))
    ax.axis("off")
    ax.set_title("Summary: resource_util vs resource_util_max\n"
                 "(GPU Mem columns measure different things \u2014 see header labels)",
                 fontsize=13, fontweight="bold", pad=20)

    col_labels = list(tbl_df.columns)
    cell_text = tbl_df.values.tolist()

    colors = []
    for row in rows:
        if "res_util_max" not in row["Trainer"]:
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
        fig.text(0.5, 0.60, "resource_util  vs  resource_util_max", ha="center",
                 fontsize=28, fontweight="bold", color="#2c3e50")
        fig.text(0.5, 0.50, "Side-by-Side Comparison", ha="center",
                 fontsize=18, color="#7f8c8d")
        fig.text(0.5, 0.42, "COMP 597 \u2014 Energy Profiling of Whisper Fine-Tuning",
                 ha="center", fontsize=14, color="#95a5a6")
        fig.text(0.5, 0.25,
                 "resource_util: per-step GPU/CPU metrics with CUDA synchronization\n"
                 "  \u2022 GPU Memory = torch.cuda.memory_allocated() (tensor memory only)\n\n"
                 "resource_util_max: per-step GPU/CPU metrics without CUDA sync\n"
                 "  \u2022 GPU Memory = pynvml.nvmlDeviceGetMemoryInfo().used (total GPU memory)",
                 ha="center", fontsize=11, color="#2c3e50", linespacing=1.6,
                 family="monospace")
        fig.text(0.5, 0.08,
                 "Blue = resource_util  |  Orange = resource_util_max",
                 ha="center", fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1"))
        pdf.savefig(fig)
        plt.close(fig)

        # Summary table
        print("Page: Summary table")
        page_summary_table(pdf)

        # GPU utilization across batch sizes
        for ds in data_sources:
            for w in workers_list:
                print("Page: GPU util \u2014 {} w={}".format(ds_label(ds), w))
                page_gpu_util(pdf, ds, w)

        # GPU vs CPU overlay
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print("Page: GPU vs CPU \u2014 {} w={} B={}".format(ds_label(ds), w, b))
                    page_gpu_cpu(pdf, ds, w, b)

        # Metric overlay (comparable metrics only, GPU mem as bar chart)
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print("Page: Overlay \u2014 {} w={} B={}".format(ds_label(ds), w, b))
                    page_overlay_comparison(pdf, ds, w, b)

        # Full resource overview (stacked, with shared Y-axes where comparable)
        for ds in data_sources:
            for w in workers_list:
                for b in batch_sizes:
                    print("Page: Overview \u2014 {} w={} B={}".format(ds_label(ds), w, b))
                    page_resource_overview(pdf, ds, w, b)

    print("\nDone. PDF saved to {}".format(OUT_PDF))


if __name__ == "__main__":
    main()
