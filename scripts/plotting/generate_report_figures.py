#!/usr/bin/env python3
"""
Generate all 7 figures referenced by report/report.tex into report/figures/.

  1. phase_time_bars_disk_w0_b32.png   — per-phase timing bar chart
  2. resource_phases_disk_w0_b32.png   — per-phase resource violin plots
  3. gpu_util_disk_w0.png              — GPU util across batch sizes (comparison)
  4. gpu_cpu_util_mila_w0_b32.png      — GPU + CPU overlay (Milabench)
  5. resources_disk_w0.png             — full resource overview (2×4 grid)
  6. overhead_vs_noop_w0.png           — trainer overhead comparison (workers=0)
  7. overhead_vs_noop_w4.png           — trainer overhead comparison (workers=4)

Uses the fixed plotting functions from plot_resources.py (corrected GPU memory
priority and SI units).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = REPO_ROOT / "logs"
OUT = REPO_ROOT / "report" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_resources import (
    load_resource_plot_df,
    plot_gpu_cpu_overlap,
    plot_overview,
    plot_phase_bars,
    plot_phases_boxplot,
)


def setup_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            pass


def load_avg(data_source, trainer, workers, batch):
    for name in ("resource_util.csv", "resource_util_steps.csv"):
        p = LOGS / data_source / trainer / f"workers_{workers}" / f"batch_{batch}" / "averaged" / name
        if p.exists():
            return load_resource_plot_df(p)
    p = LOGS / data_source / f"workers_{workers}" / f"batch_{batch}" / "averaged" / "resource_util.csv"
    if p.exists():
        return load_resource_plot_df(p)
    return None


def load_substeps(data_source, workers, batch):
    p = LOGS / data_source / f"workers_{workers}" / f"batch_{batch}" / "averaged" / "resource_util_substeps.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ── Figure 1: phase_time_bars_disk_w0_b32 ──────────────────────────
def fig1():
    phase_path = LOGS / "experiments_disk" / "workers_0" / "batch_32" / "averaged" / "phase_times.csv"
    if not phase_path.exists():
        phase_path = LOGS / "experiments_disk" / "phase_times" / "workers_0" / "batch_32" / "averaged" / "phase_times.csv"
    plot_phase_bars(phase_path, OUT / "phase_time_bars_disk_w0_b32.png")


# ── Figure 2: resource_phases_disk_w0_b32 (violins) ────────────────
def fig2():
    df_sub = load_substeps("experiments_disk", 0, 32)
    if df_sub is not None and "phase" in df_sub.columns:
        plot_phases_boxplot(df_sub, OUT / "resource_phases_disk_w0_b32.png")
    else:
        print("  WARNING: No substep data for fig2 (resource_phases_disk_w0_b32)")


# ── Figure 3: gpu_util_disk_w0 (comparison across batch sizes) ─────
def fig3():
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"32": "#2980b9", "64": "#e74c3c", "128": "#27ae60"}

    for bs in [32, 64, 128]:
        df = load_avg("experiments_disk", "resource_util", 0, bs)
        if df is None:
            df = load_avg("experiments_disk", "", 0, bs)
        if df is None or "gpu_util" not in df.columns:
            print(f"  WARNING: No data for disk w=0 B={bs}")
            continue
        ax.plot(df["step"], df["gpu_util"], linewidth=1.5,
                color=colors[str(bs)], label=f"Batch {bs}")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax.set_title("GPU Utilization Across Batch Sizes (Disk, workers=0)", fontsize=12, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_util_disk_w0.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: gpu_util_disk_w0.png")


# ── Figure 4: gpu_cpu_util_mila_w0_b32 ─────────────────────────────
def fig4():
    df = load_avg("experiments_milabench", "resource_util", 0, 32)
    if df is None:
        df = load_avg("experiments_milabench", "", 0, 32)
    if df is None:
        print("  WARNING: No data for milabench w=0 B=32")
        return
    plot_gpu_cpu_overlap(df, OUT / "gpu_cpu_util_mila_w0_b32.png")


# ── Figure 5: resources_disk_w0 ────────────────────────────────────
def fig5():
    df = load_avg("experiments_disk", "resource_util", 0, 32)
    if df is None:
        df = load_avg("experiments_disk", "", 0, 32)
    if df is None:
        print("  WARNING: No data for disk w=0 B=32")
        return
    plot_overview(df, OUT / "resources_disk_w0.png")


# ── Figures 6 & 7: overhead_vs_noop ────────────────────────────────
def fig6_7():
    csv_path = LOGS / "comparison_plots" / "trainer_overhead_data.csv"
    if not csv_path.exists():
        print("  WARNING: trainer_overhead_data.csv not found — running compare_trainer_overhead.py")
        import subprocess
        subprocess.run([sys.executable,
                        str(REPO_ROOT / "scripts" / "compare_trainer_overhead.py")],
                       cwd=str(REPO_ROOT), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not csv_path.exists():
        print("  WARNING: Still no overhead data, skipping figures 6-7")
        return

    df_all = pd.read_csv(csv_path)
    trainers_order = ["simple", "phase_times", "resource_util", "resource_util_max",
                      "codecarbon", "codecarbon_e2e"]
    colors_map = {
        "simple": "#3498db", "phase_times": "#9b59b6", "resource_util": "#2ecc71",
        "resource_util_max": "#e67e22", "codecarbon": "#e74c3c", "codecarbon_e2e": "#1abc9c",
    }
    ds_short = {"disk": "Disk", "milabench": "Mila."}

    for workers, fig_name in [(0, "overhead_vs_noop_w0.png"), (4, "overhead_vs_noop_w4.png")]:
        setup_style()
        df = df_all[(df_all["workers"] == workers) & (df_all["trainer"] != "noop")]
        if df.empty:
            continue

        data_sources = sorted(df["data_source"].unique())
        batch_sizes = sorted(df["batch_size"].unique())
        ncols = len(batch_sizes)
        nrows = len(data_sources)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for ri, ds in enumerate(data_sources):
            for ci, bs in enumerate(batch_sizes):
                ax = axes[ri][ci]
                subset = df[(df["data_source"] == ds) & (df["batch_size"] == bs)]
                ordered = [t for t in trainers_order if t in subset["trainer"].values]
                subset = subset.set_index("trainer").reindex(ordered).reset_index()
                if subset.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue

                bar_colors = [colors_map.get(t, "gray") for t in subset["trainer"]]
                bars = ax.barh(range(len(subset)), subset["overhead_pct"],
                               color=bar_colors, edgecolor="black", linewidth=0.5)
                ax.set_yticks(range(len(subset)))
                ax.set_yticklabels(subset["trainer"], fontsize=9)
                ax.set_xlabel("Overhead vs noop (%)", fontsize=9)
                ax.set_title(f"{ds_short.get(ds, ds)}, B={bs}", fontsize=10, fontweight="medium")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(5, color="red", linewidth=0.8, linestyle="--", alpha=0.4, label="5%")

                for bar, pct in zip(bars, subset["overhead_pct"]):
                    offset = 0.3 if bar.get_width() >= 0 else -0.3
                    ax.text(bar.get_width() + offset,
                            bar.get_y() + bar.get_height() / 2,
                            f"{pct:.1f}%",
                            ha="left" if bar.get_width() >= 0 else "right",
                            va="center", fontsize=8)

        fig.suptitle(f"Instrumentation Overhead vs noop (workers={workers})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / fig_name, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Generating report figures in {OUT}\n")

    print("Fig 1: phase_time_bars_disk_w0_b32")
    fig1()

    print("Fig 2: resource_phases_disk_w0_b32")
    fig2()

    print("Fig 3: gpu_util_disk_w0")
    fig3()

    print("Fig 4: gpu_cpu_util_mila_w0_b32")
    fig4()

    print("Fig 5: resources_disk_w0")
    fig5()

    print("Figs 6-7: overhead_vs_noop")
    fig6_7()

    print(f"\nDone. All figures in {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
