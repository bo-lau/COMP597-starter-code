#!/usr/bin/env python3
"""
Generate all 7 figures referenced by report/report.tex into report/figures/.

Every figure is rendered at the same FIGSIZE so they appear uniform on the
single-page figure layout in the report.
"""
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
from plot_resources import load_resource_plot_df

FIGSIZE = (10, 6)
DPI = 150


def _setup_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            pass


def _save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


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
    _setup_style()
    phase_path = LOGS / "experiments_disk" / "workers_0" / "batch_32" / "averaged" / "phase_times.csv"
    if not phase_path.exists():
        print("  WARNING: phase_times.csv not found"); return
    df = pd.read_csv(phase_path)
    phases = ["forward_ms", "backward_ms", "optimizer_ms"]
    labels = ["Forward", "Backward", "Optimizer"]
    present = [p for p in phases if p in df.columns]
    if not present:
        print("  WARNING: no phase columns"); return
    means = [df[p].mean() for p in present]
    stds = [s if pd.notna(s) else 0 for s in [df[p].std() for p in present]]
    x = np.arange(len(present))
    colors = ["#27ae60", "#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, means, 0.6, yerr=stds, color=colors[:len(present)], capsize=5, edgecolor="black")
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Average Time per Phase (mean ± std)", fontsize=12, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[phases.index(p)] for p in present], fontsize=11)
    ax.set_ylim(0, max(means) * 1.25 if means else 1)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(means) * 0.02, f"{m:.1f}±{s:.1f}", ha="center", fontsize=10)
    fig.tight_layout()
    _save(fig, "phase_time_bars_disk_w0_b32.png")


# ── Figure 2: resource_phases_disk_w0_b32 (violins) ────────────────
def fig2():
    _setup_style()
    df_sub = load_substeps("experiments_disk", 0, 32)
    if df_sub is None or "phase" not in df_sub.columns:
        print("  WARNING: No substep data"); return

    phase_order = ["forward", "backward", "optimizer"]
    phase_colors = {"forward": "#27ae60", "backward": "#e74c3c", "optimizer": "#3498db"}
    df_plot = df_sub[df_sub["phase"].isin(phase_order)].copy()

    metrics = [
        ("gpu_util", "GPU Util (%)", "GPU Utilization"),
        ("cpu_util", "Process CPU (sum %)", "CPU Utilization"),
        (("gpu_mem_pct", "gpu_mem_gb"), ("%", "GB"), "GPU Memory"),
        ("cpu_mem_gb", "GB", "CPU Memory"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        if isinstance(metric[0], tuple):
            col = next((c for c in metric[0] if c in df_plot.columns), None)
            ylabel = metric[1][0] if col == metric[0][0] else metric[1][1]
            title = metric[2]
        else:
            col, ylabel, title = metric
        if col is None or col not in df_plot.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        data = [df_plot.loc[df_plot["phase"] == p, col].dropna().values for p in phase_order]
        parts = ax.violinplot(data, positions=[0, 1, 2], widths=0.7, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(phase_colors[phase_order[i]])
            pc.set_alpha(0.7); pc.set_edgecolor("black"); pc.set_linewidth(1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(phase_order, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.set_ylabel(ylabel, fontsize=9)
        ymax = max((np.max(d) for d in data if len(d) > 0), default=1)
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "resource_phases_disk_w0_b32.png")


# ── Figure 3: gpu_util_disk_w0 (comparison across batch sizes) ─────
def fig3():
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {"32": "#2980b9", "64": "#e74c3c", "128": "#27ae60"}
    for bs in [32, 64, 128]:
        df = load_avg("experiments_disk", "resource_util", 0, bs)
        if df is None:
            df = load_avg("experiments_disk", "", 0, bs)
        if df is None or "gpu_util" not in df.columns:
            print(f"  WARNING: No data for disk w=0 B={bs}"); continue
        ax.plot(df["step"], df["gpu_util"], linewidth=1.5,
                color=colors[str(bs)], label=f"Batch {bs}")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax.set_title("GPU Utilization Across Batch Sizes (Disk, workers=0)", fontsize=12, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "gpu_util_disk_w0.png")


# ── Figure 4: gpu_cpu_util_mila_w0_b32 ─────────────────────────────
def fig4():
    _setup_style()
    df = load_avg("experiments_milabench", "resource_util", 0, 32)
    if df is None:
        df = load_avg("experiments_milabench", "", 0, 32)
    if df is None:
        print("  WARNING: No data for milabench w=0 B=32"); return

    df_plot = (df.groupby("step", as_index=False).mean(numeric_only=True)
               if "step" in df.columns else df)
    x_col = "elapsed_s" if "elapsed_s" in df_plot.columns else "step"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df_plot[x_col], df_plot["gpu_util"], linewidth=1.5, color="#2980b9", label="GPU Util (%)")
    ax.plot(df_plot[x_col], df_plot["cpu_util"], linewidth=1.5, color="#e74c3c",
            label="Process CPU (sum %, all cores)")
    ax.set_xlabel("Time (s)" if x_col == "elapsed_s" else "Step", fontsize=11)
    ax.set_ylabel("Utilization (%)", fontsize=11)
    ax.set_title("GPU vs CPU Utilization (Milabench, w=0, B=32)", fontsize=12, fontweight="medium")
    ymax = max(df_plot["gpu_util"].max(), df_plot["cpu_util"].max())
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "gpu_cpu_util_mila_w0_b32.png")


# ── Figure 5: resources_disk_w0 ────────────────────────────────────
def fig5():
    _setup_style()
    df = load_avg("experiments_disk", "resource_util", 0, 32)
    if df is None:
        df = load_avg("experiments_disk", "", 0, 32)
    if df is None:
        print("  WARNING: No data for disk w=0 B=32"); return

    df_plot = (df.groupby("step", as_index=False).mean(numeric_only=True)
               if "phase" in df.columns else df.copy())
    x_col = "elapsed_s" if "elapsed_s" in df_plot.columns else "step"
    xlabel = "Time (s)" if x_col == "elapsed_s" else "Step"

    metrics = [
        ("gpu_util", "GPU Util (%)", "GPU Utilization"),
        ("cpu_util", "CPU (sum %)", "CPU Utilization"),
        (("gpu_mem_pct", "gpu_mem_gb"), ("%", "GB"), "GPU Memory"),
        ("cpu_mem_gb", "GB", "CPU Memory"),
        ("ram_gb", "GB", "System RAM"),
        ("io_read_gb", "GB", "I/O Read"),
        ("io_write_gb", "GB", "I/O Write"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE)
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
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        ax.plot(df_plot[x_col], df_plot[col], linewidth=1.2, color="#2980b9")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="medium")
        ymax = df_plot[col].max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == "io_write_gb" and df_plot[col].max() < 1e-3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.0f}"))
            ax.set_ylabel("KB", fontsize=8)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")
    fig.supxlabel(xlabel, fontsize=9, y=0.02)
    fig.tight_layout()
    _save(fig, "resources_disk_w0.png")


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
        print("  WARNING: Still no overhead data"); return

    df_all = pd.read_csv(csv_path)
    trainers_order = ["simple", "phase_times", "resource_util",
                      "codecarbon", "codecarbon_e2e"]
    colors_map = {
        "simple": "#3498db", "phase_times": "#9b59b6", "resource_util": "#2ecc71",
        "codecarbon": "#e74c3c", "codecarbon_e2e": "#1abc9c",
    }
    ds_short = {"disk": "Disk", "milabench": "Mila."}

    for workers, fig_name in [(0, "overhead_vs_noop_w0.png"), (4, "overhead_vs_noop_w4.png")]:
        _setup_style()
        df = df_all[(df_all["workers"] == workers)
                    & (df_all["trainer"] != "noop")
                    & (df_all["trainer"] != "resource_util_max")]
        if df.empty:
            continue

        data_sources = sorted(df["data_source"].unique())
        batch_sizes = sorted(df["batch_size"].unique())
        ncols = len(batch_sizes)
        nrows = len(data_sources)

        fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, squeeze=False)

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
                ax.set_yticklabels(subset["trainer"], fontsize=7)
                ax.set_xlabel("Overhead (%)", fontsize=8)
                ax.set_title(f"{ds_short.get(ds, ds)}, B={bs}", fontsize=9, fontweight="medium")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(5, color="red", linewidth=0.8, linestyle="--", alpha=0.4)
                ax.tick_params(labelsize=7)

                for bar, pct in zip(bars, subset["overhead_pct"]):
                    offset = 0.3 if bar.get_width() >= 0 else -0.3
                    ax.text(bar.get_width() + offset,
                            bar.get_y() + bar.get_height() / 2,
                            f"{pct:.1f}%",
                            ha="left" if bar.get_width() >= 0 else "right",
                            va="center", fontsize=7)

        fig.suptitle(f"Instrumentation Overhead vs noop (workers={workers})",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, fig_name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Generating report figures in {OUT}")
    print(f"Uniform figsize: {FIGSIZE}\n")

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
