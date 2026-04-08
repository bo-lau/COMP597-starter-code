#!/usr/bin/env python3
"""
Compare training duration across all trainer_stats types vs noop baseline.
Parses tqdm elapsed times from run.log and resource_util_max_train_duration.txt.
"""
import re
import os
from typing import Optional
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"
DATA_SOURCES = ["experiments_disk", "experiments_milabench"]
TRAINERS = ["noop", "resource_util", "resource_util_max", "phase_times", "simple", "codecarbon", "codecarbon_e2e"]
TQDM_PATTERN = re.compile(r"\|\s*(\d+)/\1\s+\[(\d+:\d+(?::\d+)?)<")


def parse_tqdm_seconds(log_path: Path) -> Optional[float]:
    """Extract elapsed seconds from the final tqdm progress line."""
    text = log_path.read_text(errors="replace")
    matches = list(TQDM_PATTERN.finditer(text))
    if not matches:
        return None
    time_str = matches[-1].group(2)
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def parse_duration_txt(path: Path) -> Optional[float]:
    """Parse resource_util_max_train_duration.txt → seconds."""
    text = path.read_text().strip()
    match = re.search(r"duration_ms\s+([\d.]+)", text)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def collect_durations() -> pd.DataFrame:
    rows = []
    for ds in DATA_SOURCES:
        ds_path = LOGS_ROOT / ds
        if not ds_path.exists():
            continue
        for trainer in TRAINERS:
            trainer_path = ds_path / trainer
            if not trainer_path.exists():
                continue
            for workers_dir in sorted(trainer_path.glob("workers_*")):
                workers = int(workers_dir.name.split("_")[1])
                for batch_dir in sorted(workers_dir.glob("batch_*")):
                    batch = int(batch_dir.name.split("_")[1])
                    for run_dir in sorted(batch_dir.glob("run_*")):
                        run_num = int(run_dir.name.split("_")[1])
                        duration = None

                        if trainer == "resource_util_max":
                            dur_file = run_dir / "resource_util_max_train_duration.txt"
                            if dur_file.exists():
                                duration = parse_duration_txt(dur_file)
                            if duration is None:
                                log_file = run_dir / "run.log"
                                if log_file.exists():
                                    duration = parse_tqdm_seconds(log_file)
                        else:
                            log_file = run_dir / "run.log"
                            if log_file.exists():
                                duration = parse_tqdm_seconds(log_file)

                        if duration is not None:
                            rows.append({
                                "data_source": ds.replace("experiments_", ""),
                                "trainer": trainer,
                                "workers": workers,
                                "batch_size": batch,
                                "run": run_num,
                                "duration_s": duration,
                            })
    return pd.DataFrame(rows)


def main():
    df = collect_durations()
    if df.empty:
        print("No data found!")
        return

    print(f"Collected {len(df)} individual run measurements\n")

    avg = df.groupby(["data_source", "trainer", "workers", "batch_size"]).agg(
        mean_s=("duration_s", "mean"),
        std_s=("duration_s", "std"),
        count=("duration_s", "count"),
    ).reset_index()
    avg["std_s"] = avg["std_s"].fillna(0)

    noop = avg[avg["trainer"] == "noop"][["data_source", "workers", "batch_size", "mean_s"]].rename(
        columns={"mean_s": "noop_s"}
    )
    merged = avg.merge(noop, on=["data_source", "workers", "batch_size"], how="left")
    merged["overhead_s"] = merged["mean_s"] - merged["noop_s"]
    merged["overhead_pct"] = (merged["overhead_s"] / merged["noop_s"]) * 100

    print("=" * 100)
    print("DURATION COMPARISON: ALL TRAINERS vs NOOP BASELINE")
    print("=" * 100)

    for ds in sorted(merged["data_source"].unique()):
        ds_data = merged[merged["data_source"] == ds]
        print(f"\n{'─' * 100}")
        print(f"  Data Source: {ds}")
        print(f"{'─' * 100}")

        for workers in sorted(ds_data["workers"].unique()):
            for batch in sorted(ds_data["batch_size"].unique()):
                subset = ds_data[(ds_data["workers"] == workers) & (ds_data["batch_size"] == batch)]
                if subset.empty:
                    continue
                print(f"\n  Workers={workers}, Batch={batch}")
                print(f"  {'Trainer':<20} {'Mean (s)':>10} {'Std (s)':>10} {'Runs':>6} {'Overhead (s)':>14} {'Overhead (%)':>14}")
                print(f"  {'─' * 76}")

                for _, row in subset.sort_values("mean_s").iterrows():
                    oh_s = f"{row['overhead_s']:+.2f}" if pd.notna(row['overhead_s']) else "N/A"
                    oh_pct = f"{row['overhead_pct']:+.2f}%" if pd.notna(row['overhead_pct']) else "N/A"
                    print(f"  {row['trainer']:<20} {row['mean_s']:>10.2f} {row['std_s']:>10.2f} {int(row['count']):>6} {oh_s:>14} {oh_pct:>14}")

    out_dir = LOGS_ROOT / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainers_no_noop = [t for t in TRAINERS if t != "noop"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(trainers_no_noop)))
    trainer_colors = dict(zip(trainers_no_noop, colors))

    for ds in sorted(merged["data_source"].unique()):
        ds_data = merged[(merged["data_source"] == ds) & (merged["trainer"] != "noop")]
        if ds_data.empty:
            continue

        configs = ds_data.groupby(["workers", "batch_size"]).size().reset_index()[["workers", "batch_size"]]

        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 7), sharey=True)
        if len(configs) == 1:
            axes = [axes]

        for idx, (_, cfg) in enumerate(configs.iterrows()):
            ax = axes[idx]
            subset = ds_data[(ds_data["workers"] == cfg["workers"]) & (ds_data["batch_size"] == cfg["batch_size"])]
            subset = subset.sort_values("overhead_pct")

            bars = ax.barh(
                range(len(subset)),
                subset["overhead_pct"],
                color=[trainer_colors.get(t, "gray") for t in subset["trainer"]],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels(subset["trainer"])
            ax.set_xlabel("Overhead vs noop (%)")
            ax.set_title(f"w={cfg['workers']}, b={cfg['batch_size']}")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(5, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="5% threshold")

            for bar, pct in zip(bars, subset["overhead_pct"]):
                ax.text(
                    bar.get_width() + 0.3 if bar.get_width() >= 0 else bar.get_width() - 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%",
                    ha="left" if bar.get_width() >= 0 else "right",
                    va="center",
                    fontsize=8,
                )

            ax.legend(fontsize=7)

        fig.suptitle(f"Trainer Stats Overhead vs NOOP — {ds}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fname = out_dir / f"trainer_overhead_{ds}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"\nSaved: {fname}")

    # Combined grouped bar chart: each batch/worker config, all data sources
    for workers in sorted(merged["workers"].unique()):
        fig, axes = plt.subplots(1, len(merged["batch_size"].unique()), figsize=(6 * len(merged["batch_size"].unique()), 7), sharey=True)
        batch_sizes = sorted(merged["batch_size"].unique())
        if len(batch_sizes) == 1:
            axes = [axes]

        for ax_idx, batch in enumerate(batch_sizes):
            ax = axes[ax_idx]
            subset = merged[(merged["workers"] == workers) & (merged["batch_size"] == batch) & (merged["trainer"] != "noop")]

            data_sources = sorted(subset["data_source"].unique())
            x = np.arange(len(trainers_no_noop))
            width = 0.35
            ds_offsets = {ds: (i - (len(data_sources) - 1) / 2) * width for i, ds in enumerate(data_sources)}
            ds_colors = {"disk": "#4c72b0", "milabench": "#dd8452"}

            for ds_name in data_sources:
                ds_sub = subset[subset["data_source"] == ds_name]
                vals = []
                for t in trainers_no_noop:
                    row = ds_sub[ds_sub["trainer"] == t]
                    vals.append(row["overhead_pct"].values[0] if len(row) > 0 else 0)
                ax.bar(
                    x + ds_offsets[ds_name],
                    vals,
                    width,
                    label=ds_name,
                    color=ds_colors.get(ds_name, "gray"),
                    edgecolor="black",
                    linewidth=0.5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(trainers_no_noop, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Overhead vs noop (%)")
            ax.set_title(f"batch_size={batch}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(5, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

        fig.suptitle(f"Trainer Stats Overhead — workers={workers}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fname = out_dir / f"trainer_overhead_workers_{workers}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")

    # ================================================================
    # RESOURCE_UTIL_MAX vs ALL OTHER TRAINERS
    # ================================================================
    baselines = ["noop", "simple", "resource_util_max"]
    for baseline_name in baselines:
        baseline_data = avg[avg["trainer"] == baseline_name][
            ["data_source", "workers", "batch_size", "mean_s"]
        ].rename(columns={"mean_s": "baseline_s"})

        compared = avg.merge(baseline_data, on=["data_source", "workers", "batch_size"], how="left")
        compared["diff_s"] = compared["mean_s"] - compared["baseline_s"]
        compared["diff_pct"] = (compared["diff_s"] / compared["baseline_s"]) * 100

        print("\n\n" + "=" * 100)
        print(f"DURATION COMPARISON: ALL TRAINERS vs {baseline_name.upper()} BASELINE")
        print("=" * 100)

        for ds in sorted(compared["data_source"].unique()):
            ds_data = compared[compared["data_source"] == ds]
            print(f"\n{'─' * 100}")
            print(f"  Data Source: {ds}")
            print(f"{'─' * 100}")

            for workers in sorted(ds_data["workers"].unique()):
                for batch in sorted(ds_data["batch_size"].unique()):
                    subset = ds_data[(ds_data["workers"] == workers) & (ds_data["batch_size"] == batch)]
                    if subset.empty:
                        continue
                    baseline_val = subset[subset["trainer"] == baseline_name]["mean_s"]
                    baseline_str = f"{baseline_val.values[0]:.2f}" if len(baseline_val) > 0 else "N/A"
                    print(f"\n  Workers={workers}, Batch={batch}  (baseline {baseline_name} = {baseline_str}s)")
                    print(f"  {'Trainer':<20} {'Mean (s)':>10} {'Std (s)':>10} {'Runs':>6} {'Diff (s)':>14} {'Diff (%)':>14}")
                    print(f"  {'─' * 76}")

                    for _, row in subset.sort_values("mean_s").iterrows():
                        d_s = f"{row['diff_s']:+.2f}" if pd.notna(row['diff_s']) else "N/A"
                        d_pct = f"{row['diff_pct']:+.2f}%" if pd.notna(row['diff_pct']) else "N/A"
                        marker = " ◄" if row["trainer"] == baseline_name else ""
                        print(f"  {row['trainer']:<20} {row['mean_s']:>10.2f} {row['std_s']:>10.2f} {int(row['count']):>6} {d_s:>14} {d_pct:>14}{marker}")

        # --- Plots: overhead vs this baseline ---
        other_trainers = [t for t in TRAINERS if t != baseline_name]
        bl_colors = plt.cm.tab10(np.linspace(0, 1, len(other_trainers)))
        bl_trainer_colors = dict(zip(other_trainers, bl_colors))

        for ds in sorted(compared["data_source"].unique()):
            ds_data = compared[(compared["data_source"] == ds) & (compared["trainer"] != baseline_name)]
            if ds_data.empty:
                continue

            configs = ds_data.groupby(["workers", "batch_size"]).size().reset_index()[["workers", "batch_size"]]

            fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 7), sharey=True)
            if len(configs) == 1:
                axes = [axes]

            for idx, (_, cfg) in enumerate(configs.iterrows()):
                ax = axes[idx]
                subset = ds_data[(ds_data["workers"] == cfg["workers"]) & (ds_data["batch_size"] == cfg["batch_size"])]
                subset = subset.sort_values("diff_pct")

                bars = ax.barh(
                    range(len(subset)),
                    subset["diff_pct"],
                    color=[bl_trainer_colors.get(t, "gray") for t in subset["trainer"]],
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_yticks(range(len(subset)))
                ax.set_yticklabels(subset["trainer"])
                ax.set_xlabel(f"Difference vs {baseline_name} (%)")
                ax.set_title(f"w={cfg['workers']}, b={cfg['batch_size']}")
                ax.axvline(0, color="black", linewidth=0.8)

                for bar, pct in zip(bars, subset["diff_pct"]):
                    offset = 0.3 if bar.get_width() >= 0 else -0.3
                    ax.text(
                        bar.get_width() + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{pct:+.1f}%",
                        ha="left" if bar.get_width() >= 0 else "right",
                        va="center",
                        fontsize=8,
                    )

            fig.suptitle(f"Trainer Duration vs {baseline_name.upper()} — {ds}", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fname = out_dir / f"trainer_vs_{baseline_name}_{ds}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"\nSaved: {fname}")

        # Combined grouped bar chart by workers
        for workers in sorted(compared["workers"].unique()):
            batch_sizes = sorted(compared["batch_size"].unique())
            fig, axes = plt.subplots(1, len(batch_sizes), figsize=(6 * len(batch_sizes), 7), sharey=True)
            if len(batch_sizes) == 1:
                axes = [axes]

            for ax_idx, batch in enumerate(batch_sizes):
                ax = axes[ax_idx]
                subset = compared[
                    (compared["workers"] == workers)
                    & (compared["batch_size"] == batch)
                    & (compared["trainer"] != baseline_name)
                ]

                data_sources = sorted(subset["data_source"].unique())
                x = np.arange(len(other_trainers))
                width = 0.35
                ds_offsets = {ds_name: (i - (len(data_sources) - 1) / 2) * width for i, ds_name in enumerate(data_sources)}
                ds_colors = {"disk": "#4c72b0", "milabench": "#dd8452"}

                for ds_name in data_sources:
                    ds_sub = subset[subset["data_source"] == ds_name]
                    vals = []
                    for t in other_trainers:
                        row = ds_sub[ds_sub["trainer"] == t]
                        vals.append(row["diff_pct"].values[0] if len(row) > 0 else 0)
                    ax.bar(
                        x + ds_offsets[ds_name],
                        vals,
                        width,
                        label=ds_name,
                        color=ds_colors.get(ds_name, "gray"),
                        edgecolor="black",
                        linewidth=0.5,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(other_trainers, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel(f"Diff vs {baseline_name} (%)")
                ax.set_title(f"batch_size={batch}")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.legend(fontsize=8)

            fig.suptitle(f"Trainer Duration vs {baseline_name.upper()} — workers={workers}", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fname = out_dir / f"trainer_vs_{baseline_name}_workers_{workers}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"Saved: {fname}")

    # Duration table: absolute durations
    print("\n\n" + "=" * 100)
    print("RAW DURATION TABLE (seconds)")
    print("=" * 100)
    pivot = avg.pivot_table(
        index=["data_source", "workers", "batch_size"],
        columns="trainer",
        values="mean_s",
    )
    print(pivot.round(2).to_string())

    csv_path = out_dir / "trainer_overhead_data.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\nFull data saved to: {csv_path}")


if __name__ == "__main__":
    main()
