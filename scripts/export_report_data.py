#!/usr/bin/env python3
"""
Export all raw data used to generate the plots and tables in the report
into a single CSV file with a 'sheet' column to distinguish sections.
"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / "logs"
OUTPUT = REPO_ROOT / "report" / "report_raw_data.csv"

frames = []


def add(df: pd.DataFrame, sheet: str) -> None:
    df = df.copy()
    df.insert(0, "sheet", sheet)
    frames.append(df)


# ── Figure 1: phase_time_bars_disk_w0_b32 ──────────────────────────
# Source: legacy phase_times.csv (disk, w=0, batch 32, averaged over 3 runs)
phase_times_path = LOGS / "experiments_disk" / "workers_0" / "batch_32" / "averaged" / "phase_times.csv"
if phase_times_path.exists():
    add(pd.read_csv(phase_times_path), "fig1_phase_time_bars_disk_w0_b32")
else:
    # Fall back to per-run data
    for r in sorted((LOGS / "experiments_disk" / "workers_0" / "batch_32").glob("run_*")):
        pt = r / "phase_times.csv"
        if pt.exists():
            df = pd.read_csv(pt)
            df["run"] = r.name
            add(df, "fig1_phase_time_bars_disk_w0_b32")

# ── Figure 2: resource_phases_disk_w0_b32 (violin plots) ───────────
# Source: legacy resource_util_substeps.csv (disk, w=0, batch 32)
substeps_path = LOGS / "experiments_disk" / "workers_0" / "batch_32" / "averaged" / "resource_util_substeps.csv"
if substeps_path.exists():
    add(pd.read_csv(substeps_path), "fig2_resource_phases_disk_w0_b32")
else:
    for r in sorted((LOGS / "experiments_disk" / "workers_0" / "batch_32").glob("run_*")):
        sub = r / "resource_util_substeps.csv"
        if sub.exists():
            df = pd.read_csv(sub)
            df["run"] = r.name
            add(df, "fig2_resource_phases_disk_w0_b32")

# ── Figure 3: gpu_util_disk_w0 (GPU util across batch sizes) ───────
# Source: resource_util.csv from disk/workers_0 per batch size
for bs in [32, 64, 128]:
    # Try per-trainer directory first (resource_util_steps.csv)
    ru_path = LOGS / "experiments_disk" / "resource_util" / "workers_0" / f"batch_{bs}"
    avg_path = LOGS / "experiments_disk" / "workers_0" / f"batch_{bs}" / "averaged" / "resource_util.csv"
    if avg_path.exists():
        df = pd.read_csv(avg_path)
        df["batch_size"] = bs
        add(df, "fig3_gpu_util_disk_w0")
    elif ru_path.exists():
        for r in sorted(ru_path.glob("run_*")):
            steps = r / "resource_util_steps.csv"
            if steps.exists():
                df = pd.read_csv(steps)
                df["batch_size"] = bs
                df["run"] = r.name
                add(df, "fig3_gpu_util_disk_w0")
                break  # latest run only

# ── Figure 4: gpu_cpu_util_mila_w0_b32 ─────────────────────────────
# Source: resource_util.csv from milabench/workers_0/batch_32
mila_avg = LOGS / "experiments_milabench" / "workers_0" / "batch_32" / "averaged" / "resource_util.csv"
mila_ru = LOGS / "experiments_milabench" / "resource_util" / "workers_0" / "batch_32"
if mila_avg.exists():
    add(pd.read_csv(mila_avg), "fig4_gpu_cpu_util_mila_w0_b32")
elif mila_ru.exists():
    for r in sorted(mila_ru.glob("run_*")):
        steps = r / "resource_util_steps.csv"
        if steps.exists():
            add(pd.read_csv(steps), "fig4_gpu_cpu_util_mila_w0_b32")
            break

# ── Figure 5: resources_disk_w0 (full resource timeline) ───────────
# Source: resource_util.csv from disk/workers_0/batch_32
res_avg = LOGS / "experiments_disk" / "workers_0" / "batch_32" / "averaged" / "resource_util.csv"
res_ru = LOGS / "experiments_disk" / "resource_util" / "workers_0" / "batch_32"
if res_avg.exists():
    add(pd.read_csv(res_avg), "fig5_resources_disk_w0")
elif res_ru.exists():
    for r in sorted(res_ru.glob("run_*")):
        steps = r / "resource_util_steps.csv"
        if steps.exists():
            add(pd.read_csv(steps), "fig5_resources_disk_w0")
            break

# ── Figures 6 & 7: overhead_vs_noop (workers=0 and workers=4) ──────
# Source: trainer_overhead_data.csv
overhead_path = LOGS / "comparison_plots" / "trainer_overhead_data.csv"
if overhead_path.exists():
    df_oh = pd.read_csv(overhead_path)
    add(df_oh[df_oh["workers"] == 0].copy(), "fig6_overhead_vs_noop_w0")
    add(df_oh[df_oh["workers"] == 4].copy(), "fig7_overhead_vs_noop_w4")

# ── Table 1: End-to-end execution time ─────────────────────────────
# Source: trainer_overhead_data.csv (full)
if overhead_path.exists():
    add(pd.read_csv(overhead_path), "table1_e2e_execution_time")

# ── Table 2: Energy consumption ─────────────────────────────────────
# Source: CodeCarbon cc_full_rank_0.csv files
energy_rows = []
for ds_name, ds_dir in [("disk", "experiments_disk"), ("milabench", "experiments_milabench")]:
    for trainer in ["codecarbon", "codecarbon_e2e"]:
        trainer_path = LOGS / ds_dir / trainer
        if not trainer_path.exists():
            continue
        for wdir in sorted(trainer_path.glob("workers_*")):
            workers = int(wdir.name.split("_")[1])
            for bdir in sorted(wdir.glob("batch_*")):
                batch = int(bdir.name.split("_")[1])
                for rdir in sorted(bdir.glob("run_*")):
                    run_num = int(rdir.name.split("_")[1])
                    for cc_file in rdir.glob("*cc_full_rank_0.csv"):
                        try:
                            df_cc = pd.read_csv(cc_file)
                            for _, row in df_cc.iterrows():
                                energy_rows.append({
                                    "data_source": ds_name,
                                    "trainer": trainer,
                                    "workers": workers,
                                    "batch_size": batch,
                                    "run": run_num,
                                    "duration_s": row.get("duration", None),
                                    "emissions_kgCO2": row.get("emissions", None),
                                    "cpu_power_w": row.get("cpu_power", None),
                                    "gpu_power_w": row.get("gpu_power", None),
                                    "ram_power_w": row.get("ram_power", None),
                                    "cpu_energy_kwh": row.get("cpu_energy", None),
                                    "gpu_energy_kwh": row.get("gpu_energy", None),
                                    "ram_energy_kwh": row.get("ram_energy", None),
                                    "total_energy_kwh": row.get("energy_consumed", None),
                                })
                        except Exception:
                            pass

if energy_rows:
    add(pd.DataFrame(energy_rows), "table2_energy_consumption")

# ── Combine and write ──────────────────────────────────────────────
if frames:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(combined)} rows across {combined['sheet'].nunique()} sections to {OUTPUT}")
    print("\nSections:")
    for name, group in combined.groupby("sheet", sort=False):
        print(f"  {name}: {len(group)} rows, {len(group.columns)-1} data columns")
else:
    print("No data found!")
