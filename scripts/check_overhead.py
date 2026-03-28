#!/usr/bin/env python3
"""
Check that metrics collection adds < 5% overhead.
Runs baseline (noop) and resource_util_csv for --max-time minutes each, compares.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_training(trainer_stats: str, output_dir: Path, max_min: float, batch: int) -> float:
    cmd = [
        sys.executable, str(REPO_ROOT / "launch.py"),
        "--logging.level", "WARNING",
        "--model", "whisper",
        "--trainer", "simple",
        "--data", "synthetic_whisper",
        "--batch_size", str(batch),
        "--learning_rate", "1e-6",
        "--max_time_minutes", str(max_min),
        "--trainer_stats", trainer_stats,
    ]
    if trainer_stats == "resource_util_csv":
        cmd.extend(["--trainer_stats_configs.resource_util_csv.output_dir", str(output_dir)])
    start = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-time", type=float, default=5, help="Minutes per run")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs" / "overhead_check")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Overhead check ({args.max_time} min each, batch {args.batch}) ===\n")

    print("[1/2] Baseline (noop)...")
    t_baseline = run_training("noop", args.out_dir, args.max_time, args.batch)
    print(f"  Time: {t_baseline:.1f} s\n")

    metrics_dir = args.out_dir / "with_metrics"
    metrics_dir.mkdir(exist_ok=True)
    print("[2/2] With resource_util_csv...")
    t_metrics = run_training("resource_util_csv", metrics_dir, args.max_time, args.batch)
    print(f"  Time: {t_metrics:.1f} s\n")

    overhead = (t_metrics - t_baseline) / t_baseline * 100 if t_baseline > 0 else 0
    print("=== Result ===")
    print(f"  Baseline:     {t_baseline:.1f} s")
    print(f"  With metrics: {t_metrics:.1f} s")
    print(f"  Overhead:     {overhead:.2f}%")
    if overhead > 5:
        print("  WARNING: Overhead exceeds 5%")
    else:
        print("  OK: Overhead under 5%")
    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
