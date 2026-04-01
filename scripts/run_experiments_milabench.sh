#!/bin/bash
# Multi-run Whisper experiments: Milabench-style in-memory data (synthetic_whisper_milabench).
# 3 runs per batch size, 5 minutes each.
# Output: logs/experiments_milabench/workers_{W}/batch_{B}/run_{R}/
#
# Usage:
#   ./scripts/run_experiments_milabench.sh [BATCH1 BATCH2 BATCH3]
#   Example: ./scripts/run_experiments_milabench.sh 128 64 32
#
# DataLoader worker counts (space-separated; each gets its own directory tree):
#   WORKERS="0 4 8" ./scripts/run_experiments_milabench.sh 128 64 32
# Default batch sizes: 128 64 32 (8 4 2 commented out below). Default WORKERS: 0 (see script body)
#
# Override repeat (optional). Unique sample count n = batch_size per run (sham-bolic memory rule).
#   DATA_REPEAT=150 ./scripts/run_experiments_milabench.sh 128 64 32
# RAM grows with batch_size (n); workers>0 copies the dataset per process.

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
EXPERIMENTS_ROOT="${REPO_DIR}/logs/experiments_milabench"

DATA_REPEAT="${DATA_REPEAT:-200}"

if [ "$#" -gt 0 ]; then
  BATCH_SIZES=("$@")
else
  BATCH_SIZES=(128 64 32)
  # BATCH_SIZES=(128 64 32 8 4 2)
fi
# Default 0 only: DataLoader workers replicate the full in-memory dataset (high RAM). Set WORKERS=4 to compare.
WORKERS="${WORKERS:-0}"
NRUNS=3
MAX_TIME_MIN=5

echo "=== Experiment runner (Milabench: synthetic_whisper_milabench) ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "num_workers (each): ${WORKERS}"
echo "Runs per batch: ${NRUNS}"
echo "n=batch_size repeat=${DATA_REPEAT} (effective len = n * repeat)"
echo "Time limit: ${MAX_TIME_MIN} min per run"
echo "Output root: ${EXPERIMENTS_ROOT}/workers_<W>/batch_<B>/run_<R>"
echo ""

mkdir -p "${EXPERIMENTS_ROOT}"

for NW in ${WORKERS}; do
  EXPERIMENTS_DIR="${EXPERIMENTS_ROOT}/workers_${NW}"
  mkdir -p "${EXPERIMENTS_DIR}"
  echo "========== num_workers=${NW} -> ${EXPERIMENTS_DIR} =========="
  for BATCH in "${BATCH_SIZES[@]}"; do
    BATCH_DIR="${EXPERIMENTS_DIR}/batch_${BATCH}"
    mkdir -p "${BATCH_DIR}"
    echo "--- workers=${NW} batch=${BATCH} ---"
    for R in $(seq 1 ${NRUNS}); do
        RUN_DIR="${BATCH_DIR}/run_${R}"
        mkdir -p "${RUN_DIR}"
        echo "  Run ${R}/${NRUNS} -> ${RUN_DIR}"
        {
            "${SCRIPTS_DIR}/srun.sh" \
                --logging.level INFO \
                --model whisper \
                --trainer simple \
                --data synthetic_whisper_milabench \
                --batch_size "${BATCH}" \
                --num_workers "${NW}" \
                --learning_rate 1e-6 \
                --max_time_minutes "${MAX_TIME_MIN}" \
                --trainer_stats resource_util_csv \
                --trainer_stats_configs.resource_util_csv.output_dir "${RUN_DIR}" \
                --trainer_stats_configs.resource_util_csv.output_file resource_util.csv \
                --trainer_stats_configs.resource_util_csv.substep_output_file resource_util_substeps.csv \
                --data_configs.synthetic_whisper_milabench.repeat "${DATA_REPEAT}"
        } 2>&1 | tee "${RUN_DIR}/run.log" || true
    done
    echo ""
  done
done

echo "=== Done. Aggregate & plot (pick one workers_* dir): ==="
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/workers_0"
