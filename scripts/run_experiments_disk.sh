#!/bin/bash
# Multi-run Whisper experiments: disk-backed data (synthetic_whisper).
# 3 runs per batch size, 5 minutes each.
# Output: logs/experiments_disk/workers_{W}/batch_{B}/run_{R}/
#
# Usage:
#   ./scripts/run_experiments_disk.sh [BATCH1 BATCH2 BATCH3]
#   Example: ./scripts/run_experiments_disk.sh 128 64 32
#
# DataLoader worker counts (space-separated; each gets its own directory tree):
#   WORKERS="0 4 8" ./scripts/run_experiments_disk.sh 128 64 32
# Default batch sizes: 128 64 32 (8 4 2 commented out below). Default workers: 0 4

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
EXPERIMENTS_ROOT="${REPO_DIR}/logs/experiments_disk"

if [ "$#" -gt 0 ]; then
  BATCH_SIZES=("$@")
else
  BATCH_SIZES=(128 64 32)
  # BATCH_SIZES=(128 64 32 8 4 2)
fi
WORKERS="${WORKERS:-0 4}"
NRUNS=3
MAX_TIME_MIN=5

echo "=== Experiment runner (disk cache: synthetic_whisper) ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "num_workers (each): ${WORKERS}"
echo "Runs per batch: ${NRUNS}"
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
        "${SCRIPTS_DIR}/srun.sh" \
            --logging.level INFO \
            --model whisper \
            --trainer simple \
            --data synthetic_whisper \
            --batch_size "${BATCH}" \
            --num_workers "${NW}" \
            --learning_rate 1e-6 \
            --max_time_minutes "${MAX_TIME_MIN}" \
            --trainer_stats resource_util_csv \
            --trainer_stats_configs.resource_util_csv.output_dir "${RUN_DIR}" \
            --trainer_stats_configs.resource_util_csv.output_file 'resource_util.csv' \
            --trainer_stats_configs.resource_util_csv.substep_output_file 'resource_util_substeps.csv' \
            2>&1 | tee "${RUN_DIR}/run.log" || true
    done
    echo ""
  done
done

echo "=== Done. Aggregate & plot (pick one workers_* dir): ==="
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/workers_0"
