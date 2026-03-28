#!/bin/bash
# Multi-run Whisper experiments: disk-backed data (synthetic_whisper).
# 3 runs per batch size, 5 minutes each. Output: logs/experiments/batch_{N}/run_{R}/
#
# Usage:
#   ./scripts/run_experiments_disk.sh [BATCH1 BATCH2 BATCH3]
#   Example: ./scripts/run_experiments_disk.sh 8 4 2
#   Single batch, 3 runs: ./scripts/run_experiments_disk.sh 4
#   Default batch sizes: 8 4 2

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
EXPERIMENTS_DIR="${REPO_DIR}/logs/experiments"

BATCH_SIZES=("${@:-8 4 2}")
NRUNS=3
MAX_TIME_MIN=5

echo "=== Experiment runner (disk cache: synthetic_whisper) ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Runs per batch: ${NRUNS}"
echo "Time limit: ${MAX_TIME_MIN} min per run"
echo "Output: ${EXPERIMENTS_DIR}"
echo ""

mkdir -p "${EXPERIMENTS_DIR}"

for BATCH in "${BATCH_SIZES[@]}"; do
    BATCH_DIR="${EXPERIMENTS_DIR}/batch_${BATCH}"
    mkdir -p "${BATCH_DIR}"
    echo "--- Batch size ${BATCH} ---"
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

echo "=== Done. Aggregate & plot: ==="
echo "  python scripts/plotting/aggregate_and_plot.py"
