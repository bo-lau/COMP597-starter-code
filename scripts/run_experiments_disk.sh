#!/bin/bash
# Multi-run Whisper experiments: disk-backed data (synthetic_whisper).
# NRUNS per batch size (default 3; override with env NRUNS=1), 5 minutes each.
# Output: logs/experiments_disk/<trainer_stats>/workers_{W}/batch_{B}/run_{R}/
#
# Usage:
#   ./scripts/run_experiments_disk.sh [BATCH1 BATCH2 BATCH3]
#   Example: ./scripts/run_experiments_disk.sh 128 64 32
#
# Runs per (batch, workers) cell:
#   NRUNS=1 ./scripts/run_experiments_disk.sh 128 64 32
# DataLoader worker counts (space-separated; each gets its own directory tree):
#   WORKERS="0 4 8" ./scripts/run_experiments_disk.sh 128 64 32
# Default batch sizes: 128 64 32 (minimum 32). Default workers: 0 4
#
# Trainer stats (space-separated; each gets its own tree under experiments_disk/<name>/):
#   TRAINER_STATS="resource_util phase_times noop simple" ./scripts/run_experiments_disk.sh
# Full set (long; includes CodeCarbon): resource_util resource_util_max phase_times noop simple codecarbon codecarbon_e2e

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

if [ "$#" -gt 0 ]; then
  BATCH_SIZES=("$@")
else
  BATCH_SIZES=(128 64 32)
fi
WORKERS="${WORKERS:-0 4}"
NRUNS="${NRUNS:-3}"
MAX_TIME_MIN=5
TRAINER_STATS="${TRAINER_STATS:-resource_util resource_util_max phase_times noop simple codecarbon codecarbon_e2e}"

echo "=== Experiment runner (disk cache: synthetic_whisper) ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "num_workers (each): ${WORKERS}"
echo "Runs per batch: ${NRUNS}"
echo "TRAINER_STATS: ${TRAINER_STATS}"
echo "Time limit: ${MAX_TIME_MIN} min per run"
echo 'Output root: logs/experiments_disk/<trainer_stats>/workers_<W>/batch_<B>/run_<R>'
echo ""

for STAT in ${TRAINER_STATS}; do
  EXPERIMENTS_ROOT="${REPO_DIR}/logs/experiments_disk/${STAT}"
  mkdir -p "${EXPERIMENTS_ROOT}"
  echo "###################################################################"
  echo "# trainer_stats=${STAT} -> ${EXPERIMENTS_ROOT}"
  echo "###################################################################"
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
        case "${STAT}" in
          resource_util)
            EXTRA=(--trainer_stats resource_util --trainer_stats_configs.resource_util.output_dir "${RUN_DIR}")
            ;;
          resource_util_max)
            EXTRA=(--trainer_stats resource_util_max --trainer_stats_configs.resource_util_max.output_dir "${RUN_DIR}")
            ;;
          phase_times)
            EXTRA=(--trainer_stats phase_times --trainer_stats_configs.phase_times.output_dir "${RUN_DIR}")
            ;;
          noop|simple)
            EXTRA=(--trainer_stats "${STAT}")
            ;;
          codecarbon)
            EXTRA=(--trainer_stats codecarbon --trainer_stats_configs.codecarbon.output_dir "${RUN_DIR}")
            ;;
          codecarbon_e2e)
            EXTRA=(--trainer_stats codecarbon_e2e --trainer_stats_configs.codecarbon_e2e.output_dir "${RUN_DIR}")
            ;;
          *)
            echo "Unknown TRAINER_STATS entry: ${STAT}" >&2
            exit 1
            ;;
        esac
        "${SCRIPTS_DIR}/srun.sh" \
          --logging.level INFO \
          --model whisper \
          --trainer simple \
          --data synthetic_whisper \
          --batch_size "${BATCH}" \
          --num_workers "${NW}" \
          --learning_rate 1e-6 \
          --max_time_minutes "${MAX_TIME_MIN}" \
          "${EXTRA[@]}" \
          2>&1 | tee "${RUN_DIR}/run.log" || true
      done
      echo ""
    done
  done
done

echo "=== Done. Aggregate & plot (examples): ==="
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_disk/resource_util/workers_0"
echo "  python scripts/plotting/plot_all_experiments.sh"
