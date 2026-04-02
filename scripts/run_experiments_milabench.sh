#!/bin/bash
# Multi-run Whisper experiments: Milabench-style in-memory data (synthetic_whisper_milabench).
# NRUNS per batch size (default 3; override with env NRUNS=1), 5 minutes each.
# Output: logs/experiments_milabench/<trainer_stats>/workers_{W}/batch_{B}/run_{R}/
#
# Usage:
#   ./scripts/run_experiments_milabench.sh [BATCH1 BATCH2 BATCH3]
#   Example: ./scripts/run_experiments_milabench.sh 128 64 32
#
# Runs per (batch, workers) cell:
#   NRUNS=1 ./scripts/run_experiments_milabench.sh 128 64 32
# DataLoader worker counts (space-separated; each gets its own directory tree):
#   WORKERS="0 4 8" ./scripts/run_experiments_milabench.sh 128 64 32
# Default batch sizes: 128 64 32 (minimum 32). Default WORKERS: 0 (see script body)
#
# Constant dataset length: len = n * repeat = batch_size * repeat = MILABENCH_TOTAL_SAMPLES.
#   repeat is set per batch as MILABENCH_TOTAL_SAMPLES / batch_size (integer division).
#   Pick MILABENCH_TOTAL_SAMPLES divisible by every batch size you sweep (default: 16000).
# RAM grows with batch_size (n); workers>0 copies the dataset per process.
#
# Trainer stats (see run_experiments_disk.sh):
#   TRAINER_STATS="resource_util phase_times noop" ./scripts/run_experiments_milabench.sh

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

MILABENCH_TOTAL_SAMPLES="${MILABENCH_TOTAL_SAMPLES:-16000}"

if [ "$#" -gt 0 ]; then
  BATCH_SIZES=("$@")
else
  BATCH_SIZES=(128 64 32)
fi
WORKERS="${WORKERS:-0}"
NRUNS="${NRUNS:-3}"
MAX_TIME_MIN=5
TRAINER_STATS="${TRAINER_STATS:-resource_util resource_util_max phase_times noop simple codecarbon codecarbon_e2e}"

echo "=== Experiment runner (Milabench: synthetic_whisper_milabench) ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "num_workers (each): ${WORKERS}"
echo "Runs per batch: ${NRUNS}"
echo "TRAINER_STATS: ${TRAINER_STATS}"
echo "MILABENCH_TOTAL_SAMPLES=${MILABENCH_TOTAL_SAMPLES} (repeat = total / batch_size each run; len = total)"
echo "Time limit: ${MAX_TIME_MIN} min per run"
echo 'Output root: logs/experiments_milabench/<trainer_stats>/workers_<W>/batch_<B>/run_<R>'
echo ""

for STAT in ${TRAINER_STATS}; do
  EXPERIMENTS_ROOT="${REPO_DIR}/logs/experiments_milabench/${STAT}"
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
      REP=$((MILABENCH_TOTAL_SAMPLES / BATCH))
      if [ $((MILABENCH_TOTAL_SAMPLES % BATCH)) -ne 0 ]; then
        echo "  WARNING: MILABENCH_TOTAL_SAMPLES=${MILABENCH_TOTAL_SAMPLES} not divisible by batch_size=${BATCH}; repeat=${REP} gives len=$((BATCH * REP)) != total"
      fi
      echo "  repeat=${REP} (batch_size * repeat = $((BATCH * REP)))"
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
            "${EXTRA[@]}" \
            --data_configs.synthetic_whisper_milabench.repeat "${REP}"
        } 2>&1 | tee "${RUN_DIR}/run.log" || true
      done
      echo ""
    done
  done
done

echo "=== Done. Aggregate & plot (examples): ==="
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir logs/experiments_milabench/resource_util/workers_0"
echo "  python scripts/plotting/plot_all_experiments.sh"
