#!/bin/bash
# Resume from: codecarbon_e2e / workers_4 / batch_32 / run_2
# Remaining: run_2 and run_3

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
EXPERIMENTS_DIR="${REPO_DIR}/logs/experiments_milabench/codecarbon_e2e/workers_4"
BATCH_DIR="${EXPERIMENTS_DIR}/batch_32"
MAX_TIME_MIN=5
BATCH=32
NW=4
REP=$((16000 / BATCH))

for R in 2 3; do
  RUN_DIR="${BATCH_DIR}/run_${R}"
  mkdir -p "${RUN_DIR}"
  echo "  Run ${R}/3 -> ${RUN_DIR}"
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
      --trainer_stats codecarbon_e2e \
      --trainer_stats_configs.codecarbon_e2e.output_dir "${RUN_DIR}" \
      --data_configs.synthetic_whisper_milabench.repeat "${REP}"
  } 2>&1 | tee "${RUN_DIR}/run.log" || true
done

echo "=== Resume complete ==="
