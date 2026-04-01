#!/usr/bin/env bash
# One random smoke-test run + plots into the same output directory.
#
# Usage (from repo root, or any cwd — script cds to repo):
#   ./scripts/whisper/test_one_random_run.sh
#
# Optional env:
#   OUT_DIR          — default: logs/test_runs/<timestamp>_<random>
#   USE_LOCAL=1      — run `python3 launch.py` on this machine (no Slurm). Default is Slurm via ./scripts/srun.sh.
#   MAX_TIME_MINUTES — default: 3 (short)
#
# Examples:
#   ./scripts/whisper/test_one_random_run.sh
#   OUT_DIR=logs/my_try MAX_TIME_MINUTES=5 ./scripts/whisper/test_one_random_run.sh
#   USE_LOCAL=1 ./scripts/whisper/test_one_random_run.sh   # interactive GPU node / laptop

set -euo pipefail

WHISPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${WHISPER_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
cd "${REPO_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/logs/test_runs/${STAMP}}"
MAX_TIME_MINUTES="${MAX_TIME_MINUTES:-3}"
mkdir -p "${OUT_DIR}"

# --- random discrete config (edit arrays to taste) ---
BATCHES=(4 8 16 32 64 128)
BATCH="${BATCHES[$((RANDOM % ${#BATCHES[@]}))]}"

declare -a WORKERS_OPTS=(0 2 4)
NW="${WORKERS_OPTS[$((RANDOM % ${#WORKERS_OPTS[@]}))]}"

if (( RANDOM % 2 )); then
  DATA_NAME="synthetic_whisper_milabench"
  REPEAT_OPTS=(50 100 200)
  REP="${REPEAT_OPTS[$((RANDOM % ${#REPEAT_OPTS[@]}))]}"
  DATA_ARGS=(
    --data "${DATA_NAME}"
    --data_configs.synthetic_whisper_milabench.repeat "${REP}"
  )
else
  DATA_NAME="synthetic_whisper"
  REP=""
  DATA_ARGS=(--data "${DATA_NAME}")
fi

SMOOTH_OPTS=(1 5)
SMOOTH="${SMOOTH_OPTS[$((RANDOM % ${#SMOOTH_OPTS[@]}))]}"

{
  echo "timestamp=${STAMP}"
  echo "data=${DATA_NAME}"
  echo "batch_size=${BATCH}"
  echo "num_workers=${NW}"
  echo "max_time_minutes=${MAX_TIME_MINUTES}"
  if [[ -n "${REP}" ]]; then echo "repeat=${REP}"; fi
  echo "plot_smooth=${SMOOTH}"
} | tee "${OUT_DIR}/run_config.txt"

echo ""
echo "=== Launching (output: ${OUT_DIR}) ==="

BASE_ARGS=(
  --logging.level INFO
  --model whisper
  --trainer simple
  --batch_size "${BATCH}"
  --num_workers "${NW}"
  --learning_rate 1e-6
  --max_time_minutes "${MAX_TIME_MINUTES}"
  --trainer_stats resource_util_csv
  --trainer_stats_configs.resource_util_csv.output_dir "${OUT_DIR}"
  --trainer_stats_configs.resource_util_csv.output_file resource_util.csv
  --trainer_stats_configs.resource_util_csv.substep_output_file resource_util_substeps.csv
)

if [[ "${USE_LOCAL:-0}" == "1" ]]; then
  python3 launch.py "${BASE_ARGS[@]}" "${DATA_ARGS[@]}"
else
  "${SCRIPTS_DIR}/srun.sh" "${BASE_ARGS[@]}" "${DATA_ARGS[@]}"
fi

echo ""
echo "=== Plotting ==="
python3 "${SCRIPTS_DIR}/plotting/plot_resources.py" \
  --input "${OUT_DIR}/resource_util.csv" \
  --output-dir "${OUT_DIR}/plots" \
  --smooth "${SMOOTH}"

echo ""
echo "Done. CSV + plots: ${OUT_DIR}"
echo "  open ${OUT_DIR}/plots/resource_util.png"
