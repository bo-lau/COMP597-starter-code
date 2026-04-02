#!/bin/bash
# Whisper + Milabench-style in-memory data (synthetic_whisper_milabench): 5 min + plots.
# See docs/WHISPER_DATA_LOADING.md and scripts/whisper/README.md
#
# Custom repeat (n = --batch_size):
#   --data_configs.synthetic_whisper_milabench.repeat 50

WHISPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${WHISPER_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
OUT_DIR="${REPO_DIR}/logs/milabench_whisper"
PLOTS_DIR="${REPO_DIR}/milabench_whisper"

mkdir -p "${OUT_DIR}"
mkdir -p "${PLOTS_DIR}"

"${SCRIPTS_DIR}/srun.sh" \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --data synthetic_whisper_milabench \
    --batch_size 32 \
    --num_workers 0 \
    --learning_rate 1e-6 \
    --max_time_minutes 5 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir "${OUT_DIR}" \
    --data_configs.synthetic_whisper_milabench.repeat 200

echo ""
echo "Generating plots..."
python3 "${SCRIPTS_DIR}/plotting/plot_resources.py" \
    --input "${OUT_DIR}/resource_util_steps.csv" \
    --output-dir "${PLOTS_DIR}" \
    --smooth 1
echo "Plots saved to ${PLOTS_DIR}"
