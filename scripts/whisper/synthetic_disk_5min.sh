#!/bin/bash
# Whisper + disk-backed synthetic data (synthetic_whisper): 5 min, resource_util.
# See docs/WHISPER_DATA_LOADING.md and scripts/whisper/README.md
#
# Writes logs/resource_util_steps.csv (and summary txt).
# Plot: python scripts/plotting/plot_resources.py --input logs/resource_util_steps.csv

WHISPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${WHISPER_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

"${SCRIPTS_DIR}/srun.sh" \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --data synthetic_whisper \
    --batch_size 32 \
    --learning_rate 1e-6 \
    --max_time_minutes 5 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir "${REPO_DIR}/logs"
