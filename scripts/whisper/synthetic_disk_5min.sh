#!/bin/bash
# Whisper + disk-backed synthetic data (synthetic_whisper): 5 min, resource_util_csv.
# See docs/WHISPER_DATA_LOADING.md and scripts/whisper/README.md
#
# Writes logs/resource_util.csv and logs/resource_util_substeps.csv.
# Plot: python scripts/plotting/plot_resources.py --input logs/resource_util.csv

WHISPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${WHISPER_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

"${SCRIPTS_DIR}/srun.sh" \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --data synthetic_whisper \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --max_time_minutes 5 \
    --trainer_stats resource_util_csv \
    --trainer_stats_configs.resource_util_csv.output_dir "${REPO_DIR}/logs" \
    --trainer_stats_configs.resource_util_csv.output_file 'resource_util.csv' \
    --trainer_stats_configs.resource_util_csv.substep_output_file 'resource_util_substeps.csv'
