#!/bin/bash

# ==============================
# Run file for Whisper example with srun but redirect output to file
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

OUTPUT_FILE="${REPO_DIR}/whisper-$(date +%Y%m%d-%H%M%S).log"

echo "Running Whisper training..."
echo "Output will be saved to: ${OUTPUT_FILE}"

### run Whisper Simple Trainer with Resource Utilization Tracking and output redirected
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/whisper/stats' \
    2>&1 | tee "${OUTPUT_FILE}"

echo ""
echo "Training completed! Results saved to: ${OUTPUT_FILE}"
