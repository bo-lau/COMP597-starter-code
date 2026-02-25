#!/bin/bash

# ==============================
# Run file for Whisper example using sbatch (output saved to log file)
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run Whisper Simple Trainer with sbatch and Resource Utilization Tracking
${SCRIPTS_DIR}/sbatch.sh \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/whisper/stats'

echo "Job submitted! Output will be saved to: comp597-<node>-<jobid>.log"
echo "Check job status with: squeue -u $USER"
echo "View output with: tail -f comp597-*.log"
