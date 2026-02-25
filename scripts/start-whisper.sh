#!/bin/bash

# ==============================
# Run file for Whisper example - how to use the launch script to run Whisper with simple trainer
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run Whisper Simple Trainer with Resource Utilization Tracking
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/whisper/stats'

# Copy results to home directory for easy access (with timestamp)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="${REPO_DIR}/whisper-results"
mkdir -p "${RESULTS_DIR}"
echo "Copying results to home directory..."
./scripts/bash_srun.sh "cp /home/slurm/comp597/students/blau9/whisper/stats/resource_utilization_summary.txt /home/2023/blau9/COMP597/whisper-results/whisper-results-${TIMESTAMP}.txt 2>/dev/null && echo 'Results copied to ${RESULTS_DIR}/whisper-results-${TIMESTAMP}.txt' || echo 'Warning: Could not copy results file'"

### run Whisper with CodeCarbon tracking
# ${SCRIPTS_DIR}/srun.sh \
#     --logging.level INFO \
#     --model whisper \
#     --trainer simple \
#     --batch_size 4 \
#     --learning_rate 1e-6 \
#     --trainer_stats codecarbon \
#     --trainer_stats_configs.codecarbon.run_num 1 \
#     --trainer_stats_configs.codecarbon.project_name whisper \
#     --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/whisper/codecarbonlogs'
