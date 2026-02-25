#!/bin/bash
# Run Whisper with synthetic data (local/interactive)

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

cd "${REPO_DIR}"

python3 launch.py \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --data synthetic_whisper \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats noop \
    "$@"
