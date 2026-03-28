#!/bin/bash
# Backward-compatible alias: same as run_experiments_disk.sh (disk cache).
# Prefer: ./scripts/run_experiments_disk.sh  or  ./scripts/run_experiments_milabench.sh
SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
exec "${SCRIPTS_DIR}/run_experiments_disk.sh" "$@"
