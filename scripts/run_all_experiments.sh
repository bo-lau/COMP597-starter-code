#!/bin/bash
# Full Whisper sweep: disk cache + Milabench, every batch size × every worker count.
# Each combo runs 3×5 min (see run_experiments_*.sh).
#
# Usage:
#   ./scripts/run_all_experiments.sh [BATCH1 BATCH2 ...]
#   ./scripts/run_all_experiments.sh 128 64 32
#
# Environment:
#   WORKERS   Space-separated num_workers values (default: 0 4)
#   DATA_REPEAT  Milabench repeat multiplier (default: from run_experiments_milabench.sh; n = batch_size)
#   DATA_REPEAT  Milabench repeat (default: 200)
#
# Options:
#   --disk-only       Only logs/experiments_disk/
#   --milabench-only  Only logs/experiments_milabench/
#   --dry-run         Print commands, do not run

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

RUN_DISK=1
RUN_MILABENCH=1
DRY_RUN=0
BATCH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --disk-only) RUN_MILABENCH=0 ;;
    --milabench-only) RUN_DISK=0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) BATCH_ARGS+=("$arg") ;;
  esac
done

if [ "${#BATCH_ARGS[@]}" -eq 0 ]; then
  BATCH_SIZES=(128 64 32)
  # BATCH_SIZES=(128 64 32 8 4 2)
else
  BATCH_SIZES=("${BATCH_ARGS[@]}")
fi

WORKERS="${WORKERS:-0 4}"

echo "==================================================================="
echo " run_all_experiments.sh"
echo " Batch sizes: ${BATCH_SIZES[*]}"
echo " Workers:     ${WORKERS}"
echo " Disk:        $([ "$RUN_DISK" -eq 1 ] && echo yes || echo no)"
echo " Milabench:   $([ "$RUN_MILABENCH" -eq 1 ] && echo yes || echo no)"
echo " Dry-run:     $([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)"
echo "==================================================================="
echo ""

_run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

if [ "$RUN_DISK" -eq 1 ]; then
  echo ">>> Disk cache (synthetic_whisper) -> logs/experiments_disk/workers_*/"
  _run env WORKERS="$WORKERS" "${SCRIPTS_DIR}/run_experiments_disk.sh" "${BATCH_SIZES[@]}"
  echo ""
fi

if [ "$RUN_MILABENCH" -eq 1 ]; then
  echo ">>> Milabench (synthetic_whisper_milabench) -> logs/experiments_milabench/workers_*/"
  _run env WORKERS="$WORKERS" "${SCRIPTS_DIR}/run_experiments_milabench.sh" "${BATCH_SIZES[@]}"
  echo ""
fi

echo "=== Done ==="
echo "Aggregate & plot (pick workers_* and re-run for each):"
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir ${REPO_DIR}/logs/experiments_disk/workers_0"
echo "  python scripts/plotting/aggregate_and_plot.py --experiments-dir ${REPO_DIR}/logs/experiments_milabench/workers_0"
echo "(Repeat for workers_4, etc.)"
