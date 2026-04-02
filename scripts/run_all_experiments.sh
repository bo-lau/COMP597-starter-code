#!/bin/bash
# Full Whisper sweep: disk cache + Milabench, every batch size × every worker count
# × every trainer_stats in TRAINER_STATS (see run_experiments_disk.sh).
# Each combo runs NRUNS×5 min (default NRUNS=1 here; run_experiments_*.sh default NRUNS=3 when invoked directly).
#
# Usage:
#   ./scripts/run_all_experiments.sh [BATCH1 BATCH2 ...]
#   ./scripts/run_all_experiments.sh 128 64 32
#
# Environment:
#   NRUNS                      Runs per (batch, workers) cell (default: 1 for this script)
#   WORKERS                    Space-separated num_workers values (default: 0 4)
#   MILABENCH_TOTAL_SAMPLES    Milabench: fixed dataset len (default: 16000)
#   TRAINER_STATS              Space-separated stats (default: resource_util resource_util_max phase_times noop simple codecarbon codecarbon_e2e)
#                              CodeCarbon modes are slow; use e.g. TRAINER_STATS=resource_util for a quick sweep.
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
else
  BATCH_SIZES=("${BATCH_ARGS[@]}")
fi

NRUNS="${NRUNS:-1}"
WORKERS="${WORKERS:-0 4}"
# Full default set (disk + Milabench): resource_util, resource_util_max, phase_times, noop, simple, codecarbon, codecarbon_e2e
TRAINER_STATS="${TRAINER_STATS:-resource_util resource_util_max phase_times noop simple codecarbon codecarbon_e2e}"

echo "==================================================================="
echo " run_all_experiments.sh"
echo " Batch sizes: ${BATCH_SIZES[*]}"
echo " Workers:     ${WORKERS}"
echo " NRUNS:       ${NRUNS}"
echo " TRAINER_STATS: ${TRAINER_STATS}"
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
  echo '>>> Disk cache (synthetic_whisper) -> logs/experiments_disk/<trainer_stats>/workers_*/'
  _run env NRUNS="$NRUNS" WORKERS="$WORKERS" TRAINER_STATS="$TRAINER_STATS" "${SCRIPTS_DIR}/run_experiments_disk.sh" "${BATCH_SIZES[@]}"
  echo ""
fi

if [ "$RUN_MILABENCH" -eq 1 ]; then
  echo '>>> Milabench (synthetic_whisper_milabench) -> logs/experiments_milabench/<trainer_stats>/workers_*/'
  _run env NRUNS="$NRUNS" WORKERS="$WORKERS" TRAINER_STATS="$TRAINER_STATS" "${SCRIPTS_DIR}/run_experiments_milabench.sh" "${BATCH_SIZES[@]}"
  echo ""
fi

echo "=== Done ==="
echo "Aggregate & plot (plot_all runs aggregate for every workers_* under each trainer_stats dir):"
echo "  python scripts/plotting/plot_all_experiments.sh"
