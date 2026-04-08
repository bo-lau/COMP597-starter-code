#!/bin/bash
# Aggregate + plot every batch_* under each workers_* folder for disk and Milabench experiments.
#
# Supports both layouts:
#   - Legacy: logs/experiments_disk/workers_W/batch_*
#   - Per trainer_stats: logs/experiments_disk/<trainer_stats>/workers_W/batch_*
#
# Usage (from repo root):
#   ./scripts/plotting/plot_all_experiments.sh
#   ./scripts/plotting/plot_all_experiments.sh --disk-only
#   ./scripts/plotting/plot_all_experiments.sh --milabench-only
#
# Optional:
#   SMOOTH=5 ./scripts/plotting/plot_all_experiments.sh
#   RUNS=all ./scripts/plotting/plot_all_experiments.sh   # mean across run_1, run_2, … (legacy)
#   Default RUNS=latest: only the most recent run per batch (by CSV mtime).
#
# Output layout (any batch size, e.g. batch_128 / batch_64 / batch_32):
#   logs/experiments_disk/workers_W/plots/batch_B/*.png
#   logs/experiments_disk/resource_util/workers_W/plots/batch_B/*.png
#   logs/experiments_milabench/.../workers_W/plots/batch_B/*.png

set -uo pipefail

PLOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PLOT_DIR}/../.." && pwd)"
AGG="${PLOT_DIR}/aggregate_and_plot.py"

RUN_DISK=1
RUN_MILABENCH=1
for arg in "$@"; do
  case "$arg" in
    --disk-only) RUN_MILABENCH=0 ;;
    --milabench-only) RUN_DISK=0 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--disk-only] [--milabench-only]" >&2
      exit 1
      ;;
  esac
done

SMOOTH="${SMOOTH:-1}"
RUNS="${RUNS:-all}"
EXTRA_ARGS=(--smooth "${SMOOTH}" --runs "${RUNS}")

_plot_root() {
  local root="$1"
  local label="$2"
  if [[ ! -d "$root" ]]; then
    echo "Skip (missing): $root"
    return 0
  fi
  local found=0
  while IFS= read -r -d '' wdir; do
    found=1
    wname="${wdir#"${root}/"}"
    if [[ ! "$(find "$wdir" -maxdepth 1 -type d -name 'batch_*' -print -quit)" ]]; then
      echo "Skip $label/${wname}: no batch_* dirs"
      continue
    fi
    echo "=== ${label} / ${wname} ==="
    python3 "${AGG}" --experiments-dir "$wdir" "${EXTRA_ARGS[@]}"
    echo ""
  done < <(find "$root" -type d -name 'workers_*' -print0 | sort -zV)

  if [[ "$found" -eq 0 ]]; then
    echo "No workers_* under $root"
  fi
}

echo "Repo: ${REPO_ROOT}"
echo "Smooth: ${SMOOTH}"
echo "Runs: ${RUNS} (latest = most recent CSV per batch; all = average every run_*)"
echo ""

if [[ "$RUN_DISK" -eq 1 ]]; then
  _plot_root "${REPO_ROOT}/logs/experiments_disk" "disk"
fi
if [[ "$RUN_MILABENCH" -eq 1 ]]; then
  _plot_root "${REPO_ROOT}/logs/experiments_milabench" "milabench"
fi

echo "All requested plots finished."
