#!/bin/bash
# Check that metrics collection adds < 5% overhead.
# Runs 5-min baseline (noop) and 5-min with resource_util.
#
# Usage:
#   Local:  ./scripts/check_overhead.sh
#   Slurm:  ./scripts/check_overhead.sh --slurm

SCRIPT_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPT_DIR}/..")
OUT_DIR="${REPO_DIR}/logs/overhead_check"
mkdir -p "${OUT_DIR}"

BATCH=32
MAX_MIN=5

if [ "$1" = "--slurm" ]; then
    RUN_CMD="${SCRIPT_DIR}/srun.sh"
    echo "Using Slurm (srun)"
else
    RUN_CMD="python3 ${REPO_DIR}/launch.py"
    echo "Using local (launch.py)"
fi

echo "=== Overhead check (${MAX_MIN} min each, batch ${BATCH}) ==="
echo ""

echo "[1/2] Baseline (noop)..."
T0=$(python3 -c "import time; print(time.perf_counter())")
if [ "$1" = "--slurm" ]; then
    "${SCRIPT_DIR}/srun.sh" --logging.level WARNING --model whisper --trainer simple --data synthetic_whisper \
        --batch_size ${BATCH} --learning_rate 1e-6 --max_time_minutes ${MAX_MIN} --trainer_stats noop \
        2>&1 | tee "${OUT_DIR}/baseline.log" | tail -3
else
    python3 "${REPO_DIR}/launch.py" --logging.level WARNING --model whisper --trainer simple --data synthetic_whisper \
        --batch_size ${BATCH} --learning_rate 1e-6 --max_time_minutes ${MAX_MIN} --trainer_stats noop \
        2>&1 | tee "${OUT_DIR}/baseline.log" | tail -3
fi
T1=$(python3 -c "import time; print(time.perf_counter())")
BASELINE=$(python3 -c "print(round($T1 - $T0, 1))")
echo "  Time: ${BASELINE} s"
echo ""

echo "[2/2] With resource_util..."
mkdir -p "${OUT_DIR}/with_metrics"
T0=$(python3 -c "import time; print(time.perf_counter())")
if [ "$1" = "--slurm" ]; then
    "${SCRIPT_DIR}/srun.sh" --logging.level WARNING --model whisper --trainer simple --data synthetic_whisper \
        --batch_size ${BATCH} --learning_rate 1e-6 --max_time_minutes ${MAX_MIN} --trainer_stats resource_util \
        --trainer_stats_configs.resource_util.output_dir "${OUT_DIR}/with_metrics" \
        2>&1 | tee "${OUT_DIR}/with_metrics.log" | tail -3
else
    python3 "${REPO_DIR}/launch.py" --logging.level WARNING --model whisper --trainer simple --data synthetic_whisper \
        --batch_size ${BATCH} --learning_rate 1e-6 --max_time_minutes ${MAX_MIN} --trainer_stats resource_util \
        --trainer_stats_configs.resource_util.output_dir "${OUT_DIR}/with_metrics" \
        2>&1 | tee "${OUT_DIR}/with_metrics.log" | tail -3
fi
T1=$(python3 -c "import time; print(time.perf_counter())")
METRICS=$(python3 -c "print(round($T1 - $T0, 1))")
echo "  Time: ${METRICS} s"
echo ""

OVERHEAD=$(python3 -c "b=${BASELINE}; m=${METRICS}; print(f'{(m-b)/b*100:.2f}' if b > 0 else '0')")
echo "=== Result ==="
echo "  Baseline:     ${BASELINE} s"
echo "  With metrics: ${METRICS} s"
echo "  Overhead:     ${OVERHEAD}%"
python3 -c "
b, m = ${BASELINE}, ${METRICS}
oh = (m-b)/b*100 if b > 0 else 0
print('  WARNING: Overhead exceeds 5%' if oh > 5 else '  OK: Overhead under 5%')
"
echo ""
echo "Logs: ${OUT_DIR}"
