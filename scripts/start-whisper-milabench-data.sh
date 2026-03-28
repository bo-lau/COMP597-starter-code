#!/bin/bash
# Compatibility wrapper — see scripts/whisper/milabench_5min.sh and docs/WHISPER_DATA_LOADING.md
WH="$(cd "$(dirname "$0")" && pwd)"
exec "${WH}/whisper/milabench_5min.sh" "$@"
