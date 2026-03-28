#!/bin/bash
# Compatibility wrapper — see scripts/whisper/synthetic_disk_5min.sh and docs/WHISPER_DATA_LOADING.md
WH="$(cd "$(dirname "$0")" && pwd)"
exec "${WH}/whisper/synthetic_disk_5min.sh" "$@"
