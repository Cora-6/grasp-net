#!/bin/zsh
set -e
cd "$(dirname "$0")"
ROBOT_PYTHON="${ROBOT_PYTHON:-python3}"
export MPLCONFIGDIR="$PWD/audit/matplotlib"
mkdir -p audit
"$ROBOT_PYTHON" scripts/download_data.py --root "$PWD" --stride 16 --archives scenes.tar >> audit/download.log 2>&1 &
image_pid=$!
"$ROBOT_PYTHON" scripts/download_data.py --root "$PWD" --stride 16 --archives rect_labels.tar >> audit/download_labels.log 2>&1 &
label_pid=$!
wait "$image_pid"
wait "$label_pid"
"$ROBOT_PYTHON" scripts/pipeline.py >> audit/pipeline.log 2>&1
