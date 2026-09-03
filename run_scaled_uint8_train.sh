#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MASKRCNN_CONFIG="$(pwd)/config_scaled_uint8.ini"
#mkdir -p checkpoints_scaled_uint8
python train.py
