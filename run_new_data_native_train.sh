#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MASKRCNN_CONFIG="$(pwd)/config_new_data_native.ini"
python train.py
