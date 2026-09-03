#!/usr/bin/env bash
# Runs the Mask R-CNN data pipeline up to (but not including) training:
#   VRT build -> cut plots -> create masks -> slice plots
# Must be run from the repo root (scripts read config.ini from the cwd).

set -euo pipefail

PYTHON="/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ ! -f config.ini ]; then
    echo "config.ini not found in $REPO_ROOT (copy config_sample.ini to config.ini and fill in paths)" >&2
    exit 1
fi

echo "== [1/4] Building stacked VRT (image_merging_vrt.py) =="
"$PYTHON" scripts/image_merging_vrt.py

echo "== [2/4] Cutting plots into per-plot GeoTIFFs (cut_plots.py) =="
"$PYTHON" scripts/cut_plots.py

echo "== [3/4] Rasterizing tree masks (create_masks.py) =="
"$PYTHON" scripts/create_masks.py

echo "== [4/4] Slicing plots into training tiles (slice_plots.py) =="
"$PYTHON" scripts/slice_plots.py

echo "== Done. Ready for train.py (not run by this script). =="
