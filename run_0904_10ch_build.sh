#!/usr/bin/env bash
# Rebuilds the 0904 dataset as a 10-band stack (the usual 8 bands + EL elevation +
# LiDAR INTENSITY, appended as bands 9 and 10) and reports the per-channel stats
# needed for the native config.
#
# Run this only after data/tiff/new/el/el has been re-copied *completely* - the
# first attempt failed because that ArcInfo grid was missing its z001001x.adf block
# index, which reads fine for 48 of 59 plots and then hard-fails on the other 11.
# Step 0 below catches exactly that before any of the expensive work starts.
#
# After this finishes: paste the printed image_mean/image_std into
# config_0904_10ch_native.ini, then launch training for both configs.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain/bin/python
export MASKRCNN_CONFIG="$(pwd)/config_0904_10ch_native.ini"

echo "=========================================================================="
echo "STEP 0/5  pre-flight: are both extra bands complete and readable?"
echo "=========================================================================="
"$PY" scripts/check_extra_bands.py

echo "=========================================================================="
echo "STEP 1/5  build the 10-band VRT"
echo "=========================================================================="
"$PY" scripts/image_merging_vrt.py
"$PY" - <<'EOF'
from osgeo import gdal
ds = gdal.Open("results/stacked_10ch_out_0904.vrt")
print(f"  VRT: {ds.RasterXSize} x {ds.RasterYSize}, {ds.RasterCount} bands")
assert ds.RasterCount == 10, f"expected 10 bands, got {ds.RasterCount}"
EOF

echo "=========================================================================="
echo "STEP 2/5  cut plots"
echo "=========================================================================="
"$PY" scripts/cut_plots.py
echo "  train plots: $(find data/new_0904_10ch/cropped_plots/train -name '*.tif' | wc -l)"
echo "  test plots:  $(find data/new_0904_10ch/cropped_plots/test  -name '*.tif' | wc -l)"

echo "=========================================================================="
echo "STEP 3/5  rasterize annotations into per-plot masks"
echo "=========================================================================="
"$PY" scripts/create_masks.py
tail -n 3 utils.log

echo "=========================================================================="
echo "STEP 4/5  slice plots into 800px tiles"
echo "=========================================================================="
"$PY" scripts/slice_plots.py
echo "  train/val tiles: $(find data/new_0904_10ch/dataset_sliced_800/images      -name '*.tif' | wc -l)"
echo "  test tiles:      $(find data/new_0904_10ch/dataset_sliced_800_test/images -name '*.tif' | wc -l)"

echo "=========================================================================="
echo "STEP 5/5  measure per-channel stats for the native config"
echo "=========================================================================="
"$PY" scripts/compute_channel_stats.py --dataset_dir data/new_0904_10ch/dataset_sliced_800

echo
echo "Build complete. Paste the image_mean/image_std above into"
echo "config_0904_10ch_native.ini, then start training."
