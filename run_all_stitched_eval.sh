#!/usr/bin/env bash
# Runs scripts/evaluate_stitched_plots.py for every checkpoint in this project that
# has both a native and a scaled_uint8 counterpart trained on the same data.
#
# F (results/checkpoints_scaled_uint8) and I (results/checkpoints_new_data) are
# deliberately SKIPPED: data/new_data's ground-truth masks were rebuilt with a
# corrected class encoding for the later J/K run (see RUN_COMPARISON.md's
# "corrected 3-tier annotation scheme" update), so the on-disk test masks no
# longer match F/I's learned class semantics (F/I predict classes 1-4 under the
# OLD offset-shifted scheme; the current masks only encode classes 1-3 under the
# NEW scheme). J/K are F/I's direct, corrected replacements and are included
# below, so no run is actually missing from this comparison.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain/bin/python

run() {
  local name="$1" checkpoint="$2" dataset_dir="$3" cropped_plots_dir="$4" plot_masks_dir="$5" preproc="$6"
  local out_dir="results/stitched_metrics/${name}"
  echo "=========================================================================="
  echo "STITCHED EVAL: ${name}"
  echo "=========================================================================="
  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP ${name}: checkpoint not found: $checkpoint"; return
  fi
  if [[ ! -d "$dataset_dir/images" ]]; then
    echo "SKIP ${name}: dataset_dir not found: $dataset_dir"; return
  fi
  if [[ ! -d "$cropped_plots_dir" ]]; then
    echo "SKIP ${name}: cropped_plots_dir not found: $cropped_plots_dir"; return
  fi
  if [[ ! -d "$plot_masks_dir" ]]; then
    echo "SKIP ${name}: plot_masks_dir not found: $plot_masks_dir"; return
  fi
  "$PY" scripts/evaluate_stitched_plots.py \
    --checkpoint "$checkpoint" \
    --dataset_dir "$dataset_dir" \
    --cropped_plots_dir "$cropped_plots_dir" \
    --plot_masks_dir "$plot_masks_dir" \
    --input_preprocessing "$preproc" \
    --output_dir "$out_dir"
  echo
}

# G, H: data/new_data_chm_update (11 train plots, 3 test plots) - old offset-shifted
# 4-tier class scheme (see RUN_COMPARISON.md; not rebuilt in this pass).
run "chm_update_native" \
  "results/checkpoints_chm_update_native/maskrcnn_best.pth" \
  "data/new_data_chm_update/dataset_sliced_800_test" \
  "data/new_data/cropped_plots/test" \
  "data/new_data_chm_update/test_plot_masks" \
  "native"

run "chm_update_scaled_uint8" \
  "results/checkpoints_chm_update_scaled_uint8/maskrcnn_best.pth" \
  "data/new_data_chm_update/dataset_sliced_800_test" \
  "data/new_data/cropped_plots/test" \
  "data/new_data_chm_update/test_plot_masks" \
  "scaled_uint8"

# J, K: data/new_data (26 train plots, 6 test plots), corrected 3-tier scheme.
run "new_data_native_3class" \
  "results/checkpoints_new_data_native_3class/maskrcnn_best.pth" \
  "data/new_data/dataset_sliced_800_test" \
  "data/new_data/cropped_plots/test" \
  "data/new_data/test_plot_masks" \
  "native"

run "new_data_scaled_uint8_3class" \
  "results/checkpoints_new_data_scaled_uint8_3class/maskrcnn_best.pth" \
  "data/new_data/dataset_sliced_800_test" \
  "data/new_data/cropped_plots/test" \
  "data/new_data/test_plot_masks" \
  "scaled_uint8"

# L, M: data/new_plots (34 train plots, 9 test plots), corrected 3-tier scheme.
run "new_plots_native" \
  "results/checkpoints_new_plots_native/maskrcnn_best.pth" \
  "data/new_plots/dataset_sliced_800_test" \
  "data/new_plots/cropped_plots/test" \
  "data/new_plots/test_plot_masks" \
  "native"

run "new_plots_scaled_uint8" \
  "results/checkpoints_new_plots_scaled_uint8/maskrcnn_best.pth" \
  "data/new_plots/dataset_sliced_800_test" \
  "data/new_plots/cropped_plots/test" \
  "data/new_plots/test_plot_masks" \
  "scaled_uint8"

echo "All stitched evaluations finished."
