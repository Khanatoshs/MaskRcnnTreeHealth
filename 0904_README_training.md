# 0904 — Training README

How to run the 0904 training end to end, which files and data go into it, and what
changed most recently. Written so the runs can be reproduced later without
re-deriving anything.

Four models were trained on the 0904 annotation set:

| Run | Bands | Preprocessing | Complete F1 | Checkpoint |
|---|---|---|---|---|
| N | 8 | native | 0.514 | `results/checkpoints_0904_native/` |
| O | 8 | scaled_uint8 | 0.608 | `results/checkpoints_0904_scaled_uint8/` |
| P | 10 | native | 0.615 | `results/checkpoints_0904_10ch_native/` |
| **Q** | **10** | **scaled_uint8** | **0.630** | `results/checkpoints_0904_10ch_scaled_uint8/` |

**Q is the best model in the project.** Detailed results: [MODEL_Q_REPORT.md](MODEL_Q_REPORT.md).

---

## 1. What data and files go into a run

### 1.1 Annotations (the labels)

| File | What it is |
|---|---|
| `data/annotations/0904/Plot_59_0904.shp` | 59 plot boundaries, with a `class` column splitting them **49 train / 10 test** and a `plot_id` column naming them |
| `data/annotations/0904/Annotation0904.shp` | 2937 hand-drawn tree crown polygons, each with a `tree_class` column |

`tree_class` is **1 = healthy, 2 = mild, 3 = severe** (743 / 1060 / 1134 polygons).
It is already 1-indexed, so `[MASKS] CLASS_ID_OFFSET = 0` — see §3.1.

### 1.2 Source imagery (the input bands)

All eight base rasters live in `data/tiff/new/`. Bands 9–10 are only used by the
10-band runs (P/Q).

| Band | File | Notes |
|---|---|---|
| 1–3 | `data/tiff/new/rgb_new2.tiff` | RGB drone photo; bands 1,2,3 extracted separately |
| 4 | `data/tiff/new/chm_ge2m.tif` | CHM — canopy height (metres) |
| 5 | `data/tiff/new/ndvi_ge2m.tif` | NDVI |
| 6 | `data/tiff/new/cire_ge2m.tif` | CIRE |
| 7 | `data/tiff/new/gndvi_ge2m.tif` | GNDVI |
| 8 | `data/tiff/new/ndre_ge2m.tif` | NDRE |
| 9 | `data/tiff/new/el/el (1)/el` | **EL — elevation.** ESRI ArcInfo grid (a folder of `.adf` files), Byte, nodata 255 |
| 10 | `data/tiff/new/intensity/intensity` | **INTENSITY — LiDAR return intensity.** ArcInfo grid, Int16, nodata −32768 |

> ⚠️ **Careful with the elevation path.** `data/tiff/new/el/el` (without the `(1)`) is
> an **incomplete copy** — it is missing the `z001001x.adf` block index and the
> `z001002*`/`z001003*` files. It *opens* without error and then fails partway
> through a job, on 11 of 59 plots. The correct one is **`data/tiff/new/el/el (1)/el`**
> (15 `.adf` files; the broken one has 10). Deleting the broken copy is recommended.
> `scripts/check_extra_bands.py` detects this before any training starts.

### 1.3 Config files

| Config | Run | Bands | Preprocessing |
|---|---|---|---|
| `config_0904_native.ini` | N | 8 | native |
| `config_0904_scaled_uint8.ini` | O | 8 | scaled_uint8 |
| `config_0904_10ch_native.ini` | P | 10 | native |
| `config_0904_10ch_scaled_uint8.ini` | Q | 10 | scaled_uint8 |
| `config_stitched_eval.ini` | — | evaluation thresholds (score, NMS, min size, visualization style) |

Every script picks its config from the `MASKRCNN_CONFIG` environment variable,
falling back to `config.ini`. **Always set it explicitly** — otherwise a script
silently uses the wrong dataset.

### 1.4 Generated data (produced by the pipeline)

| Path | Contents |
|---|---|
| `results/stacked_10ch_out_0904.vrt` | the 10-band virtual stack (plus 3 small RGB band VRTs) |
| `data/new_0904/` | 8-band dataset (runs N/O) |
| `data/new_0904_10ch/` | 10-band dataset (runs P/Q) |
| &nbsp;&nbsp;`cropped_plots/{train,test}/` | 49 + 10 plot GeoTIFFs |
| &nbsp;&nbsp;`{train,test}_plot_masks/` | one mask per plot — 2424 train + 573 test tree instances |
| &nbsp;&nbsp;`dataset_sliced_800/` | 780 training/validation tiles |
| &nbsp;&nbsp;`dataset_sliced_800_test/` | 160 held-out test tiles |

### 1.5 Scripts used

| Script | Role |
|---|---|
| `scripts/check_extra_bands.py` | **pre-flight** — verifies extra bands are complete and readable |
| `scripts/image_merging_vrt.py` | stacks the source rasters into one VRT |
| `scripts/cut_plots.py` | crops the VRT into per-plot GeoTIFFs, split train/test |
| `scripts/create_masks.py` | rasterizes tree polygons into per-plot instance+class masks |
| `scripts/slice_plots.py` | cuts plots into 800 px tiles with 50% overlap |
| `scripts/compute_channel_stats.py` | measures per-band mean/std for the native config |
| `scripts/compute_anchor_sizes.py` | measures crown sizes (anchor tuning) |
| `train.py` | trains the model |
| `scripts/evaluate_stitched_plots.py` | **the evaluation to use** — whole-plot scoring |
| `run_0904_10ch_build.sh` | runs the whole 10-band data build in one command |

Python environment (has torch, rasterio, GDAL, sklearn, pycocotools):

```
/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain/bin/python
```

---

## 2. How to run the training

Run everything **from the repository root**.

### 2.1 Ten-band run (P/Q) — the current best path

```bash
cd /media/smartforest/Storage/Burma/MaskRCNN/MaskRcnnTreeHealth
PY=/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain/bin/python
```

**Step 1 — build the dataset** (pre-flight + VRT + plots + masks + tiles + stats):

```bash
./run_0904_10ch_build.sh
```

This runs six steps and stops immediately if anything is wrong. Expect:
`10 bands` · `49 train / 10 test plots` · `2424 + 573 tree instances` ·
`780 train + 160 test tiles`. It finishes by printing measured `image_mean` /
`image_std`.

**Step 2 — paste the measured statistics** printed at the end of step 1 into
`config_0904_10ch_native.ini` (the `image_mean` / `image_std` lines).

> Only the **native** config needs this. `scaled_uint8` rescales bands to fixed
> physical ranges instead, so it keeps the 0.5/0.5 placeholder convention.
> ⚠️ Normalization is **not** stored in the checkpoint — a wrong value produces
> silently wrong results, never an error. See `CLAUDE.md`.

**Step 3 — train** (~50 min each on the RTX 4070 Ti; run sequentially, one GPU):

```bash
# native (run P)
MASKRCNN_CONFIG="$(pwd)/config_0904_10ch_native.ini" $PY train.py

# scaled_uint8 (run Q)
MASKRCNN_CONFIG="$(pwd)/config_0904_10ch_scaled_uint8.ini" $PY train.py
```

Training stops early when pooled F1 stops improving (patience 30) and keeps only
the single best checkpoint, `maskrcnn_best.pth`. Progress goes to
`train_0904_10ch_native.log` / `train_0904_10ch_scaled_uint8.log`.

**Step 4 — evaluate** (see §2.3).

### 2.2 Eight-band run (N/O)

Identical, minus the two extra bands. The 8-band VRT already exists
(`results/stacked_8ch_out_new_data.vrt`), so the build is just three steps:

```bash
export MASKRCNN_CONFIG="$(pwd)/config_0904_native.ini"
$PY scripts/cut_plots.py       # -> data/new_0904/cropped_plots
$PY scripts/create_masks.py    # -> data/new_0904/{train,test}_plot_masks
$PY scripts/slice_plots.py     # -> data/new_0904/dataset_sliced_800{,_test}
$PY scripts/compute_channel_stats.py --dataset_dir data/new_0904/dataset_sliced_800

# then train both
MASKRCNN_CONFIG="$(pwd)/config_0904_native.ini"       $PY train.py
MASKRCNN_CONFIG="$(pwd)/config_0904_scaled_uint8.ini" $PY train.py
```

To rebuild that VRT from scratch instead, run `$PY scripts/image_merging_vrt.py` first.

### 2.3 Evaluation (stitched — the one to use)

```bash
$PY scripts/evaluate_stitched_plots.py \
    --checkpoint results/checkpoints_0904_10ch_scaled_uint8/maskrcnn_best.pth \
    --dataset_dir data/new_0904_10ch/dataset_sliced_800_test \
    --cropped_plots_dir data/new_0904_10ch/cropped_plots/test \
    --plot_masks_dir data/new_0904_10ch/test_plot_masks \
    --input_preprocessing scaled_uint8 \
    --visualization_style boxes \
    --output_dir results/stitched_metrics/0904_10ch_scaled_uint8
```

`--input_preprocessing` **must match how the model was trained** (`native` for P,
`scaled_uint8` for Q). `--visualization_style` is `boxes` (rectangles) or `masks`
(the predicted crown outlines tinted over each tree).

Outputs land in `--output_dir`: `stitched_metrics.json`, `stitched_metrics.csv`,
`confusion_matrix.csv` + `_heatmap.png`, and `visualizations/` with one annotated
image per test plot.

**Why "stitched"**: tiles overlap 50%, so scoring each tile separately counts a tree
near a seam twice and inflates the score (detection F1 by 0.17–0.25). This script
stitches predictions back into whole plots, merges duplicates with NMS, drops tiny
boxes, and scores once per real tree.

---

## 3. Most recent changes

Everything below is in commit `55a5b47`, except the config path fix in §3.7.

### 3.1 Class-offset bug — fixed

The newer annotation files number classes 1/2/3, but `create_masks.py` was adding
+1 (a rule for the older 0-indexed shapefile). Every tree was shifted one class up
and "healthy" was never used — which is why earlier reports showed **zero healthy
trees**. The offset is now a config value, `[MASKS] CLASS_ID_OFFSET`, defaulting to
the old behaviour so the original dataset is unaffected. The 0904 configs set `0`.

### 3.2 Extra input bands — new capability

`scripts/image_merging_vrt.py` gained `[MULTICHANNEL] extra_band_paths`, a
comma-separated list appended as bands 9, 10, … **after** the fixed eight, so band 4
stays CHM (`CHM_BAND` and the mean/std order depend on it). Blank or absent → the
usual 8-band stack. ArcInfo grid folders work directly via GDAL's AIG driver.

### 3.3 Pre-flight check — new script

`scripts/check_extra_bands.py` reads every extra band over every plot and reports
unreadable plots, value range, nodata count, and any values above 255. It exists
because a partially copied ArcInfo grid opens cleanly and only fails deep into a
job. Run before any rebuild that adds bands:

```bash
MASKRCNN_CONFIG=config_0904_10ch_native.ini $PY scripts/check_extra_bands.py
```

### 3.4 Two preprocessing traps the new bands exposed — fixed

- `DEFAULT_PREPROCESS_RANGES` only covered channels 0–7, and unlisted channels fall
  back to `(0, 1)` — which would have clipped elevation (11–241) and intensity
  (0–158) to a **constant 1.0**, silently deleting both new bands in `scaled_uint8`
  mode. Ranges added for bands 9–10.
- The RGB 0-255 → 0-1 rescale was gated on the maximum across **all** bands, so any
  band above 255 (Int16 intensity can be) would flip the test off and leave RGB
  unscaled while `image_mean`/`image_std` expected 0–1. Now gated on the RGB bands
  only. Verified identical behaviour on 8-band data.

### 3.5 Stitched evaluation — new script

`scripts/evaluate_stitched_plots.py` (see §2.3). Also added: a
`--visualization_style boxes|masks` toggle, and a `config_stitched_eval.ini` holding
the thresholds.

### 3.6 Visualization legend — improved (most recent change)

The per-plot images now carry a **much larger legend placed below the image** (it
used to be tiny and sat on top of the canopy, hiding trees). Colours and line styles
are spelled out in words:

- **healthy (cyan)** · **mild (purple)** · **severe (orange)**
- **solid box** = found, class correct
- **dash-dot box** = found, wrong class
- **white dashed box** = missed tree / false positive

### 3.7 Stale file paths — fixed

`rgb_new2.tiff` had been moved from `data/tiff/` into `data/tiff/new/`, but the
configs still pointed at the old location. This went unnoticed because runs reused
an already-built VRT rather than rebuilding one. All four 0904 configs now point at
`data/tiff/new/rgb_new2.tiff`. (The old `results/stacked_8ch_out_new_data.vrt` still
contains the dead path internally — harmless, but rebuild it if you need that VRT.)

### 3.8 Other fixes

- `scripts/create_masks.py` and `scripts/image_merging_vrt.py` now respect
  `MASKRCNN_CONFIG` (they were hardcoded to `config.ini`).
- `scripts/evaluate_separated_metrics.py` — fixed a GPU memory leak that crashed
  evaluation with CUDA out-of-memory on noisier checkpoints (predictions, including
  full-resolution masks, were accumulated on the GPU and never moved to CPU).
- `scripts/evaluate_test_set.py` and `scripts/visualize_predictions.py` — class
  names are now chosen from the checkpoint's actual class count, so 3-tier models no
  longer print the old 4-tier labels (class 3 was being shown as "moderate" instead
  of "severe").

---

## 4. Results

Stitched evaluation, held-out test set (10 plots, 160 tiles, 573 trees),
score ≥ 0.5, IoU ≥ 0.5.

| Run | Bands | Preproc | Complete F1 | Detection F1 | Grading F1 |
|---|---|---|---|---|---|
| N | 8 | native | 0.514 | 0.591 | 0.860 |
| O | 8 | scaled_uint8 | 0.608 | 0.666 | 0.913 |
| P | 10 | native | 0.615 | 0.672 | 0.914 |
| **Q** | 10 | scaled_uint8 | **0.630** | **0.682** | **0.924** |

Per-class F1 (complete task):

| Run | healthy | mild | severe |
|---|---|---|---|
| N | 0.523 | 0.439 | 0.590 |
| O | 0.617 | 0.589 | 0.620 |
| P | 0.653 | 0.573 | 0.626 |
| **Q** | 0.634 | 0.605 | 0.651 |

**What this says:** adding elevation + intensity was worth **+0.101 F1 to native**
and **+0.022 to scaled_uint8** — the largest input-side gain in the project. The
gain is mostly in *detection*, which is the bottleneck: Q finds ~68% of trees but
grades ~92% of those correctly. Of Q's errors, 181 trees were missed entirely while
only 30 were found-but-misclassified.

---

## 5. Common problems

| Symptom | Cause / fix |
|---|---|
| `Failed to open grid block index file: ...x.adf` | Incomplete ArcInfo grid copy — check `el (1)/el` is the one in use (§1.2). Run `check_extra_bands.py`. |
| `No such file or directory: data/tiff/rgb_new2.tiff` | Stale path — should be `data/tiff/new/rgb_new2.tiff` (§3.7). |
| Model trains but scores nonsense | `image_mean`/`image_std` or `--input_preprocessing` do not match the checkpoint. Normalization is not stored in the checkpoint. |
| "healthy" class shows zero ground truth | `CLASS_ID_OFFSET` wrong for the shapefile in use (§3.1). |
| A band appears to have no effect | Missing entry in `DEFAULT_PREPROCESS_RANGES` flattens it in `scaled_uint8` mode (§3.4). |
| CUDA out of memory during evaluation | Make sure no other training is running on the GPU; the separated-metrics leak itself is fixed (§3.8). |
| Script used the wrong dataset | `MASKRCNN_CONFIG` not set — it falls back to `config.ini`. |

## 6. Related documents

| Document | Contents |
|---|---|
| [MODEL_Q_REPORT.md](MODEL_Q_REPORT.md) | Full report on the best model (Q), all metrics per class |
| [MODEL_RUNS_SUMMARY.md](MODEL_RUNS_SUMMARY.md) | Every run in the project, methodology explained simply |
| [STITCHED_EVAL_COMPARISON.md](STITCHED_EVAL_COMPARISON.md) | Stitched evaluation results, all runs, chronological |
| [RUN_COMPARISON.md](RUN_COMPARISON.md) | Older tile-level results (kept as history — reads optimistically high) |
| [CLAUDE.md](CLAUDE.md) | Project conventions and hard-won rules |
