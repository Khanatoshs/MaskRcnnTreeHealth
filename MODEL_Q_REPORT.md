# Model Q — Full Report

The best-performing model produced in this project. This document is meant to stand
on its own: what Q is, how the data was built, how it was trained, how it was
scored, and every number it produced — explained plainly enough to pick up later.

- **Checkpoint**: `results/checkpoints_0904_10ch_scaled_uint8/maskrcnn_best.pth`
- **Config**: [config_0904_10ch_scaled_uint8.ini](config_0904_10ch_scaled_uint8.ini)
- **Evaluation outputs**: `results/stitched_metrics/0904_10ch_scaled_uint8/`
- **Headline**: pooled F1 **0.630** · detection F1 **0.682** · grading F1 **0.924**

---

## 1. What Q does, in plain terms

Q takes a 10-channel aerial image of a forest plot and, for every tree crown it can
find, draws a box around it and labels its health as **healthy**, **mild**, or
**severe**. It is a Mask R-CNN with a ResNet-50-FPN backbone.

The 10 input channels, in band order:

| Band | What it is |
|---|---|
| 1–3 | Red, Green, Blue (drone photo) |
| 4 | CHM — canopy height model (tree height in metres) |
| 5 | NDVI — vegetation greenness index |
| 6 | CIRE — chlorophyll index, red edge |
| 7 | GNDVI — green normalized vegetation index |
| 8 | NDRE — normalized difference red edge |
| 9 | EL — terrain elevation (added 2026-09-05) |
| 10 | INTENSITY — LiDAR return intensity (added 2026-09-05) |

Bands 9 and 10 are what separate Q from the earlier 8-band models, and they are the
single biggest input-side improvement made in this project (see §7).

## 2. How the training data was built

Five steps, all reproducible from the config above:

1. **Stack the bands** (`scripts/image_merging_vrt.py`) — the ten source rasters are
   combined into one virtual 10-band image (a VRT — a pointer file, so it costs
   almost no disk or RAM). Bands 9–10 are appended *after* the original eight via
   the `[MULTICHANNEL] extra_band_paths` setting, so band 4 stays CHM (several other
   settings depend on that position).
2. **Cut into plots** (`scripts/cut_plots.py`) — a shapefile of plot boundaries
   (`data/annotations/0904/Plot_59_0904.shp`) crops that mosaic into one GeoTIFF per
   plot: **59 plots, split 49 train / 10 test**. The split is fixed by the surveyors
   in the shapefile and never changes.
3. **Turn annotations into masks** (`scripts/create_masks.py`) — hand-drawn tree
   crown polygons (`data/annotations/0904/Annotation0904.shp`, each tagged with a
   health class) are burned into a mask image per plot. Every tree pixel gets a
   value encoding both which tree and what class it is
   (`value = class_id × 10000 + tree_number`); background stays 0. Result:
   **2424 training and 573 test tree instances**.
4. **Slice into tiles** (`scripts/slice_plots.py`) — plots are too large to feed the
   network, so each is cut into 800×800 pixel tiles with **50% overlap** (so every
   crown is complete in at least one tile). Empty tiles are dropped. Result:
   **780 train/val tiles and 160 test tiles**.
5. **Train** (`train.py`).

The test plots are held out from step 2 onward and are never seen during training or
validation — every number in §5–§6 comes from them.

## 3. How Q was trained

| Setting | Value |
|---|---|
| Architecture | Mask R-CNN, ResNet-50-FPN backbone, `trainable_layers=3` |
| Input channels | 10 |
| Classes | 4 = background + healthy / mild / severe |
| Preprocessing | `scaled_uint8` (see below) |
| Anchors | sizes `112,144,184,216,288`; aspect ratios `0.62,1.0,1.36` |
| Batch size | 2 |
| Learning rate | 1e-4, cosine decay, 10 warm-up epochs |
| Early stopping | patience 30, monitored on pooled F1 |
| Result | **best epoch 12**, stopped after 42 |
| Train/val split | by whole plot (39 train / 10 val plots), never by tile |

**Why split by plot, not by tile**: tiles overlap 50%, so a tile-level split would
put near-identical images (containing the same trees) on both sides and inflate
validation scores.

**What `scaled_uint8` means**: each band is clipped to a physically sensible range
and rescaled to 0–1 before the model sees it (elevation clipped to 0–255, intensity
to 0–160, CHM to 0–30 m, etc. — see `DEFAULT_PREPROCESS_RANGES` in
`dataset/multi_channel_dataset.py`). The alternative used elsewhere in this project,
`native`, instead leaves bands in their physical units and normalizes with measured
per-band statistics. Q uses `scaled_uint8`; its twin **P** is the identical model
trained with `native` and scores slightly lower (0.615 vs 0.630).

## 4. How Q was scored — the "stitched" method

This is the important methodological point, because it differs from how earlier
runs in this project were measured.

**The problem with the obvious approach.** The natural thing is to score each 800px
tile on its own and add up the results. But tiles overlap by 50%, so a tree near a
tile boundary appears in two tiles and can be *counted as correctly found twice*.
That inflates the score — measurably: detection F1 came out 0.17–0.25 higher this
way across every model tested.

**What was done instead** (`scripts/evaluate_stitched_plots.py`):

1. Run Q on all 160 test tiles.
2. **Stitch** — convert every predicted box from tile coordinates back into
   whole-plot coordinates, using the pixel offset stored in each tile's filename.
   Now all of a plot's predictions live in one picture instead of 16 overlapping
   ones.
3. **Merge duplicates** — the same tree detected in two overlapping tiles is now
   two overlapping boxes, so standard Non-Maximum Suppression (IoU threshold 0.3)
   collapses them into one, keeping the more confident detection.
4. **Drop implausibly small boxes** (under 20 px across) as noise.
5. **Compare against the plot's real ground truth** — read straight from the
   per-plot mask, one entry per real tree, so neither side can double-count.
   Matching is greedy by confidence, at IoU ≥ 0.5, done **per plot** so a
   prediction can never accidentally match a different plot's tree.

Evaluation settings used: `score_threshold=0.5`, `iou_threshold=0.5`,
`nms_iou_threshold=0.3`, `min_crown_side_px=20`.

For Q this took **6457 raw detections down to 1410** after merging and filtering —
most of what the model emits across overlapping tiles is duplicate views of the same
trees.

### Three ways the results are reported

- **Detection** — did we find the tree at all? (ignores health class)
- **Grading** — of the trees we found, did we get the health class right?
- **Complete (pooled)** — the whole job: found *and* correctly classified. This is
  the headline number, and it is always the lowest of the three, because it
  penalises both kinds of mistake.

### What is *not* measured

All metrics here are **bounding-box based**. Q's segmentation masks — the actual
crown outlines Mask R-CNN produces — are trained and produced (and can be rendered
with `--visualization_style masks`) but their pixel accuracy has **never been
scored**: `compute_iou` works on boxes, and the COCO AP is computed with
`iouType="bbox"`. So a prediction with a sloppy mask but a good box scores exactly
like one with a perfect crown outline. If crown *area/extent* matters downstream,
that gap needs closing.

---

## 5. Results — the headline numbers

Held-out test set: **10 plots, 160 tiles, 573 real trees.**

| Level | Precision | Recall | F1 |
|---|---|---|---|
| **Detection** (found the tree at all) | 0.679 | 0.684 | **0.682** |
| **Grading** (right class, given found) | 0.929 | 0.920 | **0.924** |
| **Complete** (found *and* classified right) | 0.627 | 0.632 | **0.630** |

Counts behind those: detection tp 392 / fp 185 / fn 181 · complete tp 362 / fp 215 /
fn 211.

**In plain terms**: Q finds about **two-thirds** of the trees, and when it finds one
it gets the health class right about **92%** of the time. The overall score is
limited by *finding* trees, not by classifying them.

## 6. Results per class

### 6a. Complete task — found *and* correctly classified

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 90 | 51 | 53 | 0.638 | 0.629 | **0.634** |
| Mild | 130 | 88 | 82 | 0.596 | 0.613 | **0.605** |
| Severe | 142 | 76 | 76 | 0.651 | 0.651 | **0.651** |
| **Pooled** | 362 | 215 | 211 | 0.627 | 0.632 | **0.630** |
| Macro average | | | | 0.629 | 0.631 | 0.630 |

Very even across classes — only 0.046 separates best (severe) from worst (mild).

### 6b. Detection only — was the tree found at all, regardless of class

| True class | Detected | Of total | Missed | Detection recall |
|---|---|---|---|---|
| Healthy | 101 | 143 | 42 | **0.706** |
| Mild | 140 | 212 | 72 | **0.660** |
| Severe | 151 | 218 | 67 | **0.693** |
| **Overall** | 392 | 573 | 181 | **0.684** |

### 6c. Grading only — of trees that were found, was the class right

| Class | Correct | Predicted | Actual | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 90 | 93 | 101 | 0.968 | 0.891 | **0.928** |
| Mild | 130 | 150 | 140 | 0.867 | 0.929 | **0.897** |
| Severe | 142 | 149 | 151 | 0.953 | 0.940 | **0.947** |
| **Macro** | | | | 0.929 | 0.920 | **0.924** |

Severe is graded most reliably (0.947); mild is the most often confused, which makes
sense — it is the middle category with neighbours on both sides.

### 6d. Confusion matrix

Rows = what the tree really is, columns = what Q called it. The `bg` column means
"never detected at all".

| true \ predicted | bg (missed) | healthy | mild | severe |
|---|---|---|---|---|
| **healthy** | 42 | **90** | 11 | 0 |
| **mild** | 72 | 3 | **130** | 7 |
| **severe** | 67 | 0 | 9 | **142** |
| *(false positives)* | — | 48 | 68 | 69 |

**The key reading**: the `bg` column (42 + 72 + 67 = 181 missed trees) dwarfs all the
class-confusion cells combined (11 + 3 + 7 + 9 = 30). Q's errors are overwhelmingly
*trees it never saw*, not trees it misjudged. Note also that healthy is never
confused with severe in either direction — the mistakes it does make are between
adjacent severity levels.

### 6e. Class-agnostic AP (COCO-style, boxes)

| Metric | Value |
|---|---|
| AP @ IoU 0.50:0.95 | 0.263 |
| AP @ IoU 0.50 | 0.634 |
| AP @ IoU 0.75 | 0.167 |
| AR @ 100 detections | 0.393 |

The gap between AP50 (0.634) and AP75 (0.167) says boxes are placed roughly right
but not tightly — good enough to identify a tree, looser on exact extent.

### 6f. Confidence-threshold sweep

The default operating point is 0.5. This shows the precision/recall trade-off:

| Threshold | Precision | Recall | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|
| 0.3 | 0.508 | 0.688 | 0.585 | 394 | 381 | 179 |
| 0.4 | 0.578 | 0.661 | 0.617 | 379 | 277 | 194 |
| **0.5** | 0.627 | 0.632 | **0.630** | 362 | 215 | 211 |
| **0.6** | 0.686 | 0.588 | **0.633** ← best | 337 | 154 | 236 |
| 0.7 | 0.725 | 0.506 | 0.596 | 290 | 110 | 283 |
| 0.8 | 0.779 | 0.375 | 0.506 | 215 | 61 | 358 |
| 0.9 | 0.864 | 0.199 | 0.323 | 114 | 18 | 459 |

Peak F1 is at 0.6 (0.633), only marginally above the default. If you need **fewer
false alarms**, 0.7 buys precision 0.725 at the cost of recall. If you need to
**miss fewer trees**, 0.3 gets recall to 0.688 but precision drops to ~0.5.

### 6g. Per-plot breakdown

| Plot | Tiles | Real trees | Raw detections | After merge+filter |
|---|---|---|---|---|
| plot_20 | 16 | 50 | 690 | 129 |
| plot_23 | 16 | 75 | 578 | 120 |
| plot_28 | 16 | 48 | 659 | 145 |
| plot_32 | 16 | 53 | 628 | 140 |
| plot_34 | 16 | 84 | 724 | 149 |
| plot_38 | 16 | 51 | 733 | 152 |
| plot_45 | 16 | 33 | 579 | 136 |
| plot_50 | 16 | 37 | 579 | 135 |
| plot_53 | 16 | 79 | 655 | 146 |
| plot_58 | 16 | 63 | 632 | 158 |

## 7. How Q compares

Same data, same settings, only the listed factor changed:

| Run | Bands | Preprocessing | Complete F1 | Detection F1 | Grading F1 |
|---|---|---|---|---|---|
| N | 8 | native | 0.514 | 0.591 | 0.860 |
| O | 8 | scaled_uint8 | 0.608 | 0.666 | 0.913 |
| P | **10** | native | 0.615 | 0.672 | 0.914 |
| **Q** | **10** | scaled_uint8 | **0.630** | **0.682** | **0.924** |

- **Adding elevation + intensity was worth +0.022 F1 to Q** (O → Q) and a much
  larger **+0.101 to the native variant** (N → P). It is the biggest input-side
  gain achieved in this project, and it helped detection most — exactly the
  bottleneck.
- **Preprocessing choice barely matters once those bands exist**: Q beats P by only
  0.015, where at 8 bands the same comparison was a 0.094 gap.
- Q is the best model in the project on all three levels. Full cross-run context:
  `STITCHED_EVAL_COMPARISON.md` and `MODEL_RUNS_SUMMARY.md`.

## 8. Bottom line, in simple terms

- Q **finds ~68% of trees** and, of those, **grades ~92% correctly** — ending at
  **63% fully correct** end to end.
- **Detection is the bottleneck, not classification.** 181 trees were missed
  entirely; only 30 were found-but-misclassified. Effort spent on finding more
  trees (more annotated plots, better detection) will pay off far more than effort
  spent on the health classifier.
- **Performance is even across the three health classes** (F1 0.605–0.651), and
  errors it does make are between neighbouring severity levels — healthy is never
  once confused with severe.
- **The numbers are honest ones.** They come from whole-plot scoring with duplicate
  detections merged, on plots the model never trained on. Earlier tile-by-tile
  numbers elsewhere in this project run optimistically high and are not comparable.
- **Known gap**: mask/segmentation quality is unmeasured (§4). If crown area or
  outline accuracy matters for downstream work, that still needs quantifying.

## 9. Files

| What | Where |
|---|---|
| Checkpoint | `results/checkpoints_0904_10ch_scaled_uint8/maskrcnn_best.pth` |
| Training config | `config_0904_10ch_scaled_uint8.ini` |
| Training log | `train_0904_10ch_scaled_uint8.log` |
| Training curves/metrics | `results/checkpoints_0904_10ch_scaled_uint8/metrics/` |
| Evaluation JSON / CSV | `results/stitched_metrics/0904_10ch_scaled_uint8/stitched_metrics.{json,csv}` |
| Confusion matrix + heatmap | `results/stitched_metrics/0904_10ch_scaled_uint8/confusion_matrix.csv` / `_heatmap.png` |
| Per-plot prediction images | `results/stitched_metrics/0904_10ch_scaled_uint8/visualizations/` (10 PNGs) |
| Dataset | `data/new_0904_10ch/` |
| Annotations | `data/annotations/0904/` |

### Reproducing the evaluation

```bash
python scripts/evaluate_stitched_plots.py \
    --checkpoint results/checkpoints_0904_10ch_scaled_uint8/maskrcnn_best.pth \
    --dataset_dir data/new_0904_10ch/dataset_sliced_800_test \
    --cropped_plots_dir data/new_0904_10ch/cropped_plots/test \
    --plot_masks_dir data/new_0904_10ch/test_plot_masks \
    --input_preprocessing scaled_uint8 \
    --visualization_style boxes \
    --output_dir results/stitched_metrics/0904_10ch_scaled_uint8
```

`--input_preprocessing scaled_uint8` is **required and must match how the model was
trained** — normalization settings are not stored inside the checkpoint, so passing
the wrong mode produces silently wrong results rather than an error.
