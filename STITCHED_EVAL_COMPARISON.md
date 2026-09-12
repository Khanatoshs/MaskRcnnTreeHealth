# Stitched-Plot Evaluation

Every metric reported in `RUN_COMPARISON.md` scores each 800px test *tile*
independently. `slice_plots.py` cuts tiles at 50% overlap, so a tree crown near a
tile seam is frequently detected — and counted — more than once across the tiles
that contain it. This document uses a new evaluation path, `scripts/evaluate_stitched_plots.py`,
that stitches tile-level predictions back into whole plots, deduplicates them, and
scores the result against the plot's true (unfragmented) ground truth instead.

## Methodology

For each checkpoint and its held-out test set:

1. **Run inference on every test tile.**
2. **Stitch**: each tile's raw predictions are transformed into the source plot's
   pixel coordinates using the y/x offset encoded in the tile filename
   (`{plot}_tile_y{y}_x{x}.tif`), so every prediction that came from the same plot
   lands in one shared coordinate frame — without touching the imagery at all.
3. **Merge duplicates**: class-agnostic NMS (`torchvision.ops.nms`, IoU threshold
   configurable via `NMS_IOU_THRESHOLD`, default 0.3) over each plot's stitched
   detections. A duplicate detection of the same tree from two overlapping tiles
   is just two overlapping boxes once stitched, so ordinary NMS merges them (keeps
   the higher-scoring one) exactly like it merges any other overlap. A genuine
   single detection near a plot's own outer edge has no duplicate to compete with,
   so it survives NMS untouched — no special "don't merge across the plot border"
   logic was needed; it falls out of using absolute plot coordinates.
4. **Filter tiny boxes**: anything smaller than `MIN_CROWN_SIDE_PX` (default 20px,
   sqrt of box area) is dropped as implausible for a tree crown at this GSD.
5. **Score against plot-level ground truth**, read directly from the mask
   `create_masks.py` wrote for that plot — one instance per tree, never
   tile-fragmented, so there is nothing to stitch on the GT side. Matching is
   done once per plot (never across plots), so pooled counts can never match a
   prediction from one plot against a different plot's ground truth.
6. **Report**: pooled per-class precision/recall/F1, a confusion matrix (CSV +
   heatmap PNG), a detection-vs-grading breakdown (same methodology as
   `evaluate_separated_metrics.py`), a class-agnostic AP/AR sweep (pycocotools),
   and a score-threshold sweep — written to JSON and CSV.
7. **Visualize**: one PNG per plot, the plot's RGB (read directly from
   `cut_plots.py`'s already-existing unsliced output — re-mosaicking it from
   overlapping tiles would need blending logic for an identical result) with GT
   and kept predictions drawn on top, in the same colour scheme and match-status
   line styles as `scripts/visualize_predictions.py` (solid = correct, dash-dot =
   wrong class, dashed = unmatched).

New config: [config_stitched_eval.ini](config_stitched_eval.ini) (score/NMS/size
thresholds, output root). Checkpoint/dataset/plot paths are per-run and passed on
the command line — see [run_all_stitched_eval.sh](run_all_stitched_eval.sh), which
runs every checkpoint below.

**A bug this surfaced and fixed along the way**: `train.py`'s
`compute_detection_metrics`, `compute_confusion_matrix`, and
`compute_grading_metrics_on_detections` match every sample in a list against
*every other sample's* ground truth when given more than one sample at once (they
flatten all boxes across the whole list before matching, rather than matching
per-sample). `evaluate_separated_metrics.py` calls them with the full multi-tile
list, so a prediction from one tile could spuriously match a same-region ground
truth box from a *different, unrelated* tile purely by coincidence of tile-local
pixel coordinates. This script avoids the bug entirely by calling those functions
once per plot (a single-sample list, which is always safe) and pooling the
results by hand; `evaluate_test_set.py`'s own `pooled_prf`/`confusion`/
`class_agnostic_ap` were already per-sample-safe and are reused directly (one
"sample" is a plot here instead of a tile). The bug itself was **not** fixed in
`train.py` and **not** re-verified against every prior `evaluate_separated_metrics.py`
run in `RUN_COMPARISON.md` — that would be a substantial separate effort; flagged
here as a known issue affecting the detection/grading numbers reported there.

## Runs

| Run | Checkpoint | Preprocessing | Dataset | Test plots | Classes |
|---|---|---|---|---|---|
| G | `results/checkpoints_chm_update_native/` | native | `data/new_data_chm_update` | 3 | 4 (bg+4, old shifted scheme*) |
| H | `results/checkpoints_chm_update_scaled_uint8/` | scaled_uint8 | `data/new_data_chm_update` | 3 | 4 (bg+4, old shifted scheme*) |
| J | `results/checkpoints_new_data_native_3class/` | native | `data/new_data` | 6 | 3 (bg+3: healthy/mild/severe) |
| K | `results/checkpoints_new_data_scaled_uint8_3class/` | scaled_uint8 | `data/new_data` | 6 | 3 (bg+3) |
| L | `results/checkpoints_new_plots_native/` | native | `data/new_plots` | 9 | 3 (bg+3) |
| M | `results/checkpoints_new_plots_scaled_uint8/` | scaled_uint8 | `data/new_plots` | 9 | 3 (bg+3) |

\* G/H's ground truth (`data/new_data_chm_update`) still carries the class-offset
bug documented in `RUN_COMPARISON.md`'s "corrected 3-tier annotation scheme"
section — displayed class names (mild/moderate/severe) are shifted one class from
their real meaning (healthy/mild/severe). Not rebuilt in this pass; results below
are internally self-consistent (same shift on both prediction and GT sides) but
should not be read by their printed class names.

**F (`results/checkpoints_scaled_uint8/`) and I (`results/checkpoints_new_data/`)
were skipped** — not possible to evaluate validly. `data/new_data`'s ground-truth
masks were rebuilt with the corrected class encoding for the later J/K run, so the
on-disk test masks (classes 1–3) no longer match F/I's learned class semantics
(trained under the old offset-shifted scheme, predicting classes 1–4 with class 1
always empty). J/K are F/I's direct, corrected replacements on the same dataset,
so no comparison is actually missing.

## Pooled test-set results (stitched, score≥0.5, IoU≥0.5)

| Run | Plots | Tiles | GT | Raw dets | Kept (post-NMS+size) | tp | fp | fn | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G (chm native) | 3 | 48 | 124 | 2245 | 351 | 62 | 99 | 62 | 0.385 | 0.500 | 0.435 | 0.335 |
| H (chm scaled_uint8) | 3 | 48 | 124 | 1340 | 289 | 57 | 100 | 67 | 0.363 | 0.460 | 0.406 | 0.307 |
| J (new_data native) | 6 | 96 | 314 | 1041 | 357 | 153 | 146 | 161 | 0.512 | 0.487 | 0.499 | 0.503 |
| K (new_data scaled_uint8) | 6 | 96 | 314 | 3293 | 739 | 172 | 149 | 142 | 0.536 | 0.548 | **0.542** | 0.554 |
| L (new_plots native) | 9 | 144 | 413 | 1331 | 481 | 243 | 188 | 170 | 0.564 | 0.588 | **0.576** | 0.579 |
| M (new_plots scaled_uint8) | 9 | 144 | 413 | 4187 | 1002 | 264 | 284 | 149 | 0.482 | 0.639 | 0.549 | 0.551 |

"Raw dets" is every detection the model produced across all tiles before any
merging; "Kept" is what survives NMS + the size filter, at *any* confidence (the
score threshold is applied when computing the metrics columns, not before). The
gap between raw and kept is almost entirely tile-seam duplicates being merged —
K and M in particular produce far more raw detections than G/H/J/L, and lose
proportionally more of them to NMS.

### Stitched vs. tile-level: the correction this analysis makes

| Run | Tile-level F1 (RUN_COMPARISON.md) | Stitched F1 | Δ | Tile-level detection F1 | Stitched detection F1 | Δ |
|---|---|---|---|---|---|---|
| G | 0.444 | 0.435 | −0.009 | 0.710 | 0.540 | **−0.170** |
| H | 0.482 | 0.406 | **−0.076** | 0.733 | 0.512 | **−0.221** |
| J | 0.533 | 0.499 | −0.034 | 0.826 | 0.600 | **−0.226** |
| K | 0.573 | 0.542 | −0.031 | 0.839 | 0.643 | **−0.196** |
| L | 0.610 | 0.576 | −0.034 | 0.879 | 0.626 | **−0.253** |
| M | 0.572 | 0.549 | −0.023 | 0.792 | 0.610 | **−0.182** |

**Every run's F1 drops once tile overlap is removed, and detection F1 drops far
more than complete-task F1 does (−0.17 to −0.25).** This is the double-counting
effect the tile-level numbers never controlled for: a tree near a tile seam
detected in two overlapping tiles was scored as two independent true positives
(one per tile-as-sample) rather than one. Stitching plus NMS collapses those
duplicates into a single detection matched against a single ground-truth crown,
so the "free" true positives disappear and — for models that over-detect (K, M)
— the surviving duplicates that don't get matched become false positives instead.
**Detection quality specifically was the most inflated statistic in every prior
report in this project**; complete-task F1 was inflated too, but less severely,
because grading (classification) errors don't benefit from the same duplication
effect.

**G vs. H reverses under stitching**: at tile level H (scaled_uint8) beat G
(native) on every metric (0.482 vs. 0.444 F1). Stitched, G beats H (0.435 vs.
0.406) — H had more raw detections (a noisier detector) whose tile-level "extra"
true positives from duplicate detections evaporate once deduplicated, exposing
worse real precision. J-vs-K and L-vs-M keep the same ranking as before
(scaled_uint8 wins new_data, native wins new_plots), though the margins shrink.

### Per-class F1 (stitched, pooled)

| Run | class 1 | class 2 | class 3 | class 4 |
|---|---|---|---|---|
| G | n/a (no GT†) | 0.536 (15/10/16) | 0.455 (30/39/33) | 0.351 (17/50/13) |
| H | n/a (no GT†) | 0.479 (17/23/14) | 0.435 (25/27/38) | 0.316 (15/50/15) |
| J (healthy/mild/severe) | 0.533 (45/29/50) | 0.466 (55/63/63) | 0.510 (53/54/48) | — |
| K | 0.608 (48/15/47) | 0.472 (63/86/55) | 0.581 (61/48/40) | — |
| L | 0.619 (86/56/50) | 0.529 (87/90/65) | 0.591 (70/42/55) | — |
| M | 0.573 (104/123/32) | 0.478 (77/93/75) | 0.601 (83/68/42) | — |

(tp/fp/fn in parentheses.) † G/H's class 1 has no ground truth for the same
class-offset-bug reason documented throughout `RUN_COMPARISON.md`.

## Detection vs grading breakdown (stitched)

| Run | Detection P | Detection R | Detection F1 | Grading macro P | Grading macro R | Grading macro F1 |
|---|---|---|---|---|---|---|
| G | 0.478 | 0.621 | 0.540 | 0.806 | 0.824 | 0.807 |
| H | 0.459 | 0.581 | 0.512 | 0.788 | 0.829 | 0.797 |
| J | 0.615 | 0.586 | 0.600 | 0.840 | 0.835 | 0.835 |
| K | 0.636 | 0.650 | 0.643 | 0.855 | 0.847 | 0.846 |
| L | 0.613 | 0.639 | 0.626 | 0.929 | 0.918 | 0.921 |
| M | 0.535 | 0.709 | 0.610 | 0.901 | 0.898 | 0.898 |

Grading (classification accuracy, given a matched detection) holds up much
better under stitching than detection does — grading macro F1 is 0.80–0.92 across
every run here, well above every detection F1 (0.51–0.64). This reframes a
takeaway from `RUN_COMPARISON.md`: earlier tile-level numbers made it look like
grading was the dominant source of end-to-end error (detection F1 was 0.79–0.88,
grading only 0.41–0.53 there). Once tile-overlap double-counting is removed,
**detection is actually the weaker half of the pipeline, not grading** — models
here are quite good at getting the health class right once they've found a tree,
they just don't find/deduplicate trees as reliably as the tile-level numbers
suggested.

### Class-agnostic AP / AR (stitched)

| Run | AP | AP50 | AP75 | AR100 |
|---|---|---|---|---|
| G | 0.142 | 0.393 | 0.065 | 0.311 |
| H | 0.153 | 0.390 | 0.065 | 0.316 |
| J | 0.191 | 0.494 | 0.113 | 0.313 |
| K | 0.215 | 0.530 | 0.135 | 0.376 |
| L | 0.211 | 0.504 | 0.136 | 0.342 |
| M | 0.246 | 0.597 | 0.152 | 0.380 |

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| G | 0.50 | 0.435 (0.435) |
| H | 0.70 | 0.442 (0.406) |
| J | 0.90 | 0.503 (0.499) |
| K | 0.50 | 0.542 (0.542) |
| L | 0.70 | 0.582 (0.576) |
| M | 0.70 | 0.574 (0.549) |

## Outputs

Per run, under `results/stitched_metrics/<run_name>/`:
- `stitched_metrics.json` — full results (pooled, per-class, macro, AP, threshold sweep, detection, grading, per-plot counts)
- `stitched_metrics.csv` — flat CSV of the same
- `confusion_matrix.csv` + `confusion_matrix_heatmap.png`
- `visualizations/<plot_id>.png` — one per test plot (36 total across all 6 runs)

## Takeaways

- **Tile-level evaluation (everywhere else in this project) systematically
  overstates model quality**, mostly through detection, because 50%-overlap
  tiles let the same tree score as a true positive more than once. The stitched
  numbers here are the more honest estimate of real-world, whole-plot
  performance; treat every F1/AP/detection number in `RUN_COMPARISON.md` as an
  optimistic upper bound rather than the number to report externally.
  Complete-task F1 drops by 0.02–0.08 and detection F1 by 0.17–0.25 once
  stitched.
- **Grading (classification), not detection, is the stronger half of the
  pipeline** once double-counting is removed — the opposite framing from every
  earlier detection-vs-grading writeup in `RUN_COMPARISON.md`.
- **K (scaled_uint8) and L (native) remain the two best-performing checkpoints**
  in this project by stitched F1 (0.542 and 0.576) and by stitched detection F1
  (0.643 and 0.626) — same overall leaders as the tile-level analysis, just with
  smaller margins.
- G-vs-H is the one ranking that flips: H looked better at tile level, G is
  better stitched. Any preprocessing-mode conclusion drawn from tile-level H
  should be treated as unreliable specifically for the CHM-update pair.
- Next step, if pursued: the `train.py` cross-sample matching bug flagged above
  affects every `evaluate_separated_metrics.py` number in `RUN_COMPARISON.md`
  (not just G/H/J/K/L/M) — worth fixing centrally and re-running rather than
  patching around it per-script, if those numbers are going to be relied on.

---

# 2026-09-04 update: 0904 annotation set (runs N, O) + mask visualization toggle

A new annotation pass arrived: `data/annotations/0904/{Plot_59_0904,Annotation0904}.shp`
— 59 plots (49 train / 10 test, up from new_plots' 43), same site as `data/new_data`/
`data/new_plots` (same VRT reused, no new raster capture), same 3-tier `tree_class`
scheme (743 healthy / 1060 mild / 1134 severe polygons). Ran the full pipeline
(`cut_plots.py` → `create_masks.py` → `slice_plots.py`, `CLASS_ID_OFFSET=0`) into
`data/new_0904/` and trained both preprocessing modes into
`results/checkpoints_0904_{native,scaled_uint8}/`, evaluated with
`scripts/evaluate_stitched_plots.py` exactly as J–M were.

## `evaluate_stitched_plots.py` gained a visualization toggle

`--visualization_style {boxes,masks}` (config: `VISUALIZATION_STYLE`). `boxes`
(default, unchanged) draws box outlines only. `masks` additionally tints each
*kept prediction's* instance mask over the crown, cropped to its own box extent
during inference (cheap regardless of raw detection count - a box-sized patch,
not a full-tile mask - and only fetched from the model at all when this style is
requested, so the default box-only path pays no extra cost). Placement uses
matplotlib's `imshow(..., extent=...)`, which scales each patch to the display
resolution automatically - no manual resizing needed. Ground truth is always
box-only (there is no predicted mask to show it). Verified on run G/`plot_1`: the
mask overlay renders correctly and pooled/detection/grading metrics are bit-for-bit
identical to the box-only run on the same checkpoint (masks affect visualization
only, never scoring). **Runs N and O below use the default `boxes` style**, per
this update's request; `masks` is available for future runs.

## Configuration

| Run | Checkpoint | Preprocessing | Dataset | Test plots | Best epoch |
|---|---|---|---|---|---|
| N | `results/checkpoints_0904_native/` | native | `data/new_0904` | 10 | 12 |
| O | `results/checkpoints_0904_scaled_uint8/` | scaled_uint8 | `data/new_0904` | 10 | 13 |

Configs: [config_0904_native.ini](config_0904_native.ini) /
[config_0904_scaled_uint8.ini](config_0904_scaled_uint8.ini). Same anchors as
every other new_data/new_plots-family run (`sizes=112,144,184,216,288`,
`ratios=0.62,1.0,1.36` — `compute_anchor_sizes.py` measured p50=177px/p99=405px
here, essentially identical to new_plots' p50=178px/p99=407px, so the same
anchors were reused rather than the script's stride-respecting default, per the
documented ablation earlier in this project). N's `image_mean`/`image_std` for
the extra 5 bands: `11.1844,0.3872,0.2343,0.3713,0.1000` /
`8.4282,0.2564,0.1610,0.2364,0.0670` (CHM/NDVI/CIRE/GNDVI/NDRE), measured on 620
training tiles; O uses the standard 0.5/0.5 scaled_uint8 placeholder.

## Pooled test-set results (stitched, score≥0.5, IoU≥0.5, 10 plots / 160 tiles / 573 GT)

| Run | Raw dets | Kept (post-NMS+size) | tp | fp | fn | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| N (native) | 6289 | 1224 | 287 | 257 | 286 | 0.528 | 0.501 | 0.514 | 0.517 |
| O (scaled_uint8) | 6270 | 1284 | 358 | 246 | 215 | 0.593 | 0.625 | **0.608** | 0.609 |

**O (scaled_uint8) clearly beats N (native) here** — F1 0.608 vs. 0.514, every
per-class F1 higher, and both higher precision *and* higher recall (not a
precision/recall trade-off this time, an unambiguous win). This is the same
direction as K-vs-J (scaled_uint8 won `data/new_data`) and the opposite of
L-vs-M (native won `data/new_plots`) — so across the four "clean" (corrected
3-tier) preprocessing comparisons run in this project (G/H excluded as
shift-scheme-caveated), scaled_uint8 now leads 3 of 4 (H, K, O) to native's 1
(L), though G/H's own comparison went the other way once stitched. Preprocessing
mode still does not have a project-wide winner - see this file's earlier
takeaways.

### Per-class F1 (stitched, pooled)

| Run | healthy | mild | severe |
|---|---|---|---|
| N | 0.523 (63/35/80) | 0.439 (100/144/112) | 0.590 (124/78/94) |
| O | 0.617 (100/81/43) | 0.589 (121/78/91) | 0.620 (137/87/81) |

(tp/fp/fn in parentheses.) O's biggest edge is `healthy` (+0.094) and `mild`
(+0.150) — N's `mild` precision (0.410) is dragged down by 144 false positives,
nearly 1.5x its true positives.

## Detection vs grading breakdown (stitched)

| Run | Detection P | Detection R | Detection F1 | Grading macro P | Grading macro R | Grading macro F1 |
|---|---|---|---|---|---|---|
| N | 0.607 | 0.576 | 0.591 | 0.898 | 0.859 | 0.860 |
| O | 0.649 | 0.684 | **0.666** | 0.912 | 0.915 | **0.913** |

Same pattern as every other stitched run: grading (0.86–0.91) is far stronger
than detection (0.59–0.67). O wins at both levels, not just one.

### Class-agnostic AP / AR

| Run | AP | AP50 | AP75 | AR100 |
|---|---|---|---|---|
| N | 0.176 | 0.494 | 0.088 | 0.327 |
| O | 0.266 | 0.627 | 0.187 | 0.385 |

### Confusion matrix (rows=true, cols=predicted; 0=background)

N (native):

| true \ pred | bg | healthy | mild | severe |
|---|---|---|---|---|
| bg | — | 35 | 103 | 76 |
| healthy | 49 | **63** | 31 | 0 |
| mild | 110 | 0 | **100** | 2 |
| severe | 84 | 0 | 10 | **124** |

O (scaled_uint8):

| true \ pred | bg | healthy | mild | severe |
|---|---|---|---|---|
| bg | — | 69 | 59 | 84 |
| healthy | 36 | **100** | 7 | 0 |
| mild | 76 | 12 | **121** | 3 |
| severe | 69 | 0 | 12 | **137** |

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| N | 0.50 | 0.514 (0.514) |
| O | 0.60 | 0.621 (0.608) |

## Takeaways

- **O (scaled_uint8) is an unambiguous win on the largest dataset in this
  project** (49 train plots) — unlike every earlier preprocessing comparison
  here, this one isn't a precision/recall trade-off; O beats N on precision,
  recall, and every per-class and detection/grading metric simultaneously.
- Detection remains the weaker half of the pipeline for both N and O (F1
  0.59–0.67) versus grading (0.86–0.91), consistent with every run in this
  document once stitched.
- N and O's raw detection counts (6289, 6270) are far higher than L/M's
  (1331, 4187) despite N/O's dataset being only ~1.4x more train plots than
  L/M's (49 vs. 34/26) - worth a closer look if raw-detection volume itself
  becomes a concern (inference cost, NMS cost), though it doesn't appear to be
  hurting O's final metrics here.
- The mask-visualization toggle is available for future runs (`--visualization_style masks`)
  but was not used for N/O per this update's request; `boxes` was used, matching
  every prior run in this document.

---

# 2026-09-05 update: adding elevation + LiDAR intensity (runs P, Q) — best results in the project

Two more source rasters were supplied as ESRI ArcInfo binary grids and appended to
the stack as bands 9 and 10, taking the 0904 dataset from 8 to 10 channels:

- **band 9 — EL**: Byte-quantised elevation grid (values 11–241 over these plots,
  nodata 255).
- **band 10 — INTENSITY**: LiDAR return intensity (Int16, 0–158 here, nodata −32768).

Everything else is held fixed against runs N/O: same `0904` annotations (59 plots,
49 train / 10 test), same anchors, same schedule, same 3-tier classes. **P and Q are
therefore a clean measurement of what those two bands are worth.**

## Getting there: three problems worth recording

1. **A silently truncated source grid.** The elevation grid arrived as a partial
   copy — missing its `z001001x.adf` block index and the `z001002*`/`z001003*`
   files. An ArcInfo grid like that *opens* perfectly (the header is intact, so
   `gdalinfo` reports the full 45660×33247 raster) and only fails when a read lands
   on an unindexed block: 11 of 59 plots, including 3 of the 10 held-out test
   plots. Caught before any training, and now guarded by a new pre-flight script,
   `scripts/check_extra_bands.py`, which reads every extra band over every plot
   footprint and reports unreadable plots, value range, nodata count, and any
   values >255. Run it before any rebuild that adds bands.
2. **A stale RGB path.** `[MULTICHANNEL] rgb_img_path` still pointed at
   `data/tiff/rgb_new2.tiff` after that file had been moved into `data/tiff/new/`.
   This had gone unnoticed because every run since the move reused an
   already-built VRT rather than rebuilding one — the same "stale non-`[TRAIN]`
   config section" pattern noted earlier in this project. The old
   `results/stacked_8ch_out_new_data.vrt` still carries the dead path; harmless,
   since the plots it produced were materialised while it was valid.
3. **Two preprocessing landmines the new bands would have hit.**
   `DEFAULT_PREPROCESS_RANGES` in `dataset/multi_channel_dataset.py` only covered
   channels 0–7, and unlisted channels fall back to `(0.0, 1.0)` — which would have
   clipped elevation (11–241) and intensity (0–158) to a **constant 1.0**, silently
   deleting both new bands in `scaled_uint8` mode. Ranges added. Separately, the
   RGB 0-255→0-1 rescale in `native` mode was gated on the max across *all* bands,
   so any band exceeding 255 (Int16 intensity can) would have flipped that test off
   and left RGB unscaled while `image_mean`/`image_std` expected 0–1 — a per-tile,
   data-dependent mis-normalization raising no error. Now gated on the RGB bands
   themselves; verified identical behaviour on the existing 8-band data.

`scripts/image_merging_vrt.py` gained `[MULTICHANNEL] extra_band_paths`, a
comma-separated list appended as bands 9, 10, … *after* the eight fixed bands, so
band 4 stays CHM (which `[MASKS] CHM_BAND` and the mean/std ordering depend on).
Absent or blank, it builds the usual 8-band stack, so existing configs are
unaffected.

**Comparability check**: the rebuilt 10-band stack measured bands 1–8 within 0.002
of the 8-band 0904 build (e.g. CHM mean 11.1860 vs 11.1844), and produced identical
plot/tile/instance counts (49/10 plots, 780/160 tiles, 573 test instances). Tile
geometry shifts by at most one pixel — the 10-band VRT is 2 px wider because its
extent is the union including the new grids — with mean per-pixel difference ~4/255
(sub-pixel resampling, not misalignment). Zero-fill from the VRT edge inside the new
bands is negligible (0.10% EL, 0.14% INTENSITY).

## Configuration

| Run | Checkpoint | Channels | Preprocessing | Best epoch / trained |
|---|---|---|---|---|
| N | `results/checkpoints_0904_native/` | 8 | native | 12 / 42 |
| O | `results/checkpoints_0904_scaled_uint8/` | 8 | scaled_uint8 | 13 / 42 |
| P | `results/checkpoints_0904_10ch_native/` | **10** | native | 19 / 49 |
| Q | `results/checkpoints_0904_10ch_scaled_uint8/` | **10** | scaled_uint8 | 12 / 42 |

Configs: [config_0904_10ch_native.ini](config_0904_10ch_native.ini) /
[config_0904_10ch_scaled_uint8.ini](config_0904_10ch_scaled_uint8.ini); dataset at
`data/new_0904_10ch/`. P's measured `image_mean`/`image_std` add
`104.4511 / 58.3868` (EL) and `34.8472 / 21.8508` (INTENSITY) to the eight existing
values; Q uses the 0.5/0.5 placeholder convention for all seven non-RGB bands.

## Pooled test-set results (stitched, score≥0.5, IoU≥0.5, 10 plots / 160 tiles / 573 GT)

| Run | Channels | tp | fp | fn | Precision | Recall | F1 | Macro F1 | AP50 |
|---|---|---|---|---|---|---|---|---|---|
| N (native, 8ch) | 8 | 287 | 257 | 286 | 0.528 | 0.501 | 0.514 | 0.517 | 0.494 |
| O (scaled_uint8, 8ch) | 8 | 358 | 246 | 215 | 0.593 | 0.625 | 0.608 | 0.609 | 0.627 |
| P (native, 10ch) | 10 | 355 | 227 | 218 | 0.610 | 0.620 | 0.615 | 0.618 | 0.626 |
| Q (scaled_uint8, 10ch) | 10 | 362 | 215 | 211 | 0.627 | 0.632 | **0.630** | 0.630 | 0.634 |

**Both 10-band runs beat both 8-band runs, and Q is the best model in this project
to date.** The two extra bands are worth **+0.101 F1 to native** (0.514 → 0.615) and
**+0.022 to scaled_uint8** (0.608 → 0.630). Native gains far more — it was the
weaker of the two at 8 channels and the extra structural information closes most of
that gap, leaving the two preprocessing modes only 0.015 apart (vs. 0.094 before).

### Per-class F1 (stitched, pooled)

| Run | healthy | mild | severe |
|---|---|---|---|
| N (8ch native) | 0.523 (63/35/80) | 0.439 (100/144/112) | 0.590 (124/78/94) |
| O (8ch scaled_uint8) | 0.617 (100/81/43) | 0.589 (121/78/91) | 0.620 (137/87/81) |
| P (10ch native) | 0.653 (99/61/44) | 0.573 (116/77/96) | 0.626 (140/89/78) |
| Q (10ch scaled_uint8) | 0.634 (90/51/53) | **0.605** (130/88/82) | **0.651** (142/76/76) |

(tp/fp/fn in parentheses.) P has the best `healthy` F1 (0.653); Q leads on `mild`
and `severe`. Every class improves over the 8-band equivalents — `mild`, which was
N's weakest class by far at 0.439, rises to 0.573 (P) / 0.605 (Q).

## Detection vs grading breakdown (stitched)

| Run | Detection P | Detection R | Detection F1 | Grading macro P | Grading macro R | Grading macro F1 |
|---|---|---|---|---|---|---|
| N (8ch native) | 0.607 | 0.576 | 0.591 | 0.898 | 0.859 | 0.860 |
| O (8ch scaled_uint8) | 0.649 | 0.684 | 0.666 | 0.912 | 0.915 | 0.913 |
| P (10ch native) | 0.667 | 0.677 | 0.672 | 0.912 | 0.919 | 0.914 |
| Q (10ch scaled_uint8) | 0.679 | 0.684 | **0.682** | 0.929 | 0.920 | **0.924** |

The gain shows up on **both** levels but is larger on detection, which is the
project's persistent bottleneck: P/Q reach 0.672/0.682 detection F1, the highest
stitched detection scores recorded here (previous best: O at 0.666, and L at 0.626
on `new_plots`). Grading also ticks up to 0.914/0.924 — Q's 0.924 is the best
grading score of any run. This is the expected shape of the result: elevation and
intensity carry structural/return information that helps *separate crowns from
background and from each other*, which is precisely the failure mode dominating the
error budget.

### Per-class grading (on detected trees only)

| Run | healthy | mild | severe |
|---|---|---|---|
| P | 0.925 (99/112/102) | 0.875 (116/130/135) | 0.943 (140/146/151) |
| Q | 0.928 (90/93/101) | 0.897 (130/150/140) | 0.947 (142/149/151) |

(correct/predicted/actual in parentheses.)

### Confusion matrices (rows=true, cols=predicted; 0=background)

P (10ch native):

| true \ pred | bg | healthy | mild | severe |
|---|---|---|---|---|
| bg | — | 48 | 63 | 83 |
| healthy | 41 | **99** | 3 | 0 |
| mild | 77 | 13 | **116** | 6 |
| severe | 67 | 0 | 11 | **140** |

Q (10ch scaled_uint8):

| true \ pred | bg | healthy | mild | severe |
|---|---|---|---|---|
| bg | — | 48 | 68 | 69 |
| healthy | 42 | **90** | 11 | 0 |
| mild | 72 | 3 | **130** | 7 |
| severe | 67 | 0 | 9 | **142** |

Off-diagonal class confusion stays small (3–13 trees per cell); the background
column still holds most of the loss (41–77 per class), reaffirming that missed
detections, not misgrading, are what cap these scores.

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| P | 0.70 | 0.627 (0.615) |
| Q | 0.60 | 0.634 (0.630) |

## Takeaways

- **Adding elevation + LiDAR intensity is the single most effective change made to
  the model inputs in this project.** +0.101 F1 for native, +0.022 for
  scaled_uint8, and the best pooled (0.630), detection (0.682) and grading (0.924)
  scores recorded anywhere in this document.
- **The benefit lands mostly on detection**, the known bottleneck — consistent with
  height/return-structure helping separate crowns from background and from each
  other in ways the purely spectral bands could not.
- **Preprocessing mode matters much less once these bands are present**: 0.630 vs
  0.615 (a 0.015 gap) versus 0.608 vs 0.514 (0.094) at 8 channels. scaled_uint8
  still edges ahead, making it 4 wins to 1 across this project's clean
  preprocessing comparisons.
- Missed detections remain the dominant error: even Q leaves 181 of 573 test trees
  undetected. More annotated plots, not more bands, is still the likeliest next
  lever — though this result shows input channels are not exhausted either.
- **Housekeeping**: `data/tiff/new/el/el` is still the truncated copy; the good one
  is `data/tiff/new/el/el (1)/el`. Deleting the broken one would remove a real trap
  for future runs, since the two differ only by a `(1)` in the path and the broken
  one fails only partway through a job.
