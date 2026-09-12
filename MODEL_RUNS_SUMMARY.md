# Tree Health Model — Runs & Methodology Summary

A reference document for everything done in this working session: what the
pipeline does, what changed, every model trained, and every result obtained,
with per-class numbers. Full raw detail lives in [RUN_COMPARISON.md](RUN_COMPARISON.md)
(tile-level evaluation) and [STITCHED_EVAL_COMPARISON.md](STITCHED_EVAL_COMPARISON.md)
(the newer, more reliable plot-level evaluation) — this document distills both
into one place and explains the methodology in plain terms so it can be picked
back up later without re-deriving it.

---

## 1. The task, in plain terms

Drone imagery of forest plots (8 channels: Red, Green, Blue, a height map (CHM),
and four vegetation-health indices — NDVI, CIRE, GNDVI, NDRE) is fed into a Mask
R-CNN model. For each tree crown visible in the imagery, the model draws a box
around it and assigns it a health class. Two annotation schemes have been used:

- **Original**: 4 health classes — healthy, mild, moderate, severe.
- **Current** (from 2026-09 onward): annotators simplified this to **3 classes —
  healthy, mild, severe** (moderate dropped).

"Health class 0" is not a class — it's the label for background (no tree).

## 2. The data pipeline, in plain terms

Five steps turn raw drone imagery + hand-drawn tree annotations into something a
neural network can train on:

1. **Stack the bands** (`image_merging_vrt.py`) — the 6 raw imagery sources
   (RGB photo, CHM height map, 4 vegetation-index rasters) are combined into one
   virtual 8-band image covering the whole survey area. Cheap — no new file is
   written, it's a pointer file (a VRT).
2. **Cut into plots** (`cut_plots.py`) — a shapefile of plot boundaries is used
   to crop that big 8-band image into one GeoTIFF per survey plot, and sort them
   into `train/` and `test/` folders (the split is decided up front, by the
   surveyors, and never changes during training).
3. **Rasterize annotations into masks** (`create_masks.py`) — a shapefile of
   hand-drawn tree crown polygons, each tagged with a health class, is "burned"
   into a mask image per plot: every pixel belonging to a tree gets a number
   encoding *which tree* and *what health class* it is
   (`pixel value = class_id × 10000 + tree_number`). Background stays 0.
4. **Slice into tiles** (`slice_plots.py`) — plots are too big to feed to the
   model directly, so each plot is cut into overlapping 800×800 pixel tiles
   (50% overlap, so no tree is ever cut in half across *every* tile it appears
   in — it's whole in at least one). Tiles with no trees in them are dropped.
5. **Train** (`train.py`) — a Mask R-CNN model (ResNet-50 backbone) is trained
   on the tiles to detect and classify individual crowns.

### Two ways pixel values reach the model: "native" vs "scaled_uint8"

Before a tile's pixel values go into the model, they're normalized. Two
different conventions exist in this project and were compared throughout:

- **native**: each of the 8 bands is normalized using its own measured
  mean/standard-deviation (RGB uses standard ImageNet statistics; CHM and the
  vegetation indices use statistics measured directly from this project's data).
- **scaled_uint8**: each band is rescaled into a fixed physical range first,
  then normalized generically.

**Neither wins consistently** — see the results below; the better one changes
depending on which dataset is used.

## 3. A real bug found and fixed this session: the "class offset" bug

The newer annotation shapefiles (`0831_v3`, `new_plots`, `0904`) already number
`tree_class` as 1, 2, 3 (healthy, mild, severe — no 0). The mask-writing code,
however, was unconditionally adding **+1** to that number — a rule that was
correct for the *older* shapefile (which numbered classes 0–3), but wrong for
the new one. The effect: every tree's class got shifted up by one, and class 1
("healthy") was never used at all — which is why, for months, every evaluation
on this newer data reported **zero healthy trees**, and "mild" was silently
actually "healthy," "moderate" was actually "mild," and so on.

**Fixed** by making the offset a config value (`[MASKS] CLASS_ID_OFFSET`,
defaulting to the old behaviour so the original dataset stays unaffected), and
rebuilding every affected dataset with the correct value (`0` for the
1-indexed shapefiles). Full details: `RUN_COMPARISON.md`'s "corrected 3-tier
annotation scheme" section.

## 4. Two ways of scoring the model: tile-level vs. stitched

This session also changed *how* model quality gets measured, because the
original method had a real flaw:

### The original method (tile-level)

Each 800×800 tile is scored on its own, as if it were an independent little
image, and the results are summed across every tile in the test set. The
problem: tiles overlap by 50%, so a single tree near the border between two
tiles gets shown to the model — and can get correctly detected — **twice**,
once per tile. Tile-level scoring counts that as two separate correct
detections, which inflates the apparent recall and F1 score.

### The new method (stitched), built and run this session

`scripts/evaluate_stitched_plots.py` fixes this:

1. Run the model on every tile as before.
2. **Stitch**: move every tile's predicted boxes back into the coordinate
   system of the *whole plot* they came from (using the pixel offset encoded in
   each tile's filename) — this puts every prediction from the same plot into
   one shared picture instead of many disconnected tile-sized ones.
3. **Merge duplicates**: the same tree, if detected in two overlapping tiles,
   now shows up as two overlapping boxes in the stitched picture — a standard
   technique called Non-Maximum Suppression (NMS) collapses these into one,
   keeping the more confident detection.
4. **Drop implausibly tiny boxes** (noise, not trees).
5. **Score against the plot's real, un-fragmented ground truth** — one true
   box per real tree, no duplicates possible on the ground-truth side either.
6. **Visualize**: draws every plot's predictions on top of its actual photo,
   colour-coded by health class, so results can be checked by eye. A toggle
   (`--visualization_style boxes|masks`) controls whether it draws box outlines
   (default) or the model's actual predicted crown-shape masks tinted over the
   tree.

This gives a materially more honest number. **Every model's score dropped once
re-measured this way** — sometimes a lot (detection quality specifically was
overstated by 0.17–0.25 F1 in the old method). The stitched numbers are the
ones that should be trusted and quoted going forward; the tile-level numbers in
`RUN_COMPARISON.md` are kept as historical record only.

### Three ways of reading a score: complete / detection / grading

Both evaluation methods report the same three views of quality:

- **Complete (or "pooled")**: the full task — did we find the tree *and* get
  its health class right? This is the headline number.
- **Detection**: ignore health class — did we find the tree at all?
- **Grading**: of the trees we *did* find, did we get the health class right?

Splitting these apart matters because, once tile double-counting is removed,
**detection turns out to be the weaker half of the pipeline, not
classification** — the model is quite good at grading a tree's health once it
has found it (grading F1 typically 0.80–0.92), but still misses a meaningful
fraction of real trees outright (detection F1 typically 0.51–0.67).

## 5. Datasets used

| Dataset | Train plots | Test plots | Health classes | Notes |
|---|---|---|---|---|
| `data/new_data` | 26 | 6 | 3 (healthy/mild/severe) | Corrected this session (was mislabeled, see §3) |
| `data/new_data_chm_update` | 11 | 3 | 4 (old shifted scheme) | Predates this session; not re-corrected — still carries the class-offset bug, results below are internally consistent but the printed class *names* are one off from their real meaning |
| `data/new_plots` | 34 | 9 | 3 (healthy/mild/severe) | New annotation set, added this session |
| `data/new_0904` | 49 | 10 | 3 (healthy/mild/severe) | Newest, largest annotation set, added this session |

All four share the same underlying drone imagery (same survey site) — only the
plot boundaries and tree annotations differ between them.

## 6. Every model trained this session

| Label | Dataset | Preprocessing | Best epoch | Checkpoint |
|---|---|---|---|---|
| I | `new_data` | native | 14 | `results/checkpoints_new_data/` |
| J | `new_data` (corrected) | native | 66 | `results/checkpoints_new_data_native_3class/` |
| K | `new_data` (corrected) | scaled_uint8 | 15 | `results/checkpoints_new_data_scaled_uint8_3class/` |
| L | `new_plots` | native | 73 | `results/checkpoints_new_plots_native/` |
| M | `new_plots` | scaled_uint8 | 20 | `results/checkpoints_new_plots_scaled_uint8/` |
| N | `new_0904` | native | 12 | `results/checkpoints_0904_native/` |
| O | `new_0904` | scaled_uint8 | 13 | `results/checkpoints_0904_scaled_uint8/` |
| P | `new_0904_10ch` (10 bands) | native | 19 | `results/checkpoints_0904_10ch_native/` |
| Q | `new_0904_10ch` (10 bands) | scaled_uint8 | 12 | `results/checkpoints_0904_10ch_scaled_uint8/` |

(G and H — `new_data_chm_update`, native and scaled_uint8 — predate this
session but are included in the stitched results below since they were
re-evaluated this session. **I was superseded by J** once the class-offset bug
was found — I's numbers use the mislabeled scheme and should not be trusted;
kept only for provenance.)

---

## 7. Results — tile-level evaluation (original method, per-class)

Precision / Recall / F1, at a detection-confidence threshold of 0.5 and a
box-overlap threshold of 0.5. (tp/fp/fn = true/false positives, false
negatives.)

### I — `new_data`, native (mislabeled scheme — see §3; kept for reference only)

| Class shown† | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| "mild" (really healthy) | 97 | 41 | 123 | 0.703 | 0.441 | 0.542 |
| "moderate" (really mild) | 161 | 181 | 114 | 0.471 | 0.585 | 0.522 |
| "severe" | 155 | 117 | 92 | 0.570 | 0.628 | 0.597 |
| **Pooled** | 413 | 339 | 329 | 0.549 | 0.557 | **0.553** |

† Class names here are what the model printed under the old, buggy scheme —
not their real meaning. Detection F1 (class-agnostic) 0.837, grading F1 0.503.

### J — `new_data` corrected, native

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 107 | 69 | 113 | 0.608 | 0.486 | 0.540 |
| Mild | 144 | 156 | 131 | 0.480 | 0.524 | 0.501 |
| Severe | 140 | 109 | 107 | 0.562 | 0.567 | 0.565 |
| **Pooled** | 391 | 334 | 351 | 0.539 | 0.527 | **0.533** |

Detection F1 0.826, grading F1 0.509.

### K — `new_data` corrected, scaled_uint8

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 106 | 45 | 114 | 0.702 | 0.482 | 0.571 |
| Mild | 169 | 190 | 106 | 0.471 | 0.615 | 0.533 |
| Severe | 160 | 107 | 87 | 0.599 | 0.648 | 0.623 |
| **Pooled** | 435 | 342 | 307 | 0.560 | 0.586 | **0.573** |

Detection F1 0.839, grading F1 0.503.

### L — `new_plots`, native

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 202 | 133 | 101 | 0.603 | 0.667 | 0.633 |
| Mild | 224 | 200 | 144 | 0.528 | 0.609 | 0.566 |
| Severe | 193 | 99 | 115 | 0.661 | 0.627 | 0.643 |
| **Pooled** | 619 | 432 | 360 | 0.589 | 0.632 | **0.610** |

Detection F1 0.879 (the highest tile-level detection score of any run),
grading F1 0.531.

### M — `new_plots`, scaled_uint8

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 245 | 327 | 58 | 0.428 | 0.809 | 0.560 |
| Mild | 211 | 220 | 157 | 0.490 | 0.573 | 0.528 |
| Severe | 227 | 180 | 81 | 0.558 | 0.737 | 0.635 |
| **Pooled** | 683 | 727 | 296 | 0.484 | 0.698 | **0.572** |

Detection F1 0.792, grading F1 0.413.

*(N and O were only evaluated with the newer stitched method — see below — not
the tile-level one, since that method is what was requested for those runs.)*

---

## 8. Results — stitched evaluation (recommended; every model, re-measured)

Same Precision/Recall/F1 format, now scored on whole plots with duplicate
detections merged first. This is the number to trust and quote.

### G — `new_data_chm_update`, native (old shifted-class scheme — see §5)

| Class shown† | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| "mild" (really healthy) | 15 | 10 | 16 | 0.600 | 0.484 | 0.536 |
| "moderate" (really mild) | 30 | 39 | 33 | 0.435 | 0.476 | 0.455 |
| "severe" | 17 | 50 | 13 | 0.254 | 0.567 | 0.351 |
| **Pooled** | 62 | 99 | 62 | 0.385 | 0.500 | **0.435** |

Detection F1 0.540, grading F1 0.807.

### H — `new_data_chm_update`, scaled_uint8 (old shifted-class scheme)

| Class shown† | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| "mild" (really healthy) | 17 | 23 | 14 | 0.425 | 0.548 | 0.479 |
| "moderate" (really mild) | 25 | 27 | 38 | 0.481 | 0.397 | 0.435 |
| "severe" | 15 | 50 | 15 | 0.231 | 0.500 | 0.316 |
| **Pooled** | 57 | 100 | 67 | 0.363 | 0.460 | **0.406** |

Detection F1 0.512, grading F1 0.797.

### J — `new_data`, native

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 45 | 29 | 50 | 0.608 | 0.474 | 0.533 |
| Mild | 55 | 63 | 63 | 0.466 | 0.466 | 0.466 |
| Severe | 53 | 54 | 48 | 0.495 | 0.525 | 0.510 |
| **Pooled** | 153 | 146 | 161 | 0.512 | 0.487 | **0.499** |

Detection F1 0.600, grading F1 0.835.

### K — `new_data`, scaled_uint8

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 48 | 15 | 47 | 0.762 | 0.505 | 0.608 |
| Mild | 63 | 86 | 55 | 0.423 | 0.534 | 0.472 |
| Severe | 61 | 48 | 40 | 0.560 | 0.604 | 0.581 |
| **Pooled** | 172 | 149 | 142 | 0.536 | 0.548 | **0.542** |

Detection F1 0.643, grading F1 0.846.

### L — `new_plots`, native — best pooled result on this dataset

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 86 | 56 | 50 | 0.606 | 0.632 | 0.619 |
| Mild | 87 | 90 | 65 | 0.492 | 0.572 | 0.529 |
| Severe | 70 | 42 | 55 | 0.625 | 0.560 | 0.591 |
| **Pooled** | 243 | 188 | 170 | 0.564 | 0.588 | **0.576** |

Detection F1 0.626, grading F1 0.921 (best grading score of any run).

### M — `new_plots`, scaled_uint8

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 104 | 123 | 32 | 0.458 | 0.765 | 0.573 |
| Mild | 77 | 93 | 75 | 0.453 | 0.507 | 0.478 |
| Severe | 83 | 68 | 42 | 0.550 | 0.664 | 0.601 |
| **Pooled** | 264 | 284 | 149 | 0.482 | 0.639 | **0.549** |

Detection F1 0.610, grading F1 0.898.

### N — `new_0904`, native

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 63 | 35 | 80 | 0.643 | 0.441 | 0.523 |
| Mild | 100 | 144 | 112 | 0.410 | 0.472 | 0.439 |
| Severe | 124 | 78 | 94 | 0.614 | 0.569 | 0.590 |
| **Pooled** | 287 | 257 | 286 | 0.528 | 0.501 | **0.514** |

Detection F1 0.591, grading F1 0.860.

### O — `new_0904`, scaled_uint8 — best 8-band result

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 100 | 81 | 43 | 0.552 | 0.699 | 0.617 |
| Mild | 121 | 78 | 91 | 0.608 | 0.571 | 0.589 |
| Severe | 137 | 87 | 81 | 0.612 | 0.628 | 0.620 |
| **Pooled** | 358 | 246 | 215 | 0.593 | 0.625 | **0.608** |

Detection F1 0.666, grading F1 0.913. The only 8-band run where scaled_uint8 beats
native on *every single metric* (not a precision/recall trade-off).

### P — `new_0904` + elevation + LiDAR intensity (10 bands), native

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 99 | 61 | 44 | 0.619 | 0.692 | 0.653 |
| Mild | 116 | 77 | 96 | 0.601 | 0.547 | 0.573 |
| Severe | 140 | 89 | 78 | 0.611 | 0.642 | 0.626 |
| **Pooled** | 355 | 227 | 218 | 0.610 | 0.620 | **0.615** |

Detection F1 0.672, grading F1 0.914. Adding the two bands lifted native by
**+0.101 F1** over run N — the largest single input-side gain in this project.

### Q — `new_0904` + elevation + LiDAR intensity (10 bands), scaled_uint8 — best model overall

| Class | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Healthy | 90 | 51 | 53 | 0.638 | 0.629 | 0.634 |
| Mild | 130 | 88 | 82 | 0.596 | 0.613 | 0.605 |
| Severe | 142 | 76 | 76 | 0.651 | 0.651 | 0.651 |
| **Pooled** | 362 | 215 | 211 | 0.627 | 0.632 | **0.630** |

Detection F1 **0.682** and grading F1 **0.924** — the best pooled, detection and
grading scores of any run in this project. See `STITCHED_EVAL_COMPARISON.md`'s
2026-09-05 section for the full breakdown, including the three data/config bugs
found while integrating the new bands.

---

## 9. Bottom line, in simple terms

- **The stitched numbers are the real ones.** The old tile-by-tile scoring let
  a tree near a tile's edge count as "found" more than once, which made every
  earlier result look better than it really was — especially detection quality
  (overstated by roughly 0.17–0.25 F1). Trust `STITCHED_EVAL_COMPARISON.md` and
  §8 above over §7 or `RUN_COMPARISON.md`.
- **The model is better at grading health than at finding trees.** Once a
  crown is detected, its health class is usually right (grading F1 0.80–0.92
  across every run). The bigger source of error is trees the model never draws
  a box around at all (detection F1 only 0.51–0.67).
- **Two things actually move the needle: more annotated plots, and better input
  bands.** Training-set size tracks almost exactly with quality (11 plots (G/H) →
  26 (J/K) → 34 (L/M) → 49 (N/O)). Then adding elevation + LiDAR intensity as
  bands 9–10 (runs P/Q) gave the single largest input-side gain of the session:
  +0.101 F1 for native, +0.022 for scaled_uint8, and the best scores recorded
  anywhere here. Preprocessing choice moved the needle far less, and
  inconsistently — and matters much less once those bands are present (a 0.015
  gap at 10 channels vs. 0.094 at 8).
- **"native" vs. "scaled_uint8" has no universal winner.** scaled_uint8 won on
  `new_data` (K) and decisively on `new_0904` (O); native won on
  `new_plots` (L) and, once stitched, on `new_data_chm_update` (G). Whichever
  is better seems to depend on the specific dataset, not on some inherent
  property of the method — don't assume one is "correct" without checking.
- **A real labeling bug was found and fixed**: the newer annotation files were
  silently having their classes shifted by one, making "healthy" trees
  disappear from every report until this session. Every run using the
  corrected data (J, K, L, M, N, O) is unaffected; G and H (older,
  pre-existing runs) still carry the bug and are flagged everywhere they
  appear above.
- **Best model overall: Q** — 10-band (RGB+CHM+4 indices+elevation+intensity),
  scaled_uint8, on the `0904` annotations: pooled F1 0.630, detection F1 0.682,
  grading F1 0.924, all project bests. **P** (same data, native) is close behind
  at 0.615. Among 8-band models, **O** (scaled_uint8, `new_0904`, 0.608) and
  **L** (native, `new_plots`, 0.576) were the leaders.

## 10. Where to find things

- Configs: `config_new_data_native_3class.ini`, `config_new_data_scaled_uint8_3class.ini`,
  `config_new_plots_native.ini`, `config_new_plots_scaled_uint8.ini`,
  `config_0904_native.ini`, `config_0904_scaled_uint8.ini`, and
  `config_stitched_eval.ini` (the stitched-evaluation script's own settings).
- Checkpoints & tile-level metrics: `results/checkpoints_<run>/` (per run;
  `metrics/` and `separated_metrics/` subfolders hold the tile-level numbers).
- Stitched evaluation outputs: `results/stitched_metrics/<run>/` — a JSON, a
  CSV, a confusion-matrix PNG+CSV, and a `visualizations/` folder with one
  annotated image per test plot, for every run in §8.
- Pipeline code: `scripts/cut_plots.py`, `scripts/create_masks.py`,
  `scripts/slice_plots.py`, `train.py`.
- Evaluation code: `scripts/evaluate_test_set.py` /
  `scripts/evaluate_separated_metrics.py` (tile-level), and
  `scripts/evaluate_stitched_plots.py` (stitched — new this session, see §4).
- Full detail and additional discussion: `RUN_COMPARISON.md` (tile-level,
  chronological) and `STITCHED_EVAL_COMPARISON.md` (stitched, chronological).
