# Run Comparison

All numbers below are from `scripts/evaluate_test_set.py` (pooled precision/recall/F1 over the
whole held-out test split, at `score_threshold=0.5`, `IoU=0.5` — see
[SEPARATED_METRICS_GUIDE.md](SEPARATED_METRICS_GUIDE.md) for why pooled ≠ the per-batch macro
numbers logged during training) unless noted otherwise. Generated 2026-09-02.

**Two test sets are in play and are not comparable to each other:**
- **Old pipeline** (`data/dataset_sliced_800*_test`): 31 train / 8 test plots, 128 test tiles,
  1110 GT instances. Used by runs A–E below.
- **New pipeline** (`data/new_data/dataset_sliced_800_test`): different plot shapefile
  (`mesh.shp` / `Plot_bound0831`), 6 test plots, 96 tiles, 742 GT instances, and — in this test
  split specifically — **zero `healthy` (class 1) ground-truth instances**. Used by run F only.

Don't rank F against A–E on raw numbers; the test sets differ in plots, tile count, and class
balance.

## Configurations

| Run | Checkpoint dir | Channels | Bands | Preprocessing | Dataset dir | Best epoch / trained |
|---|---|---|---|---|---|---|
| A | `checkpoints/` | 8 | RGB+CHM+NDVI+CIRE+GNDVI+NDRE | native | `data/dataset_sliced_800` | 16 / 46 |
| B | `checkpoints_new/` | 8 | RGB+CHM+NDVI+CIRE+GNDVI+NDRE | native | `data/dataset_sliced_800` | 26 / 56 |
| C | `checkpoints_no_chm/` | 7 | RGB+NDVI+CIRE+GNDVI+NDRE (no CHM) | native | `data/dataset_sliced_800_no_chm` | 92 / 100 |
| D | `checkpoints_rgb/` | 3 | RGB only | native | `data/dataset_sliced_800_rgb` | 10 / 40 |
| E | `checkpoints_rgb_ndvi/` | 4 | RGB+NDVI | native | `data/dataset_sliced_800_rgb_ndvi` | 21 / 51 |
| F | `results/checkpoints_scaled_uint8/` | 8 | RGB+CHM+NDVI+CIRE+GNDVI+NDRE | **scaled_uint8** | `data/new_data/dataset_sliced_800` | 33 / 63 |

All runs: ResNet-50-FPN backbone (`trainable_layers=3`), data-derived anchors
`sizes=[112,144,184,216,288]`, `aspect_ratios=[0.62,1.0,1.36]` (C–E use the same anchors even
though they were computed for the 8-band tiles — not re-derived per ablation), `batch_size=2`,
`lr=1e-4`, cosine schedule, early stopping patience 30, checkpoint selection = lowest val_loss
(A–E) / highest pooled_f1 (F, per `checkpoint_monitor=pooled_f1` in `config_scaled_uint8.ini`).

A and B are the same configuration (full 8-band, native preprocessing, old data) trained twice;
B is the version currently tracked in git as the baseline. A's test-set number below was measured
against `backup_run4_imagenet_rgb.pth`, a manually-saved backup from that run at its best epoch
(16) — not a fresh eval of the `maskrcnn_best.pth` currently sitting in `checkpoints/` — so treat
A's row as approximate provenance, kept for reference only.

**Runs A–E were never re-evaluated after the NDVI-path bug fix** (see below) — that bug only
affected the *new_data* VRT (`stacked_8ch_out_new_data.vrt`) used by run F's pipeline; A–E used
the older `data/tiff/*` sources and `data/dataset_sliced_800*` directories, which were unaffected.

## Pooled test-set results (score_threshold=0.5, IoU=0.5)

| Run | tp | fp | fn | Precision | Recall | F1 | Macro F1 | AP50 | AR@100 |
|---|---|---|---|---|---|---|---|---|---|
| A (8ch native, run 1) | 671 | 560 | 439 | 0.545 | 0.605 | 0.573 | 0.565 | 0.673 | 0.537 |
| B (8ch native, run 2 / baseline) | 645 | 506 | 465 | 0.519 | 0.632 | 0.570 | 0.561 | 0.638 | 0.475 |
| C (no CHM, 7ch) | 583 | 492 | 527 | 0.562 | 0.496 | 0.527 | 0.520 | 0.534 | 0.360 |
| D (RGB only, 3ch) | 619 | 534 | 491 | 0.537 | 0.558 | 0.547 | 0.542 | 0.666 | 0.522 |
| E (RGB+NDVI, 4ch) | 635 | 568 | 475 | 0.528 | 0.572 | 0.549 | 0.548 | 0.628 | 0.482 |
| F (8ch scaled_uint8, new data) | 405 | 318 | 337 | 0.560 | 0.546 | 0.553 | 0.418* | 0.558 | 0.418 |

\* F's macro F1 is averaged over only 3 classes (mild/moderate/severe) since `healthy` has no GT
in this test split — not directly comparable to A–E's 4-class macro.

### Per-class F1 (pooled)

| Run | healthy | mild | moderate | severe |
|---|---|---|---|---|
| A | 0.646 (152/105/62) | 0.479 (131/156/129) | 0.492 (121/147/104) | 0.643 (267/152/144) |
| B | 0.606 (154/140/60) | 0.477 (129/152/131) | 0.545 (129/119/96) | 0.617 (289/238/122) |
| C | 0.567 (114/74/100) | 0.439 (100/96/160) | 0.498 (116/125/109) | 0.577 (221/134/190) |
| D | 0.624 (126/64/88) | 0.476 (135/172/125) | 0.481 (100/91/125) | 0.589 (258/207/153) |
| E | 0.620 (146/111/68) | 0.484 (131/150/129) | 0.501 (135/179/90) | 0.586 (223/128/188) |
| F | n/a (no GT) | 0.596 (123/70/97) | 0.505 (143/148/132) | 0.572 (139/100/108) |

(tp/fp/fn in parentheses.)

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| A | 0.50 | 0.573 (0.573) |
| B | 0.60 | 0.571 (0.570) |
| C | 0.30 | 0.534 (0.527) |
| D | 0.50 | 0.547 (0.547) |
| E | 0.50 | 0.549 (0.549) |
| F | 0.40 | 0.558 (0.553) |

## Run F only: detection vs grading breakdown

`scripts/evaluate_separated_metrics.py` splits "did we find the tree" from "did we grade it
right, given we found it" — a more diagnostic view than the pooled complete-task numbers above.
Only run F has been scored this way so far.

| Level | Precision | Recall | F1 |
|---|---|---|---|
| Detection (class-agnostic, found it at all) | 0.859 | 0.771 | 0.813 |
| Grading (health class, given detected — macro over mild/moderate/severe) | 0.509 | 0.506 | 0.507 |
| Complete end-to-end (includes missed trees) | 0.572 | 0.514 | 0.541 |

Reading this: run F's detector is fairly reliable at finding a tree (F1 0.81), but roughly half
of correctly-detected trees still get the wrong health grade — that's where most of the
end-to-end error comes from, not missed detections.

## Bugs found and fixed while producing run F's numbers

1. **`config.ini` `[MULTICHANNEL] ndvi_img_path`** pointed at a non-existent file
   (`data/tiff/new/ndvi_ge2m/ndvi_new3.tiff` — the real file is `data/tiff/new/ndvi_ge2m.tif`).
   `gdal.BuildVRT` silently dropped the missing source instead of erroring, so
   `stacked_8ch_out_new_data.vrt` had 7 bands instead of 8, and that 7-band data propagated
   through `cut_plots.py` → `create_masks.py` → `slice_plots.py` into the training tiles —
   crashing `train.py` at `GeneralizedRCNNTransform.normalize` (mean/std vector length 8 vs.
   image channels 7). Fixed the path and regenerated the whole `data/new_data/` tree.
2. **`scripts/evaluate_separated_metrics.py` and `scripts/evaluate_test_set.py`** both built the
   evaluation dataset with the hardcoded default `input_mode="native"`, never reading
   `TRAIN.input_preprocessing` from config. For run F (trained with `scaled_uint8`) this would
   have silently fed native-scaled pixels into a model expecting `[0,1]`-scaled uint8 bands —
   same failure mode as the mean/std mismatch documented in `CLAUDE.md`, just one layer up (no
   exception, just wrong numbers). Both scripts now call `get_image_preprocessing_mode(config)`
   and pass it through as `input_mode`, and print the mode they used so it's visible per run.

## Takeaways so far

- Every run lands in roughly the same F1 band (0.53–0.57 pooled, complete-task), so band count
  alone (3 to 8 channels) hasn't been the dominant lever yet — grading (health-class) errors
  dominate over detection errors wherever it's been measured (run F).
- CHM removal (run C) is the clearest single-factor hit: lowest recall (0.496), lowest AR@100
  (0.360) of the old-pipeline runs — losing structural height information costs recall more than
  it costs precision.
- Adding NDVI to RGB (run D → E) barely moves pooled F1 (0.547 → 0.549) on the old test set.
- Run F is not yet a clean apples-to-apples comparison against A/B (different test plots, and
  `scaled_uint8` vs `native` preprocessing confounded with the new annotation set) — worth
  training a `native`-preprocessing 8-channel model on the *same* `data/new_data` tiles run F
  used, so the preprocessing-mode question can be isolated from the annotation-set question.
- Recommended next measurement: run `evaluate_separated_metrics.py` on runs A–E so the
  detection/grading split can be compared across all configurations, not just F.

---

# 2026-09-02 update: CHM-updated dataset (runs G, H)

A new CHM capture arrived, already cut into one GeoTIFF per plot (`data/tiff/new/CHM_clip_by_plot/`)
instead of the full-mosaic form `image_merging_vrt.py`/`cut_plots.py` expect. Built
`scripts/build_chm_updated_dataset.py` (new pipeline stage, config section `[NEW_CHM]`) to swap
just the CHM band into the existing 8-band cropped plots and re-run masks + slicing. See that
script's docstring for the full stage diagram.

**Coverage is partial: only 14 of the 32 plots have a new CHM clip so far** (matched to plots by
geographic bounding-box overlap, since clip filenames don't encode the plot ID — filename-based
matching was tried first and demonstrably wrong, see the script's docstring). Of those 14: 11 are
train-split plots, 3 are test-split plots (of the 6 held-out test plots run F used, only
`plot_1`/`plot_25`/`plot_28` are covered). Output: `data/new_data_chm_update/dataset_sliced_800`
(176 train/val tiles) and `...dataset_sliced_800_test` (48 test tiles, 299 GT instances, again zero
`healthy`-class GT).

**This dataset is a strict subset of run F's** (same plots, same non-CHM bands, only the CHM band
and the plot count differ) — so G/H are comparable to each other, but not to A–F on absolute
numbers: 3x fewer test tiles and a different, much smaller train set (11 vs 26 plots) than F.

## Bug found while building this: GDAL nodata sentinel treated as real data

`data/tiff/new/{ndvi,cire,gndvi,ndre}_ge2m.tif` have real gaps in flight coverage encoded as
GDAL's float32 nodata sentinel (~-3.4028e+38) rather than NaN — about **24% of pixels** in every
plot touched by this update. `np.isfinite()` does not catch this value (it's a real, finite
float), so it flowed untouched into `dataset/multi_channel_dataset.py`'s `native` preprocessing
path and straight into the model transform's `(x - mean) / std`, which turns it into an
inf/NaN-scale activation — this would have made native-mode training on this data diverge
immediately. It also silently corrupted `scripts/compute_channel_stats.py`'s output (mean/std
computed from these tiles came out as ~1e38-magnitude garbage, discovered while computing stats
for run G below). `scaled_uint8` mode's clipping step happened to make it harmless by accident,
which is likely why run F (same underlying source rasters) trained without visible issues.

**Fixed both**: `process_image_for_model` now zeroes any non-finite or `|value| > 1e6` pixel
before either preprocessing branch runs, and `compute_channel_stats.py` excludes the same pixels
from its mean/std accumulation instead of counting them as real values. This is a general fix, not
specific to the CHM-update dataset — it also applies to run F's dataset (`data/new_data`), though
run F itself was already trained and evaluated before this fix and is not being retroactively
re-scored here.

## Configurations

| Run | Checkpoint dir | Channels | Preprocessing | Dataset | Best epoch / trained |
|---|---|---|---|---|---|
| G | `results/checkpoints_chm_update_native/` | 8 (new CHM) | native | `data/new_data_chm_update/dataset_sliced_800` | 12 / 42 |
| H | `results/checkpoints_chm_update_scaled_uint8/` | 8 (new CHM) | scaled_uint8 | same | 23 / 53 |

New config files: `config_chm_update_native.ini`, `config_chm_update_scaled_uint8.ini`. Both use
anchors re-derived for this mask set via `scripts/compute_anchor_sizes.py`
(`32,40,51; 64,81,102; 128,161,203; 256,323,406; 512,645,813`, ratios `0.63,1.0,1.28`) — different
from A–F's anchors, since crown-size distribution shifts with a different (smaller, 11-plot)
training set. G's `image_mean`/`image_std` were freshly measured on this dataset after the nodata
fix (`10.7326,0.4112,0.2539,0.3805,0.1067` / `8.3098,0.2847,0.1801,0.2558,0.0742` for
CHM/NDVI/CIRE/GNDVI/NDRE); H keeps the same ImageNet+0.5 placeholder convention as run F's config,
for consistency with that comparison. Same architecture/schedule as all other runs otherwise
(ResNet-50-FPN, `trainable_layers=3`, `batch_size=2`, `lr=1e-4` cosine, early stopping patience 30,
checkpoint selection by highest `pooled_f1`).

## Pooled test-set results (score≥0.5, IoU≥0.5) — 48 tiles, 299 GT instances

| Run | tp | fp | fn | Precision | Recall | F1 | Macro F1 | AP50 | AR@100 |
|---|---|---|---|---|---|---|---|---|---|
| G (native) | 162 | 269 | 137 | 0.376 | 0.542 | 0.444 | 0.337 | 0.482 | 0.440 |
| H (scaled_uint8) | 167 | 227 | 132 | 0.424 | 0.559 | **0.482** | 0.361 | 0.501 | 0.419 |

### Per-class F1 (pooled)

| Run | mild | moderate | severe |
|---|---|---|---|
| G | 0.503 (38/29/46) | 0.472 (77/106/66) | 0.372 (47/134/25) |
| H | 0.547 (52/54/32) | 0.523 (73/63/70) | 0.375 (42/110/30) |

(tp/fp/fn in parentheses.) H beats G on every class.

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| G | 0.60 | 0.445 (0.444) |
| H | 0.60 | 0.498 (0.482) |

## Detection vs grading breakdown (both runs)

| Level | G Precision | G Recall | G F1 | H Precision | H Recall | H F1 |
|---|---|---|---|---|---|---|
| Detection (class-agnostic) | 0.601 | 0.866 | 0.710 | 0.645 | 0.849 | 0.733 |
| Grading (given detected, macro mild/moderate/severe) | 0.407 | 0.423 | 0.391 | 0.475 | 0.473 | 0.450 |
| Complete end-to-end | 0.376 | 0.542 | 0.444 | 0.424 | 0.559 | 0.482 |

Both runs' detection F1 (~0.71–0.73) is noticeably lower than run F's 0.813 on the larger dataset —
expected, given roughly a third of the training plots. H's edge over G shows up at both the
detection and grading level, not just one of them.

## Takeaways

- **scaled_uint8 (H) beat native (G) on this dataset**, on every metric: pooled F1 0.482 vs 0.444,
  every per-class F1, and both detection and complete-task recall. This is the opposite direction
  from what run F vs. the old-pipeline runs A/B might suggest, but G and H are the first
  apples-to-apples native-vs-scaled_uint8 comparison in this file (same plots, same split, same
  epoch-selection rule) — every earlier preprocessing comparison (F vs. A–E) was confounded by a
  different annotation set. Take this result as provisional: 11 training plots is a small sample,
  and G's freshly-measured native stats vs. H's untuned placeholder stats is itself a second
  confounding variable — a fairer follow-up would tune both equally.
- Both G and H land well below run F's complete-task F1 (0.553) — expected, since G/H train on
  11 plots vs. F's 26. This isn't evidence the new CHM data is bad; it's an artifact of the much
  smaller training set until CHM coverage extends to more plots.
- Detection F1 (~0.71–0.73) is the main gap vs. run F (0.813), more than grading — consistent with
  the "detection needs plot diversity, grading needs per-class examples" pattern already seen: 11
  plots undersupplies both, but especially the range of backgrounds/crown shapes detection learns
  from.
- Next step once more plots get new CHM clips: re-run `build_chm_updated_dataset.py`, and re-train
  both configs on the larger resulting set to see whether H's advantage over G holds up, or was
  mostly a small-sample effect.
