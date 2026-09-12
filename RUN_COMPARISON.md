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

---

# 2026-09-03 update: isolating preprocessing mode on the full new_data set (run I)

Run F (scaled_uint8) was never matched by a native-preprocessing run on the *same* `data/new_data`
tiles, so every earlier native-vs-scaled_uint8 read (F vs. A–E, or G vs. H) was confounded by
either a different annotation set (F vs. A–E) or a much smaller 11-plot training set (G/H). This
run closes that gap: **run I is run F's exact configuration — same 26 train / 6 test plots, same
`data/new_data/dataset_sliced_800(_test)` tiles, same anchors, same architecture/schedule — with
only `input_preprocessing` switched from `scaled_uint8` to `native`.** I vs. F is now the cleanest
preprocessing-mode comparison in this file.

Run F's own evaluation (pooled test-set metrics in `results/checkpoints_scaled_uint8/metrics/` and
the detection/grading breakdown in `results/checkpoints_scaled_uint8/separated_metrics/`) was
already complete and already reported above (the "Configurations"/pooled-results tables and the
"Run F only: detection vs grading breakdown" section) — nothing new needed there, it's reused
as-is for this comparison.

## Configuration

| Run | Checkpoint dir | Channels | Preprocessing | Dataset | Best epoch / trained |
|---|---|---|---|---|---|
| F | `results/checkpoints_scaled_uint8/` | 8 | scaled_uint8 | `data/new_data/dataset_sliced_800` | 33 / 63 |
| I | `results/checkpoints_new_data/` | 8 | **native** | `data/new_data/dataset_sliced_800` | 14 / 44 |

Config: [config_new_data_native.ini](config_new_data_native.ini) (frozen copy of the `config.ini`
that was already staged for this run). Same anchors as F (`sizes=112,144,184,216,288`,
`ratios=0.62,1.0,1.36`), same `batch_size=2`, `lr=1e-4` cosine, warmup 10 epochs, early stopping
patience 30, checkpoint selection by highest `pooled_f1`. I's `image_mean`/`image_std` for the
extra 5 bands were measured on `data/new_data` via `scripts/compute_channel_stats.py` (with the
nodata-sentinel fix from the G/H update already in place — see that section above):
`10.8663,0.5266,0.3391,0.5179,0.1435` / `8.3472,0.1108,0.0955,0.0751,0.0347` for
CHM/NDVI/CIRE/GNDVI/NDRE (RGB stays ImageNet). I trained much faster than F did on the same data
(~27 min for 44 epochs vs. F's ~8 hours for 63) — same GPU, same tile count; not investigated
further since loss curves and val metrics both look normal, but worth knowing if reproducing.

## Pooled test-set results — evaluation metrics (`evaluate_test_set.py`, score≥0.5, IoU≥0.5, same 96 tiles / 742 GT / 6 test plots for both runs)

| Run | tp | fp | fn | Precision | Recall | F1 | AP50 | AR@100 |
|---|---|---|---|---|---|---|---|---|
| F (scaled_uint8) | 405 | 318 | 337 | 0.560 | 0.546 | 0.553 | 0.558 | 0.418 |
| I (native) | 413 | 339 | 329 | 0.549 | 0.557 | **0.553** | 0.637 | 0.529 |

Complete-task F1 is a dead heat (0.5529 vs. 0.5529 to 4 sig figs — F is 0.552901, I is 0.552878).
I clearly wins on detection quality: AP50 0.637 vs. 0.558 and AR@100 0.529 vs. 0.418, both well
above F. Pooled precision/recall trade off in opposite directions (F: higher precision, lower
recall; I: lower precision, higher recall) and roughly cancel in F1.

### Per-class F1 (pooled)

| Run | healthy | mild | moderate | severe |
|---|---|---|---|---|
| F | n/a (no GT) | 0.596 (123/70/97) | 0.505 (143/148/132) | 0.572 (139/100/108) |
| I | n/a (no GT) | 0.542 (97/41/123) | 0.522 (161/181/114) | 0.597 (155/117/92) |

(tp/fp/fn in parentheses.) F grades `mild` better; I grades `severe` slightly better and
`moderate` better; `mild` recall is I's weak point (0.441 vs. F's 0.559 — I's mild-class fn=123
vs. F's 97).

**Macro F1 note:** the "Macro F1" this project has been reporting (0.418 for F, computed the same
way here) averages precision/recall/F1 across **all 4 classes**, scoring the absent `healthy`
class as 0 rather than excluding it — despite the earlier footnote on F's row claiming a 3-class
average. That footnote was wrong; the number itself (0.418) is the 4-class figure with a zero
folded in, confirmed against `results/checkpoints_scaled_uint8/metrics/all_evaluation.json`'s
`macro.f1`. I's equivalent 4-class figure is **0.415** (`results/checkpoints_new_data/metrics/all_evaluation.json`
→ `macro.f1`), directly comparable to F's 0.418. The classification-only 3-class macro (excluding
`healthy`, mild/moderate/severe only, matching how the per-class F1 row above should actually be
averaged) is **0.554 for I** and **0.558 for F** — nearly identical, consistent with the pooled
F1 tie.

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| F | 0.40 | 0.558 (0.553) |
| I | 0.50 | 0.553 (0.553) |

## Detection vs grading breakdown — classification metrics (`evaluate_separated_metrics.py`)

Splits "did we find the tree" (class-agnostic detection) from "did we grade it right, given we
found it" (classification only, scored on true positives). Both already-reported for F; I is new.

| Level | F Precision | F Recall | F F1 | I Precision | I Recall | I F1 |
|---|---|---|---|---|---|---|
| Detection (class-agnostic, found it at all) | 0.859 | 0.771 | 0.813 | 0.831 | 0.842 | **0.837** |
| Grading (health class, given detected — macro over mild/moderate/severe) | 0.509 | 0.506 | 0.507 | 0.514 | 0.502 | 0.503 |
| Complete end-to-end (includes missed trees) | 0.572 | 0.514 | 0.541 | 0.549 | 0.557 | 0.553 |

I's detection F1 (0.837) beats F's (0.813) — mainly recall (I: tp=625/fp=127/fn=117, R=0.842; F:
tp=572/fp=94/fn=170, R=0.771 — native finds 53 more of the 742 GT trees, at the cost of 33 more
detection false positives). Grading, given a correct detection, is a near-tie (0.503 vs. 0.507) —
the health-class confusion (mild vs. moderate vs. severe on an already-detected crown) is
essentially unaffected by preprocessing mode. The complete-task numbers land close (0.553 vs.
0.541) because I's detection gain is partly offset by marginally weaker grading precision on those
extra detections.

Full outputs: `results/checkpoints_new_data/metrics/all_evaluation.json` (pooled) and
`results/checkpoints_new_data/separated_metrics/separated_metrics_test.json` (detection/grading).

## Takeaways

- **On this dataset, native vs. scaled_uint8 makes almost no difference to complete-task F1**
  (0.553 both ways) once the annotation set and training data are held fixed — the earlier F-vs-A/B
  read (which looked like scaled_uint8 might help) and the G-vs-H read (scaled_uint8 clearly ahead)
  were both confounded, by a different annotation set and a small 11-plot subset respectively. This
  run is the first apples-to-apples test on the full 26-plot set, and it says preprocessing mode is
  not the lever — contrast with the CHM-removal effect (run C) or plot-count effect (F/I vs. G/H),
  both of which move F1 far more.
- Native's edge is entirely on the detection side (AP50 0.637 vs. 0.558, AR@100 0.529 vs. 0.418,
  detection-F1 0.837 vs. 0.813) — it finds more crowns, especially at looser IoU/recall operating
  points. Grading accuracy given a detection is indistinguishable between modes (0.503 vs. 0.507).
  If detection recall matters more than precision for the paper's use case, native is the better
  default; if the two are equally weighted, either is defensible since complete-task F1 ties.
- The macro-F1 methodology bug flagged above (footnote said 3-class, number was 4-class) affects
  every future run added to this table with a healthy-free test split — use the explicit 3-class
  figure (0.554 / 0.558 here) when a true classification-only macro is wanted, and don't trust the
  "Macro F1" column's footnotes without checking the underlying JSON.

---

# 2026-09-03 update: corrected 3-tier annotation scheme (runs J, K)

**The annotation scheme changed: `tree_class` is now 3-tier (1=healthy, 2=mild, 3=severe — no
"moderate"), not the old 4-tier scale.** Investigating this surfaced a real, pre-existing bug that
had silently affected every `data/new_data` run so far (F, G, H, I):

- `data/annotations/0831_v3/AnnotationRGB.shp` (the shapefile behind every `data/new_data` run)
  already stores `tree_class ∈ {1, 2, 3}` — 1-indexed, no 0. The *old* shapefile
  (`data/annotations/NDVI_mean3.shp`, still used by the original `data/dataset_sliced_800`
  pipeline / runs A–E) stores `tree_class ∈ {0, 1, 2, 3}` — 0-indexed, 4-tier.
- `scripts/create_masks.py` unconditionally added `+1` to `tree_class` to get the mask class_id
  (background=0, so class ids must start at 1). That's correct for the 0-indexed shapefile but
  wrong for the already-1-indexed one: it shifted every new_data tree's class up by one and left
  class 1 permanently empty.
- This exactly explains a pattern that showed up in every `data/new_data`-based run documented
  above: **"zero healthy ground truth"** in every new_data test split (F, G, H, I). It wasn't a
  real annotation gap — it was this offset bug. What was reported as "mild" for F/G/H/I was
  actually the annotator's "healthy" tag, "moderate" was actually "mild", and "severe" only ever
  reached class 4 because `tree_class` topped out at 3.
- **Fix**: `class_id_offset` is now a config value (`[MASKS] CLASS_ID_OFFSET`, default `1` so the
  old NDVI_mean3.shp pipeline is unaffected). `data/new_data`'s configs set it to `0`. Also fixed:
  `create_masks.py` was hardcoded to read `config.ini` regardless of `MASKRCNN_CONFIG` (every other
  pipeline script respects that env var) — now consistent. `scripts/evaluate_test_set.py`'s
  `CLASS_NAMES` is now picked by the checkpoint's actual `num_classes` (3-tier vs. 4-tier), so old
  and new checkpoints both print correct class names instead of the new checkpoint borrowing the
  old 4-tier labels.
- **Not fixed here, flagged for follow-up**: `scripts/build_chm_updated_dataset.py` (runs G, H)
  calls the same masking function with the same un-overridden default offset, reading from the
  same `AnnotationRGB.shp`. G and H almost certainly have the identical class-shift bug. Rebuilding
  them was out of scope for this pass — only `data/new_data` (F/I's dataset) was rebuilt.
- `data/new_data/{train,test}_plot_masks` and `dataset_sliced_800(_test)` were regenerated with the
  corrected offset. Tile counts are unchanged (416 train/val + 96 test tiles, 742 test GT
  instances) — only geometry drives tile selection, not class labels, so nothing was dropped or
  added, only relabeled. **`results/checkpoints_new_data` (run I) and `results/checkpoints_scaled_uint8`
  (run F) were deliberately left untouched** as the historical record of the pre-fix numbers; the
  corrected runs below write to new directories instead of overwriting them.

## Configuration

| Run | Checkpoint dir | Channels | Classes | Preprocessing | Dataset | Best epoch / trained |
|---|---|---|---|---|---|---|
| J | `results/checkpoints_new_data_native_3class/` | 8 | 4 (bg+3: healthy/mild/severe) | native | `data/new_data/dataset_sliced_800` (rebuilt) | 66 / 96 |
| K | `results/checkpoints_new_data_scaled_uint8_3class/` | 8 | 4 (bg+3) | scaled_uint8 | same | 15 / 45 |

Configs: [config_new_data_native_3class.ini](config_new_data_native_3class.ini) /
[config_new_data_scaled_uint8_3class.ini](config_new_data_scaled_uint8_3class.ini). Same anchors,
architecture and schedule as F/I (`sizes=112,144,184,216,288`, `ratios=0.62,1.0,1.36`,
`batch_size=2`, `lr=1e-4` cosine, early stopping patience 30, checkpoint selection by highest
`pooled_f1`). Only `num_classes` (5→4) and the mask class encoding changed; image_mean/std are
unchanged from J/I's and K's scaled_uint8 predecessors since pixel statistics don't depend on
class labels. J and K are the direct 3-class counterparts of I and F respectively — same
comparison, same test plots (`plot_1/16/25/28/31/7`), corrected labels.

## Pooled test-set results — evaluation metrics (`evaluate_test_set.py`, score≥0.5, IoU≥0.5, 96 tiles / 742 GT)

| Run | tp | fp | fn | Precision | Recall | F1 | Macro F1 | AP50 | AR@100 |
|---|---|---|---|---|---|---|---|---|---|
| J (native, 3-class) | 391 | 334 | 351 | 0.539 | 0.527 | 0.533 | 0.535 | 0.517 | 0.365 |
| K (scaled_uint8, 3-class) | 435 | 342 | 307 | 0.560 | 0.586 | **0.573** | 0.576 | 0.626 | 0.497 |

Unlike F/I's macro F1, J/K's Macro F1 here is a genuine 3-class average — `healthy` now has real
GT (220 instances) so nothing needs excluding or flagging. **K (scaled_uint8) beats J (native) on
every metric**: F1 0.573 vs 0.533, macro F1 0.576 vs 0.535, AP50 0.626 vs 0.517, AR@100 0.497 vs
0.365. This is the same direction as the G-vs-H result (scaled_uint8 ahead) and the opposite
direction from I-vs-F's near-tie — though F/I's numbers are under the old mislabeled scheme, so
this is really the first clean native-vs-scaled_uint8 read on the corrected 3-class annotations.

### Per-class F1 (pooled)

| Run | healthy | mild | severe |
|---|---|---|---|
| J (native) | 0.540 (107/69/113) | 0.501 (144/156/131) | 0.565 (140/109/107) |
| K (scaled_uint8) | 0.571 (106/45/114) | 0.533 (169/190/106) | 0.623 (160/107/87) |

(tp/fp/fn in parentheses.) K beats J on all three classes, most on `severe` (+0.058) and `healthy`
(+0.031). Total false positives are similar between the two (342 vs 334), so the gap is mostly
recall: K finds more `mild` (0.615 vs 0.524) and `severe` (0.648 vs 0.567) trees, and on `healthy`
specifically has notably fewer false positives (45 vs 69) at a similar recall (0.482 vs 0.486).

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| J | 0.30 | 0.540 (0.533) |
| K | 0.50 | 0.573 (0.573) |

## Detection vs grading breakdown — classification metrics (`evaluate_separated_metrics.py`)

| Level | J Precision | J Recall | J F1 | K Precision | K Recall | K F1 |
|---|---|---|---|---|---|---|
| Detection (class-agnostic, found it at all) | 0.836 | 0.817 | 0.826 | 0.820 | 0.858 | **0.839** |
| Grading (health class, given detected — macro over healthy/mild/severe) | 0.514 | 0.507 | 0.509 | 0.539 | 0.529 | 0.531 |
| Complete end-to-end (includes missed trees) | 0.539 | 0.527 | 0.533 | 0.560 | 0.586 | 0.573 |

K wins at both levels: detection F1 0.839 vs 0.826 (K trades a bit of precision for more recall —
tp=637/fp=140/fn=105 vs J's tp=606/fp=119/fn=136) and grading F1 0.531 vs 0.509. Per-class grading
(on detected trees only): healthy F1 J=0.472/K=0.472 (a dead tie), mild J=0.510/K=0.554, severe
J=0.543/K=0.565 — K's edge is concentrated in mild/severe, not healthy.

Full outputs: `results/checkpoints_new_data_native_3class/{metrics,separated_metrics}/` and
`results/checkpoints_new_data_scaled_uint8_3class/{metrics,separated_metrics}/`.

## Takeaways

- **On the corrected 3-class annotations, scaled_uint8 clearly beats native** — the opposite of
  I-vs-F's near-tie. Since I-vs-F was run under the mislabeled 4-class scheme, K-vs-J supersedes it
  as the preprocessing-mode comparison to trust for `data/new_data`. Combined with G-vs-H (also
  scaled_uint8 ahead, on the smaller CHM-update subset), scaled_uint8 now leads in every
  apples-to-apples comparison run on this project's newer annotations — native only looked
  competitive under the buggy 4-class labeling.
- Dropping "moderate" simplified the classification task (3 tiers instead of 4, and a real
  `healthy` class instead of an always-empty one), so J/K's numbers are **not directly comparable**
  to F/I's on raw magnitude — different class taxonomy, different denominators. Don't read "J's
  0.533 vs I's 0.553" as native getting worse; they're scoring different label sets.
- Grading errors still dominate over detection errors, same pattern as every earlier run: detection
  F1 is ~0.83, grading F1 is ~0.51-0.53 for both J and K.
- Next step, if pursued: apply the same `CLASS_ID_OFFSET` fix to the CHM-update pipeline
  (`build_chm_updated_dataset.py` / configs `config_chm_update_*.ini`) and rebuild/retrain G and H,
  since they likely carry the identical class-shift bug on the same shapefile.

---

# 2026-09-03 update: new annotation set, full pipeline rerun (runs L, M)

A new, independent annotation set arrived: `data/annotations/new_plots/{Plot_0903,Annotation_0903}.shp`
— 43 plots (34 train / 9 test, `plot_id`/`class` columns) and 2163 tree polygons. Same site as
`data/new_data` (plot bounds fall entirely inside the existing `results/stacked_8ch_out_new_data.vrt`,
so the underlying RGB+CHM+NDVI+CIRE+GNDVI+NDRE imagery was reused unchanged - only the plot
boundaries and tree annotations are new) and the same 3-tier scheme as the fixed
`AnnotationRGB.shp` (`tree_class ∈ {1,2,3}`, no 0, no "moderate": 688 healthy / 757 mild / 718
severe polygons). Ran the full pipeline from `cut_plots.py` onward into distinct `data/new_plots/`
and `results/checkpoints_new_plots_*` directories so nothing from J/K/F/I/etc. was touched.

## Pipeline steps run

1. `cut_plots.py` (VRT reused, new `Plot_0903.shp`) → `data/new_plots/cropped_plots/{train,test}`:
   34 + 9 plot GeoTIFFs.
2. `create_masks.py` (new `Annotation_0903.shp`, `CLASS_ID_OFFSET=0` - same fix as the J/K update)
   → `data/new_plots/{train,test}_plot_masks`: 1766 train + 413 test tree instances.
3. `slice_plots.py` → `data/new_plots/dataset_sliced_800(_test)`: 543 train/val tiles, 144 test
   tiles, 979 test GT instances.
4. `compute_anchor_sizes.py --masks_dir data/new_plots/train_plot_masks`: crown side-length
   p50=178px / p99=407px - nearly identical to `data/new_data`'s p50=181px / p99=403px (same
   imagery/site), so **F/I/J/K's anchors were reused** (`112,144,184,216,288` /
   `0.62,1.0,1.36`) rather than the script's freshly-computed stride-respecting pyramid, per the
   documented ablation in this file's earlier section (`Don't reinstate it...`).
5. `compute_channel_stats.py --dataset_dir data/new_plots/dataset_sliced_800` (431 training tiles,
   post nodata-sentinel fix) for the native-mode config's `image_mean`/`image_std`.
6. Trained native then scaled_uint8 (same architecture/schedule as every other run: ResNet-50-FPN
   `trainable_layers=3`, `batch_size=2`, `lr=1e-4` cosine, warmup 10, early stopping patience 30,
   checkpoint selection by highest `pooled_f1`).

## Bug found while evaluating: GPU memory leak in `evaluate_separated_metrics.py`

Scoring the scaled_uint8 checkpoint crashed with `CUDA out of memory` (10.6 GiB used by a single
eval process on an 11.5 GiB card) - the native checkpoint's evaluation had just run fine on the
same 144 tiles. Cause: the per-batch loop accumulated `model()`'s raw output dicts (boxes, labels,
scores, **and full-resolution instance masks**) into `all_predictions` without moving them off the
GPU first; `torch.cuda.empty_cache()` right after (present, but only returns *unreferenced* cached
memory to the driver - it does nothing while a live Python list still holds the tensors) couldn't
help. The noisier scaled_uint8 checkpoint produced enough more raw per-tile detections before score
thresholding that accumulated mask memory crossed the card's limit; the cleaner native checkpoint
happened to stay under it. Fixed in `scripts/evaluate_separated_metrics.py` by moving each
prediction/target dict to CPU (`{k: v.cpu() for k, v in p.items()}`) before appending - a general
fix, not specific to this run.

## Configuration

| Run | Checkpoint dir | Channels | Classes | Preprocessing | Dataset | Best epoch / trained |
|---|---|---|---|---|---|---|
| L | `results/checkpoints_new_plots_native/` | 8 | 4 (bg+3: healthy/mild/severe) | native | `data/new_plots/dataset_sliced_800` | 73 / 100 |
| M | `results/checkpoints_new_plots_scaled_uint8/` | 8 | 4 (bg+3) | scaled_uint8 | same | 20 / 50 |

Configs: [config_new_plots_native.ini](config_new_plots_native.ini) /
[config_new_plots_scaled_uint8.ini](config_new_plots_scaled_uint8.ini). L's `image_mean`/`image_std`
for the extra 5 bands: `11.3114,0.3846,0.2376,0.3712,0.1012` / `8.5418,0.2560,0.1636,0.2379,0.0680`
(CHM/NDVI/CIRE/GNDVI/NDRE); M uses the same 0.5/0.5 placeholder convention as every other
scaled_uint8 run. **This is the largest training set used in this project so far** (34 plots vs.
F/I/J/K's 26, G/H's 11) - both L and M trained noticeably longer before early-stopping than J/K did.

## Pooled test-set results — evaluation metrics (`evaluate_test_set.py`, score≥0.5, IoU≥0.5, 144 tiles / 979 GT)

| Run | tp | fp | fn | Precision | Recall | F1 | Macro F1 | AP50 | AR@100 |
|---|---|---|---|---|---|---|---|---|---|
| L (native) | 619 | 432 | 360 | 0.589 | 0.632 | **0.610** | 0.614 | 0.563 | 0.401 |
| M (scaled_uint8) | 683 | 727 | 296 | 0.484 | 0.698 | 0.572 | 0.574 | 0.688 | 0.499 |

**L (native) wins on complete-task F1** (0.610 vs. 0.572) - the opposite of J-vs-K's result on
`data/new_data`, where scaled_uint8 won. M trades a lot of precision for recall (P 0.484 vs. L's
0.589, R 0.698 vs. L's 0.632) and has much higher AP50/AR@100 (0.688/0.499 vs. 0.563/0.401) -
M's raw detector is more sensitive, but at the default 0.5 threshold that sensitivity shows up
mostly as false positives (727 vs. L's 432) rather than net F1 gain. **Both L and M score
noticeably higher than every prior `data/new_data`-based run** (J/K's 0.533/0.573, F/I's 0.553) -
consistent with this being the largest, most plot-diverse training set used so far.

### Per-class F1 (pooled)

| Run | healthy | mild | severe |
|---|---|---|---|
| L (native) | 0.633 (202/133/101) | 0.566 (224/200/144) | 0.643 (193/99/115) |
| M (scaled_uint8) | 0.560 (245/327/58) | 0.528 (211/220/157) | 0.635 (227/180/81) |

(tp/fp/fn in parentheses.) L beats M on all three classes, most on `healthy` (+0.073) where M's
false-positive count (327, more than its 245 true positives) sinks precision to 0.428 despite a
strong 0.809 recall. `severe` is the closest class between the two (0.643 vs. 0.635).

### Best F1 across the threshold sweep

| Run | Best threshold | Best F1 (default-threshold F1) |
|---|---|---|
| L | 0.50 | 0.610 (0.610) |
| M | 0.70 | 0.603 (0.572) |

Even at M's best operating threshold (0.70, trading most of its recall advantage back for
precision), it still falls short of L's F1 - native wins this comparison at every threshold M was
evaluated at.

## Detection vs grading breakdown — classification metrics (`evaluate_separated_metrics.py`)

| Level | L Precision | L Recall | L F1 | M Precision | M Recall | M F1 |
|---|---|---|---|---|---|---|
| Detection (class-agnostic, found it at all) | 0.849 | 0.911 | **0.879** | 0.671 | 0.966 | 0.792 |
| Grading (health class, given detected — macro over healthy/mild/severe) | 0.535 | 0.533 | 0.531 | 0.428 | 0.430 | 0.413 |
| Complete end-to-end (includes missed trees) | 0.589 | 0.632 | 0.610 | 0.484 | 0.698 | 0.572 |

**L's detection F1 (0.879) is the highest of any run in this project so far** (previous best: F/K
around 0.81-0.84), and it leads at every level, not just one. M's detection recall is exceptional
(0.966 - only 33 of 979 GT trees missed entirely) but comes at the cost of detection precision
(0.671, i.e. a lot of spurious boxes) and grading (0.413 vs. L's 0.531) - the false positives (464
of them at the detection level) actively hurt end-to-end quality rather than being a free lunch of
extra recall. Grading errors still dominate over detection errors for both, same pattern as every
earlier run, but the gap between the two levels is smaller here than in J/K, since detection itself
improved so much with the larger training set.

Full outputs: `results/checkpoints_new_plots_native/{metrics,separated_metrics}/` and
`results/checkpoints_new_plots_scaled_uint8/{metrics,separated_metrics}/`.

## Takeaways

- **More, more-diverse training plots (34 here vs. 26 for F/I/J/K, 11 for G/H) is the strongest
  lever pulled so far in this project** - L's detection F1 (0.879) and complete-task F1 (0.610)
  both beat every earlier run outright, well past the gap any preprocessing-mode change has
  produced. This lines up with the CLAUDE.md/earlier-takeaways pattern that detection specifically
  benefits from plot/background diversity.
- **Native beats scaled_uint8 here, reversing J/K's result** (where scaled_uint8 won on
  `data/new_data`). Combined, this project now has three native-vs-scaled_uint8 comparisons under
  the corrected 3-class scheme (G/H, J/K, L/M) and they don't agree on a consistent winner - the
  preprocessing-mode question looks dataset-dependent rather than having a universal answer, at
  least at the current per-run training-set sizes. Don't generalize a preprocessing recommendation
  from any single comparison in this file.
- M's very high detection recall (0.966) alongside its worse end-to-end numbers is a useful
  diagnostic: raw detection sensitivity is not the same as being a better model once a fixed score
  threshold and health-class grading are applied. If the project ever wants a "never miss a tree"
  operating point, M (or L swept to a low threshold) is the one to revisit - but scaled_uint8's
  default-threshold precision here needs correcting first.
- Grading F1 (~0.53 for L, ~0.41 for M) is still the dominant source of end-to-end error for both,
  same as every run measured this way in this file.
