# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mask R-CNN pipeline for tree crown instance segmentation + 4-class health classification from
multi-channel UAV/drone imagery (RGB + CHM + NDVI + CIRE + GNDVI + NDRE), for a paper on tree
health classification in Mongolia.

## Environment

There is no requirements.txt/pyproject.toml. Dependencies must be present in the active Python
environment: `torch`, `torchvision`, `rasterio`, `scikit-learn` (`train_test_split`), `numpy`,
`matplotlib`, `geopandas`, `shapely`, `scipy`, and GDAL (`osgeo.gdal`) for the VRT step.

On this machine, the pyenv environment `dltrain`
(`/home/smartforest/.pyenv/versions/3.12.11/envs/dltrain`) has the full set (torch 2.9.1+cu128,
scikit-learn, rasterio, pycocotools, scipy). The conda env `maskrcnn` has torch/rasterio but is
**missing scikit-learn**, so `train.py` (which imports `sklearn.model_selection.train_test_split`)
will fail there. Check which env actually has everything before assuming any single one works;
use `<env>/bin/python` directly rather than relying on `python3` on PATH.

## Configuration

`config.ini` is gitignored (contains local machine paths) and is read from the current working
directory by every script (`configparser.ConfigParser().read("config.ini")`), so scripts must be
run from the repo root. `config_sample.ini` is the tracked, public-safe template — copy it to
`config.ini` and fill in real paths before running anything:

```bash
cp config_sample.ini config.ini
```

Config sections map 1:1 to pipeline stages: `[TRAIN]`, `[MULTICHANNEL]`, `[CUT_PLOTS]`, `[MASKS]`,
`[SLICING]`. When adding a new tunable, add it to both `config.ini` and `config_sample.ini` and
read it with a `fallback=` default (the existing convention throughout the codebase).

## Pipeline (run from repo root, in order)

```bash
python scripts/image_merging_vrt.py   # build a VRT stacking RGB+CHM+NDVI+CIRE+GNDVI+NDRE bands (near-zero disk/RAM cost)
python scripts/cut_plots.py           # crop VRT into per-plot GeoTIFFs, split into train/ and test/ by the shapefile's `class` attribute
python scripts/create_masks.py        # rasterize tree polygons into per-plot instance+class mask GeoTIFFs (train AND test plots)
python scripts/slice_plots.py         # cut plot images+masks into overlapping tiles (config: WINDOW_SIZE, OVERLAP_PCT), drop empty tiles
python train.py                       # train Mask R-CNN, evaluate, checkpoint
```

`create_masks.py` and `slice_plots.py` each process two splits: the `PLOTS_DIR`/`MASKS_DIR`/
`OUTPUT_DIR` trio (train plots) and, when the optional `TEST_*` keys are set, the held-out test
plots into their own directories. The test set is produced by the pipeline but never read by
`train.py` — it exists for final reporting only.

`scripts/image_merging.py` is an older/alternate approach that materializes a real N-channel
GeoTIFF instead of a VRT — not part of the documented pipeline, kept for reference.

`scripts/check_tree_classes.py` is a standalone sanity check on the tree shapefile (prints unique
`tree_class` values, flags out-of-range/missing values) — run it before `create_masks.py` if class
labels look off.

`scripts/compute_anchor_sizes.py --masks_dir <dir>` derives RPN anchor `sizes`/`aspect_ratios`
from the actual crown size distribution, treating each distinct mask value as one crown; prints
values to paste into `config.ini`'s `[TRAIN]` section (`anchor_sizes`, `anchor_aspect_ratios`,
comma-separated). Pass `--legacy_connected_components` only for old class-only masks. Re-run it
whenever the masks or tile size change.

`scripts/evaluate_test_set.py` scores a checkpoint and is the only place the held-out test plots
should be read:

```bash
python scripts/evaluate_test_set.py                        # held-out test plots (paper number)
python scripts/evaluate_test_set.py --split val \
    --dataset_dir data/dataset_sliced_800                  # reproduce train.py's val split
python scripts/evaluate_test_set.py --score_threshold 0.6  # trade recall for precision
```

It reports per-class precision/recall/F1 **pooled over the whole split** (not averaged per tile),
a confusion matrix, class-agnostic detection AP (separates "found the tree" from "got the health
class right"), and a score-threshold sweep. Results are written to `<metrics_dir>/<split>_evaluation.json`.

There is no test suite, linter, or CI config in this repo.

## Architecture

**`train.py`** is a single self-contained script (build model → load data → train/val loop → save
metrics), not a package with a CLI. Logs to `train.log` (root logger is also configured, so
imported modules using `logging.getLogger(__name__)` inherit the same file handler).

- `get_multiband_maskrcnn(num_classes, in_channels, anchor_sizes=None, anchor_aspect_ratios=None)`
  builds a `torchvision` Mask R-CNN with a ResNet-50-FPN backbone whose `conv1` is replaced to
  accept an arbitrary channel count: the first 3 output channels keep pretrained RGB weights, any
  extra channels (spectral indices, CHM, etc.) are initialized from the mean RGB kernel. Backbone
  freezing is controlled by `resnet_fpn_backbone(trainable_layers=3)` (freezes `conv1`+`layer1`,
  trains `layer2..4`). Per-channel `image_mean`/`image_std` are generated to match `in_channels`.
  If `anchor_sizes` is given, a custom `AnchorGenerator` replaces torchvision's defaults — one
  size per FPN level, aspect ratios shared across levels.
- `dataset/multi_channel_dataset.py`:
  - `StackedImageInstanceMaskDataset` reads a stacked multi-channel image TIFF + a single-channel
    mask TIFF per sample. Every distinct non-zero pixel value becomes one instance; the class
    label is decoded as `value // label_divisor` (see "Mask encoding" below). Passing
    `label_divisor=None` restores the legacy behaviour where the pixel value *is* the class —
    only for reading old datasets, never for training. A legacy mask read with a divisor set
    raises a clear error rather than silently labelling everything background.
  - `TrainAugmentation` (train split only, never validation): flips, 90° rotations, scale jitter,
    brightness/contrast jitter (RGB channels only), channel dropout (non-RGB channels). Recomputes
    boxes from the transformed masks after each geometric op rather than transforming box
    coordinates analytically, to keep box/mask alignment guaranteed correct.
- Train/val split: **by plot, not by tile** (`split_paths_by_plot`, config `split_by_plot = true`).
  `slice_plots.py` cuts tiles with 50% overlap, so a tile-level split leaks badly — measured on
  this dataset, 95% of validation tiles shared ≥50% of their pixels (and the same trees) with a
  training tile, and all 31 plots appeared on both sides. Plot IDs are parsed from the tile
  filename (`{plot}_tile_y{y}_x{x}.tif`); if that parse fails the code falls back to a tile-level
  split with a warning. Don't "simplify" this back to `train_test_split` over tile paths.
- Validation runs the model **twice per batch**: once in `.train()` mode with targets (to get the
  loss dict for `val_loss`), once in `.eval()` mode without targets (to get `boxes`/`labels`/`scores`
  for metrics). This is intentional/required — torchvision's Mask R-CNN only returns losses when
  called in training mode with targets, and only returns predictions in eval mode without targets.
- Custom detection metrics (no pycocotools dependency in `train.py` itself): `compute_iou`,
  `compute_class_prf` (per-class precision/recall/F1 via greedy IoU matching at a fixed score
  threshold), `compute_global_pr_metrics` (macro mean over classes), `compute_confusion_matrix`
  (row=true/col=predicted, index 0 = background; unmatched predictions go to row 0, missed GT go
  to column 0).
  **A class absent from both the ground truth and the predictions in a batch is omitted from the
  result, not scored 0.0.** With `val_batch_size = 1` most tiles hold only 2-3 of the 4 classes, so
  scoring the absent ones zero and folding them into the macro mean understated precision/recall by
  ~0.19 on this dataset. A missing `class_N` key therefore means "not applicable to this batch";
  `compute_global_pr_metrics` returns `{}` in that case and the training loop skips the batch
  rather than averaging in an unearned zero. Don't reintroduce the zero-fill.
- Checkpointing: only the single best checkpoint (`checkpoints/maskrcnn_best.pth`) is ever saved,
  selected by **lowest val_loss** (falls back to mean F1 if val_loss is non-finite). No per-epoch
  or "latest" checkpoint is kept, so any epoch other than the current best is unrecoverable after
  the run ends. `save_final_metrics_summary`'s reported "best epoch" is looked up by the
  `best_epoch` value the training loop tracks — it must stay consistent with the checkpointing
  criterion above; don't let it drift back to recomputing "best" by a different metric.
- End-of-run artifacts land in `checkpoints/metrics/`: `final_metrics.txt`, `metrics_history.csv`,
  `metric_progression.png`, `per_class_metrics.png`, `final_confusion_matrix.csv`(+heatmap PNG),
  `confusion_matrix_progression.png`.

## Mask encoding: instance AND class in one band

`create_masks.py`'s `generate_instance_class_mask_geotiff` writes

```
mask value = class_id * LABEL_DIVISOR + instance_index      # LABEL_DIVISOR = 10000
             10003 -> class 1 (healthy), 3rd tree in the plot
             40127 -> class 4 (severe), 127th tree in the plot
```

as uint16, so every tree is its own instance while still carrying its health class. The dataset
decodes the class with `value // label_divisor`. `LABEL_DIVISOR` in `create_masks.py` and
`mask_label_divisor` in `config.ini` must stay in sync.

Class mapping is `tree_class` (0=healthy, 1=mild, 2=moderate, 3=severe) **+ 1**, so model classes
are 1-4 with 0 = background (`num_classes = 5`).

`generate_class_mask_geotiff` (one pixel value per class, no instance identity) is retained in the
same file but **must not be used for training**: it merges every same-class tree in a plot into a
single blob. That was the previous behaviour and it discarded ~94% of the annotations — 1734
annotated trees across the 31 train plots collapsed into 104 trainable instances, with median GT
boxes of 633px inside 800px tiles. If detection quality regresses sharply, check which function
`__main__` is calling before touching hyperparameters.

## Input normalization is not in the checkpoint — it must be matched by hand

`image_mean`/`image_std` live in the model's `GeneralizedRCNNTransform`, not in its
`state_dict`. Loading a checkpoint under different normalization therefore raises **no error** —
`load_state_dict` succeeds and the model silently emits garbage predictions. This has already
caused one wrong conclusion: an evaluation script that omitted these values made a healthy run
look like it had collapsed to a single class, which then motivated a bogus "fix".

Rules:
- `[TRAIN] image_mean` / `image_std` must be one value per channel, in band order
  (R,G,B,CHM,NDVI,CIRE,GNDVI,NDRE).
- **Any script that rebuilds the model to load a checkpoint must pass the same values the
  checkpoint was trained with**, not just whatever `config.ini` currently says.
  `scripts/evaluate_test_set.py` reads them from config and accepts `--image_mean` /
  `--image_std` overrides (`fallback` selects the built-in ImageNet+0.5 defaults) so older
  checkpoints can be scored correctly. It prints the values it used — check them.
- Derive the extra-band values with `scripts/compute_channel_stats.py`. Without them the
  non-RGB bands fall back to 0.5/0.5, which assumes a [0,1] range: CHM is in metres, so it
  then spans ~58 normalized units against NDRE's ~1 and dominates the first conv layer.

## Metrics: per-tile macro vs pooled

`train.py` logs a **per-batch macro** precision/recall (mean over the classes present in each
tile, then mean over tiles). With `val_batch_size = 1` and few instances of a class per tile,
a single lucky detection gives that class recall 1.0 for the tile, so this statistic runs
optimistic and is dominated by sparse tiles. `scripts/evaluate_test_set.py` instead **pools
tp/fp/fn over the whole split** and computes precision once — that is the number to report.
The two are not comparable; do not quote a training-log figure against an evaluation-script one.

## Held-out test set

`mesh.shp`'s `class` attribute splits the 39 plots into 31 train / 8 test, and `cut_plots.py`
writes them to `data/cropped_plots/{train,test}/`. The test plots are masked and sliced by the
`TEST_*` config keys into `data/dataset_sliced_800_test/` but are **never read by `train.py`** —
they are the final-reporting set. `train.py` splits only the *train* plots into train/val (by
plot). Keep it that way: evaluating on the test tiles during development would burn the one
genuinely held-out estimate.
