# MaskRcnnTreeHealth
Repository for the paper about tree health classification in Mongolia.

## Configuration setup

This repository includes a public-safe sample configuration at [config_sample.ini](config_sample.ini). To use the project locally:

1. Copy the sample file to a local file named `config.ini`.
2. Replace the placeholder values with your local dataset paths and output directories.
3. Keep `config.ini` local and do not upload it to GitHub.

Example:

```bash
cp config_sample.ini config.ini
```

Then edit `config.ini` and set your actual paths, such as:

- dataset directories
- annotation shapefiles
- output folders
- checkpoint locations

The sample file is designed so you can safely publish the repository without exposing your personal machine paths.

## Execution order

Run the scripts from the project root in this order:

1. `python scripts/image_merging_vrt.py`
	Creates a virtual raster containing the Red, Green, Blue, CHM, NDVI, CIRE, GNDVI, and NDRE bands. It does not copy the large raster data.

2. `python scripts/cut_plots.py`
	Uses the plot boundary shapefile to crop the VRT into individual plot images. The `class` attribute places each plot in `data/cropped_plots/train/` or `data/cropped_plots/test/`.

3. `python scripts/create_masks.py`
	Uses the tree annotation shapefile to create one mask TIFF per cropped plot, for both the train and test plots. Background is `0`; every other pixel encodes **both** the tree's identity and its health class as `class_id * 10000 + instance_index`, so each tree is a separate instance that still carries its class. Health classes come from the shapefile's `tree_class` attribute (`0` healthy, `1` mild, `2` moderate, `3` severe) shifted by `+1`, giving model classes `1-4` with `0` reserved for background.

4. `python scripts/slice_plots.py`
	Cuts the plot images and masks into overlapping 800 x 800 tiles, removes tiles without trees, and saves matching files under the configured dataset output directory. The held-out test plots are sliced into their own directory.

5. `python train.py`
	Loads the sliced image/mask pairs, trains the multi-channel Mask R-CNN model, evaluates detections, and saves checkpoints and metrics under `checkpoints/`. Train and validation are split **by plot**, not by tile: tiles overlap by 50%, so a random tile-level split would put spatially overlapping tiles (sharing pixels and the same trees) on both sides and inflate validation scores. The test plots are never read here.

6. `python scripts/evaluate_test_set.py`
	Scores the best checkpoint on the held-out test plots — per-class precision/recall/F1 pooled over the split, a confusion matrix, class-agnostic detection AP, and a score-threshold sweep. Use `--split val --dataset_dir data/dataset_sliced_800` to score the validation plots with the same code instead.

The input paths, output folders, channel count, tile size, overlap, training settings, and early-stopping patience are configured in `config.ini`.
