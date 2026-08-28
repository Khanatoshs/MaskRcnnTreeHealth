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
	Uses the tree annotation shapefile to create one instance-mask TIFF for each cropped plot. Background is `0`; each tree receives a unique positive ID.

4. `python scripts/slice_plots.py`
	Cuts the plot images and masks into overlapping 800 x 800 tiles, removes tiles without trees, and saves matching files under the configured dataset output directory.

5. `python train.py`
	Loads the sliced image/mask pairs, trains the multi-channel Mask R-CNN model, evaluates detections, and saves checkpoints and metrics under `checkpoints/`.

The input paths, output folders, channel count, tile size, overlap, training settings, and early-stopping patience are configured in `config.ini`.
