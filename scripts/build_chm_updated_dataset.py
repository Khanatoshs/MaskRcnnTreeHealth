"""Rebuild the 8-band dataset with an updated CHM band, for a new CHM capture
that arrives already cut into one GeoTIFF per plot (unlike the full-mosaic CHM
`image_merging_vrt.py`/`cut_plots.py` expect).

Only band 4 (CHM) changes. RGB, NDVI, CIRE, GNDVI and NDRE are reused unchanged
from the existing 8-band cropped plots (built by cut_plots.py from
`data/tiff/new`). This script does NOT touch `data/tiff/new`'s own CHM raster.

The new CHM clip filenames don't encode the plot ID (e.g. `chm_clip171.tif`
does not correspond to `plot_17`), so plots are matched to CHM clips by
geographic bounding-box overlap instead: a clip matches the one plot whose
footprint falls almost entirely inside the clip's extent
(`MIN_OVERLAP_FRACTION`, default 0.98). Clips with no confident match, and
plots with no matching clip, are skipped and logged - only plots that received
a new CHM raster end up in the output dataset.

Pipeline stage inserted between cut_plots.py and train.py:
  cut_plots.py (existing 8-band plots) --+
                                          +--> build_chm_updated_dataset.py --> train.py
  CHM_clip_by_plot/*.tif (new CHM) ------+

Run from repo root, after cut_plots.py has produced the source 8-band cropped
plots referenced by [NEW_CHM] SOURCE_TRAIN_PLOTS_DIR / SOURCE_TEST_PLOTS_DIR:
    python scripts/build_chm_updated_dataset.py
    python scripts/build_chm_updated_dataset.py --dry_run   # show matches only
"""
import argparse
import configparser
import glob
import logging
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_masks import generate_instance_class_mask_geotiff
from slice_plots import slice_plots_and_masks

logging.basicConfig(level=logging.INFO, filename="utils.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Build_CHM_Updated_Dataset")
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console)


def _bounds_in_crs(bounds, src_crs, dst_crs):
    if src_crs == dst_crs:
        return bounds
    return BoundingBox(*transform_bounds(src_crs, dst_crs, *bounds))


def _overlap_fraction(clip_bounds, plot_bounds):
    """Fraction of plot_bounds' area that falls inside clip_bounds."""
    left = max(clip_bounds.left, plot_bounds.left)
    right = min(clip_bounds.right, plot_bounds.right)
    bottom = max(clip_bounds.bottom, plot_bounds.bottom)
    top = min(clip_bounds.top, plot_bounds.top)
    if left >= right or bottom >= top:
        return 0.0
    inter_area = (right - left) * (top - bottom)
    plot_area = (plot_bounds.right - plot_bounds.left) * (plot_bounds.top - plot_bounds.bottom)
    return inter_area / plot_area if plot_area > 0 else 0.0


def match_clips_to_plots(clip_paths, plot_entries, min_overlap_fraction):
    """Match each CHM clip GeoTIFF to the one plot it (almost) fully covers.

    plot_entries: list of (plot_name, plot_path, split) tuples.
    Returns (matches, unmatched_clips):
      matches: {plot_name: (clip_path, split, overlap_fraction)}
      unmatched_clips: [(clip_path, best_plot_name_or_None, best_fraction)]
    """
    plot_info = {}
    for name, path, split in plot_entries:
        with rasterio.open(path) as src:
            plot_info[name] = (src.bounds, src.crs, path, split)

    matches = {}
    unmatched_clips = []
    for clip_path in clip_paths:
        with rasterio.open(clip_path) as src:
            clip_bounds, clip_crs = src.bounds, src.crs

        best_name, best_frac = None, 0.0
        for name, (pbounds, pcrs, _path, _split) in plot_info.items():
            cb = _bounds_in_crs(clip_bounds, clip_crs, pcrs)
            frac = _overlap_fraction(cb, pbounds)
            if frac > best_frac:
                best_frac, best_name = frac, name

        if best_name is None or best_frac < min_overlap_fraction:
            unmatched_clips.append((clip_path, best_name, best_frac))
            continue

        prior = matches.get(best_name)
        if prior is not None and prior[2] >= best_frac:
            unmatched_clips.append((clip_path, best_name, best_frac))
            continue
        if prior is not None:
            unmatched_clips.append((prior[0], best_name, prior[2]))

        _pbounds, _pcrs, _ppath, split = plot_info[best_name]
        matches[best_name] = (clip_path, split, best_frac)

    return matches, unmatched_clips


def build_chm_updated_plot(source_plot_path, chm_clip_path, chm_band, out_path):
    """Copy source_plot_path's bands as-is, replacing chm_band with chm_clip_path
    resampled (bilinear) onto the source plot's exact grid (same transform/crs/shape)."""
    with rasterio.open(source_plot_path) as src:
        profile = src.profile.copy()
        bands = src.read()
        dst_transform, dst_crs = src.transform, src.crs
        height, width = src.height, src.width

    new_chm = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(chm_clip_path) as chm_src:
        reproject(
            source=rasterio.band(chm_src, 1),
            destination=new_chm,
            src_transform=chm_src.transform,
            src_crs=chm_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    bands[chm_band - 1] = new_chm
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild 8-band cropped plots/masks/tiles with an updated, "
                    "already per-plot-clipped CHM band. Reads [NEW_CHM] from config.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only match CHM clips to plots and print the result; write nothing.")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.environ.get("MASKRCNN_CONFIG", "config.ini"))
    if not config.has_section("NEW_CHM"):
        raise SystemExit("config.ini has no [NEW_CHM] section - see config_sample.ini.")
    S = "NEW_CHM"

    chm_clip_dir = config.get(S, "CHM_CLIP_DIR")
    train_plots_dir = config.get(S, "SOURCE_TRAIN_PLOTS_DIR")
    test_plots_dir = config.get(S, "SOURCE_TEST_PLOTS_DIR", fallback="").strip()
    chm_band = config.getint(S, "CHM_BAND", fallback=4)
    min_overlap_fraction = config.getfloat(S, "MIN_OVERLAP_FRACTION", fallback=0.98)
    output_dir = config.get(S, "OUTPUT_DIR")
    tree_shapefile = config.get(S, "TREE_SHAPEFILE")
    tree_class_property = config.get(S, "TREE_CLASS_PROPERTY", fallback="tree_class")
    _min_height = config.get(S, "MIN_TREE_HEIGHT", fallback="").strip()
    min_tree_height = float(_min_height) if _min_height else None
    window_size = config.getint(S, "WINDOW_SIZE", fallback=800)
    overlap_pct = config.getfloat(S, "OVERLAP_PCT", fallback=50.0)
    _completeness = config.get(S, "MIN_INSTANCE_COMPLETENESS", fallback="").strip()
    min_instance_completeness = float(_completeness) if _completeness else None

    clip_paths = sorted(glob.glob(os.path.join(chm_clip_dir, "*.tif")))
    if not clip_paths:
        raise SystemExit(f"No CHM clip GeoTIFFs found in {chm_clip_dir}")

    plot_entries = []
    for path in sorted(glob.glob(os.path.join(train_plots_dir, "*.tif"))):
        plot_entries.append((os.path.splitext(os.path.basename(path))[0], path, "train"))
    if test_plots_dir:
        for path in sorted(glob.glob(os.path.join(test_plots_dir, "*.tif"))):
            plot_entries.append((os.path.splitext(os.path.basename(path))[0], path, "test"))
    if not plot_entries:
        raise SystemExit(f"No source cropped plots found in {train_plots_dir!r} / {test_plots_dir!r}")

    logger.info(f"Matching {len(clip_paths)} CHM clips against {len(plot_entries)} existing plots "
                f"(min_overlap_fraction={min_overlap_fraction})...")
    matches, unmatched = match_clips_to_plots(clip_paths, plot_entries, min_overlap_fraction)

    by_split = {"train": [], "test": []}
    for plot_name, (clip_path, split, frac) in sorted(matches.items()):
        by_split[split].append(plot_name)
        logger.info(f"  MATCH  {plot_name:<12} ({split}) <- {os.path.basename(clip_path)} "
                    f"(overlap={frac:.3f})")
    for clip_path, best_name, frac in unmatched:
        logger.warning(f"  SKIP   {os.path.basename(clip_path)}: no confident plot match "
                        f"(best={best_name}, overlap={frac:.3f})")

    logger.info(f"Matched {len(matches)} plots: {len(by_split['train'])} train, "
                f"{len(by_split['test'])} test. {len(unmatched)} clip(s) skipped.")

    if not matches:
        raise SystemExit("No plots matched a CHM clip - nothing to do.")
    if args.dry_run:
        logger.info("--dry_run: stopping before writing any files.")
        return

    cropped_plots_dir = os.path.join(output_dir, "cropped_plots")
    mask_dirs = {"train": os.path.join(output_dir, "train_plot_masks"),
                "test": os.path.join(output_dir, "test_plot_masks")}
    sliced_dirs = {"train": os.path.join(output_dir, "dataset_sliced_800"),
                  "test": os.path.join(output_dir, "dataset_sliced_800_test")}

    logger.info("Building CHM-updated cropped plots...")
    for plot_name, (clip_path, split, _frac) in sorted(matches.items()):
        source_path = dict((n, p) for n, p, s in plot_entries)[plot_name]
        out_path = os.path.join(cropped_plots_dir, split, f"{plot_name}.tif")
        build_chm_updated_plot(source_path, clip_path, chm_band, out_path)
        logger.info(f"  wrote {out_path}")

    logger.info(f"Loading tree annotations from {tree_shapefile}...")
    trees_gdf = gpd.read_file(tree_shapefile)
    if min_tree_height is not None:
        logger.info(f"Height filter: dropping trees below {min_tree_height} m (CHM band {chm_band})")

    for split in ("train", "test"):
        if not by_split[split]:
            continue
        plots_dir = os.path.join(cropped_plots_dir, split)
        masks_dir = mask_dirs[split]
        os.makedirs(masks_dir, exist_ok=True)
        logger.info(f"Generating masks for {split} split ({len(by_split[split])} plots)...")
        for plot_path in sorted(glob.glob(os.path.join(plots_dir, "*.tif"))):
            generate_instance_class_mask_geotiff(
                plot_path, trees_gdf, masks_dir, class_property=tree_class_property,
                min_tree_height=min_tree_height, chm_band=chm_band,
            )

    for split in ("train", "test"):
        if not by_split[split]:
            continue
        logger.info(f"Slicing {split} split into {window_size}px tiles "
                    f"({overlap_pct}% overlap)...")
        slice_plots_and_masks(
            plots_dir=os.path.join(cropped_plots_dir, split),
            masks_dir=mask_dirs[split],
            output_base_dir=sliced_dirs[split],
            window_size=window_size,
            overlap_pct=overlap_pct,
            drop_empty_masks=True,
            min_instance_completeness=min_instance_completeness,
        )

    logger.info("Done.")
    logger.info(f"Train tiles: {sliced_dirs['train']}" if by_split["train"] else "No train tiles.")
    logger.info(f"Test tiles:  {sliced_dirs['test']}" if by_split["test"] else "No test tiles.")


if __name__ == "__main__":
    main()
