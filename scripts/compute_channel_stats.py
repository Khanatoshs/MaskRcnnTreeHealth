"""Compute per-channel mean/std for the model's input normalization.

torchvision's GeneralizedRCNNTransform normalizes every input channel as
`(x - mean) / std`. `get_multiband_maskrcnn` falls back to ImageNet RGB stats
plus a hardcoded 0.5/0.5 for every extra channel, which silently assumes those
channels live in [0, 1]. That holds for the vegetation indices but not for CHM,
which is in metres (roughly 0-26 here) - so CHM ends up spanning ~58 normalized
units while RGB spans ~4 and NDRE ~1, letting the height band dominate the first
conv layer and flattening the spectral bands.

This script measures the real distribution and prints `image_mean` / `image_std`
for config.ini's [TRAIN] section.

Statistics are computed over the TRAINING plots only (the same plot-level split
train.py uses), so validation and test tiles never leak into the normalization.

Usage:
    python scripts/compute_channel_stats.py
    python scripts/compute_channel_stats.py --all_tiles     # ignore the split
"""
import argparse
import configparser
import glob
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import split_paths_by_plot

DEFAULT_NAMES = ["R", "G", "B", "CHM", "NDVI", "CIRE", "GNDVI", "NDRE"]


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=config.get("TRAIN", "dataset_dir", fallback="data/dataset_sliced_800"))
    parser.add_argument("--all_tiles", action="store_true",
                        help="Use every tile instead of only the training plots.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sample every Nth tile (speeds up a large dataset).")
    parser.add_argument("--rgb_divisor", type=float, default=255.0,
                        help="Matches the dataset's RGB scaling; use 1 to disable.")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "images", "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "masks", "*.tif")))
    if not image_paths:
        raise SystemExit(f"No tiles found in {args.dataset_dir}/images")

    if not args.all_tiles:
        train_val_split = float(config.get("TRAIN", "train_val_split", fallback="0.8"))
        image_paths, _, _, _ = split_paths_by_plot(
            image_paths, mask_paths, train_size=train_val_split, random_state=42
        )
    image_paths = image_paths[::args.stride]

    with rasterio.open(image_paths[0]) as src:
        n_bands = src.count
    names = DEFAULT_NAMES[:n_bands] + [f"band{i}" for i in range(len(DEFAULT_NAMES) + 1, n_bands + 1)]

    total = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)
    count = 0
    lo = np.full(n_bands, np.inf)
    hi = np.full(n_bands, -np.inf)

    for path in image_paths:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        # Mirror the dataset's RGB scaling so the stats describe what the model sees.
        if args.rgb_divisor != 1.0 and n_bands >= 3:
            arr[:3] /= args.rgb_divisor
        flat = arr.reshape(n_bands, -1)
        finite = np.isfinite(flat)
        if not finite.all():
            flat = np.where(finite, flat, 0.0)
        total += flat.sum(axis=1)
        total_sq += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
        lo = np.minimum(lo, flat.min(axis=1))
        hi = np.maximum(hi, flat.max(axis=1))

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))

    scope = "all tiles" if args.all_tiles else "training plots only"
    print(f"Analyzed {len(image_paths)} tiles ({scope}), {n_bands} bands\n")
    print(f"{'band':<8}{'min':>10}{'max':>10}{'mean':>11}{'std':>10}")
    for i, name in enumerate(names):
        print(f"{name:<8}{lo[i]:>10.3f}{hi[i]:>10.3f}{mean[i]:>11.4f}{std[i]:>10.4f}")

    # Show how wide each channel becomes under the current fallback normalization,
    # which is what makes the imbalance concrete.
    fb_mean = np.array([0.485, 0.456, 0.406] + [0.5] * (n_bands - 3))[:n_bands]
    fb_std = np.array([0.229, 0.224, 0.225] + [0.5] * (n_bands - 3))[:n_bands]
    spread_before = (hi - lo) / fb_std
    spread_after = (hi - lo) / std
    print(f"\n{'band':<8}{'spread (fallback)':>19}{'spread (fitted)':>18}")
    for i, name in enumerate(names):
        print(f"{name:<8}{spread_before[i]:>19.1f}{spread_after[i]:>18.1f}")
    print(f"\nmax/min spread ratio: {spread_before.max()/spread_before.min():>6.1f}x  ->"
          f"{spread_after.max()/spread_after.min():>6.1f}x after fitting")

    # The first three channels MUST keep ImageNet statistics. The backbone's 53
    # FrozenBatchNorm2d layers carry running statistics baked in during ImageNet
    # pretraining and cannot adapt (requires_grad=False), so feeding RGB at a
    # different scale mis-calibrates every one of them. Measured RGB std here is
    # ~0.13 vs ImageNet's ~0.225; substituting it makes RGB activations ~1.8x too
    # large and collapses training. Only the extra bands, which have no pretrained
    # calibration to preserve, get measured statistics - that is what stops CHM
    # (metres) from dominating the spectral indices.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    out_mean = IMAGENET_MEAN[:n_bands] + [float(v) for v in mean[3:]]
    out_std = IMAGENET_STD[:n_bands] + [float(v) for v in std[3:]]

    print("\nPaste into config.ini [TRAIN]:")
    print("  (RGB keeps ImageNet stats - see comment in this script)")
    print("image_mean = " + ",".join(f"{v:.4f}" for v in out_mean))
    print("image_std  = " + ",".join(f"{v:.4f}" for v in out_std))
    print("\nMeasured RGB stats, for reference only - do NOT use these:")
    print("  mean " + ",".join(f"{v:.4f}" for v in mean[:3])
          + "   std " + ",".join(f"{v:.4f}" for v in std[:3]))


if __name__ == "__main__":
    main()
