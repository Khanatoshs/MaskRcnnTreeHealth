"""Derive RPN anchor sizes/aspect ratios from the actual tree-crown size distribution.

Usage:
    python scripts/compute_anchor_sizes.py --masks_dir data/dataset_sliced_800/masks

Prints anchor_sizes / anchor_aspect_ratios to paste into config.ini's [TRAIN] section.

Masks written by scripts/create_masks.py encode `class_id * label_divisor +
instance_index`, so each distinct non-zero value is exactly one tree crown and
its bounding box is measured directly. Pass --legacy_connected_components to
instead measure connected components within each value, which is what you need
for old class-only masks where every same-class tree shares one pixel value.
"""
import argparse
import glob
import os

import numpy as np
import rasterio
from scipy import ndimage


def compute_crown_sizes(mask_paths, use_connected_components=False):
    widths, heights, areas, aspects = [], [], [], []
    for mp in mask_paths:
        with rasterio.open(mp) as src:
            data = src.read(1)
        for value in np.unique(data):
            if value == 0:
                continue

            if use_connected_components:
                labeled, n_cc = ndimage.label(data == value, structure=np.ones((3, 3)))
                regions = ((labeled == cc_id) for cc_id in range(1, n_cc + 1))
            else:
                regions = iter([data == value])

            for region in regions:
                ys, xs = np.where(region)
                if xs.size < 4:
                    continue
                w = xs.max() - xs.min() + 1
                h = ys.max() - ys.min() + 1
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                aspects.append(w / h)
    return np.array(widths), np.array(heights), np.array(areas), np.array(aspects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default="data/dataset_sliced_800/masks")
    parser.add_argument("--num_levels", type=int, default=5, help="FPN levels (P2..P6 = 5)")
    parser.add_argument(
        "--legacy_connected_components", action="store_true",
        help="Measure connected components inside each mask value instead of treating "
             "each value as one crown. Only for old class-only masks."
    )
    args = parser.parse_args()

    mask_paths = sorted(glob.glob(os.path.join(args.masks_dir, "*.tif")))
    if not mask_paths:
        raise SystemExit(f"No masks found in {args.masks_dir}")

    widths, heights, areas, aspects = compute_crown_sizes(
        mask_paths, use_connected_components=args.legacy_connected_components
    )
    sides = np.sqrt(areas)

    kind = "connected-component regions" if args.legacy_connected_components else "tree instances"
    print(f"Analyzed {len(mask_paths)} mask tiles, {len(sides)} {kind}.")
    print("\nCrown side length (sqrt(area)) percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2}: {np.percentile(sides, p):7.1f} px")

    print("\nAspect ratio (w/h) percentiles:")
    for p in [5, 25, 50, 75, 90, 95]:
        print(f"  p{p:>2}: {np.percentile(aspects, p):.2f}")

    # Anchor sizes must grow with each FPN level's stride (4, 8, 16, 32, 64), because
    # a level's feature map resolution is what lets it localize an object of that size.
    # Spreading the observed crown range evenly across the five levels instead - which
    # an earlier version of this script did - puts near-identical large anchors on
    # every level, including the stride-4 map whose 40,000 locations then dominate the
    # RPN's 256-anchor sample. So: keep the standard 8x-stride base and add sub-octave
    # scales, then report how well that pyramid covers the measured crowns.
    strides = [4 * 2 ** i for i in range(args.num_levels)]
    base = [8 * s for s in strides]                       # 32, 64, 128, 256, 512
    sub_octaves = [2 ** (k / 3) for k in range(3)]        # 1, 1.26, 1.587
    levels = [[int(round(b * m)) for m in sub_octaves] for b in base]
    flat = np.array(sorted(s for level in levels for s in level))

    # A crown is reachable when some anchor is within 1.45x of its size - roughly the
    # ratio at which a same-centre, same-ratio anchor still clears IoU 0.5.
    within = np.mean([bool((np.abs(np.log(flat / x)) < np.log(1.45)).any()) for x in sides]) * 100

    lo, mid, hi = np.percentile(aspects, [10, 50, 90])
    aspect_ratios = tuple(round(float(v), 2) for v in (lo, 1.0 if 0.8 < mid < 1.2 else mid, hi))

    print("\nAnchor pyramid (one row per FPN level, sizes grow with stride):")
    for s, level in zip(strides, levels):
        print(f"  stride {s:>2}: {', '.join(str(v) for v in level)}")
    print(f"\n  crowns within 1.45x of some anchor: {within:.1f}%")

    print("\nPaste into config.ini [TRAIN]:")
    print("anchor_sizes = " + "; ".join(",".join(str(v) for v in level) for level in levels))
    print(f"anchor_aspect_ratios = {','.join(str(a) for a in aspect_ratios)}")


if __name__ == "__main__":
    main()
