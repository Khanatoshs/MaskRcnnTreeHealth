"""Pre-flight check for the [MULTICHANNEL] extra_band_paths rasters.

The elevation/intensity layers arrive as ESRI ArcInfo binary grid *directories*
(a pile of .adf files). A partial copy of one of those is not obviously broken -
gdalinfo opens it and reports the full size happily, because the header is
intact - and it only fails later, mid-pipeline, when a read lands on a block
whose index file (`*x.adf`) never arrived. That is exactly what happened on the
first attempt at the 10-band build: `data/tiff/new/el/el` was missing
`z001001x.adf` and 11 of 59 plots (3 of them held-out test plots) failed to read.

This script reads a small window from every extra band over every plot, so an
incomplete copy is caught in seconds rather than twenty minutes into a rebuild -
and reports the value range, nodata count, and whether any band exceeds 255
(which matters for the RGB rescale gate in dataset/multi_channel_dataset.py).

Usage (from repo root):
    MASKRCNN_CONFIG=config_0904_10ch_native.ini python scripts/check_extra_bands.py
"""
import configparser
import glob
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def main():
    config = configparser.ConfigParser()
    config.read(os.environ.get("MASKRCNN_CONFIG", "config.ini"))

    extra = [p.strip() for p in
             config.get("MULTICHANNEL", "extra_band_paths", fallback="").split(",")
             if p.strip()]
    if not extra:
        print("No [MULTICHANNEL] extra_band_paths configured - nothing to check.")
        return 0

    # Plot outlines only supply footprints to sample, so any existing cropped-plot tree
    # works. The configured one usually doesn't exist yet (that's what the rebuild this
    # check gates is about to create), so fall back to any other dataset's plots.
    configured = config.get("CUT_PLOTS", "OUTPUT_FOLDER")
    plots = sorted(glob.glob(os.path.join(configured, "*", "*.tif")))
    used = configured
    if not plots:
        for root in sorted(glob.glob(os.path.join("data", "*", "cropped_plots"))):
            found = sorted(glob.glob(os.path.join(root, "*", "*.tif")))
            if found:
                plots, used = found, root
                break
    if not plots:
        raise SystemExit("No cropped plots found anywhere under data/*/cropped_plots "
                         "to sample footprints from.")
    if used != configured:
        print(f"(configured plots dir {configured} not built yet - "
              f"sampling footprints from {used} instead)")

    bounds = []
    for p in plots:
        with rasterio.open(p) as src:
            bounds.append((os.path.basename(p), src.bounds))
    print(f"Checking {len(extra)} extra band(s) against {len(bounds)} plot footprints\n")

    failed_overall = False
    for path in extra:
        print(f"== {path} ==")
        if not os.path.exists(path):
            print("   MISSING - path does not exist\n")
            failed_overall = True
            continue
        try:
            src = rasterio.open(path)
        except Exception as exc:
            print(f"   UNREADABLE - {exc}\n")
            failed_overall = True
            continue

        nodata = src.nodata
        unreadable, vmin, vmax, nod, over255, npix = [], np.inf, -np.inf, 0, 0, 0
        with src:
            for name, b in bounds:
                win = from_bounds(b.left, b.bottom, b.right, b.top, src.transform)
                try:
                    a = src.read(1, window=win, out_shape=(1, 96, 96)).astype(np.float64)
                except Exception:
                    unreadable.append(name)
                    continue
                if nodata is not None:
                    mask = a == nodata
                    nod += int(mask.sum())
                    a = a[~mask]
                if a.size:
                    vmin = min(vmin, float(a.min()))
                    vmax = max(vmax, float(a.max()))
                    over255 += int((a > 255).sum())
                    npix += a.size

        print(f"   dtype={src.dtypes[0]} nodata={nodata}")
        print(f"   readable plots: {len(bounds) - len(unreadable)}/{len(bounds)}")
        if unreadable:
            failed_overall = True
            print(f"   *** UNREADABLE PLOTS ({len(unreadable)}): {', '.join(unreadable[:10])}"
                  + (" ..." if len(unreadable) > 10 else ""))
            print("   *** This band is an incomplete copy - re-copy it before rebuilding.")
        if npix:
            print(f"   value range (excl. nodata): {vmin} .. {vmax}")
            print(f"   nodata pixels inside plots: {nod}")
            print(f"   values > 255: {over255}")
        print()

    if failed_overall:
        print("PRE-FLIGHT FAILED - fix the band(s) above before running the pipeline.")
        return 1
    print("PRE-FLIGHT OK - every extra band reads cleanly over every plot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
