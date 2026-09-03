"""Run one band-ablation variant end to end, in its own directories.

The question this answers is what each input band is worth: the paper claims a
multispectral benefit, and the only way to support that is to train the same model
on band subsets and compare on the same held-out plots.

    rgb        R,G,B                          3 channels
    rgb_ndvi   R,G,B,NDVI                     4 channels
    no_chm     R,G,B,NDVI,CIRE,GNDVI,NDRE     7 channels   (everything but height)

Two things are deliberately shared across variants, because varying them would
confound the comparison:

  * The pixel grid. Each variant's VRT is a band subset of the existing 8-band VRT
    (gdal.Translate -b), not a rebuild from the source rasters, so every variant
    crops to byte-identical plot geometry. Rebuilding would let the extent shift by
    a pixel or two, since the source rasters differ slightly in size.
  * The masks. They are rasterized from the shapefile and carry no band data, so
    the same ones serve every variant - and must, or the runs would differ in
    ground truth as well as in bands. This also sidesteps create_masks.py's CHM
    height filter, which none of these variants could apply (no CHM band).

Everything else - VRT, cropped plots, tiles, checkpoints, metrics, visualizations -
is written under a per-variant name so runs can be revisited and compared later.

Usage (from repo root, one variant at a time):
    python scripts/run_band_ablation.py --variant rgb
    python scripts/run_band_ablation.py --variant rgb_ndvi
    python scripts/run_band_ablation.py --variant no_chm
    python scripts/run_band_ablation.py --variant rgb --stage train   # resume a stage
"""
import argparse
import configparser
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# Band indices in results/stacked_8ch_out.vrt, 1-based, as written by
# scripts/image_merging_vrt.py: 1-3 RGB, 4 CHM, 5 NDVI, 6 CIRE, 7 GNDVI, 8 NDRE.
BAND_INDEX = {"R": 1, "G": 2, "B": 3, "CHM": 4, "NDVI": 5, "CIRE": 6, "GNDVI": 7, "NDRE": 8}

# Per-band normalization measured on the training plots by
# scripts/compute_channel_stats.py. RGB keeps ImageNet statistics because the
# backbone's FrozenBatchNorm layers are calibrated to them and cannot adapt.
BAND_STATS = {
    "R":     (0.4850, 0.2290),
    "G":     (0.4560, 0.2240),
    "B":     (0.4060, 0.2250),
    "CHM":   (10.8663, 8.3472),
    "NDVI":  (0.5266, 0.1108),
    "CIRE":  (0.3391, 0.0955),
    "GNDVI": (0.5179, 0.0751),
    "NDRE":  (0.1435, 0.0347),
}

VARIANTS = {
    "rgb":      ["R", "G", "B"],
    "rgb_ndvi": ["R", "G", "B", "NDVI"],
    "no_chm":   ["R", "G", "B", "NDVI", "CIRE", "GNDVI", "NDRE"],
}

STAGES = ["vrt", "cut", "slice", "train", "eval", "visualize"]


def paths_for(variant):
    return {
        "vrt":        f"results/stacked_{variant}.vrt",
        "plots":      f"data/cropped_plots_{variant}",
        "tiles":      f"data/dataset_sliced_800_{variant}",
        "tiles_test": f"data/dataset_sliced_800_{variant}_test",
        "ckpt":       f"checkpoints_{variant}",
        "metrics":    f"checkpoints_{variant}/metrics",
        "vis":        f"checkpoints_{variant}/visualizations",
        "config":     f"configs/config_{variant}.ini",
        "log":        f"checkpoints_{variant}/train.log",
    }


def build_vrt(variant, bands, p, source_vrt):
    """Band-subset the 8-band VRT, preserving its exact grid."""
    from osgeo import gdal
    gdal.UseExceptions()
    os.makedirs(os.path.dirname(p["vrt"]), exist_ok=True)
    band_list = [BAND_INDEX[b] for b in bands]
    out = gdal.Translate(p["vrt"], source_vrt,
                         options=gdal.TranslateOptions(format="VRT", bandList=band_list))
    if out is None:
        raise RuntimeError(f"gdal.Translate failed for {variant}")
    out = None

    import rasterio
    with rasterio.open(source_vrt) as a, rasterio.open(p["vrt"]) as b:
        if (a.width, a.height, a.transform, a.crs) != (b.width, b.height, b.transform, b.crs):
            raise RuntimeError(
                "Variant VRT grid differs from the source - masks would not align.")
        print(f"  {b.count} bands ({', '.join(bands)}), grid identical to source "
              f"({b.width}x{b.height})")


def write_config(variant, bands, p, base_config="config.ini"):
    """Derive the variant's config from the current one, changing only what differs."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve key case
    if not cfg.read(base_config):
        raise SystemExit(f"Could not read {base_config}")

    means = ",".join(f"{BAND_STATS[b][0]:.4f}" for b in bands)
    stds = ",".join(f"{BAND_STATS[b][1]:.4f}" for b in bands)

    cfg["TRAIN"]["num_input_channels"] = str(len(bands))
    cfg["TRAIN"]["image_mean"] = means
    cfg["TRAIN"]["image_std"] = stds
    cfg["TRAIN"]["dataset_dir"] = p["tiles"]
    cfg["TRAIN"]["checkpoint_dir"] = p["ckpt"]
    cfg["TRAIN"]["metrics_dir"] = p["metrics"]
    # Own log file, so variants don't interleave into one.
    cfg["TRAIN"]["log_file"] = f"{p['ckpt']}/training.log"

    cfg["CUT_PLOTS"]["VRT_FILE"] = p["vrt"]
    cfg["CUT_PLOTS"]["OUTPUT_FOLDER"] = p["plots"]

    # Masks are shared with the 8-band run on purpose - see the module docstring.
    cfg["SLICING"]["PLOTS_DIR"] = f"{p['plots']}/train"
    cfg["SLICING"]["OUTPUT_DIR"] = p["tiles"]
    cfg["SLICING"]["TEST_PLOTS_DIR"] = f"{p['plots']}/test"
    cfg["SLICING"]["TEST_OUTPUT_DIR"] = p["tiles_test"]

    cfg["EVAL"]["CHECKPOINT"] = f"{p['ckpt']}/maskrcnn_best.pth"
    cfg["EVAL"]["DATASET_DIR"] = p["tiles_test"]
    cfg["EVAL"]["OUT_DIR"] = p["metrics"]

    cfg["VISUALIZE"]["CHECKPOINT"] = f"{p['ckpt']}/maskrcnn_best.pth"
    cfg["VISUALIZE"]["DATASET_DIR"] = p["tiles"]
    cfg["VISUALIZE"]["OUT"] = f"{p['vis']}/predictions_val.png"

    os.makedirs(os.path.dirname(p["config"]), exist_ok=True)
    with open(p["config"], "w") as fh:
        fh.write(f"# Band-ablation variant '{variant}': {', '.join(bands)}\n"
                 f"# Generated by scripts/run_band_ablation.py from {base_config}.\n"
                 f"# Masks are shared with the full-band run so ground truth is identical.\n\n")
        cfg.write(fh)
    print(f"  wrote {p['config']}  ({len(bands)} channels)")
    return p["config"]


def run(cmd, config_path, log_path=None):
    env = dict(os.environ, MASKRCNN_CONFIG=os.path.abspath(config_path))
    print(f"  $ {' '.join(cmd)}")
    started = time.time()
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "ab") as fh:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=REPO, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"  FAILED ({proc.returncode}): {' '.join(cmd)}"
                         + (f" - see {log_path}" if log_path else ""))
    print(f"  done in {time.time() - started:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--source_vrt", default="results/stacked_8ch_out.vrt")
    parser.add_argument("--base_config", default="config.ini")
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="Start from this stage instead of the beginning.")
    parser.add_argument("--only", action="store_true", help="Run just --stage, then stop.")
    args = parser.parse_args()

    os.chdir(REPO)
    variant = args.variant
    bands = VARIANTS[variant]
    p = paths_for(variant)
    for key in ("ckpt", "metrics", "vis"):
        os.makedirs(p[key], exist_ok=True)

    start = STAGES.index(args.stage) if args.stage else 0
    todo = [STAGES[start]] if args.only else STAGES[start:]

    print(f"\n=== variant '{variant}': {', '.join(bands)} ({len(bands)} channels) ===")
    print(f"stages: {', '.join(todo)}\n")

    config_path = p["config"]
    if not os.path.exists(config_path) or "vrt" in todo:
        write_config(variant, bands, p, args.base_config)

    for stage in todo:
        print(f"[{variant}] {stage}")
        if stage == "vrt":
            build_vrt(variant, bands, p, args.source_vrt)
        elif stage == "cut":
            run([PYTHON, "scripts/cut_plots.py"], config_path, p["log"])
        elif stage == "slice":
            run([PYTHON, "scripts/slice_plots.py"], config_path, p["log"])
        elif stage == "train":
            run([PYTHON, "train.py"], config_path, p["log"])
        elif stage == "eval":
            run([PYTHON, "scripts/evaluate_test_set.py"], config_path)
        elif stage == "visualize":
            run([PYTHON, "scripts/visualize_predictions.py"], config_path)

    print(f"\n=== variant '{variant}' complete ===")
    print(f"  config      {p['config']}")
    print(f"  checkpoints {p['ckpt']}")
    print(f"  metrics     {p['metrics']}")
    print(f"  images      {p['vis']}")


if __name__ == "__main__":
    main()
