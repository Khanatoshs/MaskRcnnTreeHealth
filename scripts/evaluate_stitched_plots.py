"""Evaluate a checkpoint by stitching tile-level predictions back into whole plots
before scoring, instead of scoring each overlapping tile independently.

Why: scripts/evaluate_test_set.py and scripts/evaluate_separated_metrics.py score
each 800px tile on its own, but slice_plots.py cuts tiles at 50% overlap - a single
tree crown near a tile seam is very often detected (and counted) more than once
across the tiles that contain it. This script instead:

  1. Runs the checkpoint on every tile in --dataset_dir (a sliced test split).
  2. Reads the plot's original, unsliced RGB directly from --cropped_plots_dir for
     step 9's visualization. (It already exists on disk from cut_plots.py -
     re-mosaicking it from overlapping tiles would need blending logic for no
     benefit, since the identical source is right there.)
  3. Transforms each tile's raw predictions into the source plot's pixel
     coordinates, using the y/x pixel offset encoded in the tile filename
     ('{plot}_tile_y{y}_x{x}.tif') - this "stitches" every plot's predictions into
     one shared coordinate frame without needing to touch the imagery at all.
  4. Runs class-agnostic NMS (torchvision.ops.nms, IoU threshold configurable via
     NMS_IOU_THRESHOLD) over each plot's stitched detections. This is steps 4+5 of
     the spec in one pass: a duplicate detection of the same tree from two
     overlapping tiles is just two overlapping boxes once stitched, so ordinary
     NMS merges them (keeps the higher-scoring one) exactly like it would merge
     any other overlap. A genuine single detection near a plot's own outer edge
     has no duplicate to compete with, so NMS leaves it untouched - no special
     "don't merge across the plot border" logic is needed.
  5. Drops any surviving box smaller than MIN_CROWN_SIDE_PX (sqrt of box area) -
     too small to plausibly be a tree crown at this GSD.
  6. Scores the result against the plot-level ground truth, read directly from
     the mask create_masks.py wrote for that plot (one instance per tree, never
     tile-fragmented - there is nothing to stitch on the GT side). Matching is
     done once per plot (never across plots) using this project's usual box-IoU
     greedy matching, so pooled counts can never cross-match a prediction from
     one plot against another plot's ground truth.
  7. Writes a JSON, a CSV, and a confusion-matrix PNG+CSV of the resulting
     pooled/per-class metrics plus a detection-vs-grading breakdown (same
     methodology as evaluate_test_set.py / evaluate_separated_metrics.py).
  8. Renders one PNG per plot: the plot's RGB with GT and kept predictions drawn
     on top, colour-coded and styled exactly like scripts/visualize_predictions.py
     (solid=correct, dash-dot=wrong class, dashed=unmatched).

Config (config_stitched_eval.ini, or --config) holds the score/NMS/size thresholds
that are common across runs. --checkpoint/--dataset_dir/--cropped_plots_dir/
--plot_masks_dir/--input_preprocessing/--output_dir are per-run and always passed
on the command line - see run_all_stitched_eval.py, which runs this script over
every checkpoint that has both a native and a scaled_uint8 counterpart.

Usage:
    python scripts/evaluate_stitched_plots.py \\
        --checkpoint results/checkpoints_new_plots_native/maskrcnn_best.pth \\
        --dataset_dir data/new_plots/dataset_sliced_800 \\
        --cropped_plots_dir data/new_plots/cropped_plots/test \\
        --plot_masks_dir data/new_plots/test_plot_masks \\
        --input_preprocessing native \\
        --output_dir results/stitched_metrics/new_plots_native
"""
import argparse
import configparser
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms as torchvision_nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import (get_multiband_maskrcnn, collate_fn,
                    compute_detection_metrics, compute_grading_metrics_on_detections,
                    save_confusion_matrix)
from dataset.multi_channel_dataset import StackedImageInstanceMaskDataset

import evaluate_test_set as ets
import visualize_predictions as viz

TILE_RE = re.compile(r"(.+)_tile_y(\d+)_x(\d+)\.tif$")


def parse_tile_offset(path):
    """Recover (plot_id, y_offset, x_offset) from a sliced tile filename.

    slice_plots.py names tiles '{plot}_tile_y{y}_x{x}.tif', where y/x are the raw
    pixel offset of the tile's top-left corner inside the source plot raster
    (rasterio.windows.Window(x, y, w, h)) - so adding them to a tile-local box
    coordinate gives the plot-global coordinate directly, no rescaling needed.
    """
    m = TILE_RE.match(os.path.basename(path))
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_plot_gt(mask_path, label_divisor):
    """Read a plot-level instance+class mask and decode it to boxes/labels.

    Mirrors StackedImageInstanceMaskDataset's decode exactly (value // label_divisor
    = class, one box per distinct nonzero pixel value). Masks are written once per
    plot by create_masks.py, before slicing ever splits a crown across tiles, so
    this is the plot's true, unfragmented ground truth - nothing to stitch here.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        height, width = src.height, src.width
    boxes, labels = [], []
    for value in np.unique(mask):
        if value == 0:
            continue
        ys, xs = np.where(mask == value)
        ymin, ymax = float(ys.min()), float(ys.max())
        xmin, xmax = float(xs.min()), float(xs.max())
        if xmax <= xmin or ymax <= ymin:
            continue
        label = int(value) // label_divisor
        if label < 1:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64), height, width)


def class_agnostic_nms_and_size_filter(boxes, labels, scores, nms_iou_threshold, min_side_px, masks=None):
    """Steps 4+5+6 of the module docstring: merge tile-seam duplicates (NMS), then
    drop boxes too small to be a real crown.

    NMS is class-agnostic on purpose: the same physical tree detected in two
    overlapping tiles isn't guaranteed to get the same predicted health class in
    both crops, but it is still one tree - keep the higher-confidence detection
    regardless of which class either copy predicted.

    masks, if given, is a plain Python list of per-detection cropped mask patches
    (see run_inference) in the same order as boxes/labels/scores - not a numpy
    array, since patches vary in shape. Filtered in lockstep and returned as a
    list too (or None if masks was None).
    """
    if len(boxes) == 0:
        return boxes, labels, scores, masks
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    keep = torchvision_nms(boxes_t, scores_t, nms_iou_threshold).numpy()
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    if masks is not None:
        masks = [masks[i] for i in keep]

    if len(boxes) == 0:
        return boxes, labels, scores, masks
    sides = np.sqrt(np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    keep = sides >= min_side_px
    if masks is not None:
        masks = [m for m, k in zip(masks, keep) if k]
    return boxes[keep], labels[keep], scores[keep], masks


def _crop_mask_to_box(mask_full, box, pad=1):
    """Crop a tile-sized soft mask down to (roughly) its own box's pixel extent,
    thresholded to a uint8 patch. Keeps per-detection mask storage cheap (box-sized,
    not tile-sized) regardless of how many raw detections a tile produces - only
    used when --visualization_style masks asks for it."""
    h, w = mask_full.shape
    x1, y1, x2, y2 = box
    xi1 = max(0, int(np.floor(x1)) - pad); yi1 = max(0, int(np.floor(y1)) - pad)
    xi2 = min(w, int(np.ceil(x2)) + pad); yi2 = min(h, int(np.ceil(y2)) + pad)
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((0, 0), dtype=np.uint8)
    return (mask_full[yi1:yi2, xi1:xi2] > 0.5).astype(np.uint8)


def run_inference(model, device, dataset_dir, input_preprocessing, label_divisor, batch_size, collect_masks=False):
    """Run the checkpoint on every tile in dataset_dir and group raw (unfiltered,
    tile-local) predictions by source plot - step 1+3 (stitching) of the pipeline.

    collect_masks: also keep a box-cropped mask patch per detection (for
    --visualization_style masks). Off by default so the box-only path (used for
    all metrics regardless of visualization style) pays no extra cost.
    """
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(dataset_dir, "masks", "*.tif")))
    if not image_paths:
        raise SystemExit(f"No tiles found in {dataset_dir}/images")
    if len(image_paths) != len(mask_paths):
        raise SystemExit(f"image/mask count mismatch in {dataset_dir}")

    dataset = StackedImageInstanceMaskDataset(image_paths, mask_paths, label_divisor=label_divisor,
                                              input_mode=input_preprocessing)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    by_plot = {}
    idx = 0
    with torch.no_grad():
        for images, _targets in loader:
            outs = model([img.to(device) for img in images])
            for out in outs:
                path = image_paths[idx]
                plot_id, y_off, x_off = parse_tile_offset(path)
                if plot_id is None:
                    print(f"  WARNING: could not parse tile offset from {path}, skipping tile")
                    idx += 1
                    continue
                local_boxes = out["boxes"].cpu().numpy()
                boxes = local_boxes.copy()
                if len(boxes):
                    boxes[:, [0, 2]] += x_off
                    boxes[:, [1, 3]] += y_off
                labels = out["labels"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                entry = by_plot.setdefault(plot_id, {"boxes": [], "labels": [], "scores": [], "masks": [], "n_tiles": 0})
                entry["boxes"].append(boxes)
                entry["labels"].append(labels)
                entry["scores"].append(scores)
                if collect_masks:
                    masks_full = out["masks"].cpu().numpy()[:, 0] if out["masks"].shape[0] else []
                    entry["masks"].extend(_crop_mask_to_box(m, b) for m, b in zip(masks_full, local_boxes))
                entry["n_tiles"] += 1
                idx += 1
            if idx % 40 < batch_size:
                print(f"  inference {idx}/{len(image_paths)}")

    for entry in by_plot.values():
        n = sum(len(b) for b in entry["boxes"])
        entry["boxes"] = np.concatenate(entry["boxes"], axis=0) if entry["boxes"] else np.zeros((0, 4), dtype=np.float32)
        entry["labels"] = np.concatenate(entry["labels"], axis=0) if entry["labels"] else np.zeros((0,), dtype=np.int64)
        entry["scores"] = np.concatenate(entry["scores"], axis=0) if entry["scores"] else np.zeros((0,), dtype=np.float32)
        entry["masks"] = entry["masks"] if collect_masks else None
        if collect_masks:
            assert len(entry["masks"]) == n, "mask/box count mismatch while stitching"
    return by_plot


def _to_dict(boxes, labels, scores=None):
    d = {"boxes": torch.as_tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4),
         "labels": torch.as_tensor(np.asarray(labels), dtype=torch.int64)}
    if scores is not None:
        d["scores"] = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    return d


def evaluate_plots(plot_records, score_threshold, iou_threshold, num_classes):
    """Pooled/per-class, confusion, and detection-vs-grading metrics across all
    plots. compute_detection_metrics/compute_grading_metrics_on_detections (from
    train.py) match every sample in the list against every other sample's ground
    truth when given more than one sample at once - calling them once per plot
    (a single-sample list) here avoids that and pools the per-plot results by
    hand instead. evaluate_test_set.py's pooled_prf/confusion/class_agnostic_ap
    already match per-entry safely with an arbitrary-length list, so those are
    called once over every plot's entry.
    """
    cached = [{
        "plot_id": rec["plot_id"], "height": rec["height"], "width": rec["width"],
        "pred_boxes": rec["pred_boxes"], "pred_labels": rec["pred_labels"], "pred_scores": rec["pred_scores"],
        "gt_boxes": rec["gt_boxes"], "gt_labels": rec["gt_labels"],
    } for rec in plot_records]

    det_tp = det_fp = det_fn = 0
    grading_counts = {c: {"correct": 0, "predicted": 0, "actual": 0} for c in range(1, num_classes)}
    for rec in plot_records:
        pred_dict = _to_dict(rec["pred_boxes"], rec["pred_labels"], rec["pred_scores"])
        gt_dict = _to_dict(rec["gt_boxes"], rec["gt_labels"])

        det = compute_detection_metrics([pred_dict], [gt_dict], score_threshold, iou_threshold)
        det_tp += det["tp"]; det_fp += det["fp"]; det_fn += det["fn"]

        grading = compute_grading_metrics_on_detections([pred_dict], [gt_dict], score_threshold,
                                                         iou_threshold, num_classes)
        for cls_key, m in grading.items():
            c = int(cls_key.split("_")[1])
            grading_counts[c]["correct"] += m["correct"]
            grading_counts[c]["predicted"] += m["predicted"]
            grading_counts[c]["actual"] += m["actual"]

    counts = ets.pooled_prf(cached, score_threshold, iou_threshold, num_classes)
    per_class, macro = {}, []
    for cls in range(1, num_classes):
        tp, fp, fn = counts[cls]
        p, r, f = ets.prf(tp, fp, fn)
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}
        macro.append((p, r, f))
    pooled_tp = sum(v[0] for v in counts.values())
    pooled_fp = sum(v[1] for v in counts.values())
    pooled_fn = sum(v[2] for v in counts.values())
    pooled_p, pooled_r, pooled_f = ets.prf(pooled_tp, pooled_fp, pooled_fn)
    macro_p = float(np.mean([m[0] for m in macro])) if macro else 0.0
    macro_r = float(np.mean([m[1] for m in macro])) if macro else 0.0
    macro_f = float(np.mean([m[2] for m in macro])) if macro else 0.0

    threshold_sweep = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c = ets.pooled_prf(cached, thr, iou_threshold, num_classes)
        tp = sum(v[0] for v in c.values()); fp = sum(v[1] for v in c.values()); fn = sum(v[2] for v in c.values())
        p, r, f = ets.prf(tp, fp, fn)
        threshold_sweep.append({"threshold": thr, "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn})

    grading_metrics, g_macro = {}, []
    for c, v in grading_counts.items():
        if v["predicted"] == 0 and v["actual"] == 0:
            continue
        p = v["correct"] / v["predicted"] if v["predicted"] else 0.0
        r = v["correct"] / v["actual"] if v["actual"] else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        grading_metrics[c] = {"precision": p, "recall": r, "f1": f, **v}
        g_macro.append((p, r, f))
    grading_macro = {
        "precision": float(np.mean([m[0] for m in g_macro])) if g_macro else 0.0,
        "recall": float(np.mean([m[1] for m in g_macro])) if g_macro else 0.0,
        "f1": float(np.mean([m[2] for m in g_macro])) if g_macro else 0.0,
    }
    det_p, det_r, det_f = ets.prf(det_tp, det_fp, det_fn)
    confusion = ets.confusion(cached, score_threshold, iou_threshold, num_classes)

    return {
        "cached": cached,
        "per_class": per_class,
        "pooled": {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn,
                   "precision": pooled_p, "recall": pooled_r, "f1": pooled_f},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "confusion_matrix": confusion,
        "threshold_sweep": threshold_sweep,
        "detection": {"tp": det_tp, "fp": det_fp, "fn": det_fn,
                     "precision": det_p, "recall": det_r, "f1": det_f},
        "grading": grading_metrics,
        "grading_macro": grading_macro,
    }


def read_plot_rgb(plot_tif_path, max_dim=2000, percentile=2.0):
    """Bands 1-3, downsampled so large plots stay renderable, contrast-stretched
    across all three bands jointly (per-band stretching re-balances the channels
    and casts the whole tile, per visualize_predictions.py's rgb_for_display).
    Returns (rgb_float01, scale) - multiply GT/pred box coordinates by scale to
    draw them in this image's downsampled pixel space.
    """
    with rasterio.open(plot_tif_path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_dim / max(h, w))
        out_h, out_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
    rgb = rgb.astype(np.float32).transpose(1, 2, 0)
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros_like(rgb), scale
    lo, hi = np.percentile(finite, [percentile, 100 - percentile])
    if hi <= lo:
        return np.zeros_like(rgb), scale
    return np.nan_to_num(np.clip((rgb - lo) / (hi - lo), 0, 1)), scale


def overlay_mask_patches(ax, boxes_scaled, masks, colors, alpha=0.35):
    """Tint each prediction's cropped mask patch and outline it, placed at its own
    box's location via imshow's `extent` - matplotlib handles the up/downscale from
    the patch's native (tile-cropped) resolution to the display's scale, so no
    manual resizing is needed. Mirrors visualize_predictions.py's overlay_masks,
    just per-patch instead of one shared full-image canvas (patches here can be
    fewer and smaller than a full plot, so a shared canvas would waste memory)."""
    for box, mask, color in zip(boxes_scaled, masks, colors):
        if mask is None or mask.size == 0 or not mask.any():
            continue
        x1, y1, x2, y2 = box
        rgb = matplotlib.colors.to_rgb(color)
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        sel = mask > 0
        rgba[sel, :3] = rgb
        rgba[sel, 3] = alpha
        ax.imshow(rgba, extent=(x1, x2, y2, y1), interpolation="nearest", zorder=2.5)
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.9,
                        alpha=0.95, extent=(x1, x2, y2, y1), zorder=2.6)
        cs.set_path_effects(viz._halo(0.9))


def render_plot_visualization(plot_id, rgb, scale, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                              iou_threshold, class_names, out_path, pred_masks=None, style="boxes"):
    """Step 9: one PNG per plot, styled identically to visualize_predictions.py -
    same CLASS_COLORS (keyed by class_id, so a 3-tier checkpoint's class 3
    'severe' renders in the old scheme's 'moderate' orange), same match-status
    line styles (solid=correct, dash-dot=wrong class, dashed=unmatched).

    style: 'boxes' (default) draws box outlines only, as before. 'masks' also
    tints each kept prediction's instance mask over the crown (ground truth
    stays box-only either way - GT has no predicted mask to show).
    """
    order = np.argsort(-pred_scores) if len(pred_scores) else np.array([], dtype=int)
    pred_boxes, pred_labels, pred_scores = pred_boxes[order], pred_labels[order], pred_scores[order]
    if style == "masks" and pred_masks is not None:
        pred_masks = [pred_masks[i] for i in order]
    gt_status, pred_status = viz.match(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)

    h, w = rgb.shape[:2]
    fig_w = 13.0
    fig_h = max(4.0, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb)
    if style == "masks" and pred_masks is not None:
        colors = [viz.CLASS_COLORS.get(int(l), "#888888") for l in pred_labels]
        overlay_mask_patches(ax, pred_boxes * scale, pred_masks, colors)
    viz.draw_boxes(ax, gt_boxes * scale, gt_labels, gt_status, class_names=class_names, class_colors=viz.CLASS_COLORS)
    viz.draw_boxes(ax, pred_boxes * scale, pred_labels, pred_status, scores=pred_scores,
                   class_names=class_names, class_colors=viz.CLASS_COLORS)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")

    n_ok = sum(1 for s in pred_status if s == "ok")
    n_wrong = sum(1 for s in pred_status if s.startswith("wrong"))
    n_fp = sum(1 for s in pred_status if s == "fp")
    n_missed = sum(1 for s in gt_status if s == "missed")
    ax.set_title(f"{plot_id} — {len(gt_boxes)} GT crowns, {len(pred_boxes)} kept predictions | "
                f"{n_ok} correct · {n_wrong} wrong class · {n_fp} false positive · {n_missed} missed",
                fontsize=14, loc="left", fontweight="bold")

    # Legend sits BELOW the image rather than on top of it: the canopy fills the whole
    # frame, so an in-frame legend big enough to read would hide real trees. Colour
    # names are spelled out ("healthy (cyan)") so the swatch never has to be matched
    # by eye, and the line styles are described in words rather than shown bare.
    handles = [Patch(facecolor=viz.CLASS_COLORS[c], edgecolor="#000000", linewidth=0.8,
                     label=f"{class_names[c]}  ({viz.CLASS_COLOR_NAMES.get(c, 'colour')})")
               for c in sorted(class_names)]
    handles += [
        Line2D([0], [0], color="#666666", lw=3.0, ls=viz.STYLE_OK,
               label="solid box = found, class correct"),
        Line2D([0], [0], color="#666666", lw=3.0, ls=viz.STYLE_WRONG,
               label="dash-dot box = found, wrong class"),
        Line2D([0], [0], color="#999999", lw=3.0, ls=viz.STYLE_UNMATCHED,
               label="white dashed box = missed tree / false positive"),
    ]
    # ncol=2 with three entries each: matplotlib fills column-wise, so the health
    # classes land in one column and the box-style meanings in the other. ncol=3
    # splits them across columns and reads as a jumble.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.012),
              ncol=2, fontsize=15, title="How to read this image",
              title_fontsize=16, frameon=True, framealpha=0.95,
              handlelength=2.8, handleheight=1.6, borderpad=0.9,
              labelspacing=0.7, columnspacing=2.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_stitched_eval.ini")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True, help="Sliced TEST tiles dir (has images/, masks/)")
    parser.add_argument("--cropped_plots_dir", required=True, help="Unsliced source plot GeoTIFFs (RGB for viz)")
    parser.add_argument("--plot_masks_dir", required=True, help="Plot-level GT masks (create_masks.py output)")
    parser.add_argument("--input_preprocessing", required=True, choices=["native", "scaled_uint8"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    parser.add_argument("--nms_iou_threshold", type=float, default=None)
    parser.add_argument("--min_crown_side_px", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--visualization_style", choices=["boxes", "masks"], default=None,
                        help="'boxes' (default) draws box outlines only; 'masks' also tints each "
                             "kept prediction's instance mask over the crown (ground truth is "
                             "always box-only). Config key: VISUALIZATION_STYLE.")
    parser.add_argument("--skip_visualizations", action="store_true")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    S = "STITCHED_EVAL"

    def cfg(key, cast, default):
        return cast(config.get(S, key, fallback=str(default)))

    score_threshold = args.score_threshold if args.score_threshold is not None else cfg("SCORE_THRESHOLD", float, 0.5)
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else cfg("IOU_THRESHOLD", float, 0.5)
    nms_iou_threshold = args.nms_iou_threshold if args.nms_iou_threshold is not None else cfg("NMS_IOU_THRESHOLD", float, 0.3)
    min_crown_side_px = args.min_crown_side_px if args.min_crown_side_px is not None else cfg("MIN_CROWN_SIDE_PX", float, 20)
    batch_size = args.batch_size if args.batch_size is not None else cfg("BATCH_SIZE", int, 1)
    visualization_style = args.visualization_style or cfg("VISUALIZATION_STYLE", str, "boxes")
    output_dir = args.output_dir or cfg("OUTPUT_DIR", str, "results/stitched_metrics/default_run")
    os.makedirs(output_dir, exist_ok=True)
    if not args.skip_visualizations:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = ckpt.get("arch") or {}
    if not arch:
        raise SystemExit(f"{args.checkpoint} has no stored 'arch' dict - too old a checkpoint for this script.")
    num_classes = arch["num_classes"]
    in_channels = arch["in_channels"]
    label_divisor = arch.get("mask_label_divisor", 10000)
    anchor_sizes = arch.get("anchor_sizes")
    anchor_ratios = arch.get("anchor_aspect_ratios")
    image_mean = arch.get("image_mean")
    image_std = arch.get("image_std")
    class_names = ets.class_names_for(num_classes)

    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"num_classes={num_classes} in_channels={in_channels} label_divisor={label_divisor}")
    print(f"anchor_sizes={anchor_sizes} anchor_aspect_ratios={anchor_ratios}")
    print(f"image_mean={image_mean} image_std={image_std}")
    print(f"input_preprocessing={args.input_preprocessing}")
    print(f"score_threshold={score_threshold} iou_threshold={iou_threshold} "
          f"nms_iou_threshold={nms_iou_threshold} min_crown_side_px={min_crown_side_px}")
    print(f"visualization_style={visualization_style}")

    model = get_multiband_maskrcnn(num_classes=num_classes, in_channels=in_channels,
                                   anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_ratios,
                                   image_mean=image_mean, image_std=image_std)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    collect_masks = (visualization_style == "masks") and not args.skip_visualizations
    print("\nRunning inference and stitching tiles into plot coordinates...")
    by_plot = run_inference(model, device, args.dataset_dir, args.input_preprocessing, label_divisor, batch_size,
                            collect_masks=collect_masks)
    print(f"{len(by_plot)} plots, {sum(e['n_tiles'] for e in by_plot.values())} tiles total")

    plot_records = []
    for plot_id in sorted(by_plot):
        entry = by_plot[plot_id]
        mask_path = os.path.join(args.plot_masks_dir, f"{plot_id}_mask.tif")
        if not os.path.exists(mask_path):
            print(f"  WARNING: no GT mask for {plot_id} at {mask_path}, skipping plot")
            continue
        gt_boxes, gt_labels, height, width = load_plot_gt(mask_path, label_divisor)

        n_raw = len(entry["boxes"])
        boxes, labels, scores, masks = class_agnostic_nms_and_size_filter(
            entry["boxes"], entry["labels"], entry["scores"], nms_iou_threshold, min_crown_side_px,
            masks=entry["masks"])

        plot_records.append({
            "plot_id": plot_id, "height": height, "width": width,
            "pred_boxes": boxes, "pred_labels": labels, "pred_scores": scores, "pred_masks": masks,
            "gt_boxes": gt_boxes, "gt_labels": gt_labels,
            "n_tiles": entry["n_tiles"], "n_pred_raw": n_raw, "n_pred_kept": len(boxes),
        })
        print(f"  {plot_id}: {entry['n_tiles']} tiles, {n_raw} raw detections -> {len(boxes)} after "
              f"NMS+size filter, {len(gt_boxes)} GT crowns")

    if not plot_records:
        raise SystemExit("No plots had both predictions and a matching GT mask - nothing to evaluate.")

    print("\nComputing metrics...")
    results = evaluate_plots(plot_records, score_threshold, iou_threshold, num_classes)
    ap = ets.class_agnostic_ap(results["cached"], output_dir)

    print("\n" + "=" * 72)
    print(f"PER-CLASS (stitched plots, score>={score_threshold}, IoU>={iou_threshold})")
    print("=" * 72)
    print(f"  {'class':<12} {'tp':>5} {'fp':>5} {'fn':>5} {'precision':>10} {'recall':>8} {'F1':>8}")
    for cls, v in results["per_class"].items():
        print(f"  {class_names.get(cls, cls):<12} {v['tp']:>5} {v['fp']:>5} {v['fn']:>5} "
              f"{v['precision']:>10.3f} {v['recall']:>8.3f} {v['f1']:>8.3f}")
    p = results["pooled"]
    print(f"  {'POOLED':<12} {p['tp']:>5} {p['fp']:>5} {p['fn']:>5} {p['precision']:>10.3f} {p['recall']:>8.3f} {p['f1']:>8.3f}")
    m = results["macro"]
    print(f"  {'MACRO avg':<12} {'':>5} {'':>5} {'':>5} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")
    d = results["detection"]
    print(f"\nDetection (class-agnostic): tp={d['tp']} fp={d['fp']} fn={d['fn']} "
          f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")
    gm = results["grading_macro"]
    print(f"Grading (on detections, macro): P={gm['precision']:.3f} R={gm['recall']:.3f} F1={gm['f1']:.3f}")

    save_confusion_matrix(results["confusion_matrix"], os.path.join(output_dir, "confusion_matrix.csv"))

    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "dataset_dir": args.dataset_dir,
        "cropped_plots_dir": args.cropped_plots_dir,
        "plot_masks_dir": args.plot_masks_dir,
        "input_preprocessing": args.input_preprocessing,
        "num_classes": num_classes,
        "n_plots": len(plot_records),
        "n_tiles": sum(r["n_tiles"] for r in plot_records),
        "n_gt_instances": int(sum(len(r["gt_boxes"]) for r in plot_records)),
        "n_pred_raw": int(sum(r["n_pred_raw"] for r in plot_records)),
        "n_pred_kept": int(sum(r["n_pred_kept"] for r in plot_records)),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "min_crown_side_px": min_crown_side_px,
        "per_class": {class_names.get(c, str(c)): v for c, v in results["per_class"].items()},
        "pooled": results["pooled"],
        "macro": results["macro"],
        "class_agnostic_ap": ap,
        "threshold_sweep": results["threshold_sweep"],
        "detection": results["detection"],
        "grading": {class_names.get(c, str(c)): v for c, v in results["grading"].items()},
        "grading_macro": results["grading_macro"],
        "per_plot": [{"plot_id": r["plot_id"], "n_tiles": r["n_tiles"], "n_gt": len(r["gt_boxes"]),
                     "n_pred_raw": r["n_pred_raw"], "n_pred_kept": r["n_pred_kept"]} for r in plot_records],
    }
    with open(os.path.join(output_dir, "stitched_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "stitched_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "class", "tp", "fp", "fn", "correct", "predicted", "actual",
                    "precision", "recall", "f1"])
        for cls, v in results["per_class"].items():
            w.writerow(["complete", class_names.get(cls, str(cls)), v["tp"], v["fp"], v["fn"], "", "", "",
                        f'{v["precision"]:.6f}', f'{v["recall"]:.6f}', f'{v["f1"]:.6f}'])
        w.writerow(["complete", "POOLED", p["tp"], p["fp"], p["fn"], "", "", "",
                    f'{p["precision"]:.6f}', f'{p["recall"]:.6f}', f'{p["f1"]:.6f}'])
        w.writerow(["complete", "MACRO", "", "", "", "", "", "",
                    f'{m["precision"]:.6f}', f'{m["recall"]:.6f}', f'{m["f1"]:.6f}'])
        w.writerow(["detection", "ALL", d["tp"], d["fp"], d["fn"], "", "", "",
                    f'{d["precision"]:.6f}', f'{d["recall"]:.6f}', f'{d["f1"]:.6f}'])
        for cls, v in results["grading"].items():
            w.writerow(["grading", class_names.get(cls, str(cls)), "", "", "",
                        v["correct"], v["predicted"], v["actual"],
                        f'{v["precision"]:.6f}', f'{v["recall"]:.6f}', f'{v["f1"]:.6f}'])
        w.writerow(["grading", "MACRO", "", "", "", "", "", "",
                    f'{gm["precision"]:.6f}', f'{gm["recall"]:.6f}', f'{gm["f1"]:.6f}'])
        if ap:
            w.writerow(["class_agnostic_ap", "AP50", "", "", "", "", "", "", "", "", f'{ap["AP50"]:.6f}'])
            w.writerow(["class_agnostic_ap", "AR100", "", "", "", "", "", "", "", "", f'{ap["AR100"]:.6f}'])

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {os.path.join(output_dir, 'stitched_metrics.json')}")
    print(f"Saved -> {os.path.join(output_dir, 'confusion_matrix.csv')} (+ _heatmap.png)")

    if not args.skip_visualizations:
        print("\nRendering per-plot visualizations...")
        viz_dir = os.path.join(output_dir, "visualizations")
        for rec in plot_records:
            plot_tif = os.path.join(args.cropped_plots_dir, f"{rec['plot_id']}.tif")
            if not os.path.exists(plot_tif):
                print(f"  WARNING: no source plot image for {rec['plot_id']} at {plot_tif}, skipping visualization")
                continue
            keep = rec["pred_scores"] >= score_threshold
            kept_masks = [m for m, k in zip(rec["pred_masks"], keep) if k] if rec["pred_masks"] is not None else None
            rgb, scale = read_plot_rgb(plot_tif)
            out_path = os.path.join(viz_dir, f"{rec['plot_id']}.png")
            render_plot_visualization(
                rec["plot_id"], rgb, scale, rec["gt_boxes"], rec["gt_labels"],
                rec["pred_boxes"][keep], rec["pred_labels"][keep], rec["pred_scores"][keep],
                iou_threshold, class_names, out_path, pred_masks=kept_masks, style=visualization_style)
            print(f"  {rec['plot_id']} -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
