"""Render model predictions against ground truth for a handful of tiles.

Numbers say a model is wrong; a picture says how. This draws each tile twice -
ground truth on the left, predictions on the right - and marks the three error
types the metrics count separately:

    missed crown   GT with no matching prediction   (white dashed outline)
    false positive prediction matching no GT        (white dashed outline)
    wrong class    boxes match, health class differs (magenta outline + arrow label)

Usage (from repo root):
    python scripts/visualize_predictions.py
    python scripts/visualize_predictions.py --num_tiles 10 --score_threshold 0.6
    python scripts/visualize_predictions.py --checkpoint backup_run3_measured_rgb.pth \
        --image_mean 0.4664,0.4707,0.3764,10.8663,0.5266,0.3391,0.5179,0.1435 \
        --image_std  0.1232,0.1292,0.1302,8.3472,0.1108,0.0955,0.0751,0.0347

Normalization and anchors are read from config.ini and printed. They must match
the run that produced the checkpoint: they live in the model's transform, not its
state_dict, so a mismatch raises no error and silently yields nonsense.
"""
import argparse
import configparser
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import get_multiband_maskrcnn, collate_fn, compute_iou, plot_id_from_tile_path, split_paths_by_plot
from dataset.multi_channel_dataset import StackedImageInstanceMaskDataset
from torch.utils.data import DataLoader

# Overlay colours are chosen against the imagery, not against a page. The canopy is
# green, tan and brown, so the ramp runs cool -> warm through hues that do not occur
# in the scene; an earth-toned ramp disappears into the tile. Checked pairwise under
# simulated protan/deutan vision: worst pair dE 11.6 (target >= 8) and 18.6 under
# normal vision (floor 15). Every box is also labelled, so colour never carries the
# class alone.
CLASS_NAMES = {1: "healthy", 2: "mild", 3: "moderate", 4: "severe"}
CLASS_COLORS = {1: "#00E5FF", 2: "#B14DFF", 3: "#FF8A00", 4: "#FF1744"}
UNMATCHED = "#FFFFFF"

# Line style separates the match states; colour separates matched from unmatched.
STYLE_OK = "solid"
STYLE_WRONG = (0, (6, 2, 1, 2))   # dash-dot: right box, wrong health class
STYLE_UNMATCHED = (0, (4, 3))     # dashed: missed crown or false positive


def _halo(linewidth):
    """A dark stroke under every line, so it stays visible on bright canopy as well
    as on shadow. Without it a light outline vanishes over pale crowns."""
    return [path_effects.withStroke(linewidth=linewidth + 1.8, foreground="#000000")]


def rgb_for_display(image_tensor, percentile=2.0):
    """Take bands 1-3 as RGB and stretch them for viewing.

    The stretch uses one set of limits across all three bands, not per-band.
    Stretching each band to its own range re-balances the channels and casts the
    whole tile - foliage came out purple - which makes it harder, not easier, to
    judge whether a box sits on a crown.
    """
    rgb = image_tensor[:3].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros_like(rgb)
    lo, hi = np.percentile(finite, [percentile, 100 - percentile])
    if hi <= lo:
        return np.zeros_like(rgb)
    return np.nan_to_num(np.clip((rgb - lo) / (hi - lo), 0, 1))


def overlay_masks(ax, masks, labels, colors, alpha=0.30):
    """Tint each instance's mask and outline it, so crown extent - not just the
    box - is visible. The outline carries most of the signal: a light tint alone
    washes out against bright canopy."""
    if len(masks) == 0:
        return
    h, w = masks[0].shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for mask, color in zip(masks, colors):
        rgb = matplotlib.colors.to_rgb(color)
        sel = mask > 0
        # Later instances paint over earlier ones; crowns rarely overlap much.
        rgba[sel, :3] = rgb
        rgba[sel, 3] = alpha
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=[color],
                        linewidths=0.9, alpha=0.95)
        cs.set_path_effects(_halo(0.9))
    ax.imshow(rgba, interpolation="nearest")


def draw_boxes(ax, boxes, labels, statuses, scores=None):
    """statuses: 'ok' | 'missed' | 'fp' | 'wrong:<true_class>'.

    Labels are drawn as small stroked text rather than filled chips: a filled
    background on every box covered more of the tile than the boxes did, which
    defeats the point of looking at the imagery.
    """
    stroke = [path_effects.withStroke(linewidth=2.0, foreground="#000000")]
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        status = statuses[i]
        if status == "ok":
            edge, lw, ls = CLASS_COLORS.get(int(label), "#BBBBBB"), 1.7, STYLE_OK
        elif status.startswith("wrong"):
            edge, lw, ls = CLASS_COLORS.get(int(label), "#BBBBBB"), 2.0, STYLE_WRONG
        else:  # missed or fp
            edge, lw, ls = UNMATCHED, 1.7, STYLE_UNMATCHED

        patch = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                          edgecolor=edge, linewidth=lw, linestyle=ls)
        patch.set_path_effects(_halo(lw))
        ax.add_patch(patch)

        text = CLASS_NAMES.get(int(label), str(label))
        if scores is not None:
            text += f" {scores[i]:.2f}"
        if status.startswith("wrong"):
            text += f" ←{CLASS_NAMES.get(int(status.split(':')[1]), '?')}"
        elif status == "fp":
            text += " no GT"
        elif status == "missed":
            text += " missed"

        ax.text(x1 + 2, max(y1 - 4, 11), text, fontsize=6.0, color=edge,
                path_effects=stroke, va="bottom", fontweight="bold")


def match(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold):
    """Greedy highest-score-first matching, identical to the metric's."""
    gt_status = ["missed"] * len(gt_boxes)
    pred_status = []
    gt_used = [False] * len(gt_boxes)
    for box, label in zip(pred_boxes, pred_labels):
        best_iou, best = 0.0, None
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou, best = iou, j
        if best is not None and best_iou >= iou_threshold:
            gt_used[best] = True
            if int(gt_labels[best]) == int(label):
                pred_status.append("ok")
                gt_status[best] = "ok"
            else:
                pred_status.append(f"wrong:{int(gt_labels[best])}")
                gt_status[best] = f"wrong:{int(label)}"
        else:
            pred_status.append("fp")
    return gt_status, pred_status


def visualize_predictions(checkpoint, dataset_dir, out_path, num_tiles=10, split="val",
                          score_threshold=0.5, iou_threshold=0.5, config=None,
                          image_mean=None, image_std=None,
                          anchor_sizes_override=None, anchor_ratios_override=None):
    """Render `num_tiles` tiles as ground-truth / prediction pairs into one PNG."""
    config = config or configparser.ConfigParser()
    num_classes = int(config.get("TRAIN", "num_classes", fallback="5"))
    in_channels = int(config.get("TRAIN", "num_input_channels", fallback="8"))
    label_divisor = int(config.get("TRAIN", "mask_label_divisor", fallback="10000"))
    train_val_split = float(config.get("TRAIN", "train_val_split", fallback="0.8"))

    raw = config.get("TRAIN", "anchor_sizes", fallback="").strip()
    if ";" in raw:
        anchor_sizes = [[int(s) for s in lv.split(",") if s.strip()]
                        for lv in raw.split(";") if lv.strip()] or None
    else:
        anchor_sizes = [int(s) for s in raw.split(",") if s.strip()] or None
    anchor_ratios = [float(r) for r in config.get("TRAIN", "anchor_aspect_ratios", fallback="").split(",")
                     if r.strip()] or None

    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(dataset_dir, "masks", "*.tif")))
    if not image_paths:
        raise SystemExit(f"No tiles found in {dataset_dir}/images")
    if split == "val":
        _, image_paths, _, mask_paths = split_paths_by_plot(
            image_paths, mask_paths, train_size=train_val_split, random_state=42)

    # Spread the sample across the split rather than taking the first N, so the
    # picture covers several plots instead of one corner of one plot.
    n = min(num_tiles, len(image_paths))
    idx = np.linspace(0, len(image_paths) - 1, n).round().astype(int)
    image_paths = [image_paths[i] for i in idx]
    mask_paths = [mask_paths[i] for i in idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)

    # Prefer the architecture the checkpoint was trained with over whatever
    # config.ini says now; an explicit CLI override still wins.
    # Precedence: the architecture stored in the checkpoint is a fact about that
    # checkpoint, so it always wins. The LEGACY_* config keys exist only for
    # checkpoints saved before it was recorded; [TRAIN] is the last resort.
    arch = ckpt.get("arch") or {}
    if arch:
        source = "checkpoint"
        num_classes = arch.get("num_classes", num_classes)
        in_channels = arch.get("in_channels", in_channels)
        label_divisor = arch.get("mask_label_divisor", label_divisor)
        anchor_sizes = arch.get("anchor_sizes", anchor_sizes)
        anchor_ratios = arch.get("anchor_aspect_ratios", anchor_ratios)
        image_mean = arch.get("image_mean", image_mean)
        image_std = arch.get("image_std", image_std)
    else:
        source = "config.ini"
        if anchor_sizes_override is not None:
            anchor_sizes = anchor_sizes_override
            source = "config.ini LEGACY_*"
        if anchor_ratios_override is not None:
            anchor_ratios = anchor_ratios_override
        if image_mean is None:
            # No legacy override given - fall back to the current [TRAIN] values.
            image_mean = [float(v) for v in config.get("TRAIN", "image_mean", fallback="").split(",")
                          if v.strip()] or None
            image_std = [float(v) for v in config.get("TRAIN", "image_std", fallback="").split(",")
                         if v.strip()] or None

    model = get_multiband_maskrcnn(num_classes=num_classes, in_channels=in_channels,
                                   anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_ratios,
                                   image_mean=image_mean, image_std=image_std)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Checkpoint: {checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f})")
    print(f"Arch from:  {source}")
    print(f"Anchors:    sizes={anchor_sizes} ratios={anchor_ratios}")
    print(f"Norm mean:  {image_mean}")
    print(f"Norm std:   {image_std}\n")

    dataset = StackedImageInstanceMaskDataset(image_paths, mask_paths, label_divisor=label_divisor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    fig, axes = plt.subplots(n, 2, figsize=(13, 6.2 * n), squeeze=False)
    totals = {"ok": 0, "missed": 0, "fp": 0, "wrong": 0}

    with torch.no_grad():
        for row, (images, targets) in enumerate(loader):
            image = images[0]
            target = targets[0]
            out = model([image.to(device)])[0]

            keep = out["scores"].cpu().numpy() >= score_threshold
            pb = out["boxes"].cpu().numpy()[keep]
            pl = out["labels"].cpu().numpy()[keep]
            ps = out["scores"].cpu().numpy()[keep]
            pm = out["masks"].cpu().numpy()[keep, 0] if out["masks"].shape[0] else np.zeros((0, 1, 1))
            order = np.argsort(-ps)
            pb, pl, ps = pb[order], pl[order], ps[order]
            pm = pm[order] if len(pm) else pm

            gb = target["boxes"].numpy()
            gl = target["labels"].numpy()
            gm = target["masks"].numpy()

            gt_status, pred_status = match(pb, pl, gb, gl, iou_threshold)
            totals["ok"] += sum(1 for s in pred_status if s == "ok")
            totals["fp"] += sum(1 for s in pred_status if s == "fp")
            totals["wrong"] += sum(1 for s in pred_status if s.startswith("wrong"))
            totals["missed"] += sum(1 for s in gt_status if s == "missed")

            rgb = rgb_for_display(image)
            tile = os.path.basename(image_paths[row])

            ax = axes[row][0]
            ax.imshow(rgb)
            overlay_masks(ax, list(gm), gl, [CLASS_COLORS.get(int(l), "#888") for l in gl])
            draw_boxes(ax, gb, gl, gt_status)
            n_missed = sum(1 for s in gt_status if s == "missed")
            ax.set_title(f"{tile}\nground truth — {len(gb)} crowns, {n_missed} not found",
                         fontsize=9, loc="left")

            ax = axes[row][1]
            ax.imshow(rgb)
            if len(pm):
                overlay_masks(ax, [(m > 0.5).astype(np.uint8) for m in pm], pl,
                              [CLASS_COLORS.get(int(l), "#888") for l in pl])
            draw_boxes(ax, pb, pl, pred_status, scores=ps)
            n_fp = sum(1 for s in pred_status if s == "fp")
            n_wrong = sum(1 for s in pred_status if s.startswith("wrong"))
            ax.set_title(f"predictions @ score ≥ {score_threshold} — {len(pb)} boxes, "
                         f"{n_fp} without GT, {n_wrong} wrong class", fontsize=9, loc="left")

            for a in axes[row]:
                a.set_xticks([]); a.set_yticks([])
                for spine in a.spines.values():
                    spine.set_edgecolor("#CCCCCC")

    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="none", label=CLASS_NAMES[c])
               for c in sorted(CLASS_NAMES)]
    handles += [
        Line2D([0], [0], color="#444444", lw=2, ls=STYLE_UNMATCHED,
               label="unmatched — missed crown / false positive"),
        Line2D([0], [0], color="#444444", lw=2, ls=STYLE_WRONG,
               label="matched, wrong health class"),
    ]

    # Reserve a fixed strip in inches for the header, then convert to a figure
    # fraction. Using a fraction directly makes the header shrink as the figure
    # grows with tile count, which is what collided the title and legend.
    header_in = 1.05
    top = 1.0 - header_in / fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.suptitle(
        f"{os.path.basename(checkpoint)} — {n} {split} tiles · "
        f"{totals['ok']} correct, {totals['wrong']} wrong class, "
        f"{totals['fp']} false positives, {totals['missed']} missed",
        fontsize=12, y=1.0 - 0.22 / fig.get_figheight(), va="top")
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0 - 0.60 / fig.get_figheight()))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"Across {n} tiles: {totals['ok']} correct · {totals['wrong']} wrong class · "
          f"{totals['fp']} false positives · {totals['missed']} missed")
    print(f"Saved -> {out_path}")
    return out_path


def main():
    config = configparser.ConfigParser()
    config.read(os.environ.get("MASKRCNN_CONFIG", "config.ini"))
    if not config.has_section("TRAIN"):
        raise SystemExit("config.ini with a [TRAIN] section not found - run from the repo root.")

    # Settings live in config.ini's [VISUALIZE] section. The flags below only
    # override it for a one-off; nothing here has to be passed for a normal run.
    S = "VISUALIZE"
    parser = argparse.ArgumentParser(
        description="Render ground truth beside predictions. Reads [VISUALIZE] from config.ini.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--split", choices=["val", "all"], default=None)
    parser.add_argument("--num_tiles", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    def cfg(key, fallback=""):
        return config.get(S, key, fallback=fallback).strip()

    def legacy_floats(key):
        raw = cfg(key)
        return [float(v) for v in raw.split(",") if v.strip()] or None

    def legacy_anchors(key):
        raw = cfg(key)
        if not raw:
            return None
        if ";" in raw:
            return [[int(s) for s in lv.split(",") if s.strip()]
                    for lv in raw.split(";") if lv.strip()]
        return [int(s) for s in raw.split(",") if s.strip()]

    checkpoint = args.checkpoint or cfg("CHECKPOINT") or os.path.join(
        config.get("TRAIN", "checkpoint_dir", fallback="checkpoints"), "maskrcnn_best.pth")
    split = args.split or cfg("SPLIT", "val")
    dataset_dir = args.dataset_dir or cfg("DATASET_DIR") or config.get(
        "TRAIN", "dataset_dir", fallback="data/dataset_sliced_800")
    out = args.out or cfg("OUT") or os.path.join(
        config.get("TRAIN", "metrics_dir", fallback="checkpoints/metrics"),
        f"predictions_{split}.png")

    visualize_predictions(
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        out_path=out,
        num_tiles=args.num_tiles if args.num_tiles is not None else int(cfg("NUM_TILES", "10")),
        split=split,
        score_threshold=(args.score_threshold if args.score_threshold is not None
                         else float(cfg("SCORE_THRESHOLD", "0.5"))),
        iou_threshold=(args.iou_threshold if args.iou_threshold is not None
                       else float(cfg("IOU_THRESHOLD",
                                      config.get("TRAIN", "iou_threshold", fallback="0.5")))),
        config=config,
        image_mean=legacy_floats("LEGACY_IMAGE_MEAN"),
        image_std=legacy_floats("LEGACY_IMAGE_STD"),
        anchor_sizes_override=legacy_anchors("LEGACY_ANCHOR_SIZES"),
        anchor_ratios_override=legacy_floats("LEGACY_ANCHOR_ASPECT_RATIOS"),
    )


if __name__ == "__main__":
    main()
