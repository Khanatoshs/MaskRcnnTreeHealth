"""Evaluate a trained checkpoint on the held-out test plots.

These plots come from mesh.shp's `class = test` attribute and are never seen during
training or validation, so this is the number to report in the paper.

Usage (from repo root):
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --score_threshold 0.6
    python scripts/evaluate_test_set.py --dataset_dir data/dataset_sliced_800 --split val

Reports, at the chosen score threshold:
  - per-class precision / recall / F1, pooled over the whole split (not averaged
    per tile, which is noisy and weights small tiles equally with dense ones)
  - a confusion matrix (rows = true, cols = predicted, index 0 = background)
  - class-agnostic detection AP, to separate "did we find the tree" from
    "did we get its health class right"
  - a score-threshold sweep, since precision/recall trade off sharply with it
"""
import argparse
import configparser
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (get_multiband_maskrcnn, collate_fn, compute_iou,
                   plot_id_from_tile_path, split_paths_by_plot)
from dataset.multi_channel_dataset import StackedImageInstanceMaskDataset

CLASS_NAMES = {1: "healthy", 2: "mild", 3: "moderate", 4: "severe"}


def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    if not config.has_section("TRAIN"):
        raise SystemExit("config.ini with a [TRAIN] section not found - run from the repo root.")
    return config


def build_loader(dataset_dir, label_divisor, split, train_val_split, num_workers):
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(dataset_dir, "masks", "*.tif")))
    if not image_paths:
        raise SystemExit(f"No tiles found in {dataset_dir}/images")
    if len(image_paths) != len(mask_paths):
        raise SystemExit(f"image/mask count mismatch in {dataset_dir}")

    if split == "val":
        # Reproduce train.py's plot-level split so we score exactly its val plots.
        _, image_paths, _, mask_paths = split_paths_by_plot(
            image_paths, mask_paths, train_size=train_val_split, random_state=42
        )

    plots = sorted({plot_id_from_tile_path(p) for p in image_paths} - {None})
    print(f"Evaluating {len(image_paths)} tiles from {len(plots)} plots: {', '.join(plots)}\n")

    dataset = StackedImageInstanceMaskDataset(image_paths, mask_paths, label_divisor=label_divisor)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                      num_workers=num_workers)


def cache_predictions(model, loader, device):
    """Run inference once, keeping raw (unfiltered) outputs so thresholds can be swept."""
    cached = []
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            out = model([img.to(device) for img in images])[0]
            cached.append({
                "height": int(images[0].shape[-2]),
                "width": int(images[0].shape[-1]),
                "pred_boxes": out["boxes"].cpu().numpy(),
                "pred_labels": out["labels"].cpu().numpy(),
                "pred_scores": out["scores"].cpu().numpy(),
                "gt_boxes": targets[0]["boxes"].numpy(),
                "gt_labels": targets[0]["labels"].numpy(),
            })
            if idx % 25 == 0:
                print(f"  inference {idx}/{len(loader)}")
    return cached


def match_greedy(pred_boxes, gt_boxes, iou_threshold):
    """Greedy highest-score-first IoU matching. pred_boxes must be score-sorted desc."""
    gt_used = [False] * len(gt_boxes)
    matched = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, None
        for gi, gb in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_idx is not None and best_iou >= iou_threshold:
            gt_used[best_idx] = True
            matched += 1
    return matched, len(pred_boxes) - matched, sum(1 for u in gt_used if not u)


def sorted_by_score(entry, score_threshold):
    keep = entry["pred_scores"] >= score_threshold
    boxes, labels, scores = entry["pred_boxes"][keep], entry["pred_labels"][keep], entry["pred_scores"][keep]
    order = np.argsort(-scores)
    return boxes[order], labels[order]


def pooled_prf(cached, score_threshold, iou_threshold, num_classes):
    """Pool tp/fp/fn over the whole split, then compute precision once per class."""
    counts = {c: [0, 0, 0] for c in range(1, num_classes)}
    for entry in cached:
        boxes, labels = sorted_by_score(entry, score_threshold)
        for cls in range(1, num_classes):
            gt_b = entry["gt_boxes"][entry["gt_labels"] == cls]
            pr_b = boxes[labels == cls]
            if len(gt_b) == 0 and len(pr_b) == 0:
                continue
            tp, fp, fn = match_greedy(pr_b, gt_b, iou_threshold)
            counts[cls][0] += tp
            counts[cls][1] += fp
            counts[cls][2] += fn
    return counts


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def confusion(cached, score_threshold, iou_threshold, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for entry in cached:
        boxes, labels = sorted_by_score(entry, score_threshold)
        gt_used = [False] * len(entry["gt_boxes"])
        for b, l in zip(boxes, labels):
            best_iou, best_idx = 0.0, None
            for gi, gb in enumerate(entry["gt_boxes"]):
                if gt_used[gi]:
                    continue
                iou = compute_iou(b, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_idx is not None and best_iou >= iou_threshold:
                gt_used[best_idx] = True
                matrix[int(entry["gt_labels"][best_idx]), int(l)] += 1
            else:
                matrix[0, int(l)] += 1  # predicted something where there is no GT
        for gi, used in enumerate(gt_used):
            if not used:
                matrix[int(entry["gt_labels"][gi]), 0] += 1  # missed detection
    return matrix


def class_agnostic_ap(cached, out_dir):
    """COCO box AP with every class merged into one 'tree' category."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools not installed - skipping class-agnostic AP.")
        return None

    images, annotations, results = [], [], []
    ann_id = 1
    for i, entry in enumerate(cached, start=1):
        images.append({"id": i, "height": entry["height"], "width": entry["width"],
                       "file_name": f"{i}.tif"})
        for b in entry["gt_boxes"]:
            x1, y1, x2, y2 = b.tolist()
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0.0, (x2 - x1) * (y2 - y1)), "iscrowd": 0,
            })
            ann_id += 1
        for b, s in zip(entry["pred_boxes"], entry["pred_scores"]):
            x1, y1, x2, y2 = b.tolist()
            results.append({
                "image_id": i, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s),
            })

    if not annotations or not results:
        print("Not enough boxes for AP computation - skipping.")
        return None

    gt_path = os.path.join(out_dir, "_ca_gt.json")
    dt_path = os.path.join(out_dir, "_ca_dt.json")
    with open(gt_path, "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "tree"}]}, f)
    with open(dt_path, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    os.remove(gt_path); os.remove(dt_path)
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AR100": ev.stats[8]}


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=config.get("SLICING", "TEST_OUTPUT_DIR",
                                           fallback="data/dataset_sliced_800_test"))
    parser.add_argument("--split", choices=["all", "val"], default="all",
                        help="'all' scores every tile in dataset_dir (use for the test set); "
                             "'val' reproduces train.py's plot-level validation split.")
    parser.add_argument("--checkpoint",
                        default=os.path.join(config.get("TRAIN", "checkpoint_dir", fallback="checkpoints"),
                                             "maskrcnn_best.pth"))
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float,
                        default=float(config.get("TRAIN", "iou_threshold", fallback="0.5")))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--image_mean", default=None,
                        help="Comma-separated per-channel means overriding config.ini. Needed when "
                             "evaluating a checkpoint trained under different normalization; pass "
                             "'fallback' for the built-in ImageNet+0.5 defaults.")
    parser.add_argument("--image_std", default=None,
                        help="Comma-separated per-channel stds overriding config.ini.")
    args = parser.parse_args()

    num_classes = int(config.get("TRAIN", "num_classes", fallback="5"))
    in_channels = int(config.get("TRAIN", "num_input_channels", fallback="8"))
    label_divisor = int(config.get("TRAIN", "mask_label_divisor", fallback="10000"))
    train_val_split = float(config.get("TRAIN", "train_val_split", fallback="0.8"))
    num_workers = int(config.get("TRAIN", "num_workers", fallback="2"))
    out_dir = args.out_dir or config.get("TRAIN", "metrics_dir", fallback="checkpoints/metrics")
    os.makedirs(out_dir, exist_ok=True)

    _anchor_raw = config.get("TRAIN", "anchor_sizes", fallback="").strip()
    if ";" in _anchor_raw:
        anchor_sizes = [[int(s) for s in level.split(",") if s.strip()]
                        for level in _anchor_raw.split(";") if level.strip()] or None
    else:
        anchor_sizes = [int(s) for s in _anchor_raw.split(",") if s.strip()] or None
    anchor_ratios = [float(r) for r in config.get("TRAIN", "anchor_aspect_ratios", fallback="").split(",") if r.strip()] or None
    # These MUST match what the checkpoint was trained with. Evaluating with
    # different normalization silently produces garbage predictions rather than
    # an error, because normalization lives in the model's transform, not its
    # state_dict - nothing in load_state_dict can catch the mismatch.
    def _norm(cli_value, config_key):
        if cli_value is not None:
            if cli_value.strip().lower() == "fallback":
                return None
            return [float(v) for v in cli_value.split(",") if v.strip()]
        return [float(v) for v in config.get("TRAIN", config_key, fallback="").split(",") if v.strip()] or None

    image_mean = _norm(args.image_mean, "image_mean")
    image_std = _norm(args.image_std, "image_std")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_multiband_maskrcnn(num_classes=num_classes, in_channels=in_channels,
                                   anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_ratios,
                                   image_mean=image_mean, image_std=image_std)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f})")
    print(f"Dataset:    {args.dataset_dir} (split={args.split})")
    print(f"Anchors:    sizes={anchor_sizes} ratios={anchor_ratios}")
    print(f"Norm mean:  {image_mean}")
    print(f"Norm std:   {image_std}")
    print("            (these come from config.ini and must match the training run "
          "that produced this checkpoint)\n")

    loader = build_loader(args.dataset_dir, label_divisor, args.split, train_val_split, num_workers)
    cached = cache_predictions(model, loader, device)

    n_gt = sum(len(e["gt_labels"]) for e in cached)
    print(f"\n{len(cached)} tiles, {n_gt} ground-truth instances")

    thr, iou = args.score_threshold, args.iou_threshold
    print("\n" + "=" * 72)
    print(f"PER-CLASS (pooled over split, score>={thr}, IoU>={iou})")
    print("=" * 72)
    counts = pooled_prf(cached, thr, iou, num_classes)
    print(f"  {'class':<12} {'tp':>5} {'fp':>5} {'fn':>5} {'precision':>10} {'recall':>8} {'F1':>8}")
    tt = tf = tn = 0
    macro = []
    for cls in sorted(counts):
        tp, fp, fn = counts[cls]
        tt += tp; tf += fp; tn += fn
        p, r, f = prf(tp, fp, fn)
        macro.append((p, r, f))
        print(f"  {CLASS_NAMES.get(cls, cls):<12} {tp:>5} {fp:>5} {fn:>5} {p:>10.3f} {r:>8.3f} {f:>8.3f}")
    p, r, f = prf(tt, tf, tn)
    print(f"  {'POOLED':<12} {tt:>5} {tf:>5} {tn:>5} {p:>10.3f} {r:>8.3f} {f:>8.3f}")
    mp = float(np.mean([m[0] for m in macro])); mr = float(np.mean([m[1] for m in macro])); mf = float(np.mean([m[2] for m in macro]))
    print(f"  {'MACRO avg':<12} {'':>5} {'':>5} {'':>5} {mp:>10.3f} {mr:>8.3f} {mf:>8.3f}")

    print("\n" + "=" * 72)
    print("CONFUSION MATRIX (rows=true, cols=predicted, 0=background)")
    print("=" * 72)
    matrix = confusion(cached, thr, iou, num_classes)
    header = ["bg"] + [CLASS_NAMES.get(c, str(c)) for c in range(1, num_classes)]
    print(f"  {'true \\ pred':<12}" + "".join(f"{h:>10}" for h in header))
    for i in range(num_classes):
        label = "bg" if i == 0 else CLASS_NAMES.get(i, str(i))
        print(f"  {label:<12}" + "".join(f"{matrix[i, j]:>10}" for j in range(num_classes)))
    csv_path = os.path.join(out_dir, f"{args.split}_confusion_matrix.csv")
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%d")
    print(f"\n  saved -> {csv_path}")

    print("\n" + "=" * 72)
    print("CLASS-AGNOSTIC DETECTION AP (health class ignored)")
    print("=" * 72)
    ap = class_agnostic_ap(cached, out_dir)

    print("\n" + "=" * 72)
    print("SCORE THRESHOLD SWEEP (pooled)")
    print("=" * 72)
    print(f"  {'thr':>5} {'precision':>10} {'recall':>8} {'F1':>8}   {'tp':>5} {'fp':>5} {'fn':>5}")
    sweep = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c = pooled_prf(cached, t, iou, num_classes)
        tp = sum(v[0] for v in c.values()); fp = sum(v[1] for v in c.values()); fn = sum(v[2] for v in c.values())
        pp, rr, ff = prf(tp, fp, fn)
        sweep.append({"threshold": t, "precision": pp, "recall": rr, "f1": ff, "tp": tp, "fp": fp, "fn": fn})
        print(f"  {t:>5.2f} {pp:>10.3f} {rr:>8.3f} {ff:>8.3f}   {tp:>5} {fp:>5} {fn:>5}")
    best = max(sweep, key=lambda s: s["f1"])
    print(f"\n  best F1 {best['f1']:.3f} at threshold {best['threshold']:.2f}")

    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "tiles": len(cached),
        "gt_instances": n_gt,
        "score_threshold": thr,
        "iou_threshold": iou,
        "per_class": {CLASS_NAMES.get(c, str(c)): dict(zip(["tp", "fp", "fn"], counts[c])) for c in counts},
        "pooled": {"precision": p, "recall": r, "f1": f},
        "macro": {"precision": mp, "recall": mr, "f1": mf},
        "class_agnostic_ap": ap,
        "threshold_sweep": sweep,
    }
    json_path = os.path.join(out_dir, f"{args.split}_evaluation.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary -> {json_path}")


if __name__ == "__main__":
    main()
