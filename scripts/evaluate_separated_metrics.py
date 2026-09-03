#!/usr/bin/env python3
"""
Evaluate a model checkpoint using separated detection vs grading metrics.

This script isolates three aspects of model performance:
1. DETECTION: "Did we find the tree?" (class-agnostic localization accuracy)
2. GRADING: "On found trees, did we classify the health correctly?" (class accuracy given detection)
3. COMPLETE: "End-to-end performance including missed trees" (current metric)

Run from repo root:
  python scripts/evaluate_separated_metrics.py --checkpoint checkpoints_new/maskrcnn_best.pth --split test
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.multi_channel_dataset import StackedImageInstanceMaskDataset
from train import (
    get_multiband_maskrcnn,
    get_image_preprocessing_mode,
    compute_detection_metrics,
    compute_grading_metrics_on_detections,
    compute_class_counts,
    pooled_prf_from_counts,
)

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


def setup_logging(log_file):
    """Setup logging to both console and file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def load_config(config_file="config.ini"):
    """Load configuration from config.ini."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def compute_class_statistics(predictions, targets, num_classes=5):
    """Compute statistics per class to understand failure patterns."""
    class_stats = {c: {
        "gt_count": 0,
        "pred_count": 0,
        "correct_class_on_detected": 0,
        "wrong_class_on_detected": 0,
        "undetected": 0,
    } for c in range(1, num_classes)}

    for pred, target in zip(predictions or [], targets or []):
        p_labels = pred.get("labels", torch.empty(0, dtype=torch.int64))
        p_boxes = pred.get("boxes", torch.empty((0, 4), dtype=torch.float32))
        p_scores = pred.get("scores", torch.empty(0, dtype=torch.float32))

        g_labels = target.get("labels", torch.empty(0, dtype=torch.int64))
        g_boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32))

        for g_label, g_box in zip(g_labels.tolist(), g_boxes.tolist()):
            true_class = int(g_label)
            class_stats[true_class]["gt_count"] += 1

            # Find if this GT was detected
            best_pred_idx = None
            best_iou = 0.0
            for p_idx, (p_label, p_box, p_score) in enumerate(zip(p_labels, p_boxes, p_scores)):
                if float(p_score) < 0.5:
                    continue
                from train import compute_iou
                iou = compute_iou(p_box.tolist(), g_box)
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_pred_idx = p_idx

            if best_pred_idx is not None:
                pred_class = int(p_labels[best_pred_idx].item())
                if pred_class == true_class:
                    class_stats[true_class]["correct_class_on_detected"] += 1
                else:
                    class_stats[true_class]["wrong_class_on_detected"] += 1
            else:
                class_stats[true_class]["undetected"] += 1

        for p_label, p_box, p_score in zip(p_labels.tolist(), p_boxes.tolist(), p_scores.tolist()):
            if float(p_score) < 0.5:
                continue
            pred_class = int(p_label)
            class_stats[pred_class]["pred_count"] += 1

    return class_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with separated detection/grading metrics")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--dataset_dir", default="data/dataset_sliced_800",
                        help="Base dataset directory")
    parser.add_argument("--config", default="config.ini", help="Config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--output_dir", default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--no_cuda", action="store_true", help="Don't use CUDA")
    args = parser.parse_args()

    # Setup
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    config = load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"evaluation_{args.split}.log")
    logger = setup_logging(log_file)

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Device: {device}")

    # Load config parameters (fallback only - overridden below by the checkpoint's
    # own 'arch' dict when present, since anchors and normalization live outside
    # the state_dict and a mismatch loads silently with no error).
    num_classes = config.getint("TRAIN", "num_classes", fallback=5)
    num_input_channels = config.getint("TRAIN", "num_input_channels", fallback=8)
    image_mean_str = config.get("TRAIN", "image_mean", fallback="0.485,0.456,0.406,0.5,0.5,0.5,0.5,0.5")
    image_std_str = config.get("TRAIN", "image_std", fallback="0.229,0.224,0.225,0.5,0.5,0.5,0.5,0.5")
    image_mean = [float(x) for x in image_mean_str.split(",")]
    image_std = [float(x) for x in image_std_str.split(",")]
    anchor_sizes_raw = config.get("TRAIN", "anchor_sizes", fallback="").strip()
    if ";" in anchor_sizes_raw:
        anchor_sizes = [[int(s) for s in level.split(",") if s.strip()]
                        for level in anchor_sizes_raw.split(";") if level.strip()] or None
    else:
        anchor_sizes = [int(s) for s in anchor_sizes_raw.split(",") if s.strip()] or None
    anchor_ratios_raw = config.get("TRAIN", "anchor_aspect_ratios", fallback="").strip()
    anchor_aspect_ratios = [float(r) for r in anchor_ratios_raw.split(",") if r.strip()] or None

    input_preprocessing = get_image_preprocessing_mode(config)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Prefer the architecture recorded in the checkpoint over whatever config
    # currently holds - anchors only surface as an RPN head shape mismatch, and
    # normalization lives in the model's transform, so trusting a stale config
    # would silently load wrong values or crash on shape mismatch.
    arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
    arch_source = "config"
    if arch:
        arch_source = "checkpoint"
        num_classes = arch.get("num_classes", num_classes)
        num_input_channels = arch.get("in_channels", num_input_channels)
        anchor_sizes = arch.get("anchor_sizes", anchor_sizes)
        anchor_aspect_ratios = arch.get("anchor_aspect_ratios", anchor_aspect_ratios)
        image_mean = arch.get("image_mean", image_mean)
        image_std = arch.get("image_std", image_std)

    logger.info(f"Arch from: {arch_source}")
    logger.info(f"num_classes={num_classes}, num_input_channels={num_input_channels}")
    logger.info(f"anchor_sizes={anchor_sizes} anchor_aspect_ratios={anchor_aspect_ratios}")
    logger.info(f"image_mean={image_mean}")
    logger.info(f"image_std={image_std}")
    logger.info(f"input_preprocessing={input_preprocessing}")

    # Build model
    logger.info("Building model...")
    model = get_multiband_maskrcnn(
        num_classes=num_classes,
        in_channels=num_input_channels,
        anchor_sizes=anchor_sizes,
        anchor_aspect_ratios=anchor_aspect_ratios,
        image_mean=image_mean,
        image_std=image_std,
    )

    # Extract model_state_dict if checkpoint is wrapped
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load dataset
    split_dir = f"{args.split}_sliced_800_test" if args.split == "test" else "dataset_sliced_800"
    dataset_path = f"{args.dataset_dir.replace('dataset_sliced_800', split_dir.replace('_sliced_800_test', ''))}"
    
    if args.split == "test":
        dataset_path = args.dataset_dir.replace("dataset_sliced_800", "dataset_sliced_800_test")
    elif args.split == "val":
        dataset_path = args.dataset_dir

    logger.info(f"Loading dataset from: {dataset_path}")
    
    images_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")
    
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not os.path.exists(masks_dir):
        logger.error(f"Masks directory not found: {masks_dir}")
        sys.exit(1)

    # Build image and mask path lists
    image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.tif') or f.endswith('.tiff')])
    mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.tif') or f.endswith('.tiff')])
    
    if len(image_paths) != len(mask_paths):
        logger.warning(f"Number of images ({len(image_paths)}) != number of masks ({len(mask_paths)})")
    
    logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks")

    dataset = StackedImageInstanceMaskDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        label_divisor=10000,
        input_mode=input_preprocessing,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    logger.info(f"Dataset size: {len(dataset)}")

    # Run evaluation
    logger.info("Running evaluation...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, 1):
            images = [img.to(device) for img in images]
            predictions = model([img for img in images])
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Clear GPU cache after each batch to prevent OOM
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx}/{len(loader)} batches")

    logger.info(f"Total samples evaluated: {len(all_predictions)}")

    # Compute metrics at three levels
    logger.info("\n" + "="*80)
    logger.info("LEVEL 1: DETECTION METRICS (class-agnostic, did we find the tree?)")
    logger.info("="*80)
    
    detection_metrics = compute_detection_metrics(all_predictions, all_targets, args.score_threshold, iou_threshold=0.5)
    logger.info(f"TP: {detection_metrics['tp']}, FP: {detection_metrics['fp']}, FN: {detection_metrics['fn']}")
    logger.info(f"Precision: {detection_metrics['precision']:.4f}")
    logger.info(f"Recall:    {detection_metrics['recall']:.4f}")
    logger.info(f"F1:        {detection_metrics['f1']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("LEVEL 2: GRADING METRICS (on detected trees only, did we classify correctly?)")
    logger.info("="*80)
    
    grading_metrics = compute_grading_metrics_on_detections(all_predictions, all_targets, args.score_threshold, iou_threshold=0.5, num_classes=num_classes)
    
    if grading_metrics:
        for cls_id in sorted(grading_metrics.keys()):
            m = grading_metrics[cls_id]
            logger.info(
                f"{cls_id}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f} "
                f"({m['correct']}/{m['predicted']} correct out of {m['actual']} actual)"
            )
        # Macro mean
        precisions = [m['precision'] for m in grading_metrics.values()]
        recalls = [m['recall'] for m in grading_metrics.values()]
        f1s = [m['f1'] for m in grading_metrics.values()]
        logger.info(f"Macro Mean: Precision={np.mean(precisions):.4f}, Recall={np.mean(recalls):.4f}, F1={np.mean(f1s):.4f}")
    else:
        logger.info("No grading metrics available (no detected trees)")

    logger.info("\n" + "="*80)
    logger.info("LEVEL 3: COMPLETE/END-TO-END METRICS (all trees including missed)")
    logger.info("="*80)
    
    complete_counts = compute_class_counts(all_predictions, all_targets, args.score_threshold, iou_threshold=0.5, num_classes=num_classes)
    complete_metrics = pooled_prf_from_counts(complete_counts)
    
    logger.info(f"TP: {complete_metrics['pooled_tp']}, FP: {complete_metrics['pooled_fp']}, FN: {complete_metrics['pooled_fn']}")
    logger.info(f"Precision: {complete_metrics['pooled_precision']:.4f}")
    logger.info(f"Recall:    {complete_metrics['pooled_recall']:.4f}")
    logger.info(f"F1:        {complete_metrics['pooled_f1']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("FAILURE ANALYSIS: Where are the errors coming from?")
    logger.info("="*80)
    
    class_stats = compute_class_statistics(all_predictions, all_targets, num_classes)
    
    logger.info("\nPer-Class Breakdown:")
    logger.info("-" * 80)
    for cls_id in range(1, num_classes):
        stats = class_stats[cls_id]
        if stats["gt_count"] == 0:
            logger.info(f"Class {cls_id}: No ground truth instances")
            continue
        
        detection_rate = (stats["correct_class_on_detected"] + stats["wrong_class_on_detected"]) / stats["gt_count"]
        misclass_rate = stats["wrong_class_on_detected"] / max(1, stats["correct_class_on_detected"] + stats["wrong_class_on_detected"])
        miss_rate = stats["undetected"] / stats["gt_count"]
        
        logger.info(f"Class {cls_id}:")
        logger.info(f"  Ground truth count:          {stats['gt_count']}")
        logger.info(f"  Detected & correct class:    {stats['correct_class_on_detected']} ({100*stats['correct_class_on_detected']/stats['gt_count']:.1f}%)")
        logger.info(f"  Detected but wrong class:    {stats['wrong_class_on_detected']} ({100*misclass_rate*(100*detection_rate):.1f}% of errors)")
        logger.info(f"  Undetected (missed):         {stats['undetected']} ({100*miss_rate:.1f}%)")
        logger.info(f"  False positives:             {stats['pred_count'] - stats['correct_class_on_detected']}")

    # Save results to JSON
    output_file = os.path.join(args.output_dir, f"separated_metrics_{args.split}.json")
    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "score_threshold": args.score_threshold,
        "detection": detection_metrics,
        "grading": grading_metrics,
        "complete": complete_metrics,
        "class_statistics": class_stats,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
