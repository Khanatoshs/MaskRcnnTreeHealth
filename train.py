import gc
import os
import configparser
import torch
import torch.nn as nn
import numpy as np
import rasterio
import logging
import glob
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# Prefer a dataset that reads stacked multi-channel images and instance-mask TIFFs
try:
    from dataset.multi_channel_dataset import StackedImageInstanceMaskDataset as StackedDataset
except Exception:
    # Fallback to the original UAV5ChannelDataset if a stacked dataset is not available
    from dataset.multi_channel_dataset import UAV5ChannelDataset as StackedDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, filename = "train.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Train_MaskRCNN")

# =====================================================================
# 1. DYNAMIC CHANNEL MASK R-CNN MODEL BUILDER
# =====================================================================

def get_multiband_maskrcnn(num_classes, in_channels=5):
    """Build a Mask R-CNN model for arbitrary input channel counts.

    The first three channels are treated as RGB-style bands; additional channels
    are initialized from the mean RGB kernel weights to keep transfer learning
    stable for NDVI/CHM or other remote-sensing features.
    """
    if in_channels < 1:
        raise ValueError(f"in_channels must be positive, got {in_channels}")

    # 1. Load a pre-trained ResNet-50 FPN backbone
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=torchvision.models.ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )

    # 2. Modify the first Conv2d layer to accept the configured number of channels.
    old_conv = backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # 3. Transfer pre-trained weights and initialize extra bands from RGB statistics.
    with torch.no_grad():
        if in_channels >= 3:
            new_conv.weight.data[:, :3, :, :].copy_(old_conv.weight.data[:, :3, :, :])
        if in_channels > 3:
            mean_rgb_weight = old_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.weight.data[:, 3:, :, :].copy_(mean_rgb_weight.repeat(1, in_channels - 3, 1, 1))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)

    backbone.body.conv1 = new_conv

    # 4. Match torchvision's image normalization to the actual input channel count.
    image_mean = [0.485, 0.456, 0.406] + [0.5] * max(0, in_channels - 3)
    image_std = [0.229, 0.224, 0.225] + [0.5] * max(0, in_channels - 3)
    image_mean = image_mean[:in_channels]
    image_std = image_std[:in_channels]

    # 5. Initialize Mask R-CNN with modified backbone and custom per-channel normalization.
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
    )

    # 6. Replace Box Predictor Head for custom class count.
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 7. Replace Mask Predictor Head for custom class count.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_5channel_maskrcnn(num_classes):
    """Backward-compatible wrapper for the original 5-channel configuration."""
    return get_multiband_maskrcnn(num_classes=num_classes, in_channels=5)


def collate_fn(batch):
    return tuple(zip(*batch))


def flatten_metrics(metrics, prefix=""):
    """Flatten nested metric dicts into a single dict for logging."""
    flat = {}
    for key, value in metrics.items():
        new_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, new_key))
        else:
            flat[new_key] = float(value)
    return flat


def log_metrics(logger_obj, phase, epoch, metrics=None, batch_idx=None, total_batches=None):
    """Write a compact, readable metrics summary to the log file."""
    if metrics is None:
        metrics = {}

    if isinstance(metrics, dict):
        flat_metrics = flatten_metrics(metrics)
        metric_str = ", ".join(f"{k}={v:.6f}" for k, v in flat_metrics.items()) if flat_metrics else "no metrics"
    else:
        metric_str = f"items={metrics}"

    if batch_idx is not None and total_batches is not None:
        logger_obj.info(f"Epoch [{epoch + 1}] {phase} Batch {batch_idx}/{total_batches} | {metric_str}")
    else:
        logger_obj.info(f"Epoch [{epoch + 1}] {phase} | {metric_str}")


def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union for two boxes in format [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def compute_class_prf(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    """Compute per-class precision, recall, and F1 from detection predictions and ground truth targets."""
    if predictions is None:
        predictions = []
    if targets is None:
        targets = []

    all_classes = set()
    for target in targets:
        labels = target.get("labels", torch.empty(0, dtype=torch.int64))
        all_classes.update(int(label.item()) for label in labels)
    for pred in predictions:
        labels = pred.get("labels", torch.empty(0, dtype=torch.int64))
        all_classes.update(int(label.item()) for label in labels)

    if not all_classes:
        return {"class_0": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}

    metrics = {}
    for cls_id in sorted(all_classes):
        if cls_id == 0:
            continue

        gt_boxes = []
        for target in targets:
            labels = target.get("labels", torch.empty(0, dtype=torch.int64))
            boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32))
            if labels.numel() == 0:
                continue
            mask = (labels == cls_id)
            if mask.any():
                gt_boxes.extend(boxes[mask].tolist())

        pred_boxes = []
        pred_scores = []
        for pred in predictions:
            labels = pred.get("labels", torch.empty(0, dtype=torch.int64))
            boxes = pred.get("boxes", torch.empty((0, 4), dtype=torch.float32))
            scores = pred.get("scores", torch.empty(0, dtype=torch.float32))
            if labels.numel() == 0:
                continue
            mask = (labels == cls_id) & (scores >= score_threshold)
            if mask.any():
                pred_boxes.extend(boxes[mask].tolist())
                pred_scores.extend(scores[mask].tolist())

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            precision = recall = f1 = 0.0
        else:
            order = sorted(range(len(pred_boxes)), key=lambda idx: pred_scores[idx], reverse=True)
            gt_used = [False] * len(gt_boxes)
            matched = 0
            for pred_idx in order:
                best_match = None
                best_iou = 0.0
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_used[gt_idx]:
                        continue
                    iou = compute_iou(pred_boxes[pred_idx], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = gt_idx
                if best_match is not None and best_iou >= iou_threshold:
                    gt_used[best_match] = True
                    matched += 1

            tp = matched
            fp = len(pred_boxes) - matched
            fn = sum(1 for used in gt_used if not used)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"class_{cls_id}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def compute_global_pr_metrics(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    """Compute macro mean precision and recall over the classes represented in the targets/predictions."""
    class_metrics = compute_class_prf(predictions, targets, score_threshold=score_threshold, iou_threshold=iou_threshold)
    if not class_metrics:
        return {"mean_precision": 0.0, "mean_recall": 0.0}

    precisions = [v["precision"] for v in class_metrics.values()]
    recalls = [v["recall"] for v in class_metrics.values()]
    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
    }


def compute_confusion_matrix(predictions, targets, score_threshold=0.5, iou_threshold=0.5, num_classes=2):
    """Build a detection confusion matrix with rows=true class and columns=predicted class; background class=0."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)

    gt_items = []
    for target in targets:
        labels = target.get("labels", torch.empty(0, dtype=torch.int64))
        boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32))
        for label, box in zip(labels.tolist(), boxes.tolist()):
            gt_items.append({"class_id": int(label), "box": box})

    pred_items = []
    for pred in predictions:
        labels = pred.get("labels", torch.empty(0, dtype=torch.int64))
        boxes = pred.get("boxes", torch.empty((0, 4), dtype=torch.float32))
        scores = pred.get("scores", torch.empty(0, dtype=torch.float32))
        for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
            if float(score) >= score_threshold:
                pred_items.append({"class_id": int(label), "box": box, "score": float(score)})

    pred_items = sorted(pred_items, key=lambda x: x["score"], reverse=True)
    gt_used = [False] * len(gt_items)

    for pred in pred_items:
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt_item in enumerate(gt_items):
            if gt_used[gt_idx]:
                continue
            iou = compute_iou(pred["box"], gt_item["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            gt_used[best_gt_idx] = True
            gt_class = gt_items[best_gt_idx]["class_id"]
            pred_class = pred["class_id"]
            confusion[gt_class, pred_class] += 1.0
        else:
            confusion[0, pred["class_id"]] += 1.0

    for gt_idx, used in enumerate(gt_used):
        if not used:
            gt_class = gt_items[gt_idx]["class_id"]
            confusion[gt_class, 0] += 1.0

    return confusion


def save_confusion_matrix(confusion_matrix, path):
    """Save a confusion matrix to a CSV file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savetxt(path, confusion_matrix, delimiter=",", fmt="%.6f")

    heatmap_path = os.path.splitext(path)[0] + "_heatmap.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    img = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ticks = list(range(confusion_matrix.shape[0]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, f"{confusion_matrix[i, j]:.0f}", ha="center", va="center", color="black")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)


def save_metrics_history_csv(history, path):
    """Save per-epoch metrics as a CSV table."""
    if not history:
        return
    fieldnames = sorted(set().union(*(entry.keys() for entry in history)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for entry in history:
            values = []
            for name in fieldnames:
                value = entry.get(name, 0.0)
                values.append(str(value))
            f.write(",".join(values) + "\n")


def save_metric_plots(history, metrics_dir):
    """Save a simple per-epoch plotting summary for key metrics."""
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Metrics Progression")

    metric_names = [
        ("mean_precision", "Mean Precision"),
        ("mean_recall", "Mean Recall"),
        ("class_1_precision", "Class 1 Precision"),
        ("class_1_recall", "Class 1 Recall"),
    ]

    for idx, (metric_name, title) in enumerate(metric_names):
        ax = axes[idx // 2, idx % 2]
        values = [entry.get(metric_name, np.nan) for entry in history]
        ax.plot(epochs, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(metrics_dir, "metric_progression.png"), dpi=200)
    plt.close(fig)

    # confusion matrix plot by epoch if available
    if "epoch_confusion" in history[0]:
        fig, axes = plt.subplots(1, max(1, len(history)), figsize=(max(6, len(history) * 3), 4))
        if len(history) == 1:
            axes = [axes]
        for ax, entry in zip(axes, history):
            matrix = np.asarray(entry["epoch_confusion"])
            img = ax.imshow(matrix, cmap="Blues")
            ax.set_title(f"Epoch {entry['epoch']}")
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black")
            fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(metrics_dir, "confusion_matrix_progression.png"), dpi=200)
        plt.close(fig)

# =====================================================================
# 3. TRAINING & EXECUTION BOILERPLATE
# =====================================================================

def main(config):
    if not config.has_section("TRAIN"):
        raise ValueError(f"Missing [TRAIN] section in {config_path}")
    if torch.cuda.is_available() and config.get("TRAIN", "device", fallback="cuda") == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    if config.get("TRAIN", "device", fallback="cuda") == "cuda" and  torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training on device: {device}")

    # Set up class counts (Background=0, Tree=1)
    NUM_CLASSES = int(config.get("TRAIN", "num_classes", fallback="2"))
    NUM_INPUT_CHANNELS = int(config.get("TRAIN", "num_input_channels", fallback="5"))

    logger.info(f"Input channels: {NUM_INPUT_CHANNELS} | Classes: {NUM_CLASSES}")

    # Initialize Model
    model = get_multiband_maskrcnn(num_classes=NUM_CLASSES, in_channels=NUM_INPUT_CHANNELS)
    # Move model to device after construction
    model.to(device)

    # Prepare dataset paths (expects slicer output with images/ and masks/ subfolders)
    dataset_dir = config.get("TRAIN", "dataset_dir", fallback="data/dataset_sliced_640")
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

    if len(image_paths) == 0 or len(mask_paths) == 0 or len(image_paths) != len(mask_paths):
        logger.warning("Warning: image/mask pairs not found or counts mismatch in dataset dir. Falling back to dummy dataset.")
        # Fallback dummy small dataset to allow smoke runs
        rgb_files = ["tile_0_rgb.tif", "tile_1_rgb.tif"]
        ndvi_files = ["tile_0_ndvi.tif", "tile_1_ndvi.tif"]
        chm_files = ["tile_0_chm.tif", "tile_1_chm.tif"]
        dummy_annotations = [{
            'boxes': [[50.0, 50.0, 200.0, 200.0]],
            'labels': [1],
            'masks': np.zeros((1, 640, 640), dtype=np.uint8)
        }] * len(rgb_files)
        dataset = StackedDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
        val_dataset = StackedDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
    else:
        # Use stacked/mask dataset class (StackedDataset is imported above)
        if getattr(StackedDataset, "__name__", "") == 'UAV5ChannelDataset':
            logger.info("Detected legacy UAV5ChannelDataset. Please provide a StackedImageInstanceMaskDataset implementation in dataset/multi_channel_dataset.py. Falling back to dummy dataset.")
            rgb_files = ["tile_0_rgb.tif", "tile_1_rgb.tif"]
            ndvi_files = ["tile_0_ndvi.tif", "tile_1_ndvi.tif"]
            chm_files = ["tile_0_chm.tif", "tile_1_chm.tif"]
            dummy_annotations = [{
                'boxes': [[50.0, 50.0, 200.0, 200.0]],
                'labels': [1],
                'masks': np.zeros((1, 640, 640), dtype=np.uint8)
            }] * len(rgb_files)
            dataset = StackedDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
            val_dataset = StackedDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
        else:
            dataset = StackedDataset(image_paths, mask_paths)
            val_dataset = StackedDataset(image_paths, mask_paths)

    # DataLoader
    batch_size = int(config.get("TRAIN", "batch_size", fallback="1"))
    num_workers = int(config.get("TRAIN", "num_workers", fallback="4"))
    pin_memory = config.getboolean("TRAIN", "pin_memory", fallback=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer and training configuration
    learning_rate = float(config.get("TRAIN", "learning_rate", fallback="1e-4"))
    weight_decay = float(config.get("TRAIN", "weight_decay", fallback="1e-4"))
    num_epochs = int(config.get("TRAIN", "num_epochs", fallback="10"))
    iou_threshold = float(config.get("TRAIN", "iou_threshold", fallback="0.5"))
    checkpoint_dir = config.get("TRAIN", "checkpoint_dir", fallback="checkpoints")
    metrics_dir = config.get("TRAIN", "metrics_dir", fallback=os.path.join(checkpoint_dir, "metrics"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # AMP setup
    use_amp = config.getboolean("TRAIN", "amp", fallback=False)
    scaler = torch.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    best_val = float('inf')
    save_best_only = config.getboolean("TRAIN", "save_best_only", fallback=False)
    metrics_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(data_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses):
                    logger.warning(f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} produced a non-finite loss. Skipping this step.")
                    continue
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses):
                    logger.warning(f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} produced a non-finite loss. Skipping this step.")
                    continue
                losses.backward()
                optimizer.step()

            running_loss += losses.item()
            log_metrics(logger, "Train", epoch, dict(loss_dict), batch_idx, len(data_loader))

        train_loss = running_loss / max(1, len(data_loader))
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_running_loss = 0.0
        epoch_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        epoch_class_pr = defaultdict(list)
        epoch_global_pr = {"mean_precision": [], "mean_recall": []}

        with torch.no_grad():
            for val_batch_idx, (val_images, val_targets) in enumerate(val_loader, 1):
                val_images = [img.to(device) for img in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                if use_amp:
                    with torch.amp.autocast():
                        val_output = model(val_images, val_targets)
                else:
                    val_output = model(val_images, val_targets)

                if isinstance(val_output, dict):
                    val_loss_dict = val_output
                    val_losses = sum(loss for loss in val_loss_dict.values())
                    val_loss_value = val_losses.item()
                    val_running_loss += val_loss_value
                    log_metrics(logger, "Validation", epoch, val_loss_dict, val_batch_idx, len(val_loader))
                else:
                    val_loss_value = float('nan')
                    metrics = compute_class_prf(val_output, val_targets, score_threshold=0.5, iou_threshold=iou_threshold)
                    global_metrics = compute_global_pr_metrics(val_output, val_targets, score_threshold=0.5, iou_threshold=iou_threshold)
                    confusion = compute_confusion_matrix(val_output, val_targets, score_threshold=0.5, iou_threshold=iou_threshold, num_classes=NUM_CLASSES)
                    epoch_confusion += confusion
                    for key, value in metrics.items():
                        for metric_name in ["precision", "recall", "f1"]:
                            epoch_class_pr[f"{key}_{metric_name}"].append(value[metric_name])
                    epoch_global_pr["mean_precision"].append(global_metrics["mean_precision"])
                    epoch_global_pr["mean_recall"].append(global_metrics["mean_recall"])
                    log_metrics(logger, "Validation", epoch, {**metrics, **global_metrics}, val_batch_idx, len(val_loader))
                    logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Validation Confusion Matrix (rows=true, cols=pred): {confusion.tolist()}")
                    logger.warning(
                        f"Epoch [{epoch + 1}/{num_epochs}] Validation Batch {val_batch_idx}/{len(val_loader)} returned detections list instead of losses; loss aggregation skipped."
                    )

        val_loss = val_running_loss / max(1, len(val_loader))
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {val_loss:.4f}")

        epoch_summary = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "mean_precision": float(np.mean(epoch_global_pr["mean_precision"])) if epoch_global_pr["mean_precision"] else 0.0,
            "mean_recall": float(np.mean(epoch_global_pr["mean_recall"])) if epoch_global_pr["mean_recall"] else 0.0,
            "epoch_confusion": epoch_confusion.copy(),
        }

        for key, values in epoch_class_pr.items():
            epoch_summary[key] = float(np.mean(values)) if values else 0.0

        metrics_history.append(epoch_summary)
        save_confusion_matrix(epoch_confusion, os.path.join(metrics_dir, f"confusion_matrix_epoch_{epoch + 1}.csv"))
        save_metrics_history_csv(metrics_history, os.path.join(metrics_dir, "metrics_history.csv"))
        save_metric_plots(metrics_history, metrics_dir)
        if len(metrics_history) > 1:
            all_confusions = np.stack([np.asarray(entry["epoch_confusion"]) for entry in metrics_history if "epoch_confusion" in entry])
            if all_confusions.size > 0:
                final_confusion = all_confusions[-1]
                save_confusion_matrix(final_confusion, os.path.join(metrics_dir, "latest_confusion_matrix.csv"))

        # Save checkpoint (best or every epoch)
        checkpoint_path = os.path.join(checkpoint_dir, f"maskrcnn_epoch_{epoch + 1}.pth")
        if save_best_only:
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                }, checkpoint_path)
                logger.info(f"Saved BEST checkpoint to: {checkpoint_path}")
        else:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to: {checkpoint_path}")

        if torch.cuda.is_available() and config.get("TRAIN", "device", fallback="cuda") == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    config.read(config_path)
    logger.setLevel(getattr(logging, config.get('TRAIN', 'log_level', fallback='INFO')))
    try:
        logger.info("\n\nStarting training")
        main(config)
    except Exception as e:
        logger.exception(f"Error occurred: {e}", exc_info=True)
        exit(1)

    logger.info("Training script completed.")