# Separated Detection vs Segmentation/Grading Metrics Guide

## Overview

This document explains three complementary metric frameworks for evaluating Mask R-CNN performance:

1. **Current Methodology**: Per-batch macro precision/recall on complete end-to-end task
2. **New Enhancement**: Separated detection (localization) from grading (classification)  
3. **Alternate Project**: Three-level evaluation (detection/grading/complete)

## Why Separate Detection from Classification?

Your model has two distinct jobs:
1. **Detection/Localization**: "Did we find the tree at all?" (bounding box IoU >= 0.5, ignoring class)
2. **Segmentation/Grading**: "On found trees, did we classify the health correctly?" (class accuracy given detection)

Combining these into one metric makes it hard to diagnose failures:
- If F1 is low, is it because:
  - The detector isn't finding trees? (detection failure)
  - The trees are found but misclassified? (grading failure)
  - Some combination of both?

Separating them answers these questions directly.

## The Three Metrics Frameworks

### Framework 1: Current Project (Per-Batch Macro)

```
Validation Loop:
├─ For each batch (tile):
│  ├─ Match predictions to GTs by class (IoU >= 0.5)
│  ├─ Compute per-class precision/recall/F1 for this batch
│  └─ Skip classes not present in this batch (no false zeros)
├─ Average per-class metrics across batches
└─ Report: mean_precision, mean_recall, per_class_F1
```

**Pros:**
- Optimized for per-plot evaluation (what you care about operationally)
- Avoids inflating metrics with zeros for absent classes
- Lightweight to compute during training

**Cons:**
- Mixes detection and classification into one metric
- Per-batch averaging is optimistic (sparse tiles score high)
- Can't distinguish localization from classification failures

**Where used:** Training loop validation, `train.log`

---

### Framework 2: New Separated Detection/Grading (Integrated into Training)

```
For each validation batch, compute THREE separate metrics:

┌─ Detection Only (class-agnostic)
│  └─ Match predictions to GTs by IoU alone (ignore class)
│     └─ Precision = TP / (TP + FP), Recall = TP / (TP + FN)
│     └─ Reports: "Did we find 47 out of 51 trees?"
│
├─ Grading Only (on matched detections)
│  └─ Among the trees we correctly detected, what's the class accuracy?
│     └─ Per-class accuracy = correct_class / (correct + wrong_class)
│     └─ Reports: "We found 47 trees, but got the class right on 38 (81%)"
│
└─ Complete (current approach, all trees including missed)
   └─ As currently reported
   └─ Precision/Recall/F1 for all trees (missed trees count as FN)
```

**Pros:**
- Shows exactly where failures come from
- Can be logged every epoch at minimal cost
- Immediately actionable diagnostics

**Cons:**
- Requires modifying training loop (already done!)
- Three numbers to monitor instead of one

**Where used:** Added to `train.log` during validation

---

### Framework 3: Alternate Project Style (Three-Level Evaluation)

```
evaluate_test_set.py reports:

┌─ DETECTION (IoU >= 0.5 strict)
│  └─ TP=313, FP=339, FN=136
│  └─ Precision=0.480, Recall=0.697, F1=0.569
│  └─ "We found 313 trees, had 339 false alarms, missed 136"
│
├─ GRADING (per-class accuracy on the 313 found trees)
│  └─ Class 1: 95/120 correct (79%)
│  └─ Class 2: 88/110 correct (80%)
│  └─ Macro F1: 0.761
│  └─ "On found trees, classification accuracy is 82%"
│
└─ COMPLETE (all trees, any miss counts as error)
   └─ TP=280, FP=339, FN=169
   └─ Precision=0.452, Recall=0.624, F1=0.524
   └─ Accuracy=0.624 (280 correct out of 448 total)
   └─ "End-to-end: we got 280 trees completely right (find + classify)"
```

**Pros:**
- Matches alternate project's framework for direct comparison
- Three-level breakdown is most informative
- Useful for final test set reporting

**Cons:**
- Only runs on test set evaluation (post-training)
- Computationally expensive for every epoch
- Requires `scripts/evaluate_separated_metrics.py` run

**Where used:** Final evaluation script

---

## What to Expect: Detection vs Grading Breakdown

When you run training with the new separated metrics, you'll see logs like:

```
Epoch [26/100] Validation Detection (class-agnostic, IoU≥0.5):
  TP=1247, FP=892, FN=315
  Precision=0.583, Recall=0.798, F1=0.674

Epoch [26/100] Validation Grading (on 1247 detected trees):
  Class 1 (healthy):     precision=0.92, recall=0.88, f1=0.90
  Class 2 (mild):        precision=0.65, recall=0.58, f1=0.61
  Class 3 (moderate):    precision=0.45, recall=0.41, f1=0.43
  Class 4 (severe):      precision=0.52, recall=0.48, f1=0.50
  Macro Mean:            precision=0.64, recall=0.59, f1=0.61

Epoch [26/100] Validation Complete (end-to-end):
  Precision=0.521, Recall=0.686, F1=0.593
```

### Interpreting These Numbers

**Scenario 1: Detection 0.80, Grading 0.70, Complete 0.56**
- **Analysis**: Detector is working well (80% recall), but misclassifies 30% of found trees
- **Action**: Focus on classification architecture, not localization
- **Fix ideas**: More training epochs, class-specific data augmentation, loss weight adjustment

**Scenario 2: Detection 0.65, Grading 0.85, Complete 0.55**
- **Analysis**: Found 65% of trees, but classified them well (85%). Main issue is missing trees
- **Action**: Focus on recall (find more trees), not classification
- **Fix ideas**: Lower confidence threshold, smaller NMS, augmentation for detection

**Scenario 3: Detection 0.80, Grading 0.85, Complete 0.68**
- **Analysis**: Both detection and classification are working well, complete metric just lower due to pooling
- **Action**: Model is balanced, keep current training approach
- **Fix ideas**: None needed, model is good

---

## Usage

### During Training

The separated metrics are logged automatically:

```bash
python train.py
# Logs will show:
# - Current pooled metrics (unchanged)
# - Plus separated detection/grading breakdown for each epoch
```

Check `train.log` and `train_scaled_uint8.log` for the breakdown during validation.

### After Training (Full Evaluation)

Run the evaluation script on test set:

```bash
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth \
  --split test \
  --dataset_dir data/dataset_sliced_800 \
  --output_dir evaluation_results
```

This produces:
- `evaluation_results/separated_metrics_test.json` - Full metrics breakdown
- `evaluation_results/evaluation_test.log` - Detailed analysis including per-class failure modes

### Comparing Preprocessing Modes

Native vs Scaled_uint8:

```bash
# Evaluate native mode
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth --split val

# Evaluate scaled_uint8 mode
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_scaled_uint8/maskrcnn_best.pth --split val

# Compare: The detection metrics tell you if one mode finds trees better,
#          grading metrics tell you if one classifies better
```

---

## Key Insights from Project Documentation

From CLAUDE.md, the project deliberately chose per-batch metrics over pooled because:

> "A class absent from both the ground truth and the predictions in a batch is omitted from the result, not scored 0.0... With val_batch_size = 1 most tiles hold only 2-3 of the 4 classes, so ~32% of scores were fake zeros when classes were scored as zero... Omit the class entirely; callers then average only over classes that were actually present."

This means:
- Your current per-batch metrics are intentionally conservative
- Adding separated metrics doesn't change this philosophy
- Detection/Grading are computed the same way (on present classes only)

---

## Implementation Details

### Detection Metrics Function

```python
def compute_detection_metrics(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    """Class-agnostic: match predictions to targets by IoU alone"""
    # Returns: tp, fp, fn, precision, recall, f1
```

### Grading Metrics Function

```python
def compute_grading_metrics_on_detections(predictions, targets, 
                                         score_threshold=0.5, iou_threshold=0.5):
    """Per-class accuracy on matched detections only"""
    # Returns: {class_1: {precision, recall, f1}, ...}
```

### Three-Level Evaluation Function

```python
def compute_three_level_evaluation(predictions, targets, ...):
    """Combines all three frameworks for comprehensive reporting"""
    # Returns: {detection: {...}, grading: {...}, complete: {...}}
```

All are in `train.py` and available for use in any script.

---

## Next Steps

1. **Monitor training**: Check the separated detection/grading metrics in `train.log` to diagnose where failures come from
2. **Choose focus**: Based on detection vs grading performance, decide whether to:
   - Improve detection (if grading is good but detection is low)
   - Improve classification (if detection is good but grading is low)
   - General improvements (if both are low)
3. **Evaluate preprocessing**: Compare native vs scaled_uint8 using the evaluation script to see which mode is better for detection, which is better for classification
4. **Final reporting**: Use `scripts/evaluate_separated_metrics.py` on test set for final three-level metrics

---

## References

- `train.py`: Metric functions and training loop integration
- `scripts/evaluate_separated_metrics.py`: Standalone evaluation script
- `dataset/multi_channel_dataset.py`: Preprocessing modes (native vs scaled_uint8)
- `CLAUDE.md`: Project design philosophy and metric choices
