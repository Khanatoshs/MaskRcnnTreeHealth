# Implementation Summary: Separated Detection vs Segmentation Metrics

## What Was Implemented

You now have **three complementary metric frameworks** for evaluating your Mask R-CNN model:

### 1. **Current Methodology** (Unchanged)
- Per-batch macro precision/recall/F1 on complete end-to-end task
- Logs to `train.log` during training validation
- Pooled metrics for checkpoint selection
- **Status**: Already working, enhanced with detection/grading breakdown

### 2. **New Enhancement: Separated Detection/Grading Metrics** (Now Integrated into Training)
- **Detection Metrics**: Class-agnostic localization accuracy ("Did we find the tree?")
- **Grading Metrics**: Per-class classification accuracy on detected trees only ("Did we classify it correctly?")
- Logs automatically during validation to `train.log`
- **Computational Cost**: Minimal - computed from already-generated predictions

### 3. **Alternate Project Style: Three-Level Evaluation** (Standalone Script)
- Standalone evaluation for test set
- Detection/Grading/Complete reporting at evaluation time
- Detailed failure analysis by class
- **Script**: `scripts/evaluate_separated_metrics.py`

---

## Code Changes

### File: `train.py`

**Added Functions (lines 605-750):**

```python
def compute_detection_metrics(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    """Class-agnostic detection accuracy (ignoring class labels)"""
    # Returns: {tp, fp, fn, precision, recall, f1}

def compute_grading_metrics_on_detections(predictions, targets, ...):
    """Per-class classification accuracy on matched detections only"""
    # Returns: {class_1: {precision, recall, f1, correct, predicted, actual}, ...}

def compute_three_level_evaluation(predictions, targets, ...):
    """Three-level reporting: detection/grading/complete (alternate project style)"""
    # Returns: {detection: {...}, grading: {...}, complete: {...}}
```

**Modified Validation Loop (lines 1308-1340, 1347-1369):**

1. Added tracking variables at validation start:
   ```python
   epoch_detection_tp = 0
   epoch_detection_fp = 0
   epoch_detection_fn = 0
   epoch_grading_totals = {c: [0, 0, 0] for c in range(1, NUM_CLASSES)}
   ```

2. Accumulate metrics during each batch:
   ```python
   batch_detection = compute_detection_metrics(...)
   epoch_detection_tp += batch_detection.get('tp', 0)
   # ... etc for fp, fn
   
   batch_grading = compute_grading_metrics_on_detections(...)
   # Accumulate grading results per class
   ```

3. Log separated metrics after pooled metrics (lines 1398-1422):
   ```
   Epoch [26/100] DETECTION (class-agnostic): TP=1247 FP=892 FN=315 | 
      Precision=0.583 Recall=0.798 F1=0.674
   Epoch [26/100] GRADING (on detected trees): Macro Precision=0.64
   ```

---

### File: `scripts/evaluate_separated_metrics.py` (New)

Standalone evaluation script that provides:

- Full three-level evaluation on any checkpoint + split
- Detailed per-class failure analysis
- JSON output for programmatic processing
- CPU/GPU support with OOM handling
- Comprehensive logging to `evaluation_results/`

**Key Features:**
```bash
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth \
  --split test \
  --output_dir evaluation_results
```

Produces:
- `evaluation_results/separated_metrics_test.json` - Full metrics
- `evaluation_results/evaluation_test.log` - Analysis

---

### File: `SEPARATED_METRICS_GUIDE.md` (New)

Comprehensive guide explaining:
- Why separate detection from classification
- All three frameworks side-by-side
- Interpretation guide (what metrics mean)
- Usage examples
- Implementation details

---

## How to Use

### During Training: Real-Time Detection/Grading Breakdown

Run training normally:
```bash
python train.py
```

Check `train.log` for new separated metrics appearing every epoch:

```
Epoch [5/100] Pooled over val: precision=0.412 recall=0.468 F1=0.438 ...

Epoch [5/100] DETECTION (class-agnostic): TP=1152 FP=847 FN=410 | 
   Precision=0.576 Recall=0.737 F1=0.644

Epoch [5/100] GRADING (on detected trees): Macro Precision=0.591
```

**Interpretation:**
- If **DETECTION** is high but **GRADING** is low → classification needs work
- If **DETECTION** is low but **GRADING** is high → localization needs work
- If both are low → general model improvements needed

### After Training: Full Evaluation on Test Set

```bash
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth \
  --split test \
  --dataset_dir data/dataset_sliced_800

# Produces: evaluation_results/separated_metrics_test.json
#          evaluation_results/evaluation_test.log
```

Output shows:
1. **DETECTION**: Tree-finding accuracy (class-agnostic)
2. **GRADING**: Health-classification accuracy on found trees
3. **COMPLETE**: End-to-end performance including missed trees
4. **Per-Class Breakdown**: Exactly which classes are failing

### Comparing Preprocessing Modes

**Native mode:**
```bash
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth --split val
```

**Scaled_uint8 mode:**
```bash
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_scaled_uint8/maskrcnn_best.pth --split val
```

**Compare the outputs:**
- Which mode finds more trees? (Detection metrics)
- Which mode classifies better? (Grading metrics)
- Overall, which is better? (Complete metrics)

---

## Example Output: What You'll See in train.log

```
Epoch [26/100] Average Training Loss: 1.234
Epoch [26/100] Average Validation Loss: 2.301
Epoch [26/100] Pooled over val: precision=0.3926 recall=0.4686 F1=0.4306 
   (tp=596 fp=960 fn=677) | lr=1.23e-04

Epoch [26/100] DETECTION (class-agnostic): TP=1243 FP=847 FN=346 | 
   Precision=0.595 Recall=0.782 F1=0.674

Epoch [26/100] GRADING (on detected trees): Macro Precision=0.612
   (correct classifications out of found trees)

Epoch [26/100] Early stopping monitor: pooled_f1 = 0.4306 (best so far: 0.4360, no improvement)
Epoch [26/100] Early stopping: 4/30 epochs without improvement
```

**How to Read This:**
1. Training and validation losses are reasonable
2. **Detection F1 = 0.674**: Pretty good at finding trees
3. **Grading Precision = 0.612**: 61% of found trees get classified correctly
4. **Complete F1 = 0.431**: Lower because missed trees count as errors

**Action**: Focus on improving classification (grading precision 61% is lower than detection 67%)

---

## Key Insights from Implementation

### From CLAUDE.md Philosophy

The project intentionally avoids:
- Scoring absent classes as zero (keeps metrics honest)
- Mixing per-batch and split-level metrics (one consistent approach)

The new metrics **respect this philosophy**:
- Detection/Grading also omit absent classes (only score present ones)
- Both are pooled over entire validation split for checkpoint selection
- Both are logged alongside, not replacing, existing metrics

### Detection vs Classification Failure Modes

The separated metrics help distinguish:

| Scenario | Detection | Grading | Action |
|----------|-----------|---------|--------|
| High | Low | Improve classification network |
| Low | High | Improve detection network |
| Both Low | | General improvements needed |
| Both High | | Model is good, keep training |

---

## Files Affected

```
train.py                           - Added metric functions + validation loop logging
scripts/evaluate_separated_metrics.py  - NEW: Standalone evaluation script  
SEPARATED_METRICS_GUIDE.md         - NEW: Comprehensive user guide
```

---

## Next Steps

1. **Run training** to see the new separated metrics in `train.log`:
   ```bash
   python train.py
   ```

2. **After training**, evaluate on test set:
   ```bash
   python scripts/evaluate_separated_metrics.py \
     --checkpoint checkpoints_new/maskrcnn_best.pth --split test
   ```

3. **Compare preprocessing modes** using the evaluation script on both checkpoints

4. **Read SEPARATED_METRICS_GUIDE.md** for detailed interpretation guidance

5. **Debug model**: Use detection vs grading breakdown to decide what to improve next

---

## Testing the Implementation

To verify everything works:

```bash
# 1. Check syntax
python -m py_compile train.py
python -m py_compile scripts/evaluate_separated_metrics.py

# 2. Quick training test (1 epoch to verify logging works)
python train.py
# Look for "DETECTION" and "GRADING" lines in train.log

# 3. Test evaluation script
python scripts/evaluate_separated_metrics.py \
  --checkpoint checkpoints_new/maskrcnn_best.pth \
  --split val \
  --no_cuda \
  --batch_size 1
```

---

## Troubleshooting

### If detection/grading metrics don't appear in train.log

- Check that `compute_detection_metrics` and `compute_grading_metrics_on_detections` functions exist in train.py
- Verify validation loop was updated with accumulation code
- Ensure no exceptions in the try/except blocks (check line 1398-1422)

### If evaluation script runs out of GPU memory

- Use `--no_cuda` flag to run on CPU (slower but no OOM)
- Reduce `--batch_size` (default=1)
- Make sure GPU cache is cleared between batches

### If checkpoint loading fails

- Verify checkpoint was saved with `model_state_dict` wrapper
- Check config file has correct `num_classes` and `num_input_channels`
- Ensure `image_mean` and `image_std` match training values

---

## Summary

**What you can now do:**

✅ See during training which part is failing: tree detection or tree classification
✅ Compare preprocessing modes to see which finds trees better vs classifies better
✅ Get detailed per-class failure analysis on test set
✅ Use three-level evaluation for comprehensive final reporting
✅ Make data-driven decisions about what to improve next (detection vs classification)

**Key insight:** Separated metrics show you exactly where to focus optimization effort, making debugging and improvement much more efficient.
