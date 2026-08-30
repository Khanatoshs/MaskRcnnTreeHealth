# Mask R-CNN Training Debugging Notes

## Overview
This project was debugged to fix several issues in the multi-class tree health detection pipeline:

- CUDA-related validation issues
- validation loss staying at 0
- multi-class mask handling and label propagation
- metrics missing classes
- dataset class hardcoding bugs
- incorrect validation inference mode in Mask R-CNN

## Important Findings

### 1. Validation loss was always 0
Root cause:
- validation loss accumulation was not being computed correctly
- in some versions, the model was being evaluated in the wrong mode
- Mask R-CNN expects inference with `targets=None` only in eval mode
- when called in training mode, it raises:
  `AssertionError: targets should not be none when in training mode`

### 2. Multi-class support was broken in the dataset
The dataset class was hardcoding all labels to class 1:

- old behavior: `labels.append(1)`
- fixed behavior: `labels.append(int(mask_id))`

This was the key reason the model could not effectively learn multiple classes.

### 3. Metrics were not tracking all classes
The metrics logic was only calculating metrics for classes present during a batch. This could hide minority classes, especially in validation.

Fix:
- pass `num_classes` into the metric functions
- ensure all expected classes are evaluated even when absent in a single batch

### 4. Mask generation was using class property values incorrectly
The mask pipeline used the `tree_class` property and added an offset so that classes mapped as:

- 0 -> background
- 1 -> class 1
- 2 -> class 2
- 3 -> class 3
- 4 -> class 4

This aligns with the dataset and training configuration.

## File-level Notes

### `dataset/multi_channel_dataset.py`
- Fixed hardcoded labels
- now reads actual pixel class values from masks
- properly preserves multi-class instance labels

### `train.py`
- validation loss logic fixed
- class metrics generalized for all classes
- visualization updated to plot per-class metrics instead of only class 1
- inference/validation must run in eval mode when `targets=None`

### `scripts/create_masks.py`
- creates masks based on `tree_class` values
- adds 1 offset to handle background and class numbering

### `config.ini`
- includes train/validation split and class configuration
- multi-class setup is configured as 5 total classes

## Log Evidence
Observed log messages included:

- `Average Training Loss: ...`
- `Average Validation Loss: 0.0000`
- `AssertionError: targets should not be none when in training mode`

This showed that validation loss and validation inference were not being handled correctly.

## Final Diagnosis
The root cause was multi-layered:

1. class information was lost in the dataset layer
2. validation metrics were not computed for all classes
3. validation inference was invoked in the wrong training mode
4. validation loss accumulation was incorrect or zeroed out in buggy runs

## Resolution Summary
The project was corrected by:

- fixing dataset label extraction
- ensuring all 5 classes are represented in metrics
- computing validation loss correctly
- using eval mode for inference predictions
- restoring training mode after validation evaluation

## Current Status
This conversation documents the debugging workflow and the successful issue tracking that led to the final corrected implementation.
