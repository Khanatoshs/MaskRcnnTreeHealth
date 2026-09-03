import random

import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


DEFAULT_PREPROCESS_RANGES = {
    0: (0.0, 255.0),
    1: (0.0, 255.0),
    2: (0.0, 255.0),
    3: (0.0, 30.0),
    4: (0.0, 1.0),
    5: (0.0, 1.5),
    6: (0.0, 1.0),
    7: (-0.1, 0.5),
}


def process_image_for_model(img, mode="native", rgb_divisor=255.0, channel_ranges=None):
    """Convert a stacked raster array to the format expected by the model.

    mode="native" keeps the project's current behavior: RGB values are rescaled to
    [0, 1] when they arrive as uint8-like imagery, while the extra bands are left
    in their physical scale and are normalized later by the model transform through
    `image_mean` / `image_std`.

    mode="scaled_uint8" mimics the alternate project approach: each band is clipped
    to a physically meaningful range, mapped into 0..255, then divided by 255 so the
    model sees a standardized [0,1] image. This is a comparison mode for ablations;
    it is not the default because the project's native normalization is designed to
    preserve the actual spectral distribution and then apply measured stats.

    Some source rasters (e.g. the vegetation-index mosaics under data/tiff/new)
    have real gaps in flight coverage that were left as GDAL's float32 nodata
    sentinel (~-3.4e38) instead of NaN, so `np.isfinite` doesn't catch them. Left
    alone, that sentinel reaches the model transform's `(x - mean) / std` and
    produces an activation on the order of 1e38, which is inf/NaN by the time it
    reaches the first conv layer. Zero it out here, before either preprocessing
    branch, same as an untextured/no-data pixel.
    """
    arr = np.asarray(img, dtype=np.float32, copy=True)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    bad = ~np.isfinite(arr) | (np.abs(arr) > 1e6)
    if bad.any():
        arr[bad] = 0.0

    if mode == "native":
        if arr.max() > 1.0 and arr.max() <= 255.0:
            if arr.shape[0] >= 3:
                arr[:3] = arr[:3] / rgb_divisor
            else:
                arr = arr / rgb_divisor
        return arr

    if mode == "scaled_uint8":
        if channel_ranges is None:
            channel_ranges = DEFAULT_PREPROCESS_RANGES
        for idx in range(arr.shape[0]):
            lo, hi = channel_ranges.get(idx, (0.0, 1.0))
            if hi <= lo:
                continue
            arr[idx] = np.clip(arr[idx], lo, hi)
            arr[idx] = (arr[idx] - lo) / (hi - lo)
        return arr

    raise ValueError(f"Unsupported image preprocessing mode: {mode!r}. Use 'native' or 'scaled_uint8'.")


class TrainAugmentation:
    """Geometric + photometric augmentation for multi-channel UAV tiles.

    Intended for the training split only (never validation, since eval must stay
    on the unmodified distribution). Applies random flips, 90-degree rotations,
    scale jitter, brightness/contrast jitter on the RGB bands, and channel
    dropout on the non-RGB bands (NDVI/CHM/red-edge/etc). Boxes are recomputed
    from the transformed masks after each geometric op rather than transformed
    analytically, so box/mask alignment can't drift out of sync.
    """

    def __init__(self, scale_range=(0.8, 1.2), brightness=0.2, contrast=0.2,
                 channel_dropout_p=0.15, num_rgb_channels=3):
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.channel_dropout_p = channel_dropout_p
        self.num_rgb_channels = num_rgb_channels

    def __call__(self, image, target):
        masks = target["masks"]  # [N, H, W] uint8
        labels = target["labels"]

        if random.random() < 0.5:
            image = image.flip(-1)
            masks = masks.flip(-1)
        if random.random() < 0.5:
            image = image.flip(-2)
            masks = masks.flip(-2)

        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k, dims=(-2, -1))
            masks = torch.rot90(masks, k, dims=(-2, -1))

        scale = random.uniform(*self.scale_range)
        if abs(scale - 1.0) > 1e-3:
            h, w = image.shape[-2:]
            new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                                   mode="bilinear", align_corners=False).squeeze(0)
            if masks.shape[0] > 0:
                masks = F.interpolate(masks.unsqueeze(0).float(), size=(new_h, new_w),
                                       mode="nearest").squeeze(0).to(torch.uint8)
            else:
                masks = masks.new_zeros((0, new_h, new_w))

        n_rgb = min(self.num_rgb_channels, image.shape[0])
        if n_rgb > 0:
            rgb = image[:n_rgb]
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = rgb.mean(dim=(-2, -1), keepdim=True)
            rgb = ((rgb - mean) * contrast_factor + mean) * brightness_factor
            image = torch.cat([rgb.clamp(0.0, 1.0), image[n_rgb:]], dim=0)

        for c in range(self.num_rgb_channels, image.shape[0]):
            if random.random() < self.channel_dropout_p:
                image[c] = image[c].mean()

        keep_masks = []
        keep_labels = []
        boxes = []
        for i in range(masks.shape[0]):
            ys, xs = torch.where(masks[i] > 0)
            if ys.numel() == 0:
                continue
            xmin, xmax = xs.min().item(), xs.max().item()
            ymin, ymax = ys.min().item(), ys.max().item()
            if xmax <= xmin or ymax <= ymin:
                continue
            keep_masks.append(masks[i])
            keep_labels.append(labels[i])
            boxes.append([xmin, ymin, xmax, ymax])

        if keep_masks:
            target["masks"] = torch.stack(keep_masks, dim=0)
            target["labels"] = torch.stack(keep_labels, dim=0)
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        else:
            target["masks"] = masks.new_zeros((0, masks.shape[-2], masks.shape[-1]))
            target["labels"] = labels.new_zeros((0,), dtype=torch.int64)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        return image, target


class StackedImageInstanceMaskDataset(Dataset):
    """
    Expects:
      - image_paths: list of stacked multi-channel GeoTIFF paths (e.g., 5-channel or 8-channel)
      - mask_paths: list of single-channel mask TIFFs

    Mask format (written by scripts/create_masks.py):
      - 0: background (no trees)
      - otherwise: `class_id * label_divisor + instance_index`, so every tree is a
        separate instance that still carries its health class. Each distinct
        non-zero value becomes one instance with label `value // label_divisor`.

    Set `label_divisor=None` for the legacy format where the pixel value IS the
    class (all same-class trees merged into one blob) - kept only for reading old
    datasets, not for training.

    Returns (image_tensor, target) where:
      - image_tensor: FloatTensor [C, H, W], dtype=torch.float32, values in [0, 1] if normalized
      - target: dict with keys:
        - 'boxes': FloatTensor [N,4] in format [xmin, ymin, xmax, ymax]
        - 'labels': Int64Tensor [N] with class labels (1-based)
        - 'masks': UInt8Tensor [N,H,W] binary masks for each instance
        - 'image_id': Int64Tensor [1] with image index
    """
    def __init__(self, image_paths, mask_paths, transforms=None, image_norm=True, drop_alpha=False,
                 label_divisor=10000, input_mode="native", channel_ranges=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.image_norm = image_norm
        self.drop_alpha = drop_alpha
        self.label_divisor = label_divisor
        self.input_mode = input_mode
        self.channel_ranges = channel_ranges or DEFAULT_PREPROCESS_RANGES

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read stacked image (all bands)
        with rasterio.open(self.image_paths[idx]) as src:
            img = src.read().astype(np.float32)  # shape: [C, H, W]

        # Optionally drop an alpha band only when the user explicitly requests it.
        # For multi-band remote sensing data it is safer to preserve all bands by default.
        if self.drop_alpha and img.shape[0] >= 4:
            img = np.delete(img, 3, axis=0)
        if self.image_norm:
            img = process_image_for_model(
                img,
                mode=self.input_mode,
                rgb_divisor=255.0,
                channel_ranges=self.channel_ranges,
            )
        image_tensor = torch.from_numpy(img)

        # Read the single-channel mask (uint16 or uint8)
        # Mask format: 0 = background, 1-N = tree classes or instance IDs
        with rasterio.open(self.mask_paths[idx]) as src:
            mask_data = src.read(1).astype(np.uint16)  # shape: [H, W]

        # Extract unique class/instance IDs from mask
        unique_ids = np.unique(mask_data)
        unique_ids = unique_ids[unique_ids != 0]  # drop background
        
        masks = []
        boxes = []
        labels = []
        
        for mask_id in unique_ids:
            # Create binary mask for this class/instance
            bin_mask = (mask_data == mask_id).astype(np.uint8)
            pos = np.where(bin_mask)
            
            if pos[0].size == 0:
                continue
            
            ymin = float(np.min(pos[0]))
            ymax = float(np.max(pos[0]))
            xmin = float(np.min(pos[1]))
            xmax = float(np.max(pos[1]))
            
            # Skip degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            if self.label_divisor:
                label = int(mask_id) // self.label_divisor
                if label < 1:
                    raise ValueError(
                        f"{self.mask_paths[idx]}: mask value {int(mask_id)} decodes to class {label}. "
                        f"This looks like a legacy class-only mask; regenerate masks with "
                        f"scripts/create_masks.py, or pass label_divisor=None to read the old format."
                    )
            else:
                label = int(mask_id)

            masks.append(bin_mask)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(masks) == 0:
            # Return empty target following torchvision expectations
            target = {
                'boxes': torch.zeros((0,4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.uint8),
                'image_id': torch.tensor([idx], dtype=torch.int64)
            }
        else:
            masks_np = np.stack(masks, axis=0)  # [N, H, W]
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'masks': torch.tensor(masks_np, dtype=torch.uint8),
                'image_id': torch.tensor([idx], dtype=torch.int64)
            }

        # Apply transforms if provided (should handle both image and target)
        if self.transforms:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

