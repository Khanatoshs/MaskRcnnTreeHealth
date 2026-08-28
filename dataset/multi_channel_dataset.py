import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np

class StackedImageInstanceMaskDataset(Dataset):
    """
    Expects:
      - image_paths: list of stacked multi-channel GeoTIFF paths (e.g., 5-channel)
      - mask_paths: list of single-channel instance-ID mask TIFFs (0 = background, 1..N instance ids)
      - transforms: callable(image, target) -> (image, target) for augmentations (optional)
    Returns (image_tensor, target) where:
      - image_tensor: FloatTensor [C, H, W], dtype=torch.float32
      - target: dict with 'boxes' (FloatTensor [N,4]), 'labels' (Int64Tensor [N]),
                'masks' (UInt8Tensor [N,H,W]), 'image_id' (Int64Tensor [1])
    """
    def __init__(self, image_paths, mask_paths, transforms=None, image_norm=True, drop_alpha=True):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.image_norm = image_norm
        self.drop_alpha = drop_alpha

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read stacked image (all bands)
        with rasterio.open(self.image_paths[idx]) as src:
            img = src.read().astype(np.float32)  # shape: [C, H, W]

        # Optionally drop the 4th channel (alpha) if present
        if self.drop_alpha and img.shape[0] >= 4:
            # Remove band index 3 (0-based)
            img = np.delete(img, 3, axis=0)
        if self.image_norm:
            # If first 3 bands are RGB uint8, scale them; otherwise assume float32 already
            if img.dtype == np.float32:
                # Try a heuristic: if max>1 and <=255 likely needs /255 for first 3 bands
                if img.max() > 1.0 and img.max() <= 255.0:
                    # scale only first 3 channels if present
                    if img.shape[0] >= 3:
                        img[:3] = img[:3] / 255.0
                    else:
                        img = img / 255.0
        image_tensor = torch.from_numpy(img)

        # Read the single-channel instance mask (uint16 or uint8)
        with rasterio.open(self.mask_paths[idx]) as src:
            inst_mask = src.read(1).astype(np.uint16)  # shape: [H, W]

        # Convert instance mask to per-instance binary masks + boxes + labels
        instance_ids = np.unique(inst_mask)
        instance_ids = instance_ids[instance_ids != 0]  # drop background
        masks = []
        boxes = []
        labels = []
        for iid in instance_ids:
            bin_mask = (inst_mask == iid).astype(np.uint8)
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
            masks.append(bin_mask)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # default single class 'tree' -> 1

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

