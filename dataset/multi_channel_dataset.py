import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np

class StackedImageInstanceMaskDataset(Dataset):
    """
    Expects:
      - image_paths: list of stacked multi-channel GeoTIFF paths (e.g., 5-channel or 8-channel)
      - mask_paths: list of single-channel mask TIFFs (0 = background, 1..N = class/instance IDs)
    
    Mask format:
      - 0: background (no trees)
      - 1-N: tree classes (e.g., 1-4 for different health classes)
    
    Returns (image_tensor, target) where:
      - image_tensor: FloatTensor [C, H, W], dtype=torch.float32, values in [0, 1] if normalized
      - target: dict with keys:
        - 'boxes': FloatTensor [N,4] in format [xmin, ymin, xmax, ymax]
        - 'labels': Int64Tensor [N] with class labels (1-based, matching mask values)
        - 'masks': UInt8Tensor [N,H,W] binary masks for each instance/class
        - 'image_id': Int64Tensor [1] with image index
    """
    def __init__(self, image_paths, mask_paths, transforms=None, image_norm=True, drop_alpha=False):
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

        # Optionally drop an alpha band only when the user explicitly requests it.
        # For multi-band remote sensing data it is safer to preserve all bands by default.
        if self.drop_alpha and img.shape[0] >= 4:
            img = np.delete(img, 3, axis=0)
        if self.image_norm:
            # Normalize first 3 bands if they look like uint8 RGB imagery.
            if img.dtype == np.float32:
                if img.max() > 1.0 and img.max() <= 255.0:
                    if img.shape[0] >= 3:
                        img[:3] = img[:3] / 255.0
                    else:
                        img = img / 255.0
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
            
            masks.append(bin_mask)
            boxes.append([xmin, ymin, xmax, ymax])
            # Use the mask value as the class label (1-4 for tree classes)
            labels.append(int(mask_id))

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

