import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np


# =====================================================================
# 2. DATASET CLASS FOR RGB + NDVI + CHM
# =====================================================================

class UAV5ChannelDataset(Dataset):
    """
    Dataset loader for 5-channel UAV tiles.
    Expects matching list of file paths for RGB, NDVI, and CHM rasters.
    """
    def __init__(self, rgb_paths, ndvi_paths, chm_paths, annotations):
        self.rgb_paths = rgb_paths
        self.ndvi_paths = ndvi_paths
        self.chm_paths = chm_paths
        self.annotations = annotations

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 1. Read RGB image (3 channels) -> Shape: [3, H, W]
        with rasterio.open(self.rgb_paths[idx]) as src:
            rgb = src.read().astype(np.float32) / 255.0  # Scale RGB to [0, 1]

        # 2. Read NDVI raster (1 channel) -> Shape: [1, H, W]
        with rasterio.open(self.ndvi_paths[idx]) as src:
            ndvi = src.read(1).astype(np.float32)
            ndvi = np.expand_dims(ndvi, axis=0)  # Add channel dim

        # 3. Read CHM raster (1 channel) -> Shape: [1, H, W]
        with rasterio.open(self.chm_paths[idx]) as src:
            chm = src.read(1).astype(np.float32)
            chm = np.expand_dims(chm, axis=0)   # Add channel dim

        # 4. Stack into a 5-channel array -> Shape: [5, H, W]
        combined_5ch = np.concatenate([rgb, ndvi, chm], axis=0)
        image_tensor = torch.from_numpy(combined_5ch)

        # 5. Load Target Annotations
        ann = self.annotations[idx]
        target = {
            'boxes': torch.tensor(ann['boxes'], dtype=torch.float32),  # [N, 4] (xmin, ymin, xmax, ymax)
            'labels': torch.tensor(ann['labels'], dtype=torch.int64),  # [N] Class labels (1..num_classes-1)
            'masks': torch.tensor(ann['masks'], dtype=torch.uint8),    # [N, H, W] Binary instance masks
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target

