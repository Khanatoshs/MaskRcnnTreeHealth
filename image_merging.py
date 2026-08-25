import rasterio
import numpy as np
import torch

def prepare_5channel_uav_tile(rgb_path, ndvi_path, chm_path):
    # 1. Read RGB (3 bands)
    with rasterio.open(rgb_path) as src_rgb:
        rgb_data = src_rgb.read()  # Shape: [3, H, W]
        
    # 2. Read NDVI (1 band)
    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi_data = src_ndvi.read(1)  # Shape: [H, W]
        ndvi_data = np.expand_dims(ndvi_data, axis=0)  # Shape: [1, H, W]
        
    # 3. Read CHM (1 band)
    with rasterio.open(chm_path) as src_chm:
        chm_data = src_chm.read(1)  # Shape: [H, W]
        chm_data = np.expand_dims(chm_data, axis=0)  # Shape: [1, H, W]
        
    # 4. Concatenate along channel axis -> [5, H, W]
    combined_5ch = np.concatenate([rgb_data, ndvi_data, chm_data], axis=0).astype(np.float32)
    
    # 5. Convert to PyTorch Tensor
    tensor_5ch = torch.from_numpy(combined_5ch)
    return tensor_5ch