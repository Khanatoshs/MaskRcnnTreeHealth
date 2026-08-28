import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import configparser
import logging

logging.basicConfig(level=logging.INFO, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Image_Merging")

def create_n_channel_stack(input_file_dict, output_filepath):
    """
    Combines arbitrary raster channels into an N-channel GeoTIFF.
    Resamples all input rasters to match the spatial dimensions & CRS 
    of the primary base image (first entry in dictionary).

    Parameters:
    ----------
    input_file_dict : dict
        Ordered dictionary mapping channel names to filepaths.
        Example:
        {
            'rgb': 'data/rgb.tif',      # Contains 3 channels (R, G, B)
            'ndvi': 'data/ndvi.tif',    # Contains 1 channel
            'chm': 'data/chm.tif'       # Contains 1 channel
        }
    output_filepath : str
        Path where the final N-channel GeoTIFF will be saved.
    """
    
    # Use the first dataset in the dictionary as the reference master grid
    base_name, base_path = next(iter(input_file_dict.items()))
    
    with rasterio.open(base_path) as base_src:
        profile = base_src.profile.copy()
        target_height = base_src.height
        target_width = base_src.width
        target_transform = base_src.transform
        target_crs = base_src.crs
        
        logger.info(f"Master Reference Image: {base_name} ({base_path})")
        logger.info(f"Target Dimensions: {target_width}x{target_height} pixels | CRS: {target_crs}\n")

    stacked_bands = []
    channel_names_log = []

    def is_file_match(current_src):
        """
        Validates that the current raster matches the target CRS and dimensions.
        Returns True if matching, False otherwise.
        """
        if current_src.crs != target_crs or current_src.width != target_width or current_src.height != target_height:
            logger.warning(f"CRS or dimensions mismatch for {current_src.name}. Resampling to match target grid.")
            return False
        return True

    # Iterate through all input files and align/resample to reference grid
    for name, filepath in input_file_dict.items():
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"Input file not found: {filepath}")

        with rasterio.open(filepath) as src:
            num_bands = src.count

            if is_file_match(src):
                # Directly read bands without resampling
                data = src.read().astype(np.float32)
                logger.info(f"Reading {num_bands} bands from {name} ({filepath}) without resampling.")
            else:
                # Read and resample every band in the dataset to match target spatial grid
                data = src.read(
                    out_shape=(num_bands, target_height, target_width),
                    resampling=Resampling.bilinear
                ).astype(np.float32)

            # Store bands individually
            for band_idx in range(num_bands):
                stacked_bands.append(data[band_idx])
                channel_names_log.append(f"{name}_band_{band_idx + 1}")

    # Stack along the channel/band dimension -> Shape: [N, Height, Width]
    n_channel_array = np.stack(stacked_bands, axis=0)
    total_channels = n_channel_array.shape[0]

    # Update metadata profile for N-channel float output
    profile.update({
        'count': total_channels,
        'dtype': 'float32',
        'height': target_height,
        'width': target_width,
        'transform': target_transform,
        'crs': target_crs,
        'driver': 'GTiff'
    })

    # Write out the N-channel GeoTIFF (ensure directory exists)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with rasterio.open(output_filepath, 'w', **profile) as dst:
        for i in range(total_channels):
            # Ensure each band is written with consistent dtype
            dst.write(n_channel_array[i].astype('float32'), i + 1)
            # Write channel descriptive tags into GeoTIFF metadata
            dst.update_tags(i + 1, name=channel_names_log[i])

    logger.info(f"Successfully generated {total_channels}-Channel Image Stack!")
    logger.info(f"Output File: {output_filepath}")
    logger.info("Channel Map Log:")
    for idx, cname in enumerate(channel_names_log):
        logger.info(f"  - Channel {idx + 1}: {cname}")


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    config.read(config_path)
    logger.info("Started N-Channel Image Merging Process")

    # 1. Define your inputs (Order dictates channel index in resulting output)
    #    Add as many channels as you want in the future!
    inputs = {
        'rgb': config.get('MULTICHANNEL', 'rgb_img_path'),
        'ndvi': config.get('MULTICHANNEL', 'ndvi_img_path'),
        'chm': config.get('MULTICHANNEL', 'chm_img_path'),
        # Future channels (e.g., Thermal, RedEdge, Intensity) can be added seamlessly:
        # 'thermal': 'tiles/tile_0_thermal.tif', 
        # 'rededge': 'tiles/tile_0_rededge.tif',
    }
    logger.debug(f"Input files: {inputs}")
    output_stack_path = config.get('MULTICHANNEL', 'output_dir') + '/' + config.get('MULTICHANNEL', 'output_name')
    logger.debug(f"Output path: {output_stack_path}")
    # Execute channel stacking
    try:
        create_n_channel_stack(inputs, output_stack_path)
    except Exception as e:
        print(e)
        logger.error(f"Error occurred while creating N-channel stack: {e}")