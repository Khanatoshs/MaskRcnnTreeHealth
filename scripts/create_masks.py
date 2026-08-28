import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import logging
import configparser

logging.basicConfig(level=logging.INFO, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Create_Masks")

config = configparser.ConfigParser()
config_path = "config.ini"
config.read(config_path)


def generate_instance_mask_geotiff(plot_tif_path, trees_gdf, output_mask_dir):
    """
    Generates a single-channel 2D instance mask GeoTIFF for a plot tile.
    Background = 0, Tree Instances = 1, 2, 3...
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    plot_filename = os.path.basename(plot_tif_path)
    base_id = os.path.splitext(plot_filename)[0]

    with rasterio.open(plot_tif_path) as src:
        plot_crs = src.crs
        plot_transform = src.transform
        height, width = src.height, src.width
        plot_bounds = src.bounds
        meta = src.meta.copy()

    plot_polygon = box(*plot_bounds)

    if trees_gdf.crs != plot_crs:
        trees_gdf = trees_gdf.to_crs(plot_crs)

    intersecting_trees = trees_gdf[trees_gdf.intersects(plot_polygon)].copy()

    shapes = []
    if len(intersecting_trees) == 0:
        # Create empty mask if no trees are present
        instance_mask = np.zeros((height, width), dtype=np.uint16)
    else:
        intersecting_trees['geometry'] = intersecting_trees.geometry.intersection(plot_polygon)
        
        # Assign a unique integer ID to each tree instance (1-indexed)
        for instance_id, (_, row) in enumerate(intersecting_trees.iterrows(), start=1):
            geom = row.geometry
            if not geom.is_empty and geom.area > 0:
                shapes.append((geom, instance_id))

        if len(shapes) > 0:
            instance_mask = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=plot_transform,
                fill=0,
                dtype=np.uint16
            )
        else:
            instance_mask = np.zeros((height, width), dtype=np.uint16)

    # Update metadata for a single-channel uint16 mask
    meta.update({
        "count": 1,
        "dtype": "uint16",
        "compress": "lzw",
        "nodata": 0
    })

    output_mask_path = os.path.join(output_mask_dir, f"{base_id}_mask.tif")
    with rasterio.open(output_mask_path, "w", **meta) as dst:
        dst.write(instance_mask, 1)

    logger.info(f"Saved instance mask: {os.path.basename(output_mask_path)} (Found {len(shapes)} trees)")


# Batch Execution
if __name__ == "__main__":
    PLOTS_DIR = config.get('MASKS', 'PLOTS_DIR', fallback='data/cropped_plots')
    TREES_SHAPEFILE = config.get('MASKS', 'TREE_SHAPEFILE', fallback='data/Annotations/NDVI_mean3.shp')
    MASKS_OUTPUT_DIR = config.get('MASKS', 'OUTPUT_DIR', fallback='data/masks')

    logger.info(f"Starting mask generation for plots in {PLOTS_DIR} using tree annotations from {TREES_SHAPEFILE}...")

    if not os.path.exists(MASKS_OUTPUT_DIR):
        os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)

    try:
        trees_gdf = gpd.read_file(TREES_SHAPEFILE)
    except Exception as e:
        logger.error(f"Failed to read tree annotations: {e}")
        raise

    plot_files = sorted(glob.glob(os.path.join(PLOTS_DIR, "*.tif")))
    logger.info(f"Processing {len(plot_files)} plot images from {PLOTS_DIR}...")

    for plot_path in plot_files:
        try:
            generate_instance_mask_geotiff(plot_path, trees_gdf, MASKS_OUTPUT_DIR)
        except Exception as e:
            logger.exception(f"Error generating mask for {plot_path}: {e}")
    logger.info(f"Mask generation completed. Masks saved to {MASKS_OUTPUT_DIR}.")