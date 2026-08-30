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


def generate_class_mask_geotiff(plot_tif_path, trees_gdf, output_mask_dir, class_property="tree_class"):
    """
    Generates a classification mask GeoTIFF for a plot tile.
    Background = 0, Tree pixels = class_id from tree_class property
    Handles multiple tree classes based on the tree_class property.
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
    class_counts = {}
    
    if len(intersecting_trees) == 0:
        # Create empty mask if no trees are present
        class_mask = np.zeros((height, width), dtype=np.uint8)
        logger.info(f"No trees found in {base_id}")
    else:
        intersecting_trees['geometry'] = intersecting_trees.geometry.intersection(plot_polygon)
        
        # Assign class ID based on tree_class property
        for _, row in intersecting_trees.iterrows():
            geom = row.geometry
            if not geom.is_empty and geom.area > 0:
                # Get the class ID from the tree_class property and add 1
                # (tree_class values 0-3 become 1-4, background is 0)
                try:
                    class_id = int(row.get(class_property, 0)) + 1
                except (ValueError, TypeError):
                    logger.warning(f"Invalid class value for tree, using 0")
                    class_id = 0
                
                shapes.append((geom, class_id))
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        if len(shapes) > 0:
            class_mask = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=plot_transform,
                fill=0,
                dtype=np.uint8
            )
        else:
            class_mask = np.zeros((height, width), dtype=np.uint8)

    # Update metadata for classification mask (uint8 for class indices)
    meta.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 0
    })

    output_mask_path = os.path.join(output_mask_dir, f"{base_id}_mask.tif")
    with rasterio.open(output_mask_path, "w", **meta) as dst:
        dst.write(class_mask, 1)

    class_summary = ", ".join([f"class_{cid}:{count}" for cid, count in sorted(class_counts.items())])
    logger.info(f"Saved class mask: {os.path.basename(output_mask_path)} (Trees: {class_summary})")


# Batch Execution
if __name__ == "__main__":
    PLOTS_DIR = config.get('MASKS', 'PLOTS_DIR', fallback='data/cropped_plots')
    TREES_SHAPEFILE = config.get('MASKS', 'TREE_SHAPEFILE', fallback='data/Annotations/NDVI_mean3.shp')
    MASKS_OUTPUT_DIR = config.get('MASKS', 'OUTPUT_DIR', fallback='data/masks')
    TREE_CLASS_PROPERTY = config.get('MASKS', 'TREE_CLASS_PROPERTY', fallback='tree_class')

    logger.info(f"Starting mask generation for plots in {PLOTS_DIR} using tree annotations from {TREES_SHAPEFILE}...")
    logger.info(f"Using tree class property: '{TREE_CLASS_PROPERTY}'")

    if not os.path.exists(MASKS_OUTPUT_DIR):
        os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)

    try:
        trees_gdf = gpd.read_file(TREES_SHAPEFILE)
        logger.info(f"Loaded tree shapefile with columns: {list(trees_gdf.columns)}")
        
        # Check if tree_class property exists
        if TREE_CLASS_PROPERTY not in trees_gdf.columns:
            logger.warning(f"Property '{TREE_CLASS_PROPERTY}' not found in shapefile!")
            logger.warning(f"Available properties: {list(trees_gdf.columns)}")
        else:
            unique_classes = trees_gdf[TREE_CLASS_PROPERTY].unique()
            logger.info(f"Found {len(unique_classes)} unique tree classes: {sorted(unique_classes)}")
            
    except Exception as e:
        logger.error(f"Failed to read tree annotations: {e}")
        raise

    plot_files = sorted(glob.glob(os.path.join(PLOTS_DIR, "*.tif")))
    logger.info(f"Processing {len(plot_files)} plot images from {PLOTS_DIR}...")

    for plot_path in plot_files:
        try:
            # Generate instance mask (optional, for reference)
            # generate_instance_mask_geotiff(plot_path, trees_gdf, MASKS_OUTPUT_DIR)
            
            # Generate class mask (primary output with tree classifications)
            generate_class_mask_geotiff(plot_path, trees_gdf, MASKS_OUTPUT_DIR, class_property=TREE_CLASS_PROPERTY)
        except Exception as e:
            logger.exception(f"Error generating mask for {plot_path}: {e}")
    
    logger.info(f"Mask generation completed. Classification masks saved to {MASKS_OUTPUT_DIR}.")