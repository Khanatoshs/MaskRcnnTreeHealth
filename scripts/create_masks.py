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


# Mask values encode BOTH instance identity and health class as:
#     value = class_id * LABEL_DIVISOR + instance_index
# e.g. 10003 -> class 1 (healthy), 3rd tree in the plot
#      40127 -> class 4 (severe),  127th tree in the plot
# The dataset recovers the class with `value // LABEL_DIVISOR` and treats every
# distinct value as one instance, so same-class trees no longer merge into a
# single blob. Keep this constant in sync with [TRAIN] mask_label_divisor.
LABEL_DIVISOR = 10000


def generate_instance_class_mask_geotiff(plot_tif_path, trees_gdf, output_mask_dir, class_property="tree_class",
                                         min_tree_height=None, chm_band=4):
    """
    Generates a mask GeoTIFF where every tree is its own instance AND carries its
    health class, encoded as `class_id * LABEL_DIVISOR + instance_index`.

    Background = 0. This is the mask format `train.py` expects; the older
    `generate_class_mask_geotiff` (one pixel value per class) merges every
    same-class tree in a plot into a single instance and should not be used
    for training.

    min_tree_height: if set, drop trees whose height (max CHM inside the crown,
        read from `chm_band` of the plot raster) is below this many metres. The
        surveyed stands here are mature - median height ~21 m - so a 2 m cut only
        removes a handful of annotations with near-zero or negative CHM, i.e.
        ground-level noise rather than real trees.
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
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        logger.info(f"No trees found in {base_id}")
    else:
        intersecting_trees['geometry'] = intersecting_trees.geometry.intersection(plot_polygon)

        instance_index = 0
        for _, row in intersecting_trees.iterrows():
            geom = row.geometry
            if geom.is_empty or geom.area <= 0:
                continue

            try:
                class_id = int(row.get(class_property, 0)) + 1
            except (ValueError, TypeError):
                logger.warning(f"Invalid class value for a tree in {base_id}, skipping it")
                continue

            instance_index += 1
            encoded = class_id * LABEL_DIVISOR + instance_index
            if encoded > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"{base_id}: encoded mask value {encoded} exceeds uint16. "
                    f"Too many trees in one plot for LABEL_DIVISOR={LABEL_DIVISOR}."
                )

            shapes.append((geom, encoded))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        if shapes:
            instance_mask = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=plot_transform,
                fill=0,
                dtype=np.uint16
            )
        else:
            instance_mask = np.zeros((height, width), dtype=np.uint16)

    # Count sub-pixel losses before the height filter runs, so the two reasons a
    # tree can disappear stay separately attributable.
    dropped_subpixel = len(shapes) - len(set(np.unique(instance_mask).tolist()) - {0})

    # Height filter: drop instances whose crown top is below the threshold.
    # Applied after rasterization so the height is measured over the pixels the
    # model would actually see.
    dropped_short = 0
    if min_tree_height is not None and instance_mask.any():
        with rasterio.open(plot_tif_path) as src:
            if src.count < chm_band:
                logger.warning(
                    f"{base_id}: raster has {src.count} bands, no CHM at band {chm_band}; "
                    "skipping the height filter."
                )
            else:
                chm = src.read(chm_band).astype(np.float32)
                for value in np.unique(instance_mask):
                    if value == 0:
                        continue
                    crown = instance_mask == value
                    crown_chm = chm[crown]
                    crown_chm = crown_chm[np.isfinite(crown_chm)]
                    if crown_chm.size == 0 or crown_chm.max() < min_tree_height:
                        instance_mask[crown] = 0
                        dropped_short += 1
                        class_id = int(value) // LABEL_DIVISOR
                        if class_counts.get(class_id):
                            class_counts[class_id] -= 1

    meta.update({
        "count": 1,
        "dtype": "uint16",
        "compress": "lzw",
        "nodata": 0
    })

    output_mask_path = os.path.join(output_mask_dir, f"{base_id}_mask.tif")
    with rasterio.open(output_mask_path, "w", **meta) as dst:
        dst.write(instance_mask, 1)

    # Report both ways a tree can vanish - polygons too small to cover a pixel,
    # and crowns filtered out by height - so annotation loss stays visible.
    rasterized_ids = set(np.unique(instance_mask).tolist()) - {0}
    class_summary = ", ".join(f"class_{cid}:{cnt}" for cid, cnt in sorted(class_counts.items()))
    logger.info(
        f"Saved instance+class mask: {os.path.basename(output_mask_path)} | "
        f"{len(rasterized_ids)} instances rasterized ({class_summary})"
        + (f" | {dropped_subpixel} sub-pixel tree(s) dropped" if dropped_subpixel > 0 else "")
        + (f" | {dropped_short} tree(s) below {min_tree_height} m dropped" if dropped_short else "")
    )
    return len(rasterized_ids), dropped_subpixel + dropped_short


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
    _min_height = config.get('MASKS', 'MIN_TREE_HEIGHT', fallback='').strip()
    MIN_TREE_HEIGHT = float(_min_height) if _min_height else None
    CHM_BAND = int(config.get('MASKS', 'CHM_BAND', fallback='4'))
    if MIN_TREE_HEIGHT is not None:
        logger.info(f"Height filter: dropping trees below {MIN_TREE_HEIGHT} m (CHM band {CHM_BAND})")

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

    # The plot shapefile's `class` attribute splits plots into train/ and test/
    # (see cut_plots.py). Both need masks: train/ feeds training + validation,
    # test/ is the held-out set used only for final reporting.
    splits = [(PLOTS_DIR, MASKS_OUTPUT_DIR)]
    test_plots_dir = config.get('MASKS', 'TEST_PLOTS_DIR', fallback='').strip()
    test_masks_dir = config.get('MASKS', 'TEST_OUTPUT_DIR', fallback='').strip()
    if test_plots_dir and test_masks_dir:
        splits.append((test_plots_dir, test_masks_dir))

    for plots_dir, masks_dir in splits:
        if not os.path.isdir(plots_dir):
            logger.warning(f"Plots directory not found, skipping: {plots_dir}")
            continue
        os.makedirs(masks_dir, exist_ok=True)

        plot_files = sorted(glob.glob(os.path.join(plots_dir, "*.tif")))
        logger.info(f"Processing {len(plot_files)} plot images from {plots_dir} -> {masks_dir}...")

        total_instances = 0
        total_dropped = 0
        for plot_path in plot_files:
            try:
                n_inst, n_dropped = generate_instance_class_mask_geotiff(
                    plot_path, trees_gdf, masks_dir, class_property=TREE_CLASS_PROPERTY,
                    min_tree_height=MIN_TREE_HEIGHT, chm_band=CHM_BAND
                )
                total_instances += n_inst
                total_dropped += n_dropped
            except Exception as e:
                logger.exception(f"Error generating mask for {plot_path}: {e}")

        logger.info(
            f"Finished {plots_dir}: {total_instances} tree instances written to {masks_dir}"
            + (f" ({total_dropped} sub-pixel trees dropped)" if total_dropped else "")
        )

    logger.info("Mask generation completed.")