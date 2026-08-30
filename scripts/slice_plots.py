import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import logging
import configparser



config = configparser.ConfigParser()
config_path = os.path.join("config.ini")
config.read(config_path)

log_level = getattr(logging, config.get('SLICING', 'log_level', fallback='INFO'))
logging.basicConfig(level=log_level, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Slice_Plots")

def slice_plots_and_masks(
    plots_dir,
    masks_dir,
    output_base_dir,
    window_size=640,
    overlap_pct=20.0,  # overlap percentage (0-100) of window size to overlap between tiles
    drop_empty_masks=True,
    min_instance_completeness=None
):
    """
    Slices matched plot rasters and mask rasters using an overlapping sliding window.
    
    Parameters:
    -----------
    plots_dir : str
        Directory containing input plot GeoTIFFs (e.g., 5-channel).
    masks_dir : str
        Directory containing matching single-channel instance mask GeoTIFFs.
    output_base_dir : str
        Root output folder. Creates 'images/' and 'masks/' subdirectories inside.
    window_size : int
        Width/Height of sliced tiles (e.g., 640 for 640x640 px).
    overlap_pct : float
        Percentage of the window (0-100) that should overlap between adjacent
        tiles. Internally converted to pixel stride as
        `stride = window_size - int(round(window_size * overlap_pct / 100))`.
    drop_empty_masks : bool
        If True, skips writing tiles that contain no target tree instances (mask sum == 0).
    min_instance_completeness : float or None
        If set (0-1), removes an instance from a tile when less than this fraction of
        its full area in the parent plot falls inside the tile. Tiling slices crowns at
        the tile border, leaving slivers that become tiny ground-truth boxes the model
        cannot match - every COCO-'small' instance in this dataset is such a fragment,
        not a genuinely small tree. Because tiles overlap, a crown cut at one tile's
        edge is whole in a neighbour, so dropping fragments loses no trees.
    """
    # 1. Setup output directory structure
    out_images_dir = os.path.join(output_base_dir, "images")
    out_masks_dir = os.path.join(output_base_dir, "masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    # Compute stride from overlap percentage (clamped)
    if overlap_pct < 0:
        overlap_pct = 0.0
    if overlap_pct > 99.0:
        overlap_pct = 99.0
    overlap_pixels = int(round(window_size * (overlap_pct / 100.0)))
    stride = window_size - overlap_pixels
    if stride < 1:
        stride = 1

    plot_paths = sorted(glob.glob(os.path.join(plots_dir, "*.tif")))
    logger.info(f"Found {len(plot_paths)} plot rasters to process. Window={window_size}px, Overlap={overlap_pct}%, Stride={stride}px")

    total_saved_tiles = 0

    for plot_path in plot_paths:
        base_name = os.path.splitext(os.path.basename(plot_path))[0]
        
        # Locate matching mask file (expects filename format: {base_name}_mask.tif or similar match)
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.tif")
        logger.debug(f"Processing plot: {plot_path} | Expected mask: {mask_path}")
        if not os.path.exists(mask_path):
            # Fallback check if mask shares the exact same file name as plot
            mask_path = os.path.join(masks_dir, f"{base_name}.tif")
            logger.debug(f"Checking fallback mask path: {mask_path}")
            if not os.path.exists(mask_path):
                logger.warning(f"No matching mask found for {base_name}, skipping.")
                continue

        # 2. Open plot and mask handles simultaneously
        with rasterio.open(plot_path) as src_img, rasterio.open(mask_path) as src_mask:
            width = src_img.width
            height = src_img.height

            img_meta = src_img.meta.copy()
            mask_meta = src_mask.meta.copy()

            # Full pixel area of each instance in the whole plot, so per-tile
            # completeness can be measured against it.
            full_area = {}
            if min_instance_completeness:
                whole_mask = src_mask.read(1)
                values, counts = np.unique(whole_mask, return_counts=True)
                full_area = {int(v): int(c) for v, c in zip(values, counts) if v != 0}
                del whole_mask

            plot_tile_count = 0
            plot_fragments_dropped = 0

            # 3. Sliding window iteration over grid
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    # Handle boundaries (ensures edge tiles stay fixed at window_size)
                    win_w = min(window_size, width - x)
                    win_h = min(window_size, height - y)

                    # Skip edge remnants smaller than half the window size
                    if win_w < window_size // 2 or win_h < window_size // 2:
                        continue

                    window = Window(x, y, win_w, win_h)
                    
                    # Read mask window first to check instance presence
                    mask_data = src_mask.read(window=window)

                    # Drop crowns the tile boundary cut through, before the
                    # emptiness check so a tile of pure fragments is discarded too.
                    if min_instance_completeness and full_area:
                        values, counts = np.unique(mask_data, return_counts=True)
                        for value, count in zip(values, counts):
                            if value == 0:
                                continue
                            whole = full_area.get(int(value))
                            if whole and count / whole < min_instance_completeness:
                                mask_data[mask_data == value] = 0
                                plot_fragments_dropped += 1

                    # Skip saving tile if there are no tree instances present
                    if drop_empty_masks and mask_data.sum() == 0:
                        continue

                    # Read corresponding image window
                    img_data = src_img.read(window=window)

                    # Calculate local spatial transform for the tile
                    tile_transform = rasterio.windows.transform(window, src_img.transform)

                    # Update raster metadata for output writing
                    tile_img_meta = img_meta.copy()
                    tile_img_meta.update({
                        "height": win_h,
                        "width": win_w,
                        "transform": tile_transform
                    })

                    tile_mask_meta = mask_meta.copy()
                    tile_mask_meta.update({
                        "height": win_h,
                        "width": win_w,
                        "transform": tile_transform
                    })

                    # Define unique file names for tile pairs
                    tile_filename = f"{base_name}_tile_y{y}_x{x}.tif"
                    out_img_path = os.path.join(out_images_dir, tile_filename)
                    out_mask_path = os.path.join(out_masks_dir, tile_filename)

                    # Write sliced multi-channel image tile
                    with rasterio.open(out_img_path, "w", **tile_img_meta) as dst_img:
                        dst_img.write(img_data)

                    # Write matching single-channel mask tile
                    with rasterio.open(out_mask_path, "w", **tile_mask_meta) as dst_mask:
                        dst_mask.write(mask_data)

                    plot_tile_count += 1

            total_saved_tiles += plot_tile_count
            logger.info(
                f"Processed {base_name}: Generated {plot_tile_count} matched tile pairs."
                + (f" Dropped {plot_fragments_dropped} edge-truncated instance(s)."
                   if plot_fragments_dropped else "")
            )

    logger.info(f"\nCompleted! Total dataset generated: {total_saved_tiles} image/mask pairs.")
    logger.info(f"Images directory: {out_images_dir}")
    logger.info(f"Masks directory:  {out_masks_dir}")


# =====================================================================
# CONFIGURATION & EXECUTION
# =====================================================================
if __name__ == "__main__":
    PLOTS_DIR = config.get('SLICING', 'PLOTS_DIR', fallback='data/cropped_plots')
    MASKS_DIR = config.get('SLICING', 'MASKS_DIR', fallback='data/plot_masks')
    OUTPUT_DATASET_DIR = config.get('SLICING', 'OUTPUT_DIR', fallback='data/dataset_sliced_640')

    # --- Window Configuration ---
    WINDOW_SIZE = int(config.get('SLICING', 'WINDOW_SIZE', fallback='640'))
    OVERLAP_PCT = float(config.get('SLICING', 'OVERLAP_PCT', fallback='20.0'))
    _completeness = config.get('SLICING', 'MIN_INSTANCE_COMPLETENESS', fallback='').strip()
    MIN_INSTANCE_COMPLETENESS = float(_completeness) if _completeness else None
    if MIN_INSTANCE_COMPLETENESS:
        logger.info(
            f"Dropping instances less than {MIN_INSTANCE_COMPLETENESS:.0%} complete within a tile"
        )

    logger.info(f"Starting slicing of plots and masks with window size {WINDOW_SIZE}px and overlap {OVERLAP_PCT}%...")

    # Slice the held-out test plots into their own dataset directory too, so the
    # final evaluation never has to touch the train/val dataset.
    splits = [(PLOTS_DIR, MASKS_DIR, OUTPUT_DATASET_DIR)]
    test_plots_dir = config.get('SLICING', 'TEST_PLOTS_DIR', fallback='').strip()
    test_masks_dir = config.get('SLICING', 'TEST_MASKS_DIR', fallback='').strip()
    test_output_dir = config.get('SLICING', 'TEST_OUTPUT_DIR', fallback='').strip()
    if test_plots_dir and test_masks_dir and test_output_dir:
        splits.append((test_plots_dir, test_masks_dir, test_output_dir))

    for plots_dir, masks_dir, output_dir in splits:
        if not os.path.isdir(plots_dir):
            logger.warning(f"Plots directory not found, skipping: {plots_dir}")
            continue
        slice_plots_and_masks(
            plots_dir=plots_dir,
            masks_dir=masks_dir,
            output_base_dir=output_dir,
            window_size=WINDOW_SIZE,
            overlap_pct=OVERLAP_PCT,
            drop_empty_masks=True,  # Filter out background-only tiles
            min_instance_completeness=MIN_INSTANCE_COMPLETENESS
        )
        logger.info(f"Slicing completed. Dataset saved to {output_dir}.")