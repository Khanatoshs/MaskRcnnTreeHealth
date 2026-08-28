import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
import logging
import configparser

logging.basicConfig(level=logging.INFO, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Cut_Plots")

def crop_plots_from_vrt(vrt_path, plot_shapefile_path, output_dir, id_column="plot_id"):
    """
    Crops plot polygons from a 5-channel VRT raster and saves them 
    as individual 5-channel GeoTIFFs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the plot boundary shapefile
    plots_gdf = gpd.read_file(plot_shapefile_path)
    logger.info(f"Loaded {len(plots_gdf)} plot boundaries from {plot_shapefile_path}")

    # 2. Open VRT raster handle (Loads metadata only, 0 RAM used)
    with rasterio.open(vrt_path) as src:
        vrt_crs = src.crs
        logger.info(f"VRT Raster Channels: {src.count} | CRS: {vrt_crs}")
        
        # Ensure plot geometries match the raster's CRS
        if plots_gdf.crs != vrt_crs:
            logger.info(f"Reprojecting plots from {plots_gdf.crs} to {vrt_crs}...")
            plots_gdf = plots_gdf.to_crs(vrt_crs)

        # 3. Process plot by plot
        for idx, row in plots_gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            
            # Get plot ID for output naming
            plot_id = row[id_column] if id_column in row and row[id_column] else f"plot_{idx}"

            try:
                # Wrap the source with a WarpedVRT forcing a single dtype (float32)
                with WarpedVRT(src, dtype='float32') as vrt:
                    out_image, out_transform = mask(
                        vrt,
                        geom,
                        crop=True,      # Fits pixel grid tightly around the polygon bounds
                        nodata=0        # Sets area outside plot polygon to 0
                    )
                
                # Update metadata for the plot file
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "dtype": "float32",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw"  # Keeps file size small on disk
                })

                # Write out 5-channel plot GeoTIFF
                output_plot_path = os.path.join(output_dir, f"plot_{plot_id}.tif")
                with rasterio.open(output_plot_path, "w", **out_meta) as dst:
                    dst.write(out_image.astype('float32'))

                logger.info(f"Saved: plot_{plot_id}.tif | Size: {out_image.shape[2]}x{out_image.shape[1]} px")

            except ValueError as e:
                # Skips if plot geometry falls outside the raster bounds
                logger.exception(f"Error details: {e}")
                return
            except Exception as e:
                logger.exception(f"Error processing plot {plot_id}: {e}")
                return
            
    logger.info(f"\nFinished cropping all plots! Saved to: {output_dir}")


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":

    # Load configuration settings
    config = configparser.ConfigParser()
    config_path = "config.ini"
    config.read(config_path)

    logger.info(f"Starting plot cropping process ...")

    VRT_FILE = config.get("CUT_PLOTS", "VRT_FILE")
    PLOT_SHAPEFILE = config.get("CUT_PLOTS", "PLOT_SHAPEFILE")
    OUTPUT_FOLDER = config.get("CUT_PLOTS", "OUTPUT_FOLDER")

    crop_plots_from_vrt(
        vrt_path=VRT_FILE,
        plot_shapefile_path=PLOT_SHAPEFILE,
        output_dir=OUTPUT_FOLDER,
        id_column="plot_id"  # Attribute column used to name files (e.g. plot_1.tif)
    )