from osgeo import gdal
import configparser
import logging
import os

logging.basicConfig(level=logging.INFO, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Image_Merging_VRT")

def create_single_band_vrt(input_raster_path, output_vrt_path, band_number):
    """Create a one-band VRT view of a selected source-raster band."""
    vrt_options = gdal.TranslateOptions(
        format="VRT",
        bandList=[band_number],
    )
    band_vrt = gdal.Translate(output_vrt_path, input_raster_path, options=vrt_options)
    if band_vrt is None:
        raise RuntimeError(
            f"Failed to create RGB band {band_number} VRT from {input_raster_path}"
        )
    band_vrt = None

def create_stacked_vrt(input_raster_list, output_vrt_path):
    """
    Creates a multi-channel Virtual Raster (VRT) by stacking multiple rasters
    as separate bands. Consumes near-zero RAM and generates instantly.
    
    Parameters:
    -----------
    input_raster_list : list of str
        Ordered list of raster filepaths to stack.
        Order dictates output channel indices:
        Bands 1-3: RGB, Band 4: CHM, Band 5: NDVI (or order provided).
    output_vrt_path : str
        Path where the XML .vrt header file will be written.
    """
    # Force GDAL to handle exceptions gracefully in Python
    gdal.UseExceptions()
    
    # 'separate=True' stacks files as separate bands instead of mosaicing spatially
    vrt_options = gdal.BuildVRTOptions(
        separate=True,
        resampleAlg='bilinear'  # Automatically resamples if resolutions slightly differ
    )

    logger.info("Building Virtual Raster Stack...")
    
    # Generate the VRT header XML file on disk
    vrt_doc = gdal.BuildVRT(
        output_vrt_path, 
        input_raster_list, 
        options=vrt_options,
        outputDatatype=gdal.GDT_Float32
    )
    
    # Flush to disk and close handle
    vrt_doc = None
    
    logger.info(f"Success! VRT created at: {output_vrt_path}")


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config.ini")
    logger.info("Starting VRT creation process based on configuration settings...")
    output_dir = config.get("MULTICHANNEL", "output_dir")
    rgb_path = config.get("MULTICHANNEL", "rgb_img_path")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Build one-band VRT views so RGB bands 1-3 are included and alpha band 4 is excluded.
    rgb_band_vrts = []
    for band_number in (1, 2, 3):
        rgb_band_vrt = os.path.join(
            output_dir,
            f"{config.get('MULTICHANNEL', 'output_name')}_rgb_band_{band_number}.vrt",
        )
        create_single_band_vrt(rgb_path, rgb_band_vrt, band_number)
        rgb_band_vrts.append(rgb_band_vrt)

    # Define paths to the source bands (Order defines output Band 1..N).
    input_rasters = [
        *rgb_band_vrts,  # Bands 1-3 (R, G, B)
        config.get("MULTICHANNEL", "chm_img_path"),    # Band 4 (CHM)
        config.get("MULTICHANNEL", "ndvi_img_path"),   # Band 5 (NDVI)
        config.get("MULTICHANNEL", "cire_img_path"),   # Band 6 (CIRE)
        config.get("MULTICHANNEL", "gndvi_img_path"),  # Band 7 (GNDVI)
        config.get("MULTICHANNEL", "ndre_img_path"),   # Band 8 (NDRE)
    ]

    output_path = os.path.join(config.get("MULTICHANNEL","output_dir"),
                              config.get("MULTICHANNEL","output_name") + ".vrt") 

    output_vrt = output_path

    create_stacked_vrt(input_rasters, output_vrt)