from osgeo import gdal
import configparser
import logging
import os

logging.basicConfig(level=logging.INFO, filename = "utils.log" ,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Image_Merging_VRT")

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
    # Define paths to your large 6 GB files (Order defines Band 1..N)
    input_rasters = [
        config.get("MULTICHANNEL","rgb_img_path"),   # Bands 1, 2, 3 (R, G, B)
        config.get("MULTICHANNEL","chm_img_path"),   # Band 4 (Canopy Height Model)
        config.get("MULTICHANNEL","ndvi_img_path")   # Band 5 (NDVI)
    ]

    output_path = os.path.join(config.get("MULTICHANNEL","output_dir"),
                              config.get("MULTICHANNEL","output_name") + ".vrt") 

    if not os.path.exists(config.get("MULTICHANNEL","output_dir")):
        os.makedirs(config.get("MULTICHANNEL","output_dir"))
        logger.info(f"Created output directory: {config.get('MULTICHANNEL','output_dir')}")

    output_vrt = output_path

    create_stacked_vrt(input_rasters, output_vrt)