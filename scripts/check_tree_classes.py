#!/usr/bin/env python3
"""
Quick validation script to check tree_class values in the shapefile.
"""
import geopandas as gpd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

TREE_SHAPEFILE = config.get('MASKS', 'TREE_SHAPEFILE', fallback='data/annotations/NDVI_mean3.shp')
TREE_CLASS_PROPERTY = config.get('MASKS', 'TREE_CLASS_PROPERTY', fallback='tree_class')

print(f"Loading shapefile: {TREE_SHAPEFILE}")
print(f"Checking property: '{TREE_CLASS_PROPERTY}'\n")

try:
    trees_gdf = gpd.read_file(TREE_SHAPEFILE)
    print(f"Total features: {len(trees_gdf)}")
    print(f"Available columns: {list(trees_gdf.columns)}\n")
    
    if TREE_CLASS_PROPERTY not in trees_gdf.columns:
        print(f"❌ ERROR: Property '{TREE_CLASS_PROPERTY}' NOT found!")
        print(f"Available properties: {list(trees_gdf.columns)}")
    else:
        print(f"✓ Property '{TREE_CLASS_PROPERTY}' found\n")
        
        # Check unique values
        unique_values = sorted(trees_gdf[TREE_CLASS_PROPERTY].dropna().unique())
        print(f"Unique tree_class values: {unique_values}")
        print(f"Min value: {min(unique_values)}")
        print(f"Max value: {max(unique_values)}")
        print(f"Value range: {min(unique_values)}-{max(unique_values)}\n")
        
        # Check for values in expected range (0-4)
        if min(unique_values) >= 0 and max(unique_values) <= 4:
            print("✓ All values are in expected range (0-4)")
        else:
            print("⚠ WARNING: Some values are outside 0-4 range!")
            print("  You may need to remap the class values")
        
        # Check for None/NaN values
        missing = trees_gdf[TREE_CLASS_PROPERTY].isna().sum()
        if missing > 0:
            print(f"\n⚠ WARNING: {missing} features have missing tree_class values")
        else:
            print(f"\n✓ All {len(trees_gdf)} features have tree_class values")
            
except Exception as e:
    print(f"Error reading shapefile: {e}")
