#!/usr/bin/env python
"""
Create a demo GeoTIFF from the already downloaded Sentinel-2 data for the Streamlit demo.
"""
import rasterio
import glob
import numpy as np
import os

def create_demo_geotiff():
    """Create a demo GeoTIFF from downloaded Sentinel-2 data."""
    
    # Find downloaded SAFE folders
    raw_dir = Path("data/raw")
    safe_dirs = list(raw_dir.glob("S2*_Lucknow*/*.SAFE"))
    
    if not safe_dirs:
        print("No SAFE folders found in data/raw/")
        return
    
    print(f"Found {len(safe_dirs)} SAFE folders")
    
    # Use the first one that has all bands
    for safe_dir in safe_dirs:
        safe_dir = Path(safe_dir)
        print(f"Checking: {safe_dir.name}")
        
        # Find RGB bands (B4, B3, B2) at 10m resolution
        b4_files = list(safe_dir.rglob("*B04*10m.jp2"))
        b3_files = list(safe_dir.rglob("*B03*10m.jp2"))
        b2_files = list(safe_dir.rglob("*B02*10m.jp2"))
        
        if not (b4_files and b3_files and b2_files):
            print(f"  Missing RGB bands in {safe_dir.name}")
            continue
            
        b4_path = b4_files[0]
        b3_path = b3_files[0]
        b2_path = b2_files[0]
        
        print(f"  Found RGB bands:")
        print(f"  B4 (Red): {b4_path.name}")
        print(f"  B3 (Green): {b3_path.name}")
        print(f"  B2 (Blue): {b2_path.name}")
        
        # Read and create RGB GeoTIFF
        with rasterio.open(b4_path) as b4_src, \
             rasterio.open(b3_path) as b3_src, \
             rasterio.open(b2_path) as b2_src:
            
            # Read bands
            b4 = b4_src.read(1)
            b3 = b3_src.read(1)
            b2 = b2_src.read(1)
            
            # Stack RGB
            rgb = np.stack([b4, b3, b2], axis=0)  # (3, H, W)
            
            # Get metadata from B4
            meta = b4_src.meta.copy()
            meta.update({
                "count": 3,
                "dtype": rasterio.uint16,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            })
            
            # Save RGB GeoTIFF
            output_path = Path("demo_rgb.tif")
            with rasterio.open("demo_rgb.tif", "w", **meta) as dst:
                dst.write(rgb)
            
            print(f"✅ Created demo_rgb.tif ({meta['width']}x{meta['height']})")
            print(f"   CRS: {b4_src.crs}")
            print(f"   Bounds: {b4_src.bounds}")
            return "demo_rgb.tif"
    
    print("No suitable SAFE folder found with all RGB bands")
    return None


if __name__ == "__main__":
    from pathlib import Path
    create_demo_geotiff()