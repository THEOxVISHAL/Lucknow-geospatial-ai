"""
Lucknow Land Cover - Data Preparation Script
Run: python prepare_data.py
"""
import sys
sys.path.append('.')

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject
import geopandas as gpd
from pathlib import Path
import yaml
from tqdm.auto import tqdm

from src.preprocessing.compute_indices import compute_indices, stack_composite

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

AOI_PATH = Path(config['project']['aoi_path'])
RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
LABELS_DIR = Path('data/labels')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DATES = config['project']['dates']
BANDS = config['data']['bands']  # [2,3,4,5,6,7,8,8a,11,12]

print('AOI:', AOI_PATH)
print('Dates:', DATES)
print('Bands:', BANDS)

# Find SAFE folders
safe_folders = {}
for d in DATES:
    date_str = d.replace('-', '')  # YYYYMMDD
    matches = list(RAW_DIR.glob(f'*{date_str}*/'))
    if not matches:
        matches = list(RAW_DIR.glob(f'*{date_str[:6]}*/'))
    if not matches:
        matches = list(RAW_DIR.glob('S2*/'))
    if matches:
        safe_folders[d] = matches[0]
        print(f'{d}: {matches[0].name}')
    else:
        print(f'{d}: NOT FOUND')

if len(safe_folders) != 4:
    print('WARNING: Expected 4 dates, found', len(safe_folders))

def read_s2_bands(safe_path, bands, target_res=10):
    safe_path = Path(safe_path)
    
    safe_dirs = list(safe_path.glob('*.SAFE'))
    if not safe_dirs:
        raise FileNotFoundError(f'No .SAFE directory in {safe_path}')
    safe_dir = safe_dirs[0]
    
    granule_dirs = list(safe_dir.glob('GRANULE/*/IMG_DATA'))
    if not granule_dirs:
        # Try alternative structure
        granule_dirs = list(safe_dir.glob('GRANULE/L2A_*/IMG_DATA'))
    if not granule_dirs:
        raise FileNotFoundError(f'No GRANULE/IMG_DATA in {safe_dir}')
    img_data = granule_dirs[0]
    
    band_arrays = {}
    reference_shape = None
    reference_transform = None
    reference_crs = None
    
    # Band resolution mapping (10m: B2,B3,B4,B8; 20m: B5,B6,B7,B8A,B11,B12)
    band_res = {
        2: 10, 3: 10, 4: 10, 8: 10,
        5: 20, 6: 20, 7: 20, '8a': 20, 11: 20, 12: 20
    }
    # Sentinel-2 band name mapping
    band_name_map = {
        2: 'B02', 3: 'B03', 4: 'B04', 5: 'B05', 6: 'B06', 7: 'B07',
        8: 'B08', '8a': 'B8A', 11: 'B11', 12: 'B12'
    }
    
    for b in bands:
        res = band_res.get(b, 10)
        res_dir = f'R{res}m'
        if isinstance(b, str):
            band_name = band_name_map.get(b, f'B{b}')
        else:
            band_name = band_name_map.get(b, f'B{b:02d}')
        
        files = list(img_data.glob(f'{res_dir}/*{band_name}*.jp2'))
        if not files:
            files = list(img_data.rglob(f'*{band_name}*.jp2'))
        if not files:
            raise FileNotFoundError(f'Band {band_name} not found in {img_data}')
        
        with rasterio.open(files[0]) as src:
            arr = src.read(1).astype(np.float32) / 10000.0
            
            if res == 20:
                upsampled = np.empty((arr.shape[0]*2, arr.shape[1]*2), dtype=np.float32)
                reproject(
                    source=arr,
                    destination=upsampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=src.transform * src.transform.scale(0.5, 0.5),
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )
                arr = upsampled
                transform = src.transform * src.transform.scale(0.5, 0.5)
            else:
                transform = src.transform
            
            if reference_shape is None:
                reference_shape = arr.shape
                reference_transform = transform
                reference_crs = src.crs
            
            band_arrays[b] = arr
    
    stacked = np.stack([band_arrays[b] for b in bands], axis=-1)
    return stacked, reference_transform, reference_crs
# Read all 4 dates
date_composites = []

# AOI bounds (from aoi_lucknow.geojson) - WGS84
aoi_bounds_wgs84 = (80.85, 26.75, 81.05, 26.95)  # minx, miny, maxx, maxy

for date_str, safe_folder in safe_folders.items():
    print(f'Reading {date_str} from {safe_folder.name}...')
    bands, transform, crs = read_s2_bands(safe_folder, BANDS)
    
    # Convert AOI bounds to image CRS
    from rasterio.warp import transform_bounds
    aoi_bounds = transform_bounds('EPSG:4326', crs, *aoi_bounds_wgs84)
    print(f'  Image CRS: {crs}, AOI bounds in image CRS: {aoi_bounds}')
    
    # Clip to AOI bounds to save memory
    # Convert bounds to pixel coordinates
    row_min, col_min = rasterio.transform.rowcol(transform, aoi_bounds[0], aoi_bounds[3])
    row_max, col_max = rasterio.transform.rowcol(transform, aoi_bounds[2], aoi_bounds[1])
    
    # Add small buffer
    buffer = 100
    row_min = max(0, row_min - buffer)
    col_min = max(0, col_min - buffer)
    row_max = min(bands.shape[0], row_max + buffer)
    col_max = min(bands.shape[1], col_max + buffer)
    
    bands = bands[row_min:row_max, col_min:col_max, :]
    # Update transform for clipped region
    new_transform = transform * transform.translation(col_min, row_min)
    
    indices = compute_indices(bands)
    date_composites.append({'bands': bands, 'indices': indices, 'transform': new_transform, 'crs': crs})
    print(f'  Clipped bands shape: {bands.shape}')
    print(f'  Indices: {list(indices.keys())}')

# Stack multi-temporal composite
composite = stack_composite(date_composites)
print(f'Final composite shape: {composite.shape}')

# Use transform from first date
reference_transform = date_composites[0]['transform']
reference_crs = date_composites[0]['crs']

# Save 17-band composite
output_path = PROCESSED_DIR / 'lucknow_composite_17band.tif'

with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=composite.shape[0],
    width=composite.shape[1],
    count=composite.shape[2],
    dtype='float32',
    crs=crs,
    transform=reference_transform,
    compress='lzw',
    tiled=True,
    blockxsize=256,
    blockysize=256
) as dst:
    for i in range(composite.shape[2]):
        dst.write(composite[:,:,i], i+1)
    band_names = [f'B{b}' for b in BANDS] + ['NDVI','NDWI','NDBI','BSI','SAVI','EVI','NBR']
    dst.descriptions = tuple(band_names)

print(f'Saved composite to {output_path}')
print(f'Bands: {band_names}')

print('\n=== NEXT STEPS ===')
print('1. Run Dynamic World labels via Google Colab (see below)')
print('2. Place downloaded dw_labels.tif in data/labels/')
print('3. Run: python create_chips.py')

# Colab code for Dynamic World
print('\n=== COPY TO GOOGLE COLAB ===')
print('''
# Run in Google Colab:
# !pip install earthengine-api geemap -q
# import ee, geemap
# ee.Authenticate()
# ee.Initialize(project='your-gcp-project-id')

# aoi = ee.Geometry.Rectangle([80.85, 26.75, 81.05, 26.95])
# dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \\
#        .filterBounds(aoi) \\
#        .filterDate('2025-10-01', '2026-05-15') \\
#        .select('label') \\
#        .mode() \\
#        .clip(aoi)

# # Remap: DW[0-8] -> Our[0-5]
# # 0=Water->4, 1=Trees->0, 2=Grass->2, 3=Flooded->4, 4=Built->3, 5=Crops->2, 6=Shrub->1, 7=Bare->5, 8=Snow->4
# remapped = dw.remap([0,1,2,3,4,5,6,7,8], [4,0,2,4,3,2,1,5,4]).rename('label')

# task = ee.batch.Export.image.toDrive(
#     image=remapped,
#     description='lucknow_dw_labels',
#     scale=10,
#     region=aoi,
#     maxPixels=1e10,
#     fileFormat='GeoTIFF'
# )
# task.start()
# print('Export started. Check Google Drive for lucknow_dw_labels.tif')
''')