"""
Create multi-tile mosaics per date for Lucknow AOI.
Simple numpy-based tiling since all tiles are same CRS/resolution.
"""
import sys
sys.path.append('.')

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from pathlib import Path
import yaml
from tqdm.auto import tqdm

from src.preprocessing.compute_indices import compute_indices, stack_composite

with open('config.yaml') as f:
    config = yaml.safe_load(f)

AOI_PATH = Path(config['project']['aoi_path'])
RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BANDS = config['data']['bands']

# Band resolution mapping
BAND_RES = {2:10, 3:10, 4:10, 8:10, 5:20, 6:20, 7:20, '8a':20, 11:20, 12:20}
BAND_NAME_MAP = {2:'B02',3:'B03',4:'B04',5:'B05',6:'B06',7:'B07',8:'B08','8a':'B8A',11:'B11',12:'B12'}

def get_band_name(b):
    if isinstance(b, str):
        return BAND_NAME_MAP.get(b, f'B{b}')
    return BAND_NAME_MAP.get(b, f'B{b:02d}')

def read_s2_bands(safe_path, bands):
    """Read all bands from a single SAFE folder, upsample 20m to 10m."""
    safe_path = Path(safe_path)
    safe_dirs = [p for p in safe_path.glob('*.SAFE') if p.is_dir()]
    if not safe_dirs and safe_path.suffix == '.SAFE' and safe_path.is_dir():
        safe_dirs = [safe_path]
    if not safe_dirs:
        raise FileNotFoundError(f'No .SAFE directory in {safe_path}')
    safe_dir = safe_dirs[0]
    granule_dirs = list(safe_dir.glob('GRANULE/*/IMG_DATA'))
    if not granule_dirs:
        granule_dirs = list(safe_dir.glob('GRANULE/L2A_*/IMG_DATA'))
    if not granule_dirs:
        raise FileNotFoundError(f'No GRANULE/IMG_DATA in {safe_dir}')
    img_data = granule_dirs[0]

    band_arrays = {}
    ref_transform = None
    ref_crs = None
    ref_shape = None

    for b in bands:
        res = BAND_RES.get(b, 10)
        res_dir = f'R{res}m'
        band_name = get_band_name(b)
        files = list(img_data.glob(f'{res_dir}/*{band_name}*.jp2'))
        if not files:
            files = list(img_data.rglob(f'*{band_name}*.jp2'))
        if not files:
            raise FileNotFoundError(f'Band {band_name} not found in {img_data}')

        with rasterio.open(files[0]) as src:
            arr = src.read(1).astype(np.float32) / 10000.0
            if res == 20:
                upsampled = np.empty((arr.shape[0]*2, arr.shape[1]*2), dtype=np.float32)
                from rasterio.warp import reproject, Resampling
                reproject(
                    source=arr, destination=upsampled,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=src.transform * src.transform.scale(0.5, 0.5),
                    dst_crs=src.crs, resampling=Resampling.bilinear
                )
                arr = upsampled
                transform = src.transform * src.transform.scale(0.5, 0.5)
            else:
                transform = src.transform

            if ref_shape is None:
                ref_shape = arr.shape
                ref_transform = transform
                ref_crs = src.crs

            band_arrays[b] = arr

    stacked = np.stack([band_arrays[b] for b in bands], axis=-1)
    return stacked, ref_transform, ref_crs

def find_safe_for_tile(date_str, tile):
    """Find SAFE folder for a specific date and tile."""
    date_short = date_str.replace('-', '')[:8]
    matches = list(RAW_DIR.glob(f'*{date_short}*{tile}*.SAFE'))
    if not matches:
        matches = list(RAW_DIR.glob(f'*{date_str.replace("-", "")[:6]}*{tile}*.SAFE'))
    if not matches:
        matches = list(RAW_DIR.glob(f'*{tile}*.SAFE'))
    for m in matches:
        if date_str.replace('-', '')[:6] in m.name or date_str.replace('-', '')[:8] in m.name:
            return m
    if matches:
        return matches[0]
    return None

def mosaic_tiles_for_date(date_str, tiles, aoi_bounds_wgs84):
    """Simple numpy-based mosaic: place tiles in grid based on their transforms."""
    print(f'\n=== Mosaicking {date_str} ===')
    
    tile_data = []
    
    for tile in tiles:
        safe_path = find_safe_for_tile(date_str, tile)
        if not safe_path:
            print(f'  WARNING: No scene for {date_str} {tile}')
            continue
        
        print(f'  Reading {tile} from {safe_path.name}...')
        bands, transform, crs = read_s2_bands(safe_path, BANDS)
        tile_data.append({
            'tile': tile,
            'bands': bands,
            'transform': transform,
            'crs': crs
        })
    
    if not tile_data:
        raise ValueError(f'No tiles found for {date_str}')
    
    # All tiles should have same CRS
    crs = tile_data[0]['crs']
    
    # Get AOI bounds in image CRS
    aoi_bounds = transform_bounds('EPSG:4326', crs, *aoi_bounds_wgs84)
    print(f'  AOI bounds in image CRS: {aoi_bounds}')
    
    # For each tile, calculate its intersection with AOI and read only that portion
    tile_data_aoi = []
    
    for td in tile_data:
        t = td['transform']
        arr = td['bands']
        h, w = arr.shape[:2]
        
        # Tile bounds
        left = t.c
        top = t.f
        right = left + w * t.a
        bottom = top + h * t.e  # e is negative
        
        # AOI bounds
        aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_bounds
        
        # Check intersection
        if right < aoi_left or left > aoi_right or top < aoi_bottom or bottom > aoi_top:
            continue  # No overlap
        
        # Calculate overlapping window in tile pixels
        col_min = max(0, int((aoi_left - left) / 10))
        col_max = min(w, int((aoi_right - left) / 10) + 1)
        row_min = max(0, int((top - aoi_top) / 10))  # top is higher latitude
        row_max = min(h, int((top - aoi_bottom) / 10) + 1)
        
        if row_max <= row_min or col_max <= col_min:
            continue
        
        # Extract overlapping portion
        clip = arr[row_min:row_max, col_min:col_max, :]
        
        # Calculate transform for clipped portion
        clipped_transform = t * t.translation(col_min, row_min)
        
        tile_data_aoi.append({
            'tile': td['tile'],
            'bands': clip,
            'transform': clipped_transform,
            'crs': crs
        })
        print(f'  {td["tile"]}: extracted {clip.shape} (rows {row_min}-{row_max}, cols {col_min}-{col_max})')
    
    if not tile_data_aoi:
        raise ValueError(f'No tiles overlap AOI for {date_str}')
    
    # Now mosaic only the AOI portions
    # Find global bounds of overlapping portions
    all_bounds = []
    for td in tile_data_aoi:
        t = td['transform']
        h, w = td['bands'].shape[:2]
        left = t.c
        top = t.f
        right = left + w * t.a
        bottom = top + h * t.e
        all_bounds.append((left, bottom, right, top))
    
    min_left = min(b[0] for b in all_bounds)
    min_bottom = min(b[1] for b in all_bounds)
    max_right = max(b[2] for b in all_bounds)
    max_top = max(b[3] for b in all_bounds)
    
    mosaic_transform = rasterio.transform.from_bounds(
        min_left, min_bottom, max_right, max_top,
        width=int((max_right - min_left) / 10),
        height=int((max_top - min_bottom) / 10)
    )
    
    mosaic_h = int((max_top - min_bottom) / 10)
    mosaic_w = int((max_right - min_left) / 10)
    n_bands = tile_data_aoi[0]['bands'].shape[2]
    
    print(f'  Mosaic grid: {mosaic_h} x {mosaic_w} x {n_bands}')
    
    # Create empty mosaic
    mosaic = np.full((mosaic_h, mosaic_w, n_bands), np.nan, dtype=np.float32)
    
    # Place each tile
    for td in tile_data_aoi:
        t = td['transform']
        arr = td['bands']
        h, w = arr.shape[:2]
        
        col_offset = int(round((t.c - min_left) / 10))
        row_offset = int(round((max_top - t.f) / 10))
        
        r_end = min(row_offset + h, mosaic_h)
        c_end = min(col_offset + w, mosaic_w)
        r_src_end = r_end - row_offset
        c_src_end = c_end - col_offset
        
        if row_offset >= 0 and col_offset >= 0:
            mosaic[row_offset:r_end, col_offset:c_end, :] = arr[:r_src_end, :c_src_end, :]
        else:
            r_start = max(0, row_offset)
            c_start = max(0, col_offset)
            r_src_start = max(0, -row_offset)
            c_src_start = max(0, -col_offset)
            mosaic[r_start:r_end, c_start:c_end, :] = arr[r_src_start:r_src_end, c_src_start:c_src_end, :]
    
    print(f'  Mosaic shape: {mosaic.shape}')
    return mosaic, mosaic_transform, crs

def process_date(date_str, tiles, aoi_bounds_wgs84):
    bands, transform, crs = mosaic_tiles_for_date(date_str, tiles, aoi_bounds_wgs84)
    from src.preprocessing.compute_indices import compute_indices
    indices = compute_indices(bands)
    return {'date': date_str, 'bands': bands, 'indices': indices, 
            'transform': transform, 'crs': crs}

if __name__ == '__main__':
    from pathlib import Path
    import yaml
    from src.preprocessing.compute_indices import stack_composite
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    RAW_DIR = Path('data/raw')
    PROCESSED_DIR = Path('data/processed')
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    aoi_bounds_wgs84 = (80.85, 26.75, 81.05, 26.95)
    
    date_tiles = {
        "2025-10-21": ["T44RMQ", "T44RMR", "T44RNQ", "T44RNR"],
        "2025-12-13": ["T44RMQ", "T44RMR"],
        "2026-03-10": ["T44RMQ", "T44RMR", "T44RNQ"],
        "2026-04-09": ["T44RMQ", "T44RMR", "T44RNQ", "T44RNR"],
    }
    
    date_composites = []
    reference_transform = None
    reference_crs = None
    
    for date_str, tiles in date_tiles.items():
        print(f'\n{"="*50}')
        print(f'Processing {date_str} with tiles: {tiles}')
        print(f'{"="*50}')
        
        comp = process_date(date_str, tiles, aoi_bounds_wgs84)
        date_composites.append(comp)
        
        if reference_transform is None:
            reference_transform = comp['transform']
            reference_crs = comp['crs']
    
    print(f'\n{"="*50}')
    print('Creating multi-temporal composite...')
    print(f'{"="*50}')
    
    from src.preprocessing.compute_indices import stack_composite
    composite = stack_composite(date_composites)
    print(f'Final composite shape: {composite.shape}')
    
    output_path = PROCESSED_DIR / 'lucknow_composite_17band.tif'
    band_names = [f'B{b}' for b in BANDS] + ['NDVI','NDWI','NDBI','BSI','SAVI','EVI','NBR']
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=composite.shape[0],
        width=composite.shape[1],
        count=composite.shape[2],
        dtype='float32',
        crs=reference_crs,
        transform=reference_transform,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as dst:
        for i in range(composite.shape[2]):
            dst.write(composite[:,:,i], i+1)
        dst.descriptions = tuple(band_names)
    
    print(f'\nSaved 17-band composite to {output_path}')