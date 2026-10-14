#!/usr/bin/env python
"""
Free Sentinel-2 Land Cover Monitoring for Lucknow
Downloads latest Sentinel-2 data from AWS S3, runs ONNX inference, saves results.
"""
import os
import boto3
import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import zoom
from scipy.stats import mode
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from botocore.config import Config
from botocore import UNSIGNED
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIG ==========
LUCKNOW_TILES = ["44/R/Q", "44/R/N"]  # Lucknow tiles
S3_BUCKET = "sentinel-s2-l2a"
MODEL_PATH = "weights/lucknow_landcover_final_v1.onnx"
OUTPUT_DIR = Path("monitoring_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Dense Forest", "Open Forest", "Agriculture", "Urban", "Water", "Bare/Fallow"]

# Normalization stats (from training)
MEAN = np.array([0.12,0.11,0.10,0.13,0.18,0.21,0.23,0.22,0.18,0.12, 0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
STD  = np.array([0.08,0.07,0.08,0.09,0.10,0.11,0.12,0.11,0.09,0.07, 0.3,0.3,0.3,0.3,0.3,0.3,0.3], dtype=np.float32)

# Alert thresholds
ALERT_THRESHOLDS = {
    "Urban": {"max": 60, "msg": "Urban expansion exceeds 60%"},
    "Forest": {"min": 30, "msg": "Forest cover below 30%"},
    "Water": {"min": 1.0, "msg": "Water bodies critically low"}
}

# S3 client (anonymous - no credentials needed for public S3)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def get_latest_date(tile):
    """Find latest available date for a tile on AWS S3."""
    prefix = f"tiles/{tile}/"
    paginator = s3.get_paginator('list_objects_v2')
    dates = set()
    for page in paginator.paginate(Bucket='sentinel-s2-l2a', Prefix=prefix, Delimiter='/'):
        for prefix_obj in page.get('CommonPrefixes', []):
            date_str = prefix_obj['Prefix'].split('/')[-2]
            if date_str.isdigit() and len(date_str) == 8:
                dates.add(date_str)
    return max(dates) if dates else None

def download_bands(tile, date):
    """Download and preprocess 10 bands for a tile/date."""
    bands_10m = {}
    bands_20m = {}
    
    band_mapping = {
        2: "B02", 3: "B03", 4: "B04", 5: "B05", 6: "B06",
        7: "B07", 8: "B08", 8: "B8A", 11: "B11", 12: "B12"
    }
    
    for b in [2,3,4,5,6,7,8,8,11,12]:  # 8a is band 8 at 20m
        res = "R10m" if b in [2,3,4,8] else "R20m"
        band_file = f"B{b:02d}.jp2" if b != 8 else "B8A.jp2"
        
        key = f"tiles/{tile}/{date}/{res}/{band_file}"
        try:
            obj = s3.get_object(Bucket='sentinel-s2-l2a', Key=key)
            with rasterio.open(obj['Body']) as src:
                arr = src.read(1).astype(np.float32) / 10000.0
                if res == "R20m":
                    arr = zoom(arr, 2, order=1)
                bands_20m[b] = arr
        except Exception as e:
            print(f"Warning: Missing band {b} for {tile}/{date}: {e}")
            arr = np.zeros((10980, 10980), dtype=np.float32)
            bands_20m[b] = arr
    
    # Stack in correct order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    band_order = [2,3,4,5,6,7,8,8,11,12]
    bands_stack = np.stack([bands_20m[b] for b in band_order], axis=0)  # (10, H, W)
    return bands_stack

def compute_indices(bands_10):
    """Compute 7 spectral indices from 10 bands."""
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [bands_10[i] for i in range(10)]
    eps = 1e-6
    return np.stack([
        (B8 - B4) / (B8 + B4 + eps),                    # NDVI
        (B3 - B8) / (B3 + B8 + eps),                    # NDWI
        (B11 - B8) / (B11 + B8 + eps),                  # NDBI
        ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps),  # BSI
        1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps),        # SAVI
        2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps),  # EVI
        (B8 - B12) / (B8 + B12 + eps)                   # NBR
    ], axis=-1).astype(np.float32)

def preprocess_tile(bands_10):
    """10 bands -> 17 channels normalized."""
    indices = compute_indices(bands_10)
    bands_hwc = np.transpose(bands_10, (1, 2, 0))  # (H, W, 10)
    img_17ch = np.concatenate([bands_hwc[:, :, :10], indices], axis=-1)  # (H, W, 17)
    img = (img_17ch - MEAN) / (STD + 1e-6)
    return np.transpose(img, (2, 0, 1)).astype(np.float32)  # (17, H, W)

def predict_tile(chip):
    """Run inference on 512x512 chip with TTA."""
    img_tensor = preprocess_tile(chip)
    
    # TTA transforms
    transforms = [
        lambda x: x,
        lambda x: np.flip(x, axis=2),
        lambda x: np.flip(x, axis=1),
        lambda x: np.rot90(x, k=1, axes=(1,2)),
        lambda x: np.rot90(x, k=2, axes=(1,2)),
        lambda x: np.rot90(x, k=3, axes=(1,2)),
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: np.flip(x, axis=2),
        lambda x: np.flip(x, axis=1),
        lambda x: np.rot90(x, k=3, axes=(1,2)),
        lambda x: np.rot90(x, k=2, axes=(1,2)),
        lambda x: np.rot90(x, k=1, axes=(1,2)),
    ]
    
    logits_sum = None
    for t, inv_t in zip([
        lambda x: x,
        lambda x: np.flip(x, axis=2),
        lambda x: np.flip(x, axis=1),
        lambda x: np.rot90(x, k=1, axes=(1,2)),
        lambda x: np.rot90(x, k=2, axes=(1,2)),
        lambda x: np.rot90(x, k=3, axes=(1,2)),
    ], [
        lambda x: x,
        lambda x: np.flip(x, axis=2),
        lambda x: np.flip(x, axis=1),
        lambda x: np.rot90(x, k=3, axes=(1,2)),
        lambda x: np.rot90(x, k=2, axes=(1,2)),
        lambda x: np.rot90(x, k=1, axes=(1,2)),
    ]):
        img_t = t(img_tensor)
        logits = session.run([output_name], {input_name: img_t.astype(np.float32)[None]})[0]
        logits = inv_t(logits)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    
    return (logits_sum / 6).argmax(axis=1)[0]

def predict_large(bands_10, chip_size=512, stride=512):
    """Sliding window inference for full 10980x10980 image."""
    h, w = bands_10.shape[1], bands_10.shape[2]
    pred_full = np.zeros((h, w), dtype=np.uint8)
    
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            rh, cw = min(chip_size, h-r), min(chip_size, w-c)
            chip = bands_10[:, r:r+rh, c:c+cw]
            
            if chip.shape[1:] != (512, 512):
                pad_h, pad_w = 512 - rh, 512 - cw
                chip = np.pad(chip, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
            
            pred = predict_tile(chip)
            pred_full[r:r+rh, c:c+cw] = pred[:rh, :cw]
    
    return pred_full

def run_monitoring():
    """Main monitoring loop."""
    print(f"[{datetime.now()}] Starting monitoring cycle...")
    
    all_preds = []
    for tile in LUCKNOW_TILES:
        date = get_latest_date(tile)
        if not date:
            print(f"No data for {tile}")
            continue
        
        print(f"Processing {tile} ({date})...")
        bands = download_bands(tile, date)
        pred = predict_large(bands)
        all_preds.append(pred)
        
        # Save individual tile prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"pred_{tile.replace('/', '_')}_{date}.tif"
        save_geotiff(pred, out_path)
    
    # Composite (majority vote)
    if all_preds:
        composite = mode(np.stack(all_preds), axis=0, keepdims=False).mode.astype(np.uint8)
        
        # Save composite
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = OUTPUT_DIR / f"pred_composite_{timestamp}.tif"
        save_geotiff(composite, comp_path)
        
        # Compute class distribution
        stats = compute_stats(composite)
        save_report(timestamp, stats, composite)
        
        print(f"[{datetime.now()}] Cycle complete. Reports saved to {OUTPUT_DIR}")

def compute_stats(pred):
    unique, counts = np.unique(pred, return_counts=True)
    total = counts.sum()
    classes = ['Dense Forest', 'Open Forest', 'Agriculture', 'Urban', 'Water', 'Bare/Fallow']
    return {classes[u]: {"pixels": int(c), "percent": round(c/total*100, 2)} for u, c in zip(unique, counts)}

def save_report(timestamp, stats, pred):
    # Check alerts
    alerts = []
    urban_pct = stats.get("Urban", {}).get("percent", 0)
    forest_pct = stats.get("Dense Forest", {}).get("percent", 0) + stats.get("Open Forest", {}).get("percent", 0)
    water_pct = stats.get("Water", {}).get("percent", 0)
    
    if urban_pct > 60: alerts.append(f"HIGH: Urban {urban_pct:.1f}% > 60%")
    if forest_pct < 30: alerts.append(f"WARNING: Forest {forest_pct:.1f}% < 30%")
    if water_pct < 1.0: alerts.append(f"WARNING: Water {water_pct:.1f}% < 1%")
    
    # Save JSON report
    report = {
        "timestamp": timestamp,
        "class_distribution": stats,
        "alerts": alerts,
        "area_km2": pred.size * 0.0001  # 10m = 0.0001 km²
    }
    
    report_path = OUTPUT_DIR / f"report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Append to trend CSV
    trend_row = {"date": timestamp, **{k: v["percent"] for k, v in stats.items()}}
    trend_path = OUTPUT_DIR / "trends.csv"
    if trend_path.exists():
        df = pd.read_csv(trend_path)
        df = pd.concat([df, pd.DataFrame([trend_row])], ignore_index=True)
    else:
        df = pd.DataFrame([trend_row])
    df.to_csv(trend_path, index=False)
    
    # Print alerts
    if alerts:
        for a in alerts:
            print(f"  ALERT: {a}")
    else:
        print("  No alerts")

def save_geotiff(pred, path):
    with rasterio.open(
        path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1],
        count=1, dtype='uint8', crs='EPSG:32644',
        transform=rasterio.transform.from_bounds(499980, 2890200, 609780, 3000000, pred.shape[1], pred.shape[0]),
        compress='lzw'
    ) as dst:
        dst.write(pred, 1)

if __name__ == "__main__":
    run_monitoring()