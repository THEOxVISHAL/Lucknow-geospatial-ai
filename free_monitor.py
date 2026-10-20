#!/usr/bin/env python
"""
Daily Sentinel-2 Land Cover Monitoring for Lucknow
Discovers latest Sentinel-2 L2A acquisition from AWS S3,
streams required bands, runs ONNX segmentation, and saves reports & GeoTIFFs.
"""
import os
import sys
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Performance optimizations for GDAL /vsicurl/ remote S3 access
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".jp2,.tif,.xml,.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import zoom
import onnxruntime as ort

# ======================== CONFIGURATION ========================
# Lucknow city center is inside MGRS Tile 44RMQ (EPSG:32644)
LUCKNOW_TILE = "44/R/MQ"
S3_BUCKET = "sentinel-s2-l2a"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "monitoring_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SCRIPT_DIR / "weights" / "lucknow_landcover_final_v1.onnx"

CLASS_NAMES = ["Dense Forest", "Urban", "Open Forest", "Agriculture", "Water", "Bare/Fallow"]

# Normalization stats from model training
MEAN = np.array([0.12, 0.11, 0.10, 0.13, 0.18, 0.21, 0.23, 0.22, 0.18, 0.12,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)
STD  = np.array([0.08, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.11, 0.09, 0.07,
                 0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3], dtype=np.float32)

# Alert thresholds
ALERT_THRESHOLDS = {
    "Urban": {"max": 75.0, "msg": "HIGH: Urban expansion exceeds 75%"},
    "Forest": {"min": 15.0, "msg": "WARNING: Forest cover below 15%"},
    "Water": {"min": 1.0, "msg": "WARNING: Water bodies critically low (< 1%)"}
}

# Lucknow City AOI inside 44RMQ (Origin: 399960, 3000000, 10m/px)
# 2048 x 2048 pixels = 20.48 km x 20.48 km = 419.4 km²
AOI_COL_OFF = 8448
AOI_ROW_OFF = 1900
AOI_SIZE = 2048


# ======================== S3 DATA DISCOVERY ========================
def get_latest_acquisition(tile=LUCKNOW_TILE):
    """
    Find latest available date for tile on AWS S3 sentinel-s2-l2a.
    Traverses: tiles/{tile}/{year}/{month}/{day}/{sequence}/
    """
    base_url = f"https://{S3_BUCKET}.s3.amazonaws.com/"
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def list_prefixes(prefix):
        url = f"{base_url}?delimiter=/&prefix={prefix}"
        req = urllib.request.Request(url, headers={"User-Agent": "LucknowMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            root = ET.fromstring(resp.read())
        return [p.find("s3:Prefix", ns).text for p in root.findall("s3:CommonPrefixes", ns)]

    print(f"Searching latest imagery on AWS S3 for tile {tile}...")
    # 1. Years
    years = [p.split("/")[-2] for p in list_prefixes(f"tiles/{tile}/") if p.split("/")[-2].isdigit()]
    if not years:
        raise RuntimeError(f"No years found for tile {tile} on S3 bucket {S3_BUCKET}")
    latest_year = max(years, key=int)

    # 2. Months
    months = [p.split("/")[-2] for p in list_prefixes(f"tiles/{tile}/{latest_year}/") if p.split("/")[-2].isdigit()]
    if not months:
        raise RuntimeError(f"No months found for {tile}/{latest_year}")
    latest_month = max(months, key=int)

    # 3. Days
    days = [p.split("/")[-2] for p in list_prefixes(f"tiles/{tile}/{latest_year}/{latest_month}/") if p.split("/")[-2].isdigit()]
    if not days:
        raise RuntimeError(f"No days found for {tile}/{latest_year}/{latest_month}")
    latest_day = max(days, key=int)

    # 4. Sequences
    seqs = [p.split("/")[-2] for p in list_prefixes(f"tiles/{tile}/{latest_year}/{latest_month}/{latest_day}/") if p.split("/")[-2].isdigit()]
    latest_seq = max(seqs, key=int) if seqs else "0"

    date_str = f"{latest_year}{int(latest_month):02d}{int(latest_day):02d}"
    rel_path = f"tiles/{tile}/{latest_year}/{latest_month}/{latest_day}/{latest_seq}"
    
    # Try fetching tileInfo.json for cloud cover metadata
    cloud_pct = 0.0
    try:
        info_url = f"{base_url}{rel_path}/tileInfo.json"
        req = urllib.request.Request(info_url, headers={"User-Agent": "LucknowMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            cloud_pct = float(info.get("cloudyPixelPercentage", 0.0))
    except Exception as e:
        print(f"Could not read tileInfo.json: {e}")

    return rel_path, date_str, cloud_pct


# ======================== BAND STREAMING & INDICES ========================
def read_aoi_bands(tile_path, col_off=AOI_COL_OFF, row_off=AOI_ROW_OFF, size=AOI_SIZE):
    """
    Stream 10 bands for Lucknow AOI directly from S3 concurrently via GDAL /vsicurl/.
    Bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    """
    bands_info = [
        ("B02", "R10m", 1), ("B03", "R10m", 1), ("B04", "R10m", 1),
        ("B05", "R20m", 2), ("B06", "R20m", 2), ("B07", "R20m", 2),
        ("B08", "R10m", 1), ("B8A", "R20m", 2), ("B11", "R20m", 2), ("B12", "R20m", 2)
    ]

    print(f"Streaming 10 spectral bands for Lucknow window ({size}x{size}) in parallel...")

    def fetch_single_band(item):
        b_name, res, scale = item
        url = f"/vsicurl/https://{S3_BUCKET}.s3.amazonaws.com/{tile_path}/{res}/{b_name}.jp2"
        win = Window(col_off // scale, row_off // scale, size // scale, size // scale)
        try:
            with rasterio.open(url) as src:
                arr = src.read(1, window=win).astype(np.float32) / 10000.0
                if scale == 2:
                    arr = zoom(arr, 2, order=1)
                print(f"  Loaded {b_name} ({res}) successfully.")
                return b_name, arr
        except Exception as e:
            print(f"Warning: Failed to read {b_name} via vsicurl: {e}")
            return b_name, np.zeros((size, size), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_single_band, bands_info))

    res_dict = dict(results)
    ordered_arrays = [res_dict[b_name] for b_name, _, _ in bands_info]
    return np.stack(ordered_arrays, axis=0)  # (10, H, W)


def compute_indices(bands_10):
    """Compute 7 spectral indices from 10 bands."""
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [bands_10[i] for i in range(10)]
    eps = 1e-6
    return np.stack([
        (B8 - B4) / (B8 + B4 + eps),                                 # NDVI
        (B3 - B8) / (B3 + B8 + eps),                                 # NDWI
        (B11 - B8) / (B11 + B8 + eps),                               # NDBI
        ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps),   # BSI
        1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps),                     # SAVI
        2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps),            # EVI
        (B8 - B12) / (B8 + B12 + eps)                                # NBR
    ], axis=-1).astype(np.float32)


# ======================== INFERENCE ========================
def predict_window(bands_10, session, chip_size=512, stride=512):
    """Sliding window inference for the AOI window."""
    inp_name = session.get_inputs()[0].name
    h, w = bands_10.shape[1], bands_10.shape[2]

    indices = compute_indices(bands_10)          # (H, W, 7)
    bands_hwc = np.transpose(bands_10, (1, 2, 0)) # (H, W, 10)
    img_17ch = np.concatenate([bands_hwc, indices], axis=-1)
    norm = (img_17ch - MEAN) / (STD + 1e-6)
    img_chw = np.transpose(norm, (2, 0, 1)).astype(np.float32)

    pred_full = np.zeros((h, w), dtype=np.uint8)
    total_chips = ((h - 1) // stride + 1) * ((w - 1) // stride + 1)
    chip_idx = 0

    print(f"Running inference across {total_chips} chips...")
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            rh = min(chip_size, h - r)
            cw = min(chip_size, w - c)
            chip = img_chw[:, r:r+rh, c:c+cw]

            if chip.shape[1:] != (chip_size, chip_size):
                pad_h = chip_size - rh
                pad_w = chip_size - cw
                chip = np.pad(chip, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            logits = session.run(None, {inp_name: chip[None]})[0]
            pred = logits.argmax(axis=1)[0][:rh, :cw]
            pred_full[r:r+rh, c:c+cw] = pred
            chip_idx += 1

    return pred_full


def compute_stats(pred):
    """Compute class pixel counts and percentages."""
    unique, counts = np.unique(pred, return_counts=True)
    total = counts.sum()
    stats = {}
    for u, c in zip(unique, counts):
        if u < len(CLASS_NAMES):
            name = CLASS_NAMES[u]
            stats[name] = {"pixels": int(c), "percent": round(c / total * 100, 2)}
    # Ensure all classes are present even if 0 pixels
    for name in CLASS_NAMES:
        if name not in stats:
            stats[name] = {"pixels": 0, "percent": 0.0}
    return stats


def save_geotiff(pred, path, col_off=AOI_COL_OFF, row_off=AOI_ROW_OFF, size=AOI_SIZE):
    """Save georeferenced GeoTIFF in EPSG:32644 (UTM 44N)."""
    min_x = 399960.0 + col_off * 10.0
    max_y = 3000000.0 - row_off * 10.0
    max_x = min_x + size * 10.0
    min_y = max_y - size * 10.0
    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, size, size)

    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint8", crs="EPSG:32644", transform=transform, compress="lzw"
    ) as dst:
        dst.write(pred, 1)


def save_report(timestamp, date_str, stats, pred, cloud_pct):
    """Save JSON report and update trends CSV."""
    alerts = []
    urban_pct = stats.get("Urban", {}).get("percent", 0.0)
    forest_pct = stats.get("Dense Forest", {}).get("percent", 0.0) + stats.get("Open Forest", {}).get("percent", 0.0)
    water_pct = stats.get("Water", {}).get("percent", 0.0)

    if urban_pct > ALERT_THRESHOLDS["Urban"]["max"]:
        alerts.append(f"HIGH: Urban {urban_pct:.1f}% > {ALERT_THRESHOLDS['Urban']['max']:.0f}%")
    if forest_pct < ALERT_THRESHOLDS["Forest"]["min"]:
        alerts.append(f"WARNING: Forest cover {forest_pct:.1f}% < {ALERT_THRESHOLDS['Forest']['min']:.0f}%")
    if water_pct < ALERT_THRESHOLDS["Water"]["min"]:
        alerts.append(f"WARNING: Water bodies {water_pct:.1f}% < {ALERT_THRESHOLDS['Water']['min']:.1f}%")

    area_km2 = round(pred.size * 0.0001, 1)

    report = {
        "timestamp": timestamp,
        "acquisition_date": date_str,
        "cloud_cover_percent": round(cloud_pct, 2),
        "area_km2": area_km2,
        "class_distribution": stats,
        "alerts": alerts
    }

    # 1. Dated JSON report
    report_path = OUTPUT_DIR / f"report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 2. Latest JSON report
    latest_report_path = OUTPUT_DIR / "report_latest.json"
    with open(latest_report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 3. Append to trend CSV
    trend_row = {
        "date": date_str,
        "timestamp": timestamp,
        "cloud_pct": round(cloud_pct, 2),
        **{cls: stats.get(cls, {}).get("percent", 0.0) for cls in CLASS_NAMES}
    }
    trend_path = OUTPUT_DIR / "trends.csv"
    if trend_path.exists():
        try:
            df = pd.read_csv(trend_path)
            df = pd.concat([df, pd.DataFrame([trend_row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([trend_row])
    else:
        df = pd.DataFrame([trend_row])
    df.to_csv(trend_path, index=False)

    print(f"Report saved to {report_path}")
    print(f"Trends updated in {trend_path}")
    if alerts:
        print("Alerts:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("No alerts triggered.")


# ======================== MAIN RUNNER ========================
def run_monitoring():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"[{start_time}] Starting Lucknow Land Cover Monitoring Cycle...")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"ONNX Model not found at {MODEL_PATH}")

    # 1. Discover latest Sentinel-2 acquisition
    tile_path, date_str, cloud_pct = get_latest_acquisition(LUCKNOW_TILE)
    print(f"Acquisition found: {tile_path} (Date: {date_str}, Cloud: {cloud_pct:.1f}%)")

    # 2. Stream bands for Lucknow window
    bands_10 = read_aoi_bands(tile_path, AOI_COL_OFF, AOI_ROW_OFF, AOI_SIZE)

    # 3. Initialize ONNX runtime session with tuned multi-threading
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # 4. Run inference
    pred = predict_window(bands_10, session, chip_size=512, stride=512)

    # 5. Save GeoTIFF products
    pred_path = OUTPUT_DIR / f"pred_lucknow_{date_str}.tif"
    save_geotiff(pred, pred_path, AOI_COL_OFF, AOI_ROW_OFF, AOI_SIZE)
    print(f"Saved prediction GeoTIFF to {pred_path}")

    # Also save composite alias for workflow compatibility
    comp_path = OUTPUT_DIR / f"pred_composite_{timestamp}.tif"
    save_geotiff(pred, comp_path, AOI_COL_OFF, AOI_ROW_OFF, AOI_SIZE)

    # 6. Compute statistics and save report
    stats = compute_stats(pred)
    save_report(timestamp, date_str, stats, pred, cloud_pct)

    # 7. Run Autonomous Encroachment & Illegal Construction Detection
    try:
        from detect_encroachment import detect_encroachments, find_latest_and_baseline_rasters
        _, baseline_path = find_latest_and_baseline_rasters(OUTPUT_DIR)
        if baseline_path and baseline_path.exists():
            enc_result = detect_encroachments(
                latest_raster_path=pred_path,
                baseline_raster_path=baseline_path,
                output_dir=OUTPUT_DIR,
                timestamp=timestamp
            )
            latest_report_file = OUTPUT_DIR / "report_latest.json"
            if latest_report_file.exists():
                with open(latest_report_file, "r") as rf:
                    rdata = json.load(rf)
                rdata["encroachments"] = {
                    "total_expansion_sites": enc_result["total_sites"],
                    "restricted_violations": enc_result["restricted_violations"],
                    "total_violation_area_sqm": enc_result["total_violation_area_sqm"],
                    "top_alerts": enc_result["alerts"][:10]
                }
                with open(latest_report_file, "w") as wf:
                    json.dump(rdata, wf, indent=2)
    except Exception as enc_err:
        print(f"Warning: Encroachment detection skipped or failed: {enc_err}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[{datetime.now()}] Monitoring completed successfully in {elapsed:.1f}s.")


if __name__ == "__main__":
    try:
        run_monitoring()
    except Exception as e:
        print(f"ERROR: Monitoring run failed: {e}", file=sys.stderr)
        sys.exit(1)