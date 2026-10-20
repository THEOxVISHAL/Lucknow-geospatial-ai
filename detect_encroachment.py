#!/usr/bin/env python
"""
Autonomous Encroachment & Illegal Construction Detection Engine for Lucknow
Compares multi-temporal satellite segmentation rasters against official
restricted zones (Gomti Floodplain, Kukrail Forest, Green Belts, Agricultural Reserves)
to detect unauthorized urban sprawl and generate actionable enforcement reports.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape, Point, mapping

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "monitoring_results"
DEFAULT_ZONES_PATH = SCRIPT_DIR / "data" / "lucknow_restricted_zones.geojson"
DEFAULT_BASELINE_PATH = SCRIPT_DIR / "data" / "labels" / "lucknow_labels.tif"

# Class definitions from UperNet model
CLASS_MAP = {
    0: "Dense Forest",
    1: "Urban",
    2: "Open Forest / Scrub",
    3: "Agriculture",
    4: "Water",
    5: "Bare / Fallow"
}


def load_restricted_zones(zones_path: Path) -> List[Tuple[Dict[str, Any], Any]]:
    """Load restricted zone geometries and metadata from GeoJSON."""
    if not zones_path.exists():
        print(f"Warning: Restricted zones file {zones_path} not found. Running general change detection.")
        return []
    with open(zones_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    loaded_zones = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        geom = shape(feature.get("geometry"))
        loaded_zones.append((props, geom))
    return loaded_zones


def find_latest_and_baseline_rasters(results_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Automatically discover latest prediction raster and best baseline raster."""
    pred_files = sorted(results_dir.glob("pred_lucknow_*.tif"))
    if not pred_files:
        pred_files = sorted(results_dir.glob("pred_composite_*.tif"))

    if not pred_files:
        return None, None

    latest_raster = pred_files[-1]

    # If we have at least 2 temporal predictions in monitoring_results, use previous as baseline
    if len(pred_files) >= 2:
        baseline_raster = pred_files[-2]
    elif DEFAULT_BASELINE_PATH.exists():
        baseline_raster = DEFAULT_BASELINE_PATH
    else:
        baseline_raster = latest_raster

    return latest_raster, baseline_raster


def detect_encroachments(
    latest_raster_path: Path,
    baseline_raster_path: Path,
    zones_path: Path = DEFAULT_ZONES_PATH,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    min_area_sqm: int = 300,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run bi-temporal change detection and identify illegal constructions.
    """
    start_time = datetime.now()
    if not timestamp:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    zones = load_restricted_zones(zones_path)

    print(f"[{datetime.now()}] Initializing Encroachment Detection Engine...")
    print(f"  Current Raster : {latest_raster_path.name}")
    print(f"  Baseline Raster: {baseline_raster_path.name}")
    print(f"  Restricted Zones: {len(zones)} zones loaded")
    print(f"  Min Area Filter: {min_area_sqm} m² (>= {int(min_area_sqm / 100)} pixels)")

    with rasterio.open(latest_raster_path) as dst_ds:
        curr_mask = dst_ds.read(1)
        dst_transform = dst_ds.transform
        dst_crs = dst_ds.crs
        dst_meta = dst_ds.meta.copy()

    # Align baseline to current grid
    with rasterio.open(baseline_raster_path) as src_ds:
        if (src_ds.shape == curr_mask.shape) and (src_ds.transform == dst_transform) and (src_ds.crs == dst_crs):
            base_mask = src_ds.read(1)
        else:
            print("  Resampling baseline raster to match current spatial grid...")
            base_mask = np.zeros(curr_mask.shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=base_mask,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

    # 1. Detect New Urban Pixels:
    # Was Non-Urban in baseline (0, 2, 3, 4, 5) -> now Urban (1) in latest
    urban_expansion = ((base_mask != 1) & (base_mask != 0) & (curr_mask == 1)).astype(np.uint8)

    # 2. Morphological Cleaning:
    # Remove single-pixel salt noise, bridge small gaps
    cleaned = ndimage.binary_opening(urban_expansion, structure=np.ones((3, 3)))
    labeled_array, num_features = ndimage.label(cleaned, structure=np.ones((3, 3)))

    print(f"  Identified {num_features} contiguous urban expansion candidate clusters.")

    # 3. Analyze connected components
    features_geojson = []
    alerts_table = []
    encroachment_raster = np.zeros(curr_mask.shape, dtype=np.uint8)

    # Extract polygon geometries using rasterio.features.shapes
    min_pixels = max(1, int(min_area_sqm / 100))
    polygon_gen = shapes(labeled_array.astype(np.int32), mask=(labeled_array > 0), transform=dst_transform)

    site_idx = 1
    restricted_count = 0
    total_encroached_sqm = 0

    for poly_geom, label_id in polygon_gen:
        label_id = int(label_id)
        pixel_count = int(np.sum(labeled_array == label_id))
        area_sqm = pixel_count * 100

        if area_sqm < min_area_sqm:
            continue

        # Get centroid of this cluster
        cy, cx = ndimage.center_of_mass(labeled_array == label_id)
        center_x = dst_transform.c + cx * dst_transform.a
        center_y = dst_transform.f + cy * dst_transform.e
        lon, lat = transform(dst_crs, "EPSG:4326", [center_x], [center_y])
        lon, lat = lon[0], lat[0]
        centroid_point = Point(lon, lat)

        # Determine dominant previous land cover
        cluster_pixels_base = base_mask[labeled_array == label_id]
        unq, cnt = np.unique(cluster_pixels_base, return_counts=True)
        dominant_prev_idx = unq[np.argmax(cnt)]
        dominant_prev_name = CLASS_MAP.get(dominant_prev_idx, f"Class {dominant_prev_idx}")

        # Check intersection with restricted zones
        matched_zone = None
        for props, geom in zones:
            if geom.contains(centroid_point) or shape(poly_geom).intersects(geom):
                matched_zone = props
                break

        is_restricted = matched_zone is not None
        severity = matched_zone.get("severity", "CRITICAL") if is_restricted else "MONITOR"
        zone_name = matched_zone.get("name", "Unrestricted / Permitted Urban Growth") if is_restricted else "General Sprawl"
        zone_category = matched_zone.get("category", "General") if is_restricted else "General"

        site_id = f"LKO-ENC-{timestamp[:8]}-{site_idx:03d}"
        site_idx += 1

        if is_restricted:
            restricted_count += 1
            total_encroached_sqm += area_sqm
            # Mark in difference raster: 2 = Restricted Encroachment
            encroachment_raster[labeled_array == label_id] = 2
        else:
            # 1 = Permitted General Expansion
            encroachment_raster[labeled_array == label_id] = 1

        maps_url = f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"
        area_bigha = round(area_sqm / 2530.0, 2)  # 1 Pucca Bigha in UP ≈ 2530 sq.m

        site_record = {
            "site_id": site_id,
            "detected_date": timestamp[:8],
            "is_restricted_violation": is_restricted,
            "severity": severity,
            "zone_name": zone_name,
            "zone_category": zone_category,
            "previous_landcover": dominant_prev_name,
            "area_sqm": area_sqm,
            "area_bigha": area_bigha,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "maps_url": maps_url
        }

        alerts_table.append(site_record)

        # Build GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": site_record,
            "geometry": poly_geom
        }
        features_geojson.append(feature)

    # 4. Save Products
    # A. GeoJSON Vector Products
    geojson_data = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "timestamp": timestamp,
            "current_raster": latest_raster_path.name,
            "baseline_raster": baseline_raster_path.name,
            "total_sites_detected": len(features_geojson),
            "restricted_violations": restricted_count,
            "total_violation_area_sqm": total_encroached_sqm
        },
        "features": features_geojson
    }

    dated_geojson = output_dir / f"illegal_constructions_{timestamp}.geojson"
    latest_geojson = output_dir / "illegal_constructions_latest.geojson"
    with open(dated_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    with open(latest_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # B. CSV Tabular Alerts (Prioritize Restricted Encroachments First)
    alerts_df = pd.DataFrame(alerts_table)
    if not alerts_df.empty:
        # Sort so CRITICAL / HIGH violations appear at top
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "MONITOR": 3}
        alerts_df["rank"] = alerts_df["severity"].map(severity_order).fillna(4)
        alerts_df = alerts_df.sort_values(by=["rank", "area_sqm"], ascending=[True, False]).drop(columns=["rank"])

    dated_csv = output_dir / f"encroachment_alerts_{timestamp}.csv"
    latest_csv = output_dir / "encroachment_alerts_latest.csv"
    alerts_df.to_csv(dated_csv, index=False)
    alerts_df.to_csv(latest_csv, index=False)

    # C. Difference Raster GeoTIFF (0=No change, 1=General urban, 2=Restricted encroachment)
    diff_mask_path = output_dir / f"encroachment_mask_{timestamp}.tif"
    latest_diff_mask_path = output_dir / "encroachment_mask_latest.tif"
    dst_meta.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(diff_mask_path, "w", **dst_meta) as dst:
        dst.write(encroachment_raster, 1)
    with rasterio.open(latest_diff_mask_path, "w", **dst_meta) as dst:
        dst.write(encroachment_raster, 1)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n==================== ENCROACHMENT DETECTION SUMMARY ====================")
    print(f"Total Expansion Sites Detected : {len(features_geojson)}")
    print(f"Restricted Zone Violations     : {restricted_count} sites")
    print(f"Total Encroached Area          : {total_encroached_sqm:,} m² ({total_encroached_sqm / 10000:.2f} hectares)")
    print(f"Products Generated:")
    print(f"  - GeoJSON Layer : {latest_geojson}")
    print(f"  - Alerts CSV    : {latest_csv}")
    print(f"  - Difference TIF: {latest_diff_mask_path}")
    print(f"Processed in {elapsed:.2f} seconds.")
    print("========================================================================\n")

    return {
        "timestamp": timestamp,
        "total_sites": len(features_geojson),
        "restricted_violations": restricted_count,
        "total_violation_area_sqm": total_encroached_sqm,
        "geojson_path": str(latest_geojson),
        "csv_path": str(latest_csv),
        "alerts": alerts_df[alerts_df["is_restricted_violation"]].to_dict(orient="records") if not alerts_df.empty else []
    }


def main():
    parser = argparse.ArgumentParser(description="Autonomous Lucknow Encroachment Detection Engine")
    parser.add_argument("--latest", type=str, help="Path to latest prediction GeoTIFF")
    parser.add_argument("--baseline", type=str, help="Path to baseline GeoTIFF (labels or earlier run)")
    parser.add_argument("--zones", type=str, default=str(DEFAULT_ZONES_PATH), help="Path to restricted zones GeoJSON")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Output directory")
    parser.add_argument("--min-area", type=int, default=300, help="Min area in m² to consider as a site (default: 300)")

    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    latest_path = Path(args.latest) if args.latest else None
    baseline_path = Path(args.baseline) if args.baseline else None

    if not latest_path or not baseline_path:
        found_latest, found_baseline = find_latest_and_baseline_rasters(results_dir)
        latest_path = latest_path or found_latest
        baseline_path = baseline_path or found_baseline

    if not latest_path or not latest_path.exists():
        print("ERROR: No prediction GeoTIFF found. Run free_monitor.py first.")
        sys.exit(1)

    if not baseline_path or not baseline_path.exists():
        print("ERROR: No baseline GeoTIFF found.")
        sys.exit(1)

    detect_encroachments(
        latest_raster_path=latest_path,
        baseline_raster_path=baseline_path,
        zones_path=Path(args.zones),
        output_dir=results_dir,
        min_area_sqm=args.min_area
    )


if __name__ == "__main__":
    main()
