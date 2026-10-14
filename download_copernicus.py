#!/usr/bin/env python
"""
Download Sentinel-2 L2A data for Lucknow AOI from Copernicus Data Space.
Uses the Copernicus Data Space OData API.
"""
import os
import sys
import requests
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─── Configuration ─────────────────────────────────────────────────────────
# Lucknow AOI (bounding box)
LUCKNOW_BBOX = {
    "west": 80.85,
    "south": 26.75,
    "east": 81.05,
    "north": 26.95
}

# Copernicus Data Space API endpoints
BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# ─── Authentication ────────────────────────────────────────────────────────

def get_access_token(username: str, password: str) -> str:
    """Get access token from Copernicus Data Space."""
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


# ─── Search Products ───────────────────────────────────────────────────────

def build_filter_query(bbox, start_date, end_date, cloud_cover_max=10):
    """Build OData filter query for Sentinel-2 L2A products."""
    # BBOX filter (WKT format)
    wkt = f"POLYGON(({bbox['west']} {bbox['south']}, {bbox['east']} {bbox['south']}, {bbox['east']} {bbox['north']}, {bbox['west']} {bbox['north']}, {bbox['west']} {bbox['south']}))"
    
    filter_parts = [
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        f"ContentDate/Start ge {start_date}T00:00:00.000Z",
        f"ContentDate/Start le {end_date}T23:59:59.000Z",
        "Collection/Name eq 'SENTINEL-2'",
        "contains(Name,'MSIL2A')",  # Level-2A
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_max})",
    ]
    return " and ".join(filter_parts)


def search_products(token, filter_query, max_results=100):
    """Search for products matching the filter."""
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$filter": filter_query,
        "$orderby": "ContentDate/Start desc",
        "$top": max_results,
        "$count": "true",
    }
    
    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("value", [])


# ─── Download Product ──────────────────────────────────────────────────────

def download_product(token, product, output_dir, show_progress=True):
    """Download a product by its ID."""
    product_id = product["Id"]
    product_name = product["Name"]
    
    download_url = f"{BASE_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    
    output_path = Path(output_dir) / f"{product_name}.zip"
    
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return output_path
    
    print(f"Downloading: {product_name}")
    
    headers_dl = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value", 
                           headers={"Authorization": f"Bearer {token}"}, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        if show_progress:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=product_name[:30]) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    return output_path


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A data for Lucknow from Copernicus Data Space")
    parser.add_argument("--username", required=True, help="Copernicus Data Space username")
    parser.add_argument("--password", required=True, help="Copernicus Data Space password")
    parser.add_argument("--start-date", default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--cloud-cover", type=int, default=10, help="Max cloud cover %")
    parser.add_argument("--max-products", type=int, default=10, help="Max products to download")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--tiles", nargs="+", default=["T44RMQ", "T44RMR", "T44RNQ", "T44RNR"],
                        help="Sentinel-2 tile IDs for Lucknow")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Authenticate
    print("Authenticating with Copernicus Data Space...")
    token = get_access_token(args.username, args.password)
    print("✓ Authenticated")
    
    # Build filter query
    filter_query = build_filter_query(
        bbox=LUCKNOW_BBOX,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover_max=args.cloud_cover
    )
    print(f"Searching products...")
    
    # Search products
    products = search_products(token, build_filter_query(
        bbox=LUCKNOW_BBOX,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover_max=args.cloud_cover
    ), max_results=args.max_products)
    
    print(f"Found {len(products)} products")
    
    # Filter by tile if specified
    if args.tiles:
        products = [p for p in products if any(tile in p["Name"] for tile in args.tiles)]
        print(f"Filtered to {len(products)} products matching tiles: {args.tiles}")
    
    if not products:
        print("No products found matching criteria")
        return
    
    # Download products
    print(f"Downloading {len(products)} products to {args.output_dir}...")
    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}] {product['Name']}")
        try:
            download_product(args.token, product, args.output_dir)
        except Exception as e:
            print(f"  Error downloading {product['Name']}: {e}")
    
    print("\n✓ Download complete!")


if __name__ == "__main__":
    main()