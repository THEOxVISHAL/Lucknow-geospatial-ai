"""
Gemma 4 Multimodal Geospatial Intelligence Officer for Lucknow Land Cover Surveillance.
Inspects Sentinel-2 satellite imagery of detected illegal encroachments and produces
court-ready legal notices, environmental impact assessments, and tactical radio scripts.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import rasterio
from rasterio.windows import Window
from dotenv import load_dotenv
from google import genai

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

client = genai.Client(api_key=API_KEY)
GEMMA_MODEL = "gemma-4-26b-a4b-it"


def get_satellite_chip(
    lat: float, 
    lon: float, 
    chip_size: int = 512,
    save_path: Path = None
) -> Image.Image:
    """
    Extracts an RGB visual chip from Lucknow composite or Sentinel-2 raster.
    """
    composite_path = ROOT_DIR / "data" / "processed" / "lucknow_composite_17band.tif"
    demo_path = ROOT_DIR / "demo_rgb.tif"
    
    raster_path = composite_path if composite_path.exists() else demo_path

    with rasterio.open(raster_path) as src:
        # Reproject GPS lat/lon to raster CRS (EPSG:32644)
        if src.crs and not src.crs.is_geographic:
            from rasterio.warp import transform
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)
        else:
            row, col = src.index(lon, lat)

        # Ensure window is inside raster bounds
        row = max(chip_size // 2, min(src.height - chip_size // 2, row))
        col = max(chip_size // 2, min(src.width - chip_size // 2, col))

        win = Window(col - chip_size // 2, row - chip_size // 2, chip_size, chip_size)

        # For 17-band composite: B04 (Red=3), B03 (Green=2), B02 (Blue=1)
        if src.count >= 3:
            b_red = src.read(3, window=win)
            b_green = src.read(2, window=win)
            b_blue = src.read(1, window=win)
        else:
            b = src.read(1, window=win)
            b_red, b_green, b_blue = b, b, b

        def norm(b):
            p2, p98 = np.percentile(b, (2, 98))
            clipped = np.clip(b, p2, p98)
            return ((clipped - p2) / (p98 - p2 + 1e-6) * 255.0).astype(np.uint8)

        rgb = np.dstack([norm(b_red), norm(b_green), norm(b_blue)])
        img = Image.fromarray(rgb)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)

        return img


def inspect_alert_with_gemma4(
    site_id: str,
    zone_name: str,
    area_sqm: float,
    area_bigha: float,
    lat: float,
    lon: float,
    previous_landcover: str = "Agriculture",
    save_chip_dir: Path = None
) -> dict:
    """
    Submits satellite image chip and encroachment parameters to Gemma 4
    for multimodal forensic reasoning and legal enforcement order generation.
    """
    chip_path = None
    if save_chip_dir:
        chip_path = save_chip_dir / f"{site_id}_satellite_chip.png"

    satellite_image = get_satellite_chip(lat=lat, lon=lon, save_path=chip_path)

    prompt = f"""
You are the Chief Geospatial Intelligence Officer for the Lucknow Development Authority (LDA) and UP Forest Department.
Inspect this attached high-resolution Copernicus Sentinel-2 satellite image for detected illegal concrete encroachment.

[ALERT PARAMETERS]
- Site ID: {site_id}
- Protected Zone: {zone_name}
- Encroached Area: {area_sqm:,.0f} m² (~{area_bigha:.2f} Bigha)
- Coordinates: {lat:.5f}° N, {lon:.5f}° E (Lucknow, UP)
- Baseline Landcover: {previous_landcover}
- Legal Status: Strictly Protected Eco-Sensitive Zone (No-Construction Buffer)

Analyze the visual satellite image features and return a JSON object with exactly these keys:
1. "visual_forensic_findings": A 2-sentence forensic evaluation of visible ground patterns in this image (e.g., river sandbar filling, boundary walls, linear plinth work, road cuts, or construction activity).
2. "environmental_threat": Ecological threat to Lucknow city (e.g., Gomti flood retention buffer destruction, groundwater depletion, or wildlife habitat fragmentation).
3. "statutory_violation": Specific legal section violated (e.g., Section 27 of UP Urban Planning & Development Act 1973, NGT Gomti Buffer Order O.A. 6/2012, or Forest Conservation Act).
4. "legal_demolition_notice": A 3-sentence official show-cause / demolition order draft ready for the Vice-Chairman, LDA or District Magistrate to sign.
5. "tactical_radio_script": A crisp, urgent, 25-word field dispatch message for the enforcement squad and police wireless radio.

Return ONLY valid JSON.
"""

    response = client.models.generate_content(
        model=GEMMA_MODEL,
        contents=[satellite_image, prompt],
        config={'response_mime_type': 'application/json'}
    )

    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        return json.loads(raw_text.strip())
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": response.text
        }


if __name__ == "__main__":
    print(f"Running Gemma 4 Geospatial Intelligence Officer ({GEMMA_MODEL})...")
    
    chips_dir = ROOT_DIR / "monitoring_results" / "chips"
    result = inspect_alert_with_gemma4(
        site_id="LKO-ENC-20261020-467",
        zone_name="Gomti River Protected Floodplain",
        area_sqm=31800,
        area_bigha=12.57,
        lat=26.81437,
        lon=81.01223,
        previous_landcover="Agriculture",
        save_chip_dir=chips_dir
    )

    print("\n" + "=" * 65)
    print("[SATELLITE INTEL] OFFICIAL GEMMA 4 MULTIMODAL INTELLIGENCE DOSSIER")
    print("=" * 65)
    print(json.dumps(result, indent=2))

    # Save to monitoring_results
    out_json = ROOT_DIR / "monitoring_results" / "gemma_intelligence_latest.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved dossier report to: {out_json}")
