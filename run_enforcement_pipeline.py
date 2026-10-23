#!/usr/bin/env python
"""
Autonomous Enforcement & Intelligence Pipeline for Lucknow Geospatial AI.
Orchestrates:
  1. Reading latest encroachment alerts (Gomti Floodplain, Kukrail Reserve).
  2. Multi-spectral satellite chip cropping from Sentinel-2 composites.
  3. Gemma 4 Multimodal forensic audit & statutory demolition order drafting.
  4. ElevenLabs tactical field radio dispatch generation.
  5. Executive briefing summary export.
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.gemma_inspector import inspect_alert_with_gemma4
from src.tactical_radio import generate_tactical_radio_dispatch


def run_pipeline(top_n_alerts: int = 3):
    print("=" * 70)
    print("🛰️  LUCKNOW GEOSPATIAL AI : AUTONOMOUS ENFORCEMENT PIPELINE")
    print("=" * 70)

    alerts_csv = ROOT_DIR / "monitoring_results" / "encroachment_alerts_latest.csv"
    if not alerts_csv.exists():
        print(f"Error: {alerts_csv} not found.")
        return

    df = pd.read_csv(alerts_csv)
    print(f"Loaded {len(df)} total detected encroachment alerts.")

    # Filter for Critical violations (Gomti Floodplain, Kukrail Forest)
    critical_alerts = df[df["severity"] == "CRITICAL"].head(top_n_alerts)
    print(f"Inspecting Top {len(critical_alerts)} Critical Sites with Gemma 4...\n")

    chips_dir = ROOT_DIR / "monitoring_results" / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)

    dossiers = []

    for idx, (_, row) in enumerate(critical_alerts.iterrows(), 1):
        site_id = str(row["site_id"])
        zone = str(row["zone_name"])
        area_sqm = float(row["area_sqm"])
        area_bigha = float(row.get("area_bigha", area_sqm / 2529.0))
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        prev_lc = str(row.get("previous_landcover", "Agriculture"))

        print(f"[{idx}/{len(critical_alerts)}] Inspecting Site {site_id} in {zone} ({area_sqm:,.0f} m²)...")

        # 1. Gemma 4 Multimodal Analysis
        intel = inspect_alert_with_gemma4(
            site_id=site_id,
            zone_name=zone,
            area_sqm=area_sqm,
            area_bigha=area_bigha,
            lat=lat,
            lon=lon,
            previous_landcover=prev_lc,
            save_chip_dir=chips_dir
        )

        # 2. ElevenLabs Tactical Voice Generation
        radio_script = intel.get("tactical_radio_script", f"Alert: Encroachment at {site_id} in {zone}.")
        audio_file = generate_tactical_radio_dispatch(
            script_text=radio_script,
            output_filename=f"{site_id}_dispatch.mp3"
        )

        dossier_entry = {
            "site_id": site_id,
            "zone": zone,
            "area_sqm": area_sqm,
            "area_bigha": area_bigha,
            "coordinates": {"lat": lat, "lon": lon},
            "gemma_forensics": intel.get("visual_forensic_findings"),
            "environmental_threat": intel.get("environmental_threat"),
            "statutory_violation": intel.get("statutory_violation"),
            "demolition_notice": intel.get("legal_demolition_notice"),
            "tactical_radio_script": radio_script,
            "audio_file": str(audio_file) if audio_file else None
        }
        dossiers.append(dossier_entry)
        print(f"  ✓ Legal notice & tactical dispatch completed for {site_id}.\n")

    # Save complete dossiers
    dossier_path = ROOT_DIR / "monitoring_results" / "gemma_enforcement_dossiers.json"
    with open(dossier_path, "w", encoding="utf-8") as f:
        json.dump(dossiers, f, indent=2)

    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Dossiers saved to: {dossier_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(top_n_alerts=1)
