#!/usr/bin/env python
"""
Generate GitHub Actions Step Summary for Lucknow Land Cover & Encroachment Surveillance.
"""
import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    run_number = sys.argv[1] if len(sys.argv) > 1 else "1"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    results_dir = Path("monitoring_results")

    # Find latest report JSON
    report_file = results_dir / "report_latest.json"
    if not report_file.exists():
        reports = sorted(results_dir.glob("report_*.json"))
        report_file = reports[-1] if reports else None

    lines = []
    lines.append("# 🛰️ Lucknow Geospatial AI: Surveillance Intelligence Report\n")

    # 1. Pipeline Run Details
    acq_date = "20261019"
    cloud_pct = 0.0
    area_km2 = 419.4

    if report_file and report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            acq_date = data.get("acquisition_date", acq_date)
            cloud_pct = data.get("cloud_cover_percent", cloud_pct)
            area_km2 = data.get("area_km2", area_km2)
        except Exception:
            data = {}
    else:
        data = {}

    lines.append(f"> **Pipeline Run:** #{run_number} | **Sentinel-2 Acquisition:** `{acq_date}` | **Cloud Cover:** `{cloud_pct:.1f}%` (Threshold: 20%) | **AOI:** `{area_km2:.1f} km²`\n")
    lines.append(f"[🌐 **Open Live Interactive Web Dashboard**](https://theoxvishal.github.io/Lucknow-geospatial-ai/)\n")

    # 2. Critical Encroachment Surveillance (Primary Focus)
    alerts_csv = results_dir / "encroachment_alerts_latest.csv"
    geojson_file = results_dir / "illegal_constructions_latest.geojson"
    dossiers_file = results_dir / "gemma_enforcement_dossiers.json"

    lines.append("## 🚨 Illegal Construction & Encroachment Telemetry\n")
    lines.append("> **Autonomous Spatial Policy**: Strictly monitoring **Gomti River Protected Floodplain** (250m Buffer) & **Kukrail Reserve Forest & Sanctuary Buffer**.\n")

    enc_sites = []
    if alerts_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(alerts_csv)
            # Filter rows where violation is true or severity is CRITICAL
            v_df = df[(df["is_restricted_violation"].astype(str).str.lower() == "true") | (df["severity"] == "CRITICAL")]
            if not v_df.empty:
                enc_sites = v_df.to_dict(orient="records")
        except Exception as e:
            print(f"CSV read error: {e}")

    # Fallback to GeoJSON if CSV failed or empty
    if not enc_sites and geojson_file.exists():
        try:
            with open(geojson_file, "r", encoding="utf-8") as gf:
                gdata = json.load(gf)
            enc_sites = [f["properties"] for f in gdata.get("features", [])]
        except Exception:
            pass

    if enc_sites:
        total_sites = len(enc_sites)
        total_sqm = sum(float(s.get("area_sqm", 0)) for s in enc_sites)
        total_bigha = sum(float(s.get("area_bigha", float(s.get("area_sqm", 0)) / 2529.0)) for s in enc_sites)
        total_ha = total_sqm / 10000.0

        gomti_count = sum(1 for s in enc_sites if "gomti" in str(s.get("zone_name", "")).lower())
        kukrail_count = sum(1 for s in enc_sites if "kukrail" in str(s.get("zone_name", "")).lower())

        lines.append(f"### ⚠️ **CRITICAL ALERTS: {total_sites} Illegal Encroachment Sites Detected**\n")
        lines.append(f"- **Total Concrete Encroached Area:** `{total_ha:.2f} Hectares` (**{total_bigha:.1f} Bigha** / `{total_sqm:,.0f} m²`)")
        lines.append(f"- **Kukrail Reserve Forest & Sanctuary Buffer:** `{kukrail_count} Sites` (Deforestation & concrete encroachment)")
        lines.append(f"- **Gomti River Protected Floodplain:** `{gomti_count} Sites` (Ecological riverbed obstruction & flood hazard)\n")

        # Top 10 High-Impact Sites Table
        lines.append("### 📍 Top Critical Sites Telemetry & GPS Coordinates\n")
        lines.append("| Site ID | Severity | Protected Zone | Latitude | Longitude | Area (Bigha) | Google Maps |")
        lines.append("|:---|:---:|:---|:---:|:---:|:---:|:---:|")

        # Sort by area descending
        sorted_sites = sorted(enc_sites, key=lambda x: float(x.get("area_sqm", 0)), reverse=True)

        for site in sorted_sites[:10]:
            site_id = site.get("site_id", "N/A")
            sev = site.get("severity", "CRITICAL")
            zone = site.get("zone_name", "Restricted Zone")
            lat = float(site.get("latitude", 0.0))
            lon = float(site.get("longitude", 0.0))
            area_bigha = float(site.get("area_bigha", float(site.get("area_sqm", 0)) / 2529.0))
            map_url = site.get("maps_url", f"https://www.google.com/maps?q={lat},{lon}")
            lines.append(f"| `{site_id}` | **{sev}** | {zone} | `{lat:.5f}° N` | `{lon:.5f}° E` | **{area_bigha:.2f} Bigha** | [Open Map ↗]({map_url}) |")

        # Full 66 sites in expandable details
        if len(sorted_sites) > 10:
            lines.append("\n<details>")
            lines.append(f"<summary><b>🔍 Click to view all {len(sorted_sites)} Encroachment Sites & GPS Coordinates</b></summary>\n")
            lines.append("| # | Site ID | Protected Zone | Latitude | Longitude | Area (m²) | Area (Bigha) | Direct Maps Link |")
            lines.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|")
            for idx, site in enumerate(sorted_sites, 1):
                site_id = site.get("site_id", "N/A")
                zone = site.get("zone_name", "Restricted Zone")
                lat = float(site.get("latitude", 0.0))
                lon = float(site.get("longitude", 0.0))
                sqm = float(site.get("area_sqm", 0))
                bigha = float(site.get("area_bigha", sqm / 2529.0))
                map_url = site.get("maps_url", f"https://www.google.com/maps?q={lat},{lon}")
                lines.append(f"| {idx} | `{site_id}` | {zone} | `{lat:.5f}` | `{lon:.5f}` | {sqm:,.0f} m² | {bigha:.2f} | [Open in Google Maps]({map_url}) |")
            lines.append("</details>\n")
    else:
        lines.append("✅ **No illegal construction detected** in designated restricted zones.\n")

    # 3. Gemma 4 Multimodal Intelligence Forensic Audit
    if dossiers_file.exists():
        try:
            with open(dossiers_file, "r", encoding="utf-8") as df_file:
                dossiers = json.load(df_file)
            if dossiers:
                lines.append("## 🧠 Google Gemma 4 Multimodal Forensic Audits\n")
                lines.append("> Autonomous satellite chip visual assessment & statutory demolition order drafting by `gemma-4-26b-a4b-it`.\n")

                for d in dossiers[:2]:
                    site_id = d.get("site_id")
                    zone = d.get("zone")
                    sqm = d.get("area_sqm", 0)
                    findings = d.get("visual_forensic_findings", "Visual evidence analyzed.")
                    threat = d.get("environmental_threat", "Ecological damage identified.")
                    demolition = d.get("statutory_demolition_order", "")

                    lines.append(f"### 🏛️ Forensic Dossier: `{site_id}` ({zone} — {sqm:,.0f} m²)")
                    lines.append(f"- **Visual Forensic Proof:** {findings}")
                    lines.append(f"- **Environmental Threat Matrix:** {threat}")
                    if demolition:
                        excerpt = demolition.strip().replace("\n", " ")[:280] + "..."
                        lines.append(f"- **Statutory Notice Citation:** *\"{excerpt}\"*")
                    lines.append("")
        except Exception as e:
            print(f"Dossier read error: {e}")

    # 4. Land Cover Class Distribution
    dist = data.get("class_distribution", {})
    if dist:
        lines.append("## 📊 Land Cover Classification (UperNet-B5 17-Band ONNX)\n")
        lines.append("| Class Index | Class Name | Coverage (%) | Pixel Count | Area (km²) |")
        lines.append("|:---:|:---|:---:|:---:|:---:|")
        sorted_classes = sorted(dist.items(), key=lambda x: x[1].get("percent", 0), reverse=True)
        for idx, (cls, stats) in enumerate(sorted_classes):
            px = stats.get("pixels", 0)
            pct = stats.get("percent", 0.0)
            km2 = (px * 100) / 1e6  # 10m x 10m pixel = 100 m²
            lines.append(f"| {idx} | **{cls}** | {pct:.2f}% | {px:,} | {km2:.2f} km² |")

    lines.append("\n---\n")
    lines.append("*🚀 Lucknow Geospatial AI Platform | Built with ESA Copernicus Sentinel-2, ONNX Runtime & Google Gemma 4.*")

    summary_text = "\n".join(lines) + "\n"

    # Write to GITHUB_STEP_SUMMARY if available
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
