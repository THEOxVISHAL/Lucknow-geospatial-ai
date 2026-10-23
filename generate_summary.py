#!/usr/bin/env python
"""
Generate GitHub Actions Step Summary for Lucknow Land Cover Monitor.
"""
import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    run_number = sys.argv[1] if len(sys.argv) > 1 else "N/A"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    results_dir = Path("monitoring_results")

    # Find latest report JSON
    report_file = results_dir / "report_latest.json"
    if not report_file.exists():
        reports = sorted(results_dir.glob("report_*.json"))
        report_file = reports[-1] if reports else None

    lines = []
    lines.append("## Lucknow Land Cover Monitoring Report\n")

    if report_file and report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            acq_date = data.get("acquisition_date", "N/A")
            cloud_pct = data.get("cloud_cover_percent", 0.0)
            area_km2 = data.get("area_km2", 0.0)

            lines.append(f"**Run:** #{run_number} | **Acquisition Date:** {acq_date} | **Cloud Cover:** {cloud_pct:.1f}% (Max Allowed: 20.0%)\n")
            lines.append("### Class Distribution\n")
            lines.append("| Class | Percentage | Pixels |")
            lines.append("|-------|------------|--------|")
            dist = data.get("class_distribution", {})
            sorted_classes = sorted(dist.items(), key=lambda x: x[1].get("percent", 0), reverse=True)
            for cls, stats in sorted_classes:
                lines.append(f"| {cls} | {stats['percent']:.2f}% | {stats['pixels']:,} |")

            lines.append("\n### Area Monitored\n")
            lines.append(f"**{area_km2:.1f} km²** (Lucknow city)\n")

            alerts = data.get("alerts", [])
            if alerts:
                lines.append("### Environmental Threshold Alerts\n")
                for a in alerts:
                    lines.append(f"- {a}")
            else:
                lines.append("### Environmental Thresholds\n")
                lines.append("All land cover indicators within normal thresholds.\n")

            # Encroachment & Illegal Construction Section
            enc_data = data.get("encroachments", {})
            alerts_csv_file = results_dir / "encroachment_alerts_latest.csv"

            lines.append("### 🚨 Illegal Construction & Encroachment Surveillance\n")
            if enc_data.get("restricted_violations", 0) > 0:
                v_count = enc_data["restricted_violations"]
                v_area = enc_data["total_violation_area_sqm"]
                lines.append(f"⚠️ **{v_count} unauthorized encroachment sites** detected in protected zones ({v_area:,} m² / {v_area/10000:.2f} hectares).\n")
                lines.append("| Site ID | Severity | Protected Zone | Area | Coordinates | Map Link |")
                lines.append("|---------|----------|----------------|------|-------------|----------|")
                for site in enc_data.get("top_alerts", [])[:5]:
                    lines.append(f"| `{site['site_id']}` | **{site['severity']}** | {site['zone_name']} | {site['area_sqm']:,} m² | `{site['latitude']}, {site['longitude']}` | [Open Map]({site['maps_url']}) |")
                lines.append("\n*Full vector GeoJSON and CSV alerts available in downloaded artifacts.*")
            elif alerts_csv_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(alerts_csv_file)
                    v_df = df[df.get("is_restricted_violation", False)]
                    if not v_df.empty:
                        lines.append(f"⚠️ **{len(v_df)} unauthorized encroachment sites** detected ({v_df['area_sqm'].sum():,} m²).\n")
                        lines.append("| Site ID | Severity | Protected Zone | Area | Coordinates | Map Link |")
                        lines.append("|---------|----------|----------------|------|-------------|----------|")
                        for _, site in v_df.head(5).iterrows():
                            lines.append(f"| `{site['site_id']}` | **{site['severity']}** | {site['zone_name']} | {site['area_sqm']:,} m² | `{site['latitude']}, {site['longitude']}` | [Open Map]({site['maps_url']}) |")
                        lines.append("\n*Full vector GeoJSON and CSV alerts available in downloaded artifacts.*")
                    else:
                        lines.append("✅ **No illegal construction detected** in designated restricted zones.")
                except Exception:
                    lines.append("✅ **No illegal construction detected** in designated restricted zones.")
            else:
                lines.append("✅ **No illegal construction detected** in designated restricted zones.")

            lines.append("\n---\n*Artifacts available for 30 days. Download from Actions page.*")
        except Exception as e:
            lines.append(f"Error parsing report {report_file}: {e}")
    else:
        lines.append("No report generated yet.")

    summary_text = "\n".join(lines) + "\n"

    # Write to GITHUB_STEP_SUMMARY if available
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
