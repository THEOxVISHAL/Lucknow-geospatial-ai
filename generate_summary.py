#!/usr/bin/env python
"""
Generate GitHub Actions Step Summary for Lucknow Land Cover Monitor.
"""
import os
import sys
import json
from pathlib import Path


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

            lines.append(f"**Run:** #{run_number} | **Acquisition Date:** {acq_date} | **Cloud Cover:** {cloud_pct:.1f}%\n")
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
                lines.append("### Alerts\n")
                for a in alerts:
                    lines.append(f"- {a}")
            else:
                lines.append("### No Alerts\n")
                lines.append("All indicators within normal thresholds.")

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
