@echo off
title Lucknow Geospatial AI - Autonomous Satellite Surveillance
echo ==============================================================================
echo   LUCKNOW GEOSPATIAL AI : AUTONOMOUS SENTINEL-2 SURVEILLANCE PIPELINE
echo   Direct ESA Copernicus S3 Stream ^| Zero-Upload ^| Gemma 4 Multimodal Agent
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking latest Sentinel-2 acquisition from European Space Agency (AWS S3)...
echo       (Streaming 419.4 sq km window of Lucknow directly - No file upload needed)
venv\Scripts\python.exe free_monitor.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Satellite surveillance monitoring failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Running Gemma 4 Multimodal Visual Audits on Gomti and Kukrail Encroachments...
venv\Scripts\python.exe run_enforcement_pipeline.py

echo.
echo [3/3] Launching Live Geospatial Surveillance Dashboard...
start "" "index.html"

echo.
echo ==============================================================================
echo   SUCCESS: Surveillance cycle complete!
echo   Results updated in results\encroachments_data.js and monitoring_results\
echo ==============================================================================
timeout /t 5
