import zipfile
from pathlib import Path
import shutil

# Source folders (your downloaded ZIPs)
source_folders = [
    Path(r"D:\lucknow_landcover\data\2025-10-01 to 2025-11-30"),
    Path(r"D:\lucknow_landcover\data\2025-12-01 to 2026-01-31"),
    Path(r"D:\lucknow_landcover\data\2026-02-01 to 2026-03-01"),
    Path(r"D:\lucknow_landcover\data\2026-04-01 to 2026-08-19"),
]

# Target raw folder
raw_dir = Path(r"D:\lucknow_landcover\data\raw")

# Lucknow is covered by tiles T44RMQ and T44RMR
# We want the clearest scene from each period covering these tiles

for i, src_folder in enumerate(source_folders):
    print(f"\n=== Period {i+1}: {src_folder.name} ===")
    
    # Find all ZIPs
    zips = list(src_folder.glob("*.zip"))
    print(f"Found {len(zips)} ZIP files")
    
    # Prefer T44RMQ (covers Lucknow city), then T44RMR
    preferred_tiles = ["T44RMQ", "T44RMR"]
    
    selected_zip = None
    for tile in preferred_tiles:
        matches = [z for z in zips if tile in z.name]
        if matches:
            # Pick the first one (you could add cloud cover logic here)
            selected_zip = matches[0]
            print(f"  Selected: {selected_zip.name} (tile {tile})")
            break
    
    if selected_zip is None:
        # Fallback: first ZIP
        selected_zip = zips[0]
        print(f"  Fallback: {selected_zip.name}")
    
    # Extract to raw/ with clean name
    date_str = selected_zip.name.split("_")[2][:8]  # e.g., 20251014
    target_folder = raw_dir / f"S2A_{date_str}_Lucknow"
    
    if target_folder.exists():
        print(f"  Already extracted: {target_folder.name}")
        continue
    
    print(f"  Extracting to {target_folder.name}...")
    with zipfile.ZipFile(selected_zip, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    print(f"  Done!")

print("\n=== Extraction Complete ===")
for f in raw_dir.iterdir():
    print(f"  {f.name}")