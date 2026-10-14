import rasterio
from rasterio.windows import Window
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold

def create_spatial_chips(
    composite_path: str,
    label_path: str,
    aoi_path: str,
    output_dir: str,
    chip_size: int = 512,
    stride: int = 256,
    min_valid_frac: float = 0.5
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(composite_path) as src_img, \
         rasterio.open(label_path) as src_lbl:
        
        transform = src_img.transform
        crs = src_img.crs
        h, w = src_img.height, src_img.width
        
        # Read AOI bounds directly from GeoJSON (avoid geopandas/fiona issue)
        with open(aoi_path) as f:
            aoi_geojson = json.load(f)
        
        # Get bounds from first feature
        coords = aoi_geojson['features'][0]['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        aoi_bounds_wgs84 = (min(lons), min(lats), max(lons), max(lats))
        
        # Convert AOI bounds to image CRS
        from rasterio.warp import transform_bounds
        bounds = transform_bounds('EPSG:4326', crs, *aoi_bounds_wgs84)
        
        row_min, col_min = rasterio.transform.rowcol(transform, bounds[0], bounds[3])
        row_max, col_max = rasterio.transform.rowcol(transform, bounds[2], bounds[1])
        
        # Clamp to image bounds
        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(h, row_max)
        col_max = min(w, col_max)
        
        grid_size_px = int(1000 / abs(transform.a))
        
        chips = []
        groups = []
        
        for r in range(row_min, row_max - chip_size + 1, stride):
            for c in range(col_min, col_max - chip_size + 1, stride):
                window = Window(c, r, chip_size, chip_size)
                
                img_chip = src_img.read(window=window)
                lbl_chip = src_lbl.read(1, window=window)
                
                valid_frac = (img_chip != 0).all(axis=0).mean()
                if valid_frac < min_valid_frac:
                    continue
                
                if (lbl_chip == 5).mean() > 0.9:
                    continue
                
                grid_r = r // grid_size_px
                grid_c = c // grid_size_px
                group_id = f"{grid_r}_{grid_c}"
                
                chips.append({
                    "image": img_chip.transpose(1,2,0),
                    "label": lbl_chip,
                    "window": (r, c, chip_size, chip_size),
                    "group": group_id
                })
                groups.append(group_id)
        
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(chips, groups=groups))
        train_idx, val_idx = splits[0][0], splits[0][1]
        test_idx = splits[1][1]
        
        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_dir = Path(output_dir) / split_name
            split_dir.mkdir(exist_ok=True)
            
            for i, chip_idx in enumerate(idx):
                chip = chips[chip_idx]
                np.savez_compressed(
                    split_dir / f"chip_{i:05d}.npz",
                    image=chip["image"].astype(np.float32),
                    label=chip["label"].astype(np.uint8),
                    window=chip["window"],
                    group=chip["group"]
                )
        
        meta = {
            "chip_size": chip_size,
            "stride": stride,
            "transform": transform.to_gdal(),
            "crs": crs.to_string(),
            "n_chips": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "class_names": {0:"Dense_Forest",1:"Open_Forest",2:"Agriculture",3:"Urban",4:"Water",5:"Bare_Fallow"}
        }
        with open(Path(output_dir) / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Created chips: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return meta