import numpy as np

def compute_indices(img: np.ndarray) -> dict:
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [img[..., i] for i in range(10)]
    eps = 1e-6
    indices = {
        "NDVI": (B8 - B4) / (B8 + B4 + eps),
        "NDWI": (B3 - B8) / (B3 + B8 + eps),
        "NDBI": (B11 - B8) / (B11 + B8 + eps),
        "BSI": ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps),
        "SAVI": 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps),
        "EVI": 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps),
        "NBR": (B8 - B12) / (B8 + B12 + eps),
    }
    for k, v in indices.items():
        indices[k] = np.clip(v, -1, 1).astype(np.float32)
    return indices

def stack_composite(date_composites: list) -> np.ndarray:
    bands_stack = np.stack([d['bands'] for d in date_composites], axis=0)  # (N, H, W, C)
    bands_median = np.median(bands_stack, axis=0)  # (H, W, C)
    
    # Stack indices properly
    index_names = list(date_composites[0]['indices'].keys())
    indices_median_list = []
    for name in index_names:
        idx_stack = np.stack([d['indices'][name] for d in date_composites], axis=0)  # (N, H, W)
        idx_median = np.median(idx_stack, axis=0)  # (H, W)
        indices_median_list.append(idx_median)
    indices_median = np.stack(indices_median_list, axis=-1)  # (H, W, 7)
    
    return np.concatenate([bands_median, indices_median], axis=-1)  # (H, W, 17)