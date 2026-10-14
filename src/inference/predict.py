import onnxruntime as ort
import numpy as np
import torch
from pathlib import Path

class TTAWrapper:
    def __init__(self, onnx_path: str, device="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
    
    def predict_tta(self, img: np.ndarray, tta_transforms=None) -> np.ndarray:
        """
        img: (H, W, C) float32 normalized
        Returns: (H, W) uint8 class map
        """
        if tta_transforms is None:
            tta_transforms = ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]
        
        h, w = img.shape[:2]
        logits_sum = None
        count = 0
        
        transforms = {
            "original": lambda x: x,
            "hflip": lambda x: np.flip(x, axis=1),
            "vflip": lambda x: np.flip(x, axis=0),
            "rot90": lambda x: np.rot90(x, k=1, axes=(0,1)),
            "rot180": lambda x: np.rot90(x, k=2, axes=(0,1)),
            "rot270": lambda x: np.rot90(x, k=3, axes=(0,1)),
        }
        
        inv_transforms = {
            "original": lambda x: x,
            "hflip": lambda x: np.flip(x, axis=1),
            "vflip": lambda x: np.flip(x, axis=0),
            "rot90": lambda x: np.rot90(x, k=3, axes=(0,1)),
            "rot180": lambda x: np.rot90(x, k=2, axes=(0,1)),
            "rot270": lambda x: np.rot90(x, k=1, axes=(0,1)),
        }
        
        for t_name in tta_transforms:
            t = transforms[t_name]
            inv_t = inv_transforms[t_name]
            
            img_t = t(img.copy()).transpose(2,0,1)[None]  # (1,C,H,W)
            logits = self.session.run(None, {self.input_name: img_t.astype(np.float32)})[0]  # (1,C,H,W)
            logits = logits.squeeze(0).transpose(1,2,0)  # (H,W,C)
            logits = inv_t(logits)
            
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits
            count += 1
        
        return (logits_sum / count).argmax(axis=-1).astype(np.uint8)

def predict_large_aoi(onnx_path: str, composite_path: str, output_path: str, 
                      chip_size=512, stride=512, device="cuda", tta=True):
    """Sliding window inference on full AOI with optional TTA."""
    predictor = TTAWrapper(onnx_path, device)
    
    import rasterio
    from rasterio.windows import Window
    
    with rasterio.open(composite_path) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
        h, w = src.height, src.width
        
        with rasterio.open(output_path, "w", **meta) as dst:
            total_chips = ((h - chip_size) // stride + 1) * ((w - chip_size) // stride + 1)
            for idx, r in enumerate(range(0, h, stride)):
                for c in range(0, w, stride):
                    window = Window(c, r, min(chip_size, w-c), min(chip_size, h-r))
                    img_chip = src.read(window=window).transpose(1,2,0)  # (H,W,C)
                    
                    if img_chip.shape[:2] != (chip_size, chip_size):
                        pad_h = chip_size - img_chip.shape[0]
                        pad_w = chip_size - img_chip.shape[1]
                        img_chip = np.pad(img_chip, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
                    
                    if tta:
                        pred = predictor.predict_tta(img_chip)
                    else:
                        # Single forward pass
                        img_t = img_chip.transpose(2,0,1)[None]
                        logits = predictor.session.run(None, {predictor.input_name: img_t.astype(np.float32)})[0]
                        pred = logits.argmax(axis=1).squeeze(0).astype(np.uint8)
                    
                    pred = pred[:window.height, :window.width]
                    dst.write(pred[None], 1, window=window)
                    
                    if idx % 50 == 0:
                        print(f"  Processed {idx}/{total_chips} chips...")
    
    print(f"Saved prediction to {output_path}")