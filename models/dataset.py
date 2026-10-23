import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional
import yaml


class LandCoverDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: int = 256,
        bands: List[str] = None,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
        
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        if self.augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ], p=0.3),
                A.Normalize(mean=[0]*10, std=[1]*10, max_pixel_value=10000),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0]*10, std=[1]*10, max_pixel_value=10000),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
        
        image = np.transpose(image, (1, 2, 0))
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].long()
        
        return image, mask


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    
    image_dir = os.path.join(data_cfg.get("data_root", "data"), "images")
    mask_dir = os.path.join(data_cfg.get("data_root", "data"), "masks")
    
    dataset = LandCoverDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        patch_size=data_cfg["patch_size"],
        augment=train_cfg.get("augmentations", False),
    )
    
    total = len(dataset)
    train_size = int(data_cfg["train_split"] * total)
    val_size = int(data_cfg["val_split"] * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


class InferenceDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        patch_size: int = 512,
        overlap: int = 64,
        bands: List[str] = None,
    ):
        self.image_path = image_path
        self.patch_size = patch_size
        self.overlap = overlap
        
        with rasterio.open(image_path) as src:
            self.height = src.height
            self.width = src.width
            self.transform = src.transform
            self.crs = src.crs
            self.profile = src.profile
        
        self.step = patch_size - overlap
        self.patches = self._generate_patches()
        
        self.transform = A.Compose([
            A.Normalize(mean=[0]*10, std=[1]*10, max_pixel_value=10000),
            ToTensorV2(),
        ])
    
    def _generate_patches(self) -> List[Tuple[int, int]]:
        patches = []
        for y in range(0, self.height, self.step):
            for x in range(0, self.width, self.step):
                patches.append((x, y))
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x, y = self.patches[idx]
        
        with rasterio.open(self.image_path) as src:
            window = Window(x, y, self.patch_size, self.patch_size)
            patch = src.read(window=window).astype(np.float32)
        
        patch = np.transpose(patch, (1, 2, 0))
        
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            pad_h = self.patch_size - patch.shape[0]
            pad_w = self.patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        augmented = self.transform(image=patch)
        patch = augmented["image"]
        
        return patch, (x, y)


if __name__ == "__main__":
    import yaml
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    print("Dataset module loaded successfully")