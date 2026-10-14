import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(".")
from src.models.unet import get_model, ModelEMA
from src.models.losses import CombinedLoss

class ChipDataset(Dataset):
    def __init__(self, chip_dir: str, augment=None, normalize=True, class_balanced=False):
        self.files = sorted(Path(chip_dir).glob("*.npz"))
        self.augment = augment
        self.normalize = normalize
        
        # ImageNet-style stats for Sentinel-2 (10 bands + 7 indices)
        self.mean = np.array([0.12, 0.11, 0.10, 0.13, 0.18, 0.21, 0.23, 0.22, 0.18, 0.12,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.std = np.array([0.08, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.11, 0.09, 0.07,
                             0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        
        # Class-balanced sampling
        if class_balanced:
            self._compute_sampling_weights()
        else:
            self.sample_weights = None
    
    def _compute_sampling_weights(self):
        """Compute sampling weights for class-balanced sampling."""
        class_counts = np.zeros(6)
        for f in self.files:
            lbl = np.load(f)["label"]
            for c in range(6):
                class_counts[c] += (lbl == c).sum()
        
        # Weight by inverse class frequency
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 6
        
        # Weight each chip by its rarest class
        chip_weights = []
        for f in self.files:
            lbl = np.load(f)["label"]
            present_classes = np.unique(lbl)
            chip_weight = class_weights[present_classes].max()
            chip_weights.append(chip_weight)
        
        self.sample_weights = np.array(chip_weights)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img = data["image"]
        lbl = data["label"]
        
        # Ensure label matches image spatial dimensions
        if lbl.shape != img.shape[:2]:
            pad_h = img.shape[0] - lbl.shape[0]
            pad_w = img.shape[1] - lbl.shape[1]
            if pad_h > 0 or pad_w > 0:
                lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=5)
        
        if self.augment:
            aug = self.augment(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]
            # After ToTensorV2, img is (C, H, W) tensor, lbl is (H, W) tensor
            if isinstance(img, torch.Tensor):
                if self.normalize:
                    mean_t = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
                    std_t = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)
                    img = (img - mean_t) / (std_t + 1e-6)
                if isinstance(lbl, torch.Tensor):
                    lbl = lbl.long()
                else:
                    lbl = torch.from_numpy(lbl).long()
                return img.float(), lbl
        
        if self.normalize:
            img = (img - self.mean) / (self.std + 1e-6)
        
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


def get_augmentations(config, curriculum_epoch=None, max_curriculum_epochs=None):
    """Get augmentations with curriculum learning support."""
    augs = []
    
    # Basic geometric augmentations
    if config["augmentation"]["p_rotate90"]:
        augs.append(A.RandomRotate90(p=config["augmentation"]["p_rotate90"]))
    if config["augmentation"]["p_flip"]:
        augs.append(A.Flip(p=config["augmentation"]["p_flip"]))
    
    # Curriculum learning: start with basic, add complex later
    curriculum_progress = 1.0
    if curriculum_epoch is not None and max_curriculum_epochs is not None:
        curriculum_progress = min(1.0, curriculum_epoch / max_curriculum_epochs)
    
    # Photometric augmentations
    if config["augmentation"]["p_brightness_contrast"]:
        p = config["augmentation"]["p_brightness_contrast"] * curriculum_progress
        augs.append(A.RandomBrightnessContrast(p=p))
    if config["augmentation"]["p_gaussian_noise"]:
        p = config["augmentation"]["p_gaussian_noise"] * curriculum_progress
        augs.append(A.GaussNoise(p=p))
    # Disable gamma augmentation entirely - causes NaN issues
# if config["augmentation"]["p_gamma"]:
#     p = config["augmentation"]["p_gamma"] * curriculum_progress
#     augs.append(A.RandomGamma(gamma_limit=(98, 102), p=p))
    
    # Geometric augmentations (more aggressive later)
    if config["augmentation"]["p_elastic"]:
        p = config["augmentation"]["p_elastic"] * curriculum_progress
        augs.append(A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=p))
    if config["augmentation"]["p_grid_distortion"]:
        p = config["augmentation"]["p_grid_distortion"] * curriculum_progress
        augs.append(A.GridDistortion(p=p))
    if config["augmentation"]["p_optical_distortion"]:
        p = config["augmentation"]["p_optical_distortion"] * curriculum_progress
        augs.append(A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=p))
    
    # Advanced augmentations (only in later epochs)
    if config["augmentation"]["p_grid_dropout"]:
        p = config["augmentation"]["p_grid_dropout"] * curriculum_progress
        augs.append(A.GridDropout(ratio=0.5, p=p))
    if config["augmentation"]["p_downscale"]:
        p = config["augmentation"]["p_downscale"] * curriculum_progress
        augs.append(A.Compose([
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
            A.Resize(512, 512, p=1.0),
        ], p=p))
    if config["augmentation"]["p_crop_borders"]:
        p = config["augmentation"]["p_crop_borders"] * curriculum_progress
        augs.append(A.Compose([
            A.RandomCropFromBorders(p=1.0),
            A.Resize(512, 512, p=1.0),
        ], p=p))
    
    augs.append(ToTensorV2())
    return A.Compose(augs, is_check_shapes=False)


def compute_class_weights(chip_dir: str, num_classes: int = 6):
    """Compute pixel-level class weights."""
    counts = np.zeros(num_classes)
    for f in Path(chip_dir).glob("*.npz"):
        lbl = np.load(f)["label"]
        for c in range(num_classes):
            counts[c] += (lbl == c).sum()
    freq = counts / counts.sum()
    weights = 1 / np.sqrt(freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def validate_with_tta(model, loader, device, config, tta_transforms=None):
    """Validation with Test-Time Augmentation."""
    model.eval()
    from torchmetrics import JaccardIndex
    metric = JaccardIndex(task="multiclass", num_classes=config["model"]["classes"],
                          ignore_index=-1).to(device)
    
    if tta_transforms is None:
        tta_transforms = ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]
    
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            
            # TTA predictions
            logits_sum = None
            count = 0
            
            for t_name in tta_transforms:
                if t_name == "original":
                    img_t = img
                elif t_name == "hflip":
                    img_t = torch.flip(img, dims=[3])
                elif t_name == "vflip":
                    img_t = torch.flip(img, dims=[2])
                elif t_name == "rot90":
                    img_t = torch.rot90(img, k=1, dims=[2, 3])
                elif t_name == "rot180":
                    img_t = torch.rot90(img, k=2, dims=[2, 3])
                elif t_name == "rot270":
                    img_t = torch.rot90(img, k=3, dims=[2, 3])
                
                logits = model(img_t)
                
                # Inverse transform
                if t_name == "hflip":
                    logits = torch.flip(logits, dims=[3])
                elif t_name == "vflip":
                    logits = torch.flip(logits, dims=[2])
                elif t_name == "rot90":
                    logits = torch.rot90(logits, k=3, dims=[2, 3])
                elif t_name == "rot180":
                    logits = torch.rot90(logits, k=2, dims=[2, 3])
                elif t_name == "rot270":
                    logits = torch.rot90(logits, k=1, dims=[2, 3])
                
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
                count += 1
            
            preds = (logits_sum / count).argmax(dim=1)
            metric.update(preds, lbl)
    
    return metric.compute().item()


def train(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Curriculum learning setup
    curriculum_epochs = config["data"].get("curriculum_epochs", 30)
    use_curriculum = config["data"].get("curriculum_learning", True)
    
    # Data loaders
    train_augs = get_augmentations(config, curriculum_epoch=None, max_curriculum_epochs=None)
    val_augs = A.Compose([ToTensorV2()])
    
    train_ds = ChipDataset("data/chips/train", augment=train_augs, class_balanced=config["data"].get("use_class_balanced_sampling", True))
    val_ds = ChipDataset("data/chips/val", augment=val_augs)
    test_ds = ChipDataset("data/chips/test", augment=val_augs)
    
    # WeightedRandomSampler for class-balanced sampling
    if config["data"].get("use_class_balanced_sampling", True) and train_ds.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                                  sampler=sampler, num_workers=config["training"]["num_workers"], pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                                  shuffle=True, num_workers=config["training"]["num_workers"], pin_memory=True)
    
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=config["training"]["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=config["training"]["num_workers"])
    
    # Model
    model = get_model(config).to(device)
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.995)).to(device)
    
    # Class weights
    class_weights = compute_class_weights("data/chips/train").to(device)
    
    # Loss
    criterion = CombinedLoss(config)
    criterion.focal.alpha = class_weights
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["training"]["T_0"], T_mult=config["training"]["T_mult"])
    
    def warmup_lr(epoch):
        if epoch < config["training"]["warmup_epochs"]:
            return (epoch + 1) / config["training"]["warmup_epochs"]
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
    
    # Gradient clipping
    gradient_clip = config["training"].get("gradient_clip", 1.0)
    
    # Mixup/CutMix
    mixup_alpha = config["training"].get("mixup_alpha", 0.2)
    cutmix_alpha = config["training"].get("cutmix_alpha", 1.0)
    
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["mixed_precision"])
    
    best_miou = 0
    patience = 0
    
    for epoch in range(config["training"]["epochs"]):
        # Curriculum learning: update augmentations
        if use_curriculum and epoch < config["data"].get("curriculum_epochs", 30):
            train_augs = get_augmentations(config, curriculum_epoch=epoch, 
                                          max_curriculum_epochs=config["data"].get("curriculum_epochs", 30))
            train_ds.augment = train_augs
        
        model.train()
        train_loss = 0
        
        for img, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            img, lbl = img.to(device), lbl.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["training"]["mixed_precision"]):
                logits = model(img)
                loss = criterion(logits, lbl)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            # Check for NaN gradients
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                print("WARNING: NaN gradients detected, skipping step")
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("gradient_clip", 1.0))
            
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(model)
            train_loss += loss.item()
        
        if epoch < config["training"]["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Validation with EMA weights
        ema.apply_shadow()
        miou = validate_with_tta(model, val_loader, device, config) if epoch % 5 == 0 else \
               validate(model, val_loader, device, config)
        ema.restore(model)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            patience = 0
            torch.save(model.state_dict(), f"weights/best_{config['project']['name']}.pth")
            print(f"  New best mIoU: {miou:.4f}")
        else:
            patience += 1
            if patience >= config["training"]["early_stop_patience"]:
                print("Early stopping")
                break
    
    # Final test with EMA
    ema.apply_shadow()
    test_miou = validate_with_tta(model, test_loader, device, config)
    ema.restore(model)
    print(f"Final Test mIoU: {test_miou:.4f}")
    
    # Export ONNX
    export_onnx(model, config)


def validate(model, loader, device, config):
    model.eval()
    from torchmetrics import JaccardIndex
    metric = JaccardIndex(task="multiclass", num_classes=config["model"]["classes"],
                          ignore_index=-1).to(device)
    
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            logits = model(img)
            preds = logits.argmax(dim=1)
            metric.update(preds, lbl)
    
    return metric.compute().item()


def export_onnx(model, config):
    model.eval()
    dummy = torch.randn(1, config["model"]["in_channels"], 512, 512)
    torch.onnx.export(
        model, dummy, f"weights/{config['project']['name']}.onnx",
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 2: "H", 3: "W"},
                      "logits": {0: "batch", 2: "H", 3: "W"}},
        opset_version=17, do_constant_folding=True
    )
    print("ONNX exported")


if __name__ == "__main__":
    train()