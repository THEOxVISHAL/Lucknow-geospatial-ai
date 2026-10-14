#!/usr/bin/env python
"""
Resume training from checkpoint with all bug fixes.
"""
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import sys
import os
from pathlib import Path
sys.path.append(".")
from src.models.unet import get_model, ModelEMA, freeze_encoder, unfreeze_encoder
from src.models.losses import CombinedLoss
from src.training.train import (
    ChipDataset, get_augmentations, compute_class_weights,
    validate_with_tta, validate, export_onnx
)

def train_resume(config_path: str = "config.yaml", resume_epoch: int = 1):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Resuming from epoch {resume_epoch}")
    
    # Curriculum learning setup
    curriculum_epochs = config["data"].get("curriculum_epochs", 30)
    use_curriculum = config["data"].get("curriculum_learning", True)
    
    # Freeze encoder tracking - use GLOBAL epoch, not relative to resume
    freeze_encoder_epochs = config["training"].get("freeze_encoder_epochs", 10)
    
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
    from src.models.unet import get_model, freeze_encoder, unfreeze_encoder
    model = get_model(config).to(device)
    
    # Load checkpoint - try best first, then epoch checkpoint
    ckpt_path_best = f"weights/best_{config['project']['name']}.pth"
    ckpt_path_epoch = f"weights/checkpoint_epoch_{resume_epoch-1}_{config['project']['name']}.pth"
    
    model_loaded = False
    
    # Try best checkpoint first
    if os.path.exists(ckpt_path_best):
        print(f"Loading best checkpoint from {ckpt_path_best}")
        try:
            state_dict = torch.load(ckpt_path_best, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded best checkpoint from {ckpt_path_best}")
            model_loaded = True
        except RuntimeError as e:
            print(f"Best checkpoint incompatible, trying epoch checkpoint: {e}")
    
    # If best not loaded, try epoch checkpoint
    if not model_loaded and os.path.exists(f"weights/checkpoint_epoch_{resume_epoch-1}_{config['project']['name']}.pth"):
        ckpt_path_epoch = f"weights/checkpoint_epoch_{resume_epoch-1}_{config['project']['name']}.pth"
        print(f"Loading epoch checkpoint from {ckpt_path_epoch}")
        try:
            ckpt = torch.load(ckpt_path_epoch, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded model weights from epoch checkpoint")
                model_loaded = True
            else:
                # Old format - direct state dict
                model.load_state_dict(ckpt)
                print(f"Loaded model weights from epoch checkpoint (legacy format)")
                model_loaded = True
        except RuntimeError as e:
            print(f"Epoch checkpoint incompatible: {e}")
    
    if not model_loaded:
        print(f"No compatible checkpoint found, starting fresh")
    
    # Freeze encoder based on GLOBAL epoch (not relative to resume)
    freeze_encoder_epochs = config["training"].get("freeze_encoder_epochs", 10)
    if resume_epoch <= freeze_encoder_epochs:
        freeze_encoder(model.encoder, freeze=True)
        print(f"Encoder frozen (global epoch {resume_epoch} <= {freeze_encoder_epochs})")
    else:
        print(f"Encoder already unfrozen (global epoch {resume_epoch} > {freeze_encoder_epochs})")
    
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.995)).to(device)
    
    # Class weights
    from src.training.train import compute_class_weights
    class_weights = compute_class_weights("data/chips/train").to(device)
    
    # Loss
    from src.models.losses import CombinedLoss
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
    
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.995)).to(device)
    
    # Load optimizer, scheduler, scaler, EMA state if available
    ckpt_path_full = f"weights/checkpoint_epoch_{resume_epoch-1}_{config['project']['name']}.pth"
    if os.path.exists(ckpt_path_full):
        print(f"Loading optimizer/scheduler/scaler/EMA state from {ckpt_path_full}")
        ckpt = torch.load(ckpt_path_full, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("Loaded optimizer state")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Loaded scheduler state")
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("Loaded scaler state")
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
            print("Loaded EMA state")
    
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
    
    # Load best mIoU from checkpoint
    best_miou = 0.0
    ckpt_path_best = f"weights/best_{config['project']['name']}.pth"
    if os.path.exists(ckpt_path_best):
        try:
            best_ckpt = torch.load(ckpt_path_best, map_location=device)
            if 'best_miou' in best_ckpt:
                best_miou = best_ckpt['best_miou']
                print(f"Loaded best mIoU from best checkpoint: {best_miou:.4f}")
        except:
            pass
    
    # If best checkpoint doesn't exist, try epoch checkpoint
    if best_miou == 0.0:
        ckpt_path_epoch = f"weights/checkpoint_epoch_{resume_epoch-1}_{config['project']['name']}.pth"
        if os.path.exists(ckpt_path_epoch):
            try:
                ckpt = torch.load(ckpt_path_epoch, map_location=device)
                if 'best_miou' in ckpt:
                    loaded_miou = ckpt['best_miou']
                    # Sanity check: if loaded value is around old hardcoded 0.78 (0.77-0.79), ignore it
                    # The old hardcoded value was exactly 0.78, so only flag values in that narrow range
                    if 0.77 <= loaded_miou <= 0.79:  # Old hardcoded value was 0.78
                        print(f"Warning: Loaded best_mIoU {loaded_miou:.4f} seems to be old hardcoded value, ignoring")
                        best_miou = 0.0
                    else:
                        best_miou = loaded_miou
                        print(f"Loaded best mIoU from epoch checkpoint: {best_miou:.4f}")
            except:
                pass
    
    # For fresh runs or if checkpoint has corrupted value, start from 0
    if best_miou == 0.0 or (0.77 <= best_miou <= 0.79):  # Only reject the old hardcoded 0.78 value
        best_miou = 0.0
        print(f"Using fresh best mIoU: {best_miou:.4f}")
    else:
        print(f"Loaded best mIoU: {best_miou:.4f}")
    freeze_encoder_epochs = config["training"].get("freeze_encoder_epochs", 10)
    curriculum_epochs = config["data"].get("curriculum_epochs", 30)
    use_curriculum = config["data"].get("curriculum_learning", True)
    
    # Data loaders
    train_augs = get_augmentations(config, curriculum_epoch=None, max_curriculum_epochs=None)
    val_augs = A.Compose([ToTensorV2()])
    
    train_ds = ChipDataset("data/chips/train", augment=train_augs, class_balanced=config["data"].get("use_class_balanced_sampling", True))
    val_ds = ChipDataset("data/chips/val", augment=val_augs)
    test_ds = ChipDataset("data/chips/test", augment=val_augs)
    
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
    
    # Model (already created above, but re-create for clean state)
    # Note: model, ema, optimizer, scheduler already created above
    # Just need to set up data loaders properly
    
    # Freeze encoder based on GLOBAL epoch (not relative to resume)
    freeze_encoder_epochs = config["training"].get("freeze_encoder_epochs", 10)
    if resume_epoch <= freeze_encoder_epochs:
        freeze_encoder(model.encoder, freeze=True)
        print(f"Encoder frozen (global epoch {resume_epoch} <= {freeze_encoder_epochs})")
    else:
        print(f"Encoder already unfrozen (global epoch {resume_epoch} > {freeze_encoder_epochs})")
    
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.995)).to(device)
    
    # Class weights
    from src.training.train import compute_class_weights
    class_weights = compute_class_weights("data/chips/train").to(device)
    
    # Loss
    from src.models.losses import CombinedLoss
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
    
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.995)).to(device)
    
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
    
    # Use actual best mIoU from training
    best_miou = 0.0  # Start fresh, track actual best from this run
    patience = 0
    freeze_encoder_epochs = config["training"].get("freeze_encoder_epochs", 10)
    curriculum_epochs = config["data"].get("curriculum_epochs", 30)
    use_curriculum = config["data"].get("curriculum_learning", True)
    
    for epoch in range(resume_epoch - 1, config["training"]["epochs"]):
        global_epoch = epoch + 1
        
        # Unfreeze encoder at global epoch freeze_encoder_epochs (only once)
        if global_epoch == freeze_encoder_epochs:
            unfreeze_encoder(model.encoder)
            print(f"Encoder unfrozen at global epoch {global_epoch}")
        
        # Curriculum learning: update augmentations
        if use_curriculum and epoch < curriculum_epochs:
            train_augs = get_augmentations(config, curriculum_epoch=epoch, 
                                          max_curriculum_epochs=config["data"].get("curriculum_epochs", 30))
            train_ds.augment = train_augs
        
        model.train()
        train_loss = 0
        
        for img, lbl in tqdm(train_loader, desc=f"Epoch {global_epoch}/{config['training']['epochs']}"):
            img, lbl = img.to(device), lbl.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["training"]["mixed_precision"]):
                logits = model(img)
                loss = criterion(logits, lbl)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
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
        
        if global_epoch < config["training"]["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Validation with EMA weights - use GLOBAL epoch for frequency
        ema.apply_shadow()
        if global_epoch % 5 == 0:
            miou = validate_with_tta(model, val_loader, device, config)
        else:
            miou = validate(model, val_loader, device, config)
        ema.restore(model)
        
        print(f"Epoch {global_epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val mIoU: {miou:.4f}")
        
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
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'best_miou': best_miou,
            'epoch': global_epoch,
        }, f"weights/checkpoint_epoch_{global_epoch}_{config['project']['name']}.pth")
    
    # Final test with EMA
    ema.apply_shadow()
    test_miou = validate_with_tta(model, test_loader, device, config)
    ema.restore(model)
    print(f"Final Test mIoU: {test_miou:.4f}")
    
    # Export ONNX
    export_onnx(model, config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-epoch", type=int, default=1)
    args = parser.parse_args()
    train_resume(args.config, args.resume_epoch)