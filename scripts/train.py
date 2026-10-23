import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import build_model, get_loss_fn, get_optimizer, get_scheduler
from models.dataset import get_dataloaders


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    iou_per_class = jaccard_score(targets_flat, preds_flat, average=None, labels=range(num_classes), zero_division=0)
    f1_per_class = f1_score(targets_flat, preds_flat, average=None, labels=range(num_classes), zero_division=0)
    
    miou = np.mean(iou_per_class)
    mf1 = np.mean(f1_per_class)
    
    return {
        "miou": miou,
        "mf1": mf1,
        "iou_per_class": iou_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    mixed_precision: bool,
) -> dict:
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(masks.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets, model.module.num_classes if hasattr(model, "module") else 7)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


def validate_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    mixed_precision: bool,
) -> dict:
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            if mixed_precision:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets, model.module.num_classes if hasattr(model, "module") else 7)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_miou": best_miou,
        "config": config,
    }, path)


def plot_metrics(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history["train_miou"], label="Train")
    axes[0, 1].plot(epochs, history["val_miou"], label="Val")
    axes[0, 1].set_title("mIoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, history["train_mf1"], label="Train")
    axes[1, 0].plot(epochs, history["val_mf1"], label="Val")
    axes[1, 0].set_title("mF1")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history["lr"])
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(config).to(device)
    criterion = get_loss_fn(config).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, config["training"]["epochs"])
    scaler = GradScaler() if config["training"]["mixed_precision"] else None
    
    train_loader, val_loader, _ = get_dataloaders(config)
    
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_miou": [], "val_miou": [],
        "train_mf1": [], "val_mf1": [],
        "lr": [],
    }
    
    best_miou = 0
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config["training"]["mixed_precision"]
        )
        
        val_metrics = validate_epoch(
            model, val_loader, criterion, device,
            config["training"]["mixed_precision"]
        )
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()
        
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_miou"].append(train_metrics["miou"])
        history["val_miou"].append(val_metrics["miou"])
        history["train_mf1"].append(train_metrics["mf1"])
        history["val_mf1"].append(val_metrics["mf1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f}, mF1: {train_metrics['mf1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}, mF1: {val_metrics['mf1']:.4f}")
        
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config,
                          os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved! mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
        
        if epoch % config["logging"]["save_interval"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config,
                          os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    plot_metrics(history, os.path.join(checkpoint_dir, "training_curves.png"))
    print("Training complete!")


if __name__ == "__main__":
    main()