import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    
    model = smp.Unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["num_classes"],
        activation=model_cfg["activation"],
    )
    
    return model


def get_loss_fn(config: Dict[str, Any]):
    train_cfg = config["training"]
    class_weights = torch.tensor(train_cfg["class_weights"], dtype=torch.float32)
    
    if train_cfg["loss"] == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif train_cfg["loss"] == "focal":
        return smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
    elif train_cfg["loss"] == "dice":
        return smp.losses.DiceLoss(mode="multiclass")
    elif train_cfg["loss"] == "jaccard":
        return smp.losses.JaccardLoss(mode="multiclass")
    else:
        raise ValueError(f"Unknown loss: {train_cfg['loss']}")


def get_optimizer(model: nn.Module, config: Dict[str, Any]):
    train_cfg = config["training"]
    
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], epochs: int):
    train_cfg = config["training"]
    
    if train_cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif train_cfg["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif train_cfg["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    elif train_cfg["scheduler"] == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {train_cfg['scheduler']}")


if __name__ == "__main__":
    import yaml
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    model = build_model(config)
    print(f"Model: {config['model']['architecture']}")
    print(f"Encoder: {config['model']['encoder']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Classes: {config['model']['num_classes']}")