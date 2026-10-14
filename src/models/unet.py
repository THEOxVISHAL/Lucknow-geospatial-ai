import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import timm

def get_model(config: dict) -> nn.Module:
    arch = config["model"]["architecture"]
    encoder = config["model"]["encoder"]
    encoder_weights = config["model"]["encoder_weights"]
    in_channels = config["model"]["in_channels"]
    classes = config["model"]["classes"]
    activation = config["model"]["activation"]
    
    if arch == "UperNet":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=config["model"]["in_channels"],
            classes=config["model"]["classes"],
            activation=config["model"]["activation"],
            decoder_channels=config["model"].get("decoder_channels", [256, 256, 256, 256]),
            decoder_use_batchnorm=config["model"].get("decoder_use_batchnorm", True),
            encoder_depth=5,
        )
    elif arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=config["model"]["in_channels"],
            classes=config["model"]["classes"],
            activation=config["model"]["activation"],
            encoder_depth=5,
        )
    else:  # Default U-Net
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=config["model"]["in_channels"],
            classes=config["model"]["classes"],
            activation=config["model"]["activation"],
            decoder_use_batchnorm=True,
            decoder_channels=[256, 128, 64, 32, 16],
            encoder_depth=5,
        )
    
    # Handle non-3-channel input for pretrained encoders
    if in_channels != 3:
        _adapt_first_conv(model, in_channels)
    
    # Enable gradient checkpointing for memory efficiency (especially for large models like B7)
    if hasattr(model.encoder, 'set_grad_checkpointing'):
        model.encoder.set_grad_checkpointing(True)
    
    return model

def freeze_encoder(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze encoder parameters."""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = not freeze
            if freeze:
                print(f"Froze: {name}")
            else:
                print(f"Unfroze: {name}")

def unfreeze_encoder(model: nn.Module):
    """Unfreeze all encoder parameters."""
    freeze_encoder(model, freeze=False)

def _adapt_first_conv(model: nn.Module, in_channels: int):
    """Adapt first conv layer for non-3-channel input."""
    # Find first conv layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            # Create new conv layer
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            # Initialize by averaging RGB weights
            with torch.no_grad():
                new_conv.weight[:, :3] = module.weight
                if in_channels > 3:
                    new_conv.weight[:, 3:] = module.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            
            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, new_conv)
            break

class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.995):
        super().__init__()
        self.module = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].lerp_(v, 1 - self.decay)
    
    def apply_shadow(self):
        self.module.load_state_dict(self.shadow)
    
    def restore(self, model: nn.Module):
        model.load_state_dict(self.shadow)