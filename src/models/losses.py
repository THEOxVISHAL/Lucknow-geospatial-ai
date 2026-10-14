import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                       ignore_index=self.ignore_index, 
                                       label_smoothing=self.label_smoothing)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                       ignore_index=self.ignore_index)
        
        # Clamp for numerical stability
        ce_loss = torch.clamp(ce_loss, 0, 50)
        pt = torch.exp(-ce_loss)
        # Use log for stability
        focal_loss = self.gamma * (1 - pt) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets.clamp(0, self.alpha.size(0)-1)]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, mode='multiclass', ignore_index=-100, smooth=1.0):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        true_1_hot = F.one_hot(targets.clamp(0, num_classes-1), num_classes).permute(0,3,1,2).float()
        probs = F.softmax(inputs, dim=1)
        
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            true_1_hot = true_1_hot * mask
        
        dims = (0, 2, 3)
        intersection = (probs * true_1_hot).sum(dims)
        cardinality = (probs + true_1_hot).sum(dims)
        # Clamp to avoid division by zero
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth + 1e-8)
        return 1 - dice.mean().clamp(0, 1)

class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index=-100, per_class=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_class = per_class
    
    def forward(self, inputs, targets):
        return self.lovasz_softmax_flat(inputs, targets)
    
    def lovasz_softmax_flat(self, logits, labels):
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, logits.size(1))
        labels = labels.view(-1)
        
        if self.ignore_index >= 0:
            mask = labels != self.ignore_index
            logits = logits[mask]
            labels = labels[mask]
        
        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        C = logits.size(1)
        losses = []
        for c in range(C):
            target_c = (labels == c).float()
            if target_c.sum() == 0:
                continue
            logits_c = logits[:, c]
            # Use sigmoid with clamping
            probs_c = torch.sigmoid(logits_c).clamp(1e-7, 1-1e-7)
            errors = (target_c - probs_c).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            target_c_sorted = target_c[perm]
            grad = self.lovasz_grad(target_c_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)
    
    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / (union + 1e-8)
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:p-1]
        return jaccard

class BoundaryLoss(nn.Module):
    def __init__(self, theta=5, ignore_index=-100):
        super().__init__()
        self.theta = theta
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        n, c, h, w = inputs.shape
        
        probs = F.softmax(inputs, dim=1)
        one_hot = F.one_hot(targets.clamp(0, c-1), c).permute(0, 3, 1, 2).float()
        
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            one_hot = one_hot * mask
        
        kernel_size = 3
        padding = kernel_size // 2
        
        # Compute boundary maps with clamping
        pred_boundary = probs - F.max_pool2d(-probs, kernel_size, stride=1, padding=padding)
        true_boundary = one_hot - F.max_pool2d(-one_hot, kernel_size, stride=1, padding=padding)
        
        # Clamp boundary values
        pred_boundary = pred_boundary.clamp(-1, 1)
        true_boundary = true_boundary.clamp(-1, 1)
        
        # Boundary loss (L1 between boundary maps)
        loss = F.l1_loss(pred_boundary, true_boundary)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.focal = FocalLoss(
            gamma=config["loss"]["focal_gamma"],
            alpha=config["loss"].get("alpha"),
            label_smoothing=config["loss"].get("label_smoothing", 0.0)
        )
        self.dice = DiceLoss(mode="multiclass")
        self.lovasz = LovaszSoftmax()
        self.boundary = BoundaryLoss()
        
        self.focal_w = config["loss"]["focal_weight"]
        self.dice_w = config["loss"]["dice_weight"]
        self.lovasz_w = config["loss"].get("lovasz_weight", 0.0)
        self.boundary_w = config["loss"].get("boundary_weight", 0.0)
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        lovasz_loss = self.lovasz(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        
        # Check individual losses for NaN
        losses = {
            'focal': focal_loss, 'dice': dice_loss, 
            'lovasz': lovasz_loss, 'boundary': boundary_loss
        }
        for name, l in losses.items():
            if torch.isnan(l) or torch.isinf(l):
                print(f"WARNING: {name} loss is NaN/Inf: {l}")
                losses[name] = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        
        loss = (
            self.focal_w * losses['focal'] +
            self.dice_w * losses['dice'] +
            self.lovasz_w * losses['lovasz'] +
            self.boundary_w * losses['boundary']
        )
        
        # Final check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return loss