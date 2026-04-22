"""
Loss functions for ControlNet fine-tuning.

Phase 1: Diffusion loss only (MSE on noise prediction).
Phase 2: Combined loss = diffusion + lambda * segmentation consistency loss.

The segmentation loss uses CE + Dice + boundary term. Set lambda_seg=0.0
(the default) to stay in Phase 1 and train with diffusion loss only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PHASE 1: Diffusion Loss
# ============================================================================

def diffusion_loss(noise_pred: torch.Tensor, noise_true: torch.Tensor) -> torch.Tensor:
    """
    Standard DDPM epsilon-prediction loss.

    The model is trained to predict the noise added to the latent at a given
    timestep. MSE is theoretically derived from the ELBO of the diffusion
    model (Ho et al. 2020) and is the correct choice for continuous noise.

    Args:
        noise_pred: Predicted noise from UNet, shape (B, C, H, W)
        noise_true: Ground-truth noise added to latents, shape (B, C, H, W)

    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(noise_pred, noise_true, reduction="mean")


# ============================================================================
# PHASE 2: Segmentation Consistency Loss (to be implemented)
# ============================================================================

def dice_loss(pred_probs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss over all classes.

    Args:
        pred_probs: Softmax probabilities, shape (B, C, H, W)
        target: One-hot encoded ground truth, shape (B, C, H, W)
        smooth: Smoothing to avoid division by zero

    Returns:
        Scalar Dice loss
    """
    intersection = (pred_probs * target).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def boundary_loss(pred_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple boundary-weighted CE loss using Sobel edge detection.
    Upweights pixels near class boundaries to improve edge quality.

    Args:
        pred_probs: Softmax probabilities, shape (B, C, H, W)
        target: One-hot encoded ground truth, shape (B, C, H, W)

    Returns:
        Scalar boundary loss
    """
    # Detect boundaries via Sobel on the class index map
    target_ids = target.argmax(dim=1).float().unsqueeze(1)  # (B, 1, H, W)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=pred_probs.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    grad_x = F.conv2d(target_ids, sobel_x, padding=1)
    grad_y = F.conv2d(target_ids, sobel_y, padding=1)
    raw_edges = grad_x.abs() + grad_y.abs()  # (B, 1, H, W)
    edge_map = (raw_edges / raw_edges.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)).clamp(0, 1)

    # Weight CE loss by edge proximity (1 + edge_map upweights boundary pixels)
    weight = (1.0 + edge_map).squeeze(1)  # (B, H, W)
    log_probs = torch.log(pred_probs.clamp(min=1e-8))
    target_ids_long = target.argmax(dim=1)  # (B, H, W)

    ce = F.nll_loss(log_probs, target_ids_long, reduction="none")  # (B, H, W)
    return (ce * weight).mean()


def segmentation_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    num_classes: int,
    boundary_weight: float = 0.1,
) -> torch.Tensor:
    """
    Phase 2 segmentation consistency loss: CE + Dice + boundary term.

    Run the frozen segmentor S on the generated image x̂ and compare
    the output to the ground-truth mask y:
        L_seg = CE(S(x̂), y) + Dice(S(x̂), y) + boundary_weight * boundary(S(x̂), y)

    Args:
        pred_logits: Segmentor output logits, shape (B, num_classes, H, W)
        target_mask: Ground-truth class IDs, shape (B, H, W), values in [0, num_classes)
        num_classes: Number of segmentation classes
        boundary_weight: Weight for boundary term (0.1–0.2 recommended)

    Returns:
        Scalar segmentation consistency loss
    """
    # CE loss
    ce = F.cross_entropy(pred_logits, target_mask, reduction="mean")

    # Convert to probabilities and one-hot for Dice + boundary
    pred_probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
    target_onehot = F.one_hot(target_mask, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    dice = dice_loss(pred_probs, target_onehot)
    boundary = boundary_loss(pred_probs, target_onehot)

    return ce + dice + boundary_weight * boundary


# ============================================================================
# Combined Loss (Phase 1 + Phase 2)
# ============================================================================

def combined_loss(
    noise_pred: torch.Tensor,
    noise_true: torch.Tensor,
    lambda_seg: float = 0.0,
    pred_logits: torch.Tensor = None,
    target_mask: torch.Tensor = None,
    num_classes: int = None,
    boundary_weight: float = 0.1,
) -> dict:
    """
    Combined training loss: L = L_diffusion + lambda_seg * L_seg

    lambda_seg=0.0 (default) → Phase 1: diffusion loss only.
    lambda_seg>0.0           → Phase 2: add segmentation consistency.

    The λ value should be ramped from 0.1 → 0.5 during Phase 2 training
    to avoid destabilizing the diffusion loss early in training.

    Args:
        noise_pred: UNet predicted noise, shape (B, C, H, W)
        noise_true: Ground-truth noise, shape (B, C, H, W)
        lambda_seg: Weight for segmentation loss (0.0 = Phase 1)
        pred_logits: Segmentor output logits (required if lambda_seg > 0)
        target_mask: Ground-truth class IDs (required if lambda_seg > 0)
        num_classes: Number of segmentation classes (required if lambda_seg > 0)
        boundary_weight: Weight for boundary term in seg loss

    Returns:
        dict with keys: 'loss' (total), 'diffusion_loss', 'seg_loss'
    """
    diff_loss = diffusion_loss(noise_pred, noise_true)
    seg_loss_val = torch.tensor(0.0, device=noise_pred.device)

    if lambda_seg > 0.0:
        if pred_logits is None or target_mask is None or num_classes is None:
            raise ValueError(
                "pred_logits, target_mask, and num_classes are required when lambda_seg > 0"
            )
        seg_loss_val = segmentation_loss(
            pred_logits, target_mask, num_classes, boundary_weight
        )

    total = diff_loss + lambda_seg * seg_loss_val

    return {
        "loss": total,
        "diffusion_loss": diff_loss,
        "seg_loss": seg_loss_val,
    }
