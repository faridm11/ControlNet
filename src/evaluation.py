"""
Evaluation metrics for ControlNet training.
Includes mIoU (mask adherence) and FID (realism).
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tempfile
import shutil

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("⚠️ torchmetrics not available. FID computation will be disabled.")


def compute_miou(pred_masks: torch.Tensor, target_masks: torch.Tensor, num_classes: int = 8):
    """
    Compute mean Intersection over Union (mIoU) for segmentation masks.
    
    Args:
        pred_masks: Predicted masks (B, H, W) with class IDs
        target_masks: Target masks (B, H, W) with class IDs  
        num_classes: Number of classes
        
    Returns:
        mIoU score (float)
    """
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred_masks == cls)
        target_cls = (target_masks == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union == 0:
            # Class not present in either prediction or target
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def compute_pixel_accuracy(pred_masks: torch.Tensor, target_masks: torch.Tensor):
    """Compute pixel-wise accuracy."""
    correct = (pred_masks == target_masks).sum().item()
    total = pred_masks.numel()
    return correct / total


class ControlNetEvaluator:
    """
    Evaluator for ControlNet models.
    Computes mIoU (mask adherence) and FID (realism).
    """
    
    def __init__(self, segmentor_model=None, device='cuda'):
        """
        Args:
            segmentor_model: Pretrained segmentation model for mIoU evaluation
            device: Device to run evaluation on
        """
        self.device = device
        self.segmentor = segmentor_model
        
        # Initialize FID metric if available
        self.fid_metric = None
        if FID_AVAILABLE:
            # normalize=False because we pass uint8 [0,255] images
            self.fid_metric = FrechetInceptionDistance(normalize=False).to(device)
    
    def load_segmentor(self, checkpoint_path: str):
        """Load pretrained segmentation model for mIoU computation."""
        # TODO: Implement loading of your segmentation model
        # This should load the model from sensation-sidewalk-segmentation
        raise NotImplementedError("Segmentor loading not implemented yet")
    
    @torch.inference_mode()
    def evaluate_miou(
        self,
        model,
        dataloader,
        pipeline,
        control_strength: float = 1.0,
        num_inference_steps: int = 20,
    ):
        """
        Evaluate mIoU by generating images and running segmentation.
        
        Args:
            model: DiffusionControlNet model
            dataloader: Validation dataloader
            pipeline: Inference pipeline
            control_strength: ControlNet conditioning scale
            num_inference_steps: Number of DDIM steps
            
        Returns:
            dict with mIoU and pixel accuracy
        """
        if self.segmentor is None:
            print("⚠️ No segmentor loaded, skipping mIoU evaluation")
            return {'miou': 0.0, 'pixel_acc': 0.0}
        
        model.eval()
        self.segmentor.eval()
        
        # Deterministic generator for stable metric comparison
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        all_pred_masks = []
        all_target_masks = []
        
        for batch in tqdm(dataloader, desc="Evaluating mIoU"):
            masks = batch['mask'].to(self.device)
            mask_rgb = batch['mask_rgb']
            prompts = batch['prompt']
            
            # Generate images for each sample
            for i in range(len(prompts)):
                # Convert mask to PIL for pipeline (ensure CPU for numpy)
                mask_np = (mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                control_img = Image.fromarray(mask_np)
                
                # Generate image
                generated = pipeline(
                    prompt=prompts[i],
                    image=control_img,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                ).images[0]
                
                # Run segmentation on generated image
                gen_tensor = torch.from_numpy(np.array(generated)).permute(2, 0, 1).float() / 255.0
                gen_tensor = gen_tensor.unsqueeze(0).to(self.device)
                pred_logits = self.segmentor(gen_tensor)  # (1, C, H, W) or (1, H, W)
                
                # Handle both (1, C, H, W) and (1, H, W) output formats
                if pred_logits.ndim == 4:  # (1, C, H, W)
                    pred_mask = torch.argmax(pred_logits, dim=1)  # (1, H, W)
                else:  # (1, H, W)
                    pred_mask = pred_logits
                
                all_pred_masks.append(pred_mask.cpu())
                all_target_masks.append(masks[i].cpu())
        
        # Compute metrics
        pred_masks = torch.cat(all_pred_masks, dim=0)
        target_masks = torch.stack(all_target_masks, dim=0)
        
        miou = compute_miou(pred_masks, target_masks)
        pixel_acc = compute_pixel_accuracy(pred_masks, target_masks)
        
        return {
            'miou': miou,
            'pixel_acc': pixel_acc
        }
    
    @torch.inference_mode()
    def evaluate_fid(
        self,
        model,
        dataloader,
        pipeline,
        control_strength: float = 1.0,
        num_inference_steps: int = 20,
    ):
        """
        Evaluate FID between generated and real images.
        
        Args:
            model: DiffusionControlNet model
            dataloader: Validation dataloader
            pipeline: Inference pipeline
            control_strength: ControlNet conditioning scale
            num_inference_steps: Number of DDIM steps
            
        Returns:
            FID score (float)
        """
        if not FID_AVAILABLE or self.fid_metric is None:
            print("⚠️ FID metric not available")
            return float('inf')
        
        model.eval()
        self.fid_metric.reset()
        
        # Deterministic generator for stable metric comparison
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        for batch in tqdm(dataloader, desc="Evaluating FID"):
            real_images = batch['image']  # (B, 3, H, W) in [-1, 1]
            mask_rgb = batch['mask_rgb']
            prompts = batch['prompt']
            
            # Convert real images to [0, 255] uint8 for FID
            real_images = ((real_images * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            self.fid_metric.update(real_images.to(self.device), real=True)
            
            # Generate images
            generated_images = []
            for i in range(len(prompts)):
                mask_np = (mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                control_img = Image.fromarray(mask_np)
                
                generated = pipeline(
                    prompt=prompts[i],
                    image=control_img,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                ).images[0]
                
                gen_tensor = torch.from_numpy(np.array(generated)).permute(2, 0, 1)
                generated_images.append(gen_tensor)
            
            # Convert generated images to batch tensor
            gen_batch = torch.stack(generated_images, dim=0).to(torch.uint8)
            self.fid_metric.update(gen_batch.to(self.device), real=False)
        
        fid_score = self.fid_metric.compute().item()
        return fid_score
    
    def evaluate_control_strength_sweep(
        self,
        model,
        dataloader,
        pipeline,
        control_strengths=[0.5, 1.0, 1.5],
        num_inference_steps: int = 20,
    ):
        """
        Evaluate mIoU and FID across different control strengths.
        
        Returns:
            dict mapping control_strength -> {miou, fid}
        """
        results = {}
        
        for strength in control_strengths:
            print(f"\nEvaluating with control strength: {strength}")
            
            # Compute mIoU
            miou_metrics = self.evaluate_miou(
                model, dataloader, pipeline,
                control_strength=strength,
                num_inference_steps=num_inference_steps
            )
            
            # Compute FID
            fid = self.evaluate_fid(
                model, dataloader, pipeline,
                control_strength=strength,
                num_inference_steps=num_inference_steps
            )
            
            results[strength] = {
                'miou': miou_metrics['miou'],
                'pixel_acc': miou_metrics['pixel_acc'],
                'fid': fid
            }
            
            print(f"  mIoU: {miou_metrics['miou']:.4f}")
            print(f"  Pixel Acc: {miou_metrics['pixel_acc']:.4f}")
            print(f"  FID: {fid:.2f}")
        
        return results
