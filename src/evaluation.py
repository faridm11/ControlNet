"""
Evaluation metrics for ControlNet training.
Computes FID (realism) and mIoU (mask adherence).
mIoU requires a segmentor to be loaded via load_segmentor() — not yet implemented.
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("torchmetrics not available — FID computation disabled.")


def compute_miou(pred_masks: torch.Tensor, target_masks: torch.Tensor, num_classes: int = 35):
    """
    Compute mean Intersection over Union.

    Args:
        pred_masks:   (N, H, W) predicted class IDs
        target_masks: (N, H, W) ground truth class IDs

    Returns:
        float: mIoU averaged over classes present in either prediction or target
    """
    ious = []
    for cls in range(num_classes):
        pred_cls   = (pred_masks == cls)
        target_cls = (target_masks == cls)
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        iou = (pred_cls & target_cls).sum().item() / union
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_accuracy(pred_masks: torch.Tensor, target_masks: torch.Tensor):
    """Compute pixel-wise accuracy."""
    return (pred_masks == target_masks).sum().item() / pred_masks.numel()


class ControlNetEvaluator:
    """
    Evaluator for ControlNet models.
    - FID: always available (measures realism)
    - mIoU: not yet implemented — returns 0.0 until Phase 2
    """

    def __init__(self, device: str = "cuda"):
        self.device    = device
        self.segmentor = None

        self.fid_metric = None
        if FID_AVAILABLE:
            self.fid_metric = FrechetInceptionDistance(normalize=False).to(device)

    def load_segmentor(self):
        """Segmentor not yet implemented — mIoU will remain 0.0 until Phase 2."""
        print("Segmentor not implemented yet — mIoU will be 0.0")

    @torch.inference_mode()
    def evaluate_miou(
        self,
        model,
        dataloader,
        pipeline,
        control_strength: float = 1.0,
        num_inference_steps: int = 35,
    ):
        """Returns zeros until segmentor is implemented in Phase 2."""
        return {'miou': 0.0, 'pixel_acc': 0.0}

    @torch.inference_mode()
    def evaluate_fid(
        self,
        model,
        dataloader,
        pipeline,
        control_strength: float = 1.0,
        num_inference_steps: int = 35,
    ):
        """
        Compute FID between generated images and real validation images.

        Returns:
            float: FID score (lower is better), or inf if torchmetrics unavailable
        """
        if not FID_AVAILABLE or self.fid_metric is None:
            print("FID metric not available.")
            return float('inf')

        model.eval()
        self.fid_metric.reset()
        generator = torch.Generator(device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="FID")):
            generator.manual_seed(42 + batch_idx)

            real_images = batch['image']   # (B, 3, H, W) in [-1, 1]
            mask_rgb    = batch['mask_rgb']
            prompts     = batch['prompt']

            # Real images → uint8 [0, 255]
            real_uint8 = ((real_images * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            self.fid_metric.update(real_uint8.to(self.device), real=True)

            generated_images = []
            for i in range(len(prompts)):
                mask_np     = (mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                control_img = Image.fromarray(mask_np, mode='RGB')

                generated = pipeline(
                    prompt=prompts[i],
                    image=control_img,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=config.GUIDANCE_SCALE,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                ).images[0]

                gen_tensor = torch.from_numpy(np.array(generated)).permute(2, 0, 1)
                generated_images.append(gen_tensor)

            gen_batch = torch.stack(generated_images, dim=0).to(torch.uint8).contiguous()
            self.fid_metric.update(gen_batch.to(self.device), real=False)

        return self.fid_metric.compute().item()
