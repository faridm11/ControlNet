"""
Evaluation metrics for ControlNet training.

All generation-based metrics are computed in a SINGLE pass over the val set
(generation is the expensive part — we run the pipeline once and feed every
metric from the same generated images):

  - FID   distribution realism. UNRELIABLE at n=50 (Inception covariance is
          rank-deficient), kept only for continuity / relative trend.
  - KID   unbiased small-sample alternative to FID — the one to trust at n=50.
  - LPIPS perceptual distance between each generated image and its PAIRED
          ground-truth image. Per-sample → reliable even with 50 images.
  - CLIP  image↔prompt alignment (clip_prompt) and image↔GT similarity
          (clip_img). Per-sample → reliable at n=50.
  - mIoU  placeholder (0.0) until the Phase-2 segmentor is added.

Generation uses deterministic per-batch seeds so the numbers are reproducible
and comparable across runs.
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TM_IMAGE_AVAILABLE = True
except ImportError:
    TM_IMAGE_AVAILABLE = False
    print("torchmetrics image metrics not available — FID/KID disabled.")

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("torchmetrics LPIPS not available — LPIPS disabled.")

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("transformers CLIP not available — CLIP score disabled.")


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

    FID/KID/LPIPS/CLIP are all available now. mIoU returns 0.0 until Phase 2.
    The CLIP model is loaded lazily and cached; FID/KID/LPIPS are constructed
    fresh per evaluate() call (cheap, and avoids stale feature buffers).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.segmentor = None
        self._clip_model = None
        self._clip_processor = None
        # kept for backward-compat with code that calls .fid_metric.reset()
        self.fid_metric = None

    def load_segmentor(self):
        """Segmentor not yet implemented — mIoU will remain 0.0 until Phase 2."""
        print("Segmentor not implemented yet — mIoU will be 0.0")

    @torch.inference_mode()
    def evaluate_miou(self, model, dataloader, pipeline, control_strength: float = 1.0,
                      num_inference_steps: int = 35):
        """Returns zeros until segmentor is implemented in Phase 2."""
        return {'miou': 0.0, 'pixel_acc': 0.0}

    # -----------------------------------------------------------------------
    # CLIP helpers
    # -----------------------------------------------------------------------
    def _ensure_clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_ID).eval().to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_ID)

    @staticmethod
    def _pooled(feats):
        """transformers 5.x returns BaseModelOutputWithPooling; 4.x returns a tensor."""
        return feats.pooler_output if hasattr(feats, "pooler_output") else feats

    @torch.inference_mode()
    def _clip_scores(self, gen_pils, real_pils, prompts):
        """Return (clip_prompt, clip_img): mean cosine sim image↔prompt and image↔GT."""
        self._ensure_clip()
        proc, model = self._clip_processor, self._clip_model

        img_in = proc(images=gen_pils, return_tensors="pt").to(self.device)
        gen_emb = self._pooled(model.get_image_features(pixel_values=img_in["pixel_values"]))
        gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)

        txt_in = proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        txt_emb = self._pooled(model.get_text_features(
            input_ids=txt_in["input_ids"], attention_mask=txt_in["attention_mask"]))
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        real_in = proc(images=real_pils, return_tensors="pt").to(self.device)
        real_emb = self._pooled(model.get_image_features(pixel_values=real_in["pixel_values"]))
        real_emb = real_emb / real_emb.norm(dim=-1, keepdim=True)

        clip_prompt = (gen_emb * txt_emb).sum(-1).mean().item()
        clip_img = (gen_emb * real_emb).sum(-1).mean().item()
        return clip_prompt, clip_img

    # -----------------------------------------------------------------------
    # Main generation-based evaluation (single pass, all metrics)
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self, model, dataloader, pipeline, control_strength: float = 1.0,
                 num_inference_steps: int = 35):
        """
        Run the pipeline once over the whole val set and compute every metric.

        Returns a dict: {fid, kid_mean, kid_std, lpips, clip_prompt, clip_img}.
        Metrics that are unavailable or not requested are returned as None.
        """
        model.eval()

        fid = kid = lpips = None
        if TM_IMAGE_AVAILABLE and config.EVAL_FID:
            fid = FrechetInceptionDistance(normalize=False).to(self.device)
            self.fid_metric = fid
        if TM_IMAGE_AVAILABLE and config.EVAL_KID:
            kid = KernelInceptionDistance(subset_size=config.EVAL_KID_SUBSET,
                                          normalize=False).to(self.device)
        if LPIPS_AVAILABLE and config.EVAL_LPIPS:
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)

        lpips_vals = []
        clip_prompt_vals = []
        clip_img_vals = []
        n_samples = 0

        generator = torch.Generator(device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
            generator.manual_seed(42 + batch_idx)  # deterministic → reproducible across runs

            real_images = batch['image']      # (B, 3, H, W) in [-1, 1]
            mask_rgb    = batch['mask_rgb']
            prompts     = batch['prompt']

            # real images → uint8 [0, 255] and PIL (for FID/KID/LPIPS/CLIP)
            real_01    = (real_images * 0.5 + 0.5).clamp(0, 1)
            real_uint8 = (real_01 * 255).to(torch.uint8)

            gen_pils = []
            for i in range(len(prompts)):
                mask_np     = (mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                control_img = Image.fromarray(mask_np, mode='RGB')
                gen = pipeline(
                    prompt=prompts[i],
                    image=control_img,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=config.GUIDANCE_SCALE,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                ).images[0]
                gen_pils.append(gen)

            gen_uint8 = torch.stack(
                [torch.from_numpy(np.array(g)).permute(2, 0, 1) for g in gen_pils]
            ).to(torch.uint8).contiguous()

            n_samples += len(prompts)

            # FID / KID (uint8 [0,255], normalize=False)
            if fid is not None:
                fid.update(real_uint8.to(self.device), real=True)
                fid.update(gen_uint8.to(self.device), real=False)
            if kid is not None:
                kid.update(real_uint8.to(self.device), real=True)
                kid.update(gen_uint8.to(self.device), real=False)

            # LPIPS (paired gen vs real GT, float [0,1])
            if lpips is not None:
                gen_01 = gen_uint8.float().to(self.device) / 255.0
                lpips_vals.append(lpips(gen_01, real_01.to(self.device)).item())

            # CLIP (image↔prompt, image↔GT)
            if CLIP_AVAILABLE and config.EVAL_CLIP:
                real_pils = [Image.fromarray(r.permute(1, 2, 0).cpu().numpy()) for r in real_uint8]
                cp, ci = self._clip_scores(gen_pils, real_pils, list(prompts))
                clip_prompt_vals.append(cp)
                clip_img_vals.append(ci)

        results = {
            'fid': None, 'kid_mean': None, 'kid_std': None,
            'lpips': None, 'clip_prompt': None, 'clip_img': None,
        }

        if fid is not None:
            try:
                results['fid'] = fid.compute().item()
            except RuntimeError:
                print("FID skipped — need at least 2 samples.")
        if kid is not None and n_samples >= config.EVAL_KID_SUBSET:
            try:
                km, ks = kid.compute()
                results['kid_mean'], results['kid_std'] = km.item(), ks.item()
            except (RuntimeError, ValueError) as e:
                print(f"KID skipped — {e}")
        elif kid is not None:
            print(f"KID skipped — need >= {config.EVAL_KID_SUBSET} samples (got {n_samples}).")
        if lpips_vals:
            results['lpips'] = float(np.mean(lpips_vals))
        if clip_prompt_vals:
            results['clip_prompt'] = float(np.mean(clip_prompt_vals))
            results['clip_img'] = float(np.mean(clip_img_vals))

        model.train()
        return results

    # -----------------------------------------------------------------------
    # Backward-compatible FID-only entry point (kept for older callers)
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate_fid(self, model, dataloader, pipeline, control_strength: float = 1.0,
                     num_inference_steps: int = 35):
        """Compute only FID (delegates to evaluate())."""
        res = self.evaluate(model, dataloader, pipeline, control_strength, num_inference_steps)
        return res['fid'] if res['fid'] is not None else float('inf')
