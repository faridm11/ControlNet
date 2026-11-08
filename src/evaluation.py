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

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config
from .segmentor import (load_segmentor, segmentor_available, segmentor_logits,
                        confusion_matrix, miou_from_confusion, SEG_NUM_CLASSES)

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

    Computes FID/KID/LPIPS/CLIP and segmentor mIoU in one generation pass.
    The CLIP model and the segmentor are loaded lazily and cached (the trainer hands
    over the model's own frozen segmentor when the Phase-2 loss already loaded it).
    FID/KID/LPIPS are constructed fresh per evaluate() call (cheap, no stale buffers).
    The segmentor's result on the REAL images (miou_real) is computed once per val set
    and cached: images, masks and segmentor never change.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.segmentor = None
        self._real_conf_cache = {}  # id(val dataset) -> confusion matrix of the segmentor on the real photos
        self._clip_model = None
        self._clip_processor = None
        # kept for backward-compat with code that calls .fid_metric.reset()
        self.fid_metric = None

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

        Returns a dict: {fid, kid_mean, kid_std, lpips, clip_prompt, clip_img, miou, pixel_acc,
        miou_real, pixel_acc_real, iou_per_class}.
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

        # Segmentor mIoU: generated images vs the mask, and real images vs the mask (the ceiling)
        use_miou = config.EVAL_MIOU and segmentor_available()
        if use_miou and self.segmentor is None:
            shared = getattr(model, "_segmentor", None)
            self.segmentor = shared[0] if shared else load_segmentor(self.device)
        conf_gen = torch.zeros(SEG_NUM_CLASSES, SEG_NUM_CLASSES, dtype=torch.long)
        real_key = id(dataloader.dataset)
        need_real = use_miou and real_key not in self._real_conf_cache
        conf_real = torch.zeros(SEG_NUM_CLASSES, SEG_NUM_CLASSES, dtype=torch.long)

        lpips_vals = []
        clip_prompt_vals = []
        clip_img_vals = []
        n_samples = 0

        generator = torch.Generator(device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
            generator.manual_seed(config.EVAL_SEED + batch_idx)  # deterministic → reproducible across runs

            real_images = batch['image']      # (B, 3, H, W) in [-1, 1]
            mask_rgb    = batch['mask_rgb']
            prompts     = batch['prompt']

            # real images → uint8 [0, 255] and PIL (for FID/KID/LPIPS/CLIP)
            real_01    = (real_images * 0.5 + 0.5).clamp(0, 1)
            real_uint8 = (real_01 * 255).to(torch.uint8)

            gen_pils = []
            for i in range(len(prompts)):
                mask_np     = np.round(mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                control_img = Image.fromarray(mask_np, mode='RGB')
                gen = pipeline(
                    prompt=prompts[i],
                    image=control_img,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=config.GUIDANCE_SCALE,
                    negative_prompt=config.NEGATIVE_PROMPT or None,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                ).images[0]
                gen_pils.append(gen)

            if config.EVAL_SAVE_DIR:
                save_dir = Path(config.EVAL_SAVE_DIR); save_dir.mkdir(parents=True, exist_ok=True)
                for g, name in zip(gen_pils, batch['image_name']):
                    g.save(save_dir / Path(name).with_suffix(".png").name)

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

            # Segmentor mIoU (generated and real, same masks)
            if use_miou:
                target = batch['mask_seg'].to(self.device)
                # one image at a time: the Mapillary judge (Swin-L at 1024 px) is large, and during training it
                # shares the GPU with the model being trained
                gen_01_all = gen_uint8.float().to(self.device) / 255.0
                for j in range(gen_01_all.shape[0]):
                    pred_gen = segmentor_logits(self.segmentor, gen_01_all[j:j + 1]).argmax(1)
                    conf_gen += confusion_matrix(pred_gen, target[j:j + 1]).cpu()
                    if need_real:
                        pred_real = segmentor_logits(self.segmentor, real_01[j:j + 1].float().to(self.device)).argmax(1)
                        conf_real += confusion_matrix(pred_real, target[j:j + 1]).cpu()

            # CLIP (image↔prompt, image↔GT)
            if CLIP_AVAILABLE and config.EVAL_CLIP:
                real_pils = [Image.fromarray(r.permute(1, 2, 0).cpu().numpy()) for r in real_uint8]
                cp, ci = self._clip_scores(gen_pils, real_pils, list(prompts))
                clip_prompt_vals.append(cp)
                clip_img_vals.append(ci)

        results = {
            'fid': None, 'kid_mean': None, 'kid_std': None,
            'lpips': None, 'clip_prompt': None, 'clip_img': None,
            'miou': None, 'pixel_acc': None, 'miou_real': None, 'pixel_acc_real': None,
        }
        if use_miou:
            if need_real:
                self._real_conf_cache[real_key] = conf_real
            conf_real = self._real_conf_cache[real_key]
            results['miou'], results['pixel_acc'], per_class = miou_from_confusion(conf_gen)
            results['miou_real'], results['pixel_acc_real'], _ = miou_from_confusion(conf_real)
            results['iou_per_class'] = per_class

        if fid is not None:
            try:
                results['fid'] = fid.compute().item()
            except RuntimeError:
                print("FID skipped — need at least 2 samples.")
        if kid is not None and n_samples > config.EVAL_KID_SUBSET:
            try:
                # KID draws random subsets from the global RNG — fix it so the same images give the same number.
                # fork_rng restores the CPU AND all CUDA generators afterwards. (The first version restored only
                # the CPU generator; torch.manual_seed also resets CUDA, so the training epoch after every
                # evaluation repeated the same noise and timesteps.)
                cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(0)
                    km, ks = kid.compute()
                results['kid_mean'], results['kid_std'] = km.item(), ks.item()
            except (RuntimeError, ValueError) as e:
                print(f"KID skipped — {e}")
        elif kid is not None:
            print(f"KID skipped — need > {config.EVAL_KID_SUBSET} samples (got {n_samples}).")
        if lpips_vals:
            results['lpips'] = float(np.mean(lpips_vals))
        if clip_prompt_vals:
            results['clip_prompt'] = float(np.mean(clip_prompt_vals))
            results['clip_img'] = float(np.mean(clip_img_vals))

        # Free this call's metric models before returning. Without this, several evaluate() calls in one
        # process (e.g. several control strengths) piled up Inception/LPIPS copies and ran a 10 GB GPU out
        # of memory (guidance sweep, 2026-10-18).
        del fid, kid, lpips
        self.fid_metric = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
