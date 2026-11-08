"""
Step 15a — check before SANPO Phase 2 (segmentation loss with the Mapillary judge). No training,
no weights changed, nothing saved except the report.

1. Judge accuracy on the REAL SANPO evaluation photos (fixed 150-image subset) with the judge's
   input at 512 px (no enlargement) and at 1024 px (enlarged 2x) -> pick the better size.
2. For each size, on 30 training batches of the SANPO v2 model (LoRA from job 1848221):
   - peak GPU memory of a forward + backward with the segmentation loss on;
   - diffusion loss and segmentation loss values;
   - how strongly each loss pushes the LoRA weights: gradient norm of each loss separately.
   The ratio |grad seg| / |grad diffusion| is used to pick loss weights that keep the segmentation
   term a moderate share of the update.
Run from DDPM/ with DDPM_DATA_ROOT=data_sanpo, DDPM_SEG_JUDGE=mapillary, DDPM_USE_SEGMENTOR_LOSS=true,
DDPM_LORA_TARGETS=attn_ff (see run_check.sh).
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data import create_train_dataset, create_val_dataset, create_dataloader
from src.models.diffusion_controlnet import DiffusionControlNet
from src.segmentor import confusion_matrix, miou_from_confusion, segmentor_logits, SEG_NUM_CLASSES

assert config.DATASET_NAME == "sanpo" and config.USE_SEGMENTOR_LOSS and os.environ.get("DDPM_SEG_JUDGE") == "mapillary"
LORA = ROOT / "controlnet_testing/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_LoRA.pt"
OUT = Path(__file__).parent / "check_report.json"
N_BATCHES = 30
device = torch.device("cuda")
report = {"lora": str(LORA)}

model = DiffusionControlNet(pretrained=True, device=device)
model.load_checkpoint(str(LORA))
judge = model._segmentor[0]

# ---- 1. judge accuracy on real photos at both input sizes
val = create_val_dataset(resolution=512)  # fixed 150-image subset
vl = DataLoader(val, batch_size=1, num_workers=4)
for scale in (1.0, 2.0):
    judge.scale = scale
    conf = torch.zeros(SEG_NUM_CLASSES, SEG_NUM_CLASSES, dtype=torch.long)
    with torch.inference_mode():
        for b in vl:
            pred = segmentor_logits(judge, (b["image"] * 0.5 + 0.5).to(device)).argmax(1)
            conf += confusion_matrix(pred, b["mask_seg"].to(device)).cpu()
    miou, acc, per = miou_from_confusion(conf)
    report[f"real_photos_scale{scale:g}"] = {"miou": miou, "pixel_acc": acc, "per_class": per}
    print(f"judge on {len(val)} real SANPO photos, input {int(512 * scale)} px: mIoU {miou:.4f}, pixel acc {acc:.4f} | "
          + ", ".join(f"{k} {v:.2f}" for k, v in per.items()), flush=True)

# ---- 2. memory and loss strength during training
train = create_train_dataset(resolution=512)
tl = create_dataloader(train, batch_size=4, shuffle=True, num_workers=8, pin_memory=False)
params = [p for p in model.controlnet.parameters() if p.requires_grad]
model.train()
SCALE_LOSS = 1024.0  # fp16 autocast: scale losses before backward so small gradients do not underflow
for scale in (1.0, 2.0):
    judge.scale = scale
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    diff_l, seg_l, g_diff, g_seg, applied, t0 = [], [], [], [], 0, time.time()
    for i, b in enumerate(tl):
        if i >= N_BATCHES:
            break
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(b["image"].to(device), b["mask_rgb"].to(device), b["prompt"],
                        mask_seg=b["mask_seg"].to(device), seg_weight=1.0)
        d, s = out["diffusion_loss"], out["seg_loss"]
        gd = torch.autograd.grad(d * SCALE_LOSS, params, retain_graph=s is not None, allow_unused=True)
        g_diff.append(torch.sqrt(sum((g.float() ** 2).sum() for g in gd if g is not None)).item() / SCALE_LOSS)
        diff_l.append(d.item())
        if s is not None:
            gs = torch.autograd.grad(s * SCALE_LOSS, params, allow_unused=True)
            g_seg.append(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)).item() / SCALE_LOSS)
            seg_l.append(s.item())
            applied += 1
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    ratio = float(np.mean(g_seg) / np.mean(g_diff)) if g_seg else float("nan")
    report[f"train_scale{scale:g}"] = {
        "batches": N_BATCHES, "batches_with_seg_loss": applied, "peak_memory_gb": peak,
        "diffusion_loss": float(np.mean(diff_l)), "seg_loss": float(np.mean(seg_l)) if seg_l else None,
        "grad_norm_diffusion": float(np.mean(g_diff)), "grad_norm_seg_per_unit_weight": float(np.mean(g_seg)) if g_seg else None,
        "ratio_seg_over_diffusion_per_unit_weight": ratio, "seconds": time.time() - t0}
    print(f"input {int(512 * scale)} px: peak memory {peak:.1f} GB | seg loss on {applied}/{N_BATCHES} batches | "
          f"diffusion loss {np.mean(diff_l):.4f}, seg loss {np.mean(seg_l):.3f} | grad norm diffusion {np.mean(g_diff):.5f}, "
          f"seg (weight 1) {np.mean(g_seg):.4f} -> ratio {ratio:.1f} | {time.time() - t0:.0f} s", flush=True)
    for share in (0.5, 1.0, 1.5, 3.0):
        print(f"   weight for seg gradient = {share:g} x diffusion gradient: {share / ratio:.4f}")

OUT.write_text(json.dumps(report, indent=2))
print(f"report: {OUT}")
