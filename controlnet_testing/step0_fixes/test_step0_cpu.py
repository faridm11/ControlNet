"""
Step 0 — CPU-only checks of the code fixes (no GPU, no model weights loaded).

Run from DDPM/:
    python controlnet_testing/step0_fixes/test_step0_cpu.py

Checks:
  1. config env overrides parse to the right types
  2. inference.load_mask turns a class-ID label PNG into the training conditioning image
  3. Trainer bookkeeping: CSV rows, KID-based checkpoint selection, python-float metadata
  4. a checkpoint written by save_checkpoint loads with the model's loader semantics
  5. plotting copes with the baseline row (epoch 0, train_loss = nan)
  6. the fixed validation timesteps are what we expect
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
from PIL import Image

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# 1. config env overrides (fresh interpreter so the import sees the variables)
out = subprocess.run(
    [sys.executable, "-c",
     "from src import config as c; print(repr((c.CONTROLNET_LORA_LR, c.CONTROLNET_LORA_RANK, "
     "c.USE_UNET_LORA, c.BATCH_SIZE, c.CLASS_VERSION, c.NUM_CLASSES, c.VAL_PROMPTS_FILE.name)))"],
    env={**os.environ, "DDPM_CONTROLNET_LORA_LR": "1e-4", "DDPM_CONTROLNET_LORA_RANK": "16",
         "DDPM_USE_UNET_LORA": "true", "DDPM_BATCH_SIZE": "4", "DDPM_CLASS_VERSION": "v1",
         "DDPM_PROMPTS_FILE_NAME": "prompts_crop.csv"},
    capture_output=True, text=True,
)
check("config env overrides", out.stdout.strip() == "(0.0001, 16, True, 4, 'v1', 35, 'prompts_crop.csv')",
      out.stdout.strip() or out.stderr.strip()[-300:])

from src import config
from src.data.dataset import ControlNetDataset

# 2. inference.load_mask == dataset conditioning image
import inference

ds = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE,
                       resolution=512, augment_masks=False)
max_diff = 0
for idx in (0, 7, 23):
    sample = ds[idx]
    expected = (sample["mask_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    got = np.array(inference.load_mask(str(config.VAL_MASKS_DIR / sample["image_name"]), 512))
    max_diff = max(max_diff, int(np.abs(expected.astype(int) - got.astype(int)).max()))
check("inference.load_mask matches dataset conditioning", max_diff == 0, f"max abs pixel diff = {max_diff}")

# 3 + 4. Trainer bookkeeping with a stub (no model, no GPU)
from src.train import Trainer

tmp = Path(tempfile.mkdtemp(prefix="step0_"))
config.CHECKPOINT_DIR = tmp
config.CHECKPOINT_METRIC = "kid"


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.other = torch.nn.Parameter(torch.ones(2))


class _Model:
    controlnet = _Stub()
    unet = _Stub()


t = object.__new__(Trainer)
t.model = _Model()
t.optimizer = torch.optim.AdamW(t.model.controlnet.parameters(), lr=1e-4)
t.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda s: 1.0)
t.global_step = 0
t.best_val_loss, t.best_kid, t.best_epoch = float("inf"), float("inf"), None
t.run_id = "cpu_test"
t.metrics_log_path = tmp / "metrics_cpu_test.csv"
t.metrics_log_path.write_text("step,epoch,train_loss,val_loss,lr,grad_norm,fid,kid,lpips,clip_prompt,clip_img,miou,miou_real,seg_loss,seg_weight\n")


def gm(kid):
    return {"fid": 150.0, "kid_mean": kid, "kid_std": 0.01, "lpips": 0.6, "clip_prompt": 0.27, "clip_img": 0.74}


# baseline, then epochs 1..4: KID best at epoch 2, val loss best at epoch 4
history = [(0, float("nan"), 0.160, gm(0.080)), (1, 0.15, np.float64(0.159), None),
           (2, 0.15, np.float64(0.158), gm(0.040)), (3, 0.15, np.float64(0.157), None),
           (4, 0.15, np.float64(0.156), gm(0.050))]
for ep, tr, va, g in history:
    t.global_step = ep * 100
    t._log_epoch_row(ep, tr, va, 1e-4 if ep else 0.0, np.float64(0.003) if ep else None, g)
    t._update_best(ep, va, g)

check("KID selects epoch 2 (val loss would pick 4)", t.best_epoch == 2, f"best_epoch = {t.best_epoch}")
rows = t.metrics_log_path.read_text().strip().splitlines()
check("CSV has baseline + 4 epoch rows, 15 columns each (seg_weight added in step 16)",
      len(rows) == 6 and all(len(r.split(",")) == 15 for r in rows), rows[1])

ckpt = torch.load(tmp / "cpu_test_LoRA.pt", map_location="cpu", weights_only=True)
check("checkpoint loads even with weights_only=True (metadata is plain python)",
      ckpt["epochs_done"] == 2 and type(ckpt["best_val_loss"]) is float and list(ckpt["controlnet_lora"]) == ["lora_A"],
      f"epochs_done={ckpt['epochs_done']}, keys={list(ckpt['controlnet_lora'])}")

# 5. plotting with the baseline row
from src.plotting import plot_training_curves

plot_training_curves(t.metrics_log_path, tmp, run_id="cpu_test")
made = sorted(p.name for p in (tmp / "plots").glob("*.png"))
check("plots written with baseline row present (overview + diagnostics)", made == ["cpu_test_diagnostics.png", "cpu_test_overview.png"], ", ".join(made))

# 6. validation timesteps
k = config.VAL_NUM_TIMESTEPS
ts = [int((i + 0.5) * 1000 / k) for i in range(k)]
check("validation timesteps", ts == [62, 187, 312, 437, 562, 687, 812, 937], str(ts))

print(f"\n{sum(results)}/{len(results)} checks passed. Scratch dir: {tmp}")
sys.exit(0 if all(results) else 1)
