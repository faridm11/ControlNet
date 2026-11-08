"""
Step 12 — CPU tests for the ten code-review findings of 2026-10-17 (one check or more per fix).

Run from DDPM/:  HF_HOME=... PYTHONPATH=$PWD:<segdeps> python controlnet_testing/step12_review_fixes/test_review_fixes.py
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
PY = sys.executable

import numpy as np
import torch

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def run(args, env=None):
    return subprocess.run([PY] + args, capture_output=True, text=True, env={**os.environ, **(env or {})})


# 1. prompt generation reads the label space from the data root
tmp_name = "prompts_meta_reviewtest.csv"
r = run(["data_prep/generate_prompts.py", "--data-root", "data_sanpo", "--split", "test", "--crop", "512",
         "--min-area", "0.002", "--variant", "meta", "--out-name", tmp_name])
new = Path("data_sanpo/test/prompts") / tmp_name
same = new.exists() and new.read_text() == Path("data_sanpo/test/prompts/prompts_meta.csv").read_text()
check("1. SANPO prompts without --label-space: label space read from LABEL_SPACE, output identical to prompts_meta.csv",
      same and "Label space: sanpo" in r.stdout, r.stdout.strip().splitlines()[0] if r.stdout else r.stderr[-200:])
new.unlink(missing_ok=True)
r = run(["data_prep/generate_prompts.py", "--data-root", "data_sanpo", "--split", "test", "--label-space", "v1",
         "--out-name", tmp_name])
check("1. a --label-space that contradicts the data root stops with an error", r.returncode != 0 and "disagrees" in (r.stderr + r.stdout))
(Path("data_sanpo/test/prompts") / tmp_name).unlink(missing_ok=True)

# 3. invalid CHECKPOINT_METRIC
r = run(["-c", "from src import config"], {"DDPM_CHECKPOINT_METRIC": "kid_mean"})
check("3. CHECKPOINT_METRIC typo ('kid_mean') is rejected at start-up", r.returncode != 0 and "CHECKPOINT_METRIC must be" in r.stderr)

from src import config
from src.data.dataset import ControlNetDataset, _RAW_TO_SEG
from src.segmentor import SEG_IGNORE

# 2. segmentor target follows the augmented mask
ds = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE, resolution=512,
                       augment_masks=True, mask_aug_params=dict(jitter_prob=0, dilate_erode_prob=0, elastic_prob=0,
                                                                occlusion_prob=1.0, occlusion_patches=6, occlusion_size=400))
plain = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE, resolution=512, augment_masks=False)
import random
random.seed(1)
a, p = ds[5], plain[5]
occluded = (a["mask"] == 0) & (p["mask"] != 0)  # pixels the augmentation turned into void
check("2. occlusions exist in the augmented sample", occluded.sum() > 1000, f"{int(occluded.sum())} occluded pixels")
check("2. segmentor target is 'ignore' exactly where the conditioning mask was occluded",
      bool((a["mask_seg"][occluded] == SEG_IGNORE).all()), f"{int((a['mask_seg'][occluded] != SEG_IGNORE).sum())} mismatches")
keep = ~occluded & (p["mask"] != 0)
check("2. outside the occlusions the target is unchanged", torch.equal(a["mask_seg"][keep], p["mask_seg"][keep]))

# 4. final evaluation reuses the metrics of the selected checkpoint
from src.train import Trainer


class _Model:
    def eval(self): pass
    def train(self): pass


class _NoEval:
    def evaluate(self, *a, **k):
        raise AssertionError("evaluate() should not be called when metrics can be reused")


t = object.__new__(Trainer)
t.model, t.evaluator, t.val_loader, t.pipeline, t.run_id = _Model(), _NoEval(), None, None, "reviewtest"
config.LOG_DIR = Path(tempfile.mkdtemp())
stored = {"fid": 150.0, "kid_mean": 0.02, "kid_std": 0.01, "lpips": 0.6, "clip_prompt": 0.3, "clip_img": 0.7,
          "miou": 0.5, "pixel_acc": 0.9, "miou_real": 0.8, "iou_per_class": {"road": 0.5}}
try:
    out = t.evaluate_full(control_strengths=[config.CONTROLNET_CONDITIONING_SCALE], reuse_metrics=stored)
    check("4. final eval reuses the stored metrics (no second generation pass)", out["results"][0]["kid_mean"] == 0.02)
except AssertionError as e:
    check("4. final eval reuses the stored metrics (no second generation pass)", False, str(e))
t.best_val_loss, t.best_kid, t.best_epoch, t.best_gen_metrics = float("inf"), float("inf"), None, None
t.save_checkpoint = lambda: None
config.CHECKPOINT_METRIC = "kid"
t._update_best(3, 0.15, stored)
check("4. _update_best stores the metrics of the saved epoch", t.best_gen_metrics is stored and t.best_epoch == 3)

# 5. start-of-run wording
src = Path("src/train.py").read_text()
check("5. epoch 0 is described as 'start of run', not 'untrained baseline'",
      "untrained baseline" not in src and "START OF RUN" in src and "INIT_LORA_CHECKPOINT" in src)

# 6. + 10. plotting
from src.plotting import _read, plot_training_curves
_, col_old, old = _read("outputs/logs/metrics_1698836.csv")
_, col_new, new_fmt = _read("controlnet_testing/step6_phase2/objects_segloss_w0.01/logs/metrics_1836958.csv")
check("6. pre-audit CSV detected as old format (epochs get shifted by one)", old and not new_fmt)
import matplotlib.pyplot as plt
captured = {}
orig_savefig = plt.Figure.savefig
def spy(self, path, *a, **k):
    captured[Path(path).name] = [t.get_text() for ax in self.axes for t in ax.texts]
    return orig_savefig(self, path, *a, **k)
plt.Figure.savefig = spy
plot_training_curves("controlnet_testing/step6_phase2/objects_segloss_w0.01/logs/metrics_1836958.csv", tempfile.mkdtemp(), run_id="x")
plot_training_curves("outputs/logs/metrics_1698836.csv", tempfile.mkdtemp(), run_id="old")
plt.Figure.savefig = orig_savefig
check("10. diagnostics figure has no reference labels; overview has them", not captured["x_diagnostics.png"]
      and "start of run" in captured["x_overview.png"])
check("6. old-format overview has no 'start of run' reference (it has no evaluation-only row)",
      "start of run" not in captured["old_overview.png"])

# 7. + 8. + 9. evaluator: shared segmentor, cached real-image result, stubs removed
from src.evaluation import ControlNetEvaluator
from torch.utils.data import Subset
from src.data import create_dataloader


class _CountingSeg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return torch.zeros(x.shape[0], 10, x.shape[2], x.shape[3])


class _FakePipe:  # returns the input mask as the "generated" image
    def __call__(self, prompt, image, **k):
        class R: pass
        r = R(); r.images = [image]
        return r


for k in ("EVAL_FID", "EVAL_KID", "EVAL_LPIPS", "EVAL_CLIP"):
    setattr(config, k, False)
seg = _CountingSeg()
m = _Model()
m._segmentor = [seg]
loader = create_dataloader(Subset(plain, [0, 1, 2, 3]), batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
ev = ControlNetEvaluator(device="cpu")
r1 = ev.evaluate(m, loader, _FakePipe(), num_inference_steps=1)
calls1 = seg.calls
r2 = ev.evaluate(m, loader, _FakePipe(), num_inference_steps=1)
check("7. evaluator uses the model's segmentor instead of loading a second copy", ev.segmentor is seg)
check("8. real-image segmentation runs once per val set (4 images, one call per image: 8 calls, then 4)", (calls1, seg.calls - calls1) == (8, 4),
      f"first evaluate {calls1} calls, second {seg.calls - calls1}")
check("8. cached miou_real equals the first result", r1["miou_real"] == r2["miou_real"])
check("9. dead stubs load_segmentor / evaluate_miou removed",
      not hasattr(ControlNetEvaluator, "load_segmentor") and not hasattr(ControlNetEvaluator, "evaluate_miou"))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
