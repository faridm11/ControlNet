"""CPU checks for the λ ramp (step 16): start → target over the warmup, constant after; λ column in the CSV
reaches the diagnostics plot; defaults keep the old behaviour (0 → target)."""
import os, sys, csv, tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.update(DDPM_SEGMENTOR_LOSS_WEIGHT_START="0.05", DDPM_SEGMENTOR_LOSS_WEIGHT="0.4",
                  DDPM_SEGMENTOR_LOSS_WARMUP_STEPS="6344", DDPM_USE_SEGMENTOR_LOSS="true")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
from src.train import Trainer
from src.plotting import plot_training_curves

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

w = lambda step: Trainer._seg_weight(SimpleNamespace(global_step=step))
check("λ at first step ≈ 0.05", abs(w(0) - 0.05) < 1e-4)
check("λ at half of training ≈ 0.225", abs(w(3171) - 0.225) < 1e-3)
check("λ at last step = 0.4", abs(w(6343) - 0.4) < 1e-9)
check("λ constant after the ramp", w(10000) == 0.4)
check("λ increases monotonically", all(w(s) < w(s + 1) for s in range(0, 6343, 97)))
check("λ per epoch (793 steps): " + ", ".join(f"{w(793 * e - 1):.3f}" for e in range(1, 9)),
      abs(w(793 * 4 - 1) - 0.225) < 1e-3)

config.SEGMENTOR_LOSS_WEIGHT_START, config.SEGMENTOR_LOSS_WEIGHT, config.SEGMENTOR_LOSS_WARMUP_STEPS = 0.0, 0.006, 793
check("default start 0 reproduces the earlier run (0 → 0.006 over 793)", abs(w(792) - 0.006) < 1e-12 and abs(w(0) - 0.006 / 793) < 1e-12)

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "metrics_x.csv"
    with open(p, "w") as f:
        f.write("step,epoch,train_loss,val_loss,lr,grad_norm,fid,kid,lpips,clip_prompt,clip_img,miou,miou_real,seg_loss,seg_weight\n")
        f.write("0,0,,0.1,0.0001,,150,0.03,0.7,0.3,0.75,0.45,0.62,,\n")
        for e in range(1, 5):
            f.write(f"{793*e},{e},0.12,0.1,0.0001,0.5,140,0.01,0.66,0.3,0.8,0.5,0.62,1.0,{0.05+0.0875*e:.4f}\n")
    plot_training_curves(p, d, run_id="t")
    check("plots written with the λ column", (Path(d) / "plots/t_diagnostics.png").exists())
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
