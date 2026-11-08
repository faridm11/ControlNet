"""
Training curves from the per-run metrics CSV written by Trainer.

Called at the end of every epoch (train.py) and usable by hand:
    python -c "from src.plotting import plot_training_curves; plot_training_curves('outputs/logs/metrics_<run>.csv', 'outputs', run_id='<run>')"

Writes two figures into <output_dir>/plots/, prefixed with the run id (which carries the
dataset name, e.g. sanpo_1838500):

  <run>_overview.png     — what the run achieved, one panel per metric over epochs completed:
                           KID, segmentor mIoU (with the real-photo ceiling), LPIPS to the paired
                           real photo, CLIP image↔prompt, and validation loss. Epoch 0 (start of
                           the run: the pretrained model, or INIT_LORA_CHECKPOINT) is drawn as a
                           dashed reference line in every panel.
  <run>_diagnostics.png  — whether training behaved: train loss, gradient norm (before clipping),
                           learning rate, and the Phase-2 segmentation loss when it was used.

FID is not plotted: with 50–150 images it is unreliable (KID is the small-sample replacement);
it stays in the CSV and the final-evaluation JSON. Arrows in panel titles: ↓ lower is better, ↑ higher is better.

CSV formats: current runs write "epochs completed" with an evaluation-only row at epoch 0
(train_loss = nan). Runs from before 2026-10-10 wrote the 0-based epoch index of the trained
epoch and have no evaluation-only row; they are detected and shifted by one, without a reference.
"""

import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # no display on HPC nodes
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

LINE = "#2a78d6"       # one series per panel → one colour
REF = "#8c8b86"        # reference lines (baseline, ceiling)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _read(metrics_csv_path):
    with open(metrics_csv_path) as f:
        rows = list(csv.DictReader(f))
    def col(name):
        out = []
        for r in rows:
            v = (r.get(name) or "").strip()
            try:
                out.append(float(v))
            except ValueError:
                out.append(None)
        return out
    # Old trainer: 0-based index of a TRAINED epoch, no evaluation-only row → shift by one
    first_train = col("train_loss")[0] if rows else None
    old_format = bool(rows) and float(rows[0].get("epoch", 1)) == 0 and first_train is not None and not np.isnan(first_train)
    return rows, col, old_format


def _panel(ax, epochs, values, title, lower_better=None, ceiling=None, ceiling_label="", reference=True):
    pts = [(e, v) for e, v in zip(epochs, values) if v is not None and not np.isnan(v)]
    if not pts:
        ax.set_visible(False)
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=LINE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
    if reference and xs[0] == 0 and len(xs) > 1:  # epoch 0 = start of run, drawn as reference
        ax.axhline(ys[0], color=REF, linewidth=1, linestyle="--")
        ax.annotate("start of run", (xs[-1], ys[0]), textcoords="offset points", xytext=(0, 3),
                    ha="right", fontsize=8, color=INK2)
    if ceiling is not None:
        ax.axhline(ceiling, color=REF, linewidth=1, linestyle=":")
        ax.annotate(ceiling_label, (xs[-1], ceiling), textcoords="offset points", xytext=(0, -10),
                    ha="right", fontsize=8, color=INK2)
    suffix = "" if lower_better is None else (" (lower is better)" if lower_better else " (higher is better)")
    ax.set_title(title + suffix)
    ax.set_xlabel("Epochs completed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
    ax.margins(x=0.05)


def plot_training_curves(metrics_csv_path, output_dir, run_id: str = "") -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping plots")
        return
    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        print(f"Metrics CSV not found: {metrics_csv_path} — skipping plots")
        return
    rows, col, old_format = _read(metrics_csv_path)
    if not rows:
        print("No data in metrics CSV — skipping plots")
        return

    _style()
    prefix = f"{run_id}_" if run_id else ""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = col("epoch")
    if old_format:
        epochs = [e + 1 if e is not None else None for e in epochs]
    real_miou = [v for v in col("miou_real") if v is not None]

    # ---- overview: results
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    _panel(axes[0, 0], epochs, col("kid"), "KID ↓", lower_better=None)
    _panel(axes[0, 1], epochs, col("miou"), "Segmentor mIoU ↑", lower_better=None,
           ceiling=real_miou[0] if real_miou else None, ceiling_label="segmentor on real photos")
    _panel(axes[0, 2], epochs, col("lpips"), "LPIPS to real photo ↓", lower_better=None)
    _panel(axes[1, 0], epochs, col("clip_prompt"), "CLIP image–prompt ↑", lower_better=None)
    _panel(axes[1, 1], epochs, col("clip_img"), "CLIP image–real photo ↑", lower_better=None)
    _panel(axes[1, 2], epochs, col("val_loss"), "Validation loss ↓", lower_better=None)
    fig.suptitle(f"Run {run_id or metrics_csv_path.stem}: results per evaluation", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}overview.png", dpi=150)
    plt.close(fig)

    # ---- diagnostics: training behaviour
    seg = col("seg_loss")
    panels = [("train_loss", "Train loss (epoch mean; noisy by nature)"), ("grad_norm", "Gradient norm (before clipping)"),
              ("lr", "Learning rate (end of epoch)")]
    if any(v is not None for v in seg):
        panels.append(("seg_loss", "Segmentation loss (Phase 2)"))
    if any(v is not None for v in col("seg_weight")):
        panels.append(("seg_weight", "Seg-loss weight λ (end of epoch)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6))
    for ax, (name, title) in zip(np.atleast_1d(axes), panels):
        vals = col(name)
        vals = [None if e == 0 else v for e, v in zip(epochs, vals)]  # epoch 0 = evaluation only, no training
        _panel(ax, epochs, vals, title, reference=False)
    fig.suptitle(f"Run {run_id or metrics_csv_path.stem}: training diagnostics", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, "Train loss depends on the random noise levels drawn each epoch, so it does not fall smoothly; "
             "the validation loss (fixed noise) in the overview figure is the convergence signal.",
             fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # room for the note at the bottom
    fig.savefig(plots_dir / f"{prefix}diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {plots_dir / (prefix + 'overview.png')} and {prefix}diagnostics.png")
