"""
Training curve visualization.
Reads the metrics CSV produced by Trainer and saves PNG plots.

Usage (called automatically at end of each epoch in train.py):
    from src.plotting import plot_training_curves
    plot_training_curves(metrics_csv_path, output_dir)

Or call manually after training:
    python -c "
    from src.plotting import plot_training_curves
    plot_training_curves('logs/metrics_12345.csv', 'outputs')
    "
"""

from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (works on HPC without display)
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _smooth(values: list, window: int = 20) -> list:
    """Apply a rolling average for smoother curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # Pad front to preserve length alignment
    pad = [values[0]] * (window - 1)
    return pad + smoothed.tolist()


def plot_training_curves(metrics_csv_path: str | Path, output_dir: str | Path,
                         run_id: str = "") -> None:
    """
    Read the training metrics CSV and save loss + FID plots as PNGs.

    The CSV has columns:
        step, epoch, train_loss, val_loss, miou, fid, control_strength, lr

    Files are produced in output_dir/plots/, prefixed with run_id so multiple
    runs don't overwrite each other:
        <run_id>_loss_curve.png       — smoothed train loss + val loss over steps
        <run_id>_fid_curve.png        — FID score per epoch (when available)
        <run_id>_grad_norm_curve.png  — gradient norm per epoch

    Args:
        metrics_csv_path: Path to the metrics CSV file
        output_dir: Root output directory (plots/ subfolder will be created)
        run_id: Run identifier prefixed to every output filename (e.g. SLURM job id)
    """
    prefix = f"{run_id}_" if run_id else ""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping plot generation")
        return

    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        print(f"Metrics CSV not found: {metrics_csv_path} — skipping plots")
        return

    # -------------------------------------------------------------------------
    # Parse CSV
    # -------------------------------------------------------------------------
    steps, epochs, train_losses, val_losses = [], [], [], []
    fid_values, kid_values, lpips_values, grad_norms = [], [], [], []

    def _opt(row, key):
        raw = row.get(key, "").strip()
        return float(raw) if raw else None

    with open(metrics_csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                steps.append(int(row["step"]))
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                fid_values.append(_opt(row, "fid"))
                kid_values.append(_opt(row, "kid"))
                lpips_values.append(_opt(row, "lpips"))
                grad_norms.append(_opt(row, "grad_norm"))
            except (ValueError, KeyError):
                continue

    if not steps:
        print("No data in metrics CSV — skipping plots")
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Plot 1: Train loss + Val loss
    # -------------------------------------------------------------------------
    smooth_window = max(1, len(train_losses) // 20)  # ~5% of data
    train_smooth = _smooth(train_losses, window=smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_smooth, label="Train loss (smoothed)", color="steelblue", linewidth=1.5)
    ax.plot(steps, train_losses, alpha=0.2, color="steelblue", linewidth=0.8)
    ax.plot(steps, val_losses, label="Val loss", color="darkorange",
            linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Global step")
    ax.set_ylabel("MSE loss (noise prediction)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / f"{prefix}loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {loss_path}")

    # -------------------------------------------------------------------------
    # Plot 2: FID per epoch
    # -------------------------------------------------------------------------
    fid_pairs = [(e, f) for e, f in zip(epochs, fid_values) if f is not None]
    if fid_pairs:
        fid_epochs, fid_scores = zip(*fid_pairs)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fid_epochs, fid_scores, color="seagreen", linewidth=1.5,
                marker="o", markersize=4, label="FID")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID (lower is better)")
        ax.set_title("FID Score Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fid_path = plots_dir / f"{prefix}fid_curve.png"
        fig.savefig(fid_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fid_path}")
    else:
        print("No FID values in CSV — skipping FID plot")

    # -------------------------------------------------------------------------
    # Plot 2b/2c: KID and LPIPS per epoch (the reliable small-sample metrics)
    # -------------------------------------------------------------------------
    def _plot_epoch_metric(values, fname, ylabel, title, color, lower_better=True):
        pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not pairs:
            return
        xs, ys = zip(*pairs)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=4, label=ylabel)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel + (" (lower is better)" if lower_better else " (higher is better)"))
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = plots_dir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    _plot_epoch_metric(kid_values, f"{prefix}kid_curve.png", "KID",
                       "KID Over Training (unbiased — trust over FID at n=50)", "purple")
    _plot_epoch_metric(lpips_values, f"{prefix}lpips_curve.png", "LPIPS vs GT",
                       "LPIPS to Ground-Truth Over Training", "teal")

    # -------------------------------------------------------------------------
    # Plot 3: Gradient norm per epoch
    # -------------------------------------------------------------------------
    gn_pairs = [(e, g) for e, g in zip(epochs, grad_norms) if g is not None]
    if gn_pairs:
        gn_epochs, gn_vals = zip(*gn_pairs)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gn_epochs, gn_vals, color="crimson", linewidth=1.5,
                marker="o", markersize=4, label="Avg grad norm")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient norm (after clipping)")
        ax.set_title("Gradient Norm Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        gn_path = plots_dir / f"{prefix}grad_norm_curve.png"
        fig.savefig(gn_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {gn_path}")
    else:
        print("No grad norm values in CSV — skipping grad norm plot")
