"""
Build controlnet_testing/RUN_INDEX.md — one table row per training run found on disk, so every
log, plot, sample folder and checkpoint can be located from its run id without memory.

Scans every folder under DDPM (except _archive_not_in_report/, which holds superseded and bug-affected runs) for
logs/metrics_<run_id>.csv, and reads next to it: <run_id>_config.json (settings),
<run_id>_final_eval.json (final metrics), plots/, samples/, checkpoints/.
Also lists the evaluation JSON folders and the downstream results.

Re-run after every round:   python controlnet_testing/reporting/index_runs.py
"""

import csv
import glob
import json
from datetime import datetime
from pathlib import Path

DDPM = Path(__file__).resolve().parents[2]
T = DDPM / "controlnet_testing"

# Known issues per run id (see EXPERIMENT_LOG.md)
RNG_BUG = {"1836710", "1836716", "1836717", "1836718", "1836719", "1836957", "1836958", "1836959", "1836696"}
PRE_AUDIT = {"1581867", "1693046", "1698835", "1698836", "1587490"}


def fmt(v, p=3):
    try:
        return f"{float(v):.{p}f}"
    except (TypeError, ValueError):
        return "–"


rows = []
for csv_path in sorted(glob.glob(str(DDPM / "**/logs/metrics_*.csv"), recursive=True)):
    csv_path = Path(csv_path)
    if "/data" in str(csv_path.relative_to(DDPM))[:6] or str(csv_path.relative_to(DDPM)).startswith("_archive"):
        continue
    run_id = csv_path.stem.replace("metrics_", "")
    run_dir = csv_path.parent.parent
    data = list(csv.DictReader(open(csv_path)))
    cfg_file = csv_path.parent / f"{run_id}_config.json"
    cfg = json.load(open(cfg_file)) if cfg_file.exists() else {}
    fin_file = csv_path.parent / f"{run_id}_final_eval.json"
    fin = json.load(open(fin_file))["results"][0] if fin_file.exists() else {}
    kids = [(float(r["kid"]), int(float(r["epoch"]))) for r in data if r.get("kid")]
    best = min(kids) if kids else None
    epochs_done = max((int(float(r["epoch"])) for r in data), default=0)
    ckpt = run_dir / "checkpoints"
    kept = [p.name for p in ckpt.iterdir() if p.name.startswith(run_id + "_")] if ckpt.exists() else []
    notes = []
    if run_id in PRE_AUDIT:
        notes.append("old pipeline (before 2026-10-10 audit)")
    if run_id.split("_")[-1] in RNG_BUG:
        notes.append("RNG bug: noise repeated after evaluations")
    if not fin:
        notes.append("no final evaluation (running, failed or pre-audit)")
    rows.append({
        "run": run_id,
        "dataset": cfg.get("DATASET_NAME", "sidewalk"),
        "name": cfg.get("RUN_NAME", "") or run_dir.name,
        "folder": str(run_dir.relative_to(DDPM)),
        "settings": (f"LR {cfg.get('CONTROLNET_LORA_LR', '?')}, r{cfg.get('CONTROLNET_LORA_RANK', '?')}, "
                     f"{cfg.get('PROMPTS_FILE_NAME', 'prompts.csv').replace('.csv', '')}"
                     + (f", seg-loss {cfg.get('SEGMENTOR_LOSS_WEIGHT_START', 0.0)}→{cfg.get('SEGMENTOR_LOSS_WEIGHT')}"
                        if cfg.get("USE_SEGMENTOR_LOSS") else "")
                     + (f", checkpoint = {cfg['CHECKPOINT_METRIC']}" if cfg.get("CHECKPOINT_METRIC", "kid") != "kid" else "")) if cfg else "–",
        "epochs": epochs_done,
        "best_kid": f"{best[0]:.4f} @ ep {best[1]}" if best else "–",
        # mIoU 0.0 without miou_real = the old placeholder from before the segmentor was added
        "final": (f"KID {fmt(fin.get('kid_mean'), 4)}, mIoU "
                  + (fmt(fin.get('miou')) if fin.get("miou_real") is not None else "not measured")) if fin else "–",
        "files": ", ".join(sorted(kept)) or "none",
        "notes": "; ".join(notes),
    })

lines = [
    "# Run index",
    "",
    f"Generated {datetime.now():%Y-%m-%d %H:%M} by `reporting/index_runs.py`. One row per training run; the run id is in every "
    "file name of that run (`logs/metrics_<run>.csv`, `logs/<run>_config.json`, `logs/<run>_final_eval.json`, "
    "`plots/<run>_*.png`, `samples/<run or job…>_epoch…/`, `checkpoints/<run>_LoRA.pt`). Runs on dataset 2 start with `sanpo_`.",
    "",
    "Merged model copies (`<run>_Full/`, 1.4 GB each) of superseded runs were deleted on 2026-10-17 to save space; "
    "the LoRA file `<run>_LoRA.pt` is kept and rebuilds them (`src/models/diffusion_controlnet.py: load_checkpoint` + `create_pipeline`).",
    "",
    "| Run | Dataset | Name | Folder | Settings | Epochs | Best KID | Final eval | Checkpoint files kept | Notes |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in sorted(rows, key=lambda r: (r["folder"].startswith("controlnet_testing/FINAL") is False, r["folder"], r["run"])):
    lines.append(f"| {r['run']} | {r['dataset']} | {r['name']} | `{r['folder']}` | {r['settings']} | {r['epochs']} | "
                 f"{r['best_kid']} | {r['final']} | {r['files']} | {r['notes']} |")

lines += ["", "## Evaluation result folders (three-seed and standalone evaluations)", "",
          "| Folder | JSON files |", "|---|---|"]
for d in sorted({Path(p).parent for p in glob.glob(str(T / "**/eval_*.json"), recursive=True)}):
    lines.append(f"| `{d.relative_to(DDPM)}` | {len(list(d.glob('eval_*.json')))} |")

lines += ["", "## Downstream segmentor results (`step20_downstream_sanpo/results/`, 522 real SANPO test photos)", "",
          "| Setting | Start weights | Real photos | Test mIoU per seed |", "|---|---|---|---|"]
for f in sorted(glob.glob(str(T / "step20_downstream_sanpo/results/*.json"))):
    d = json.load(open(f))
    per_seed = " / ".join(f"{r['miou']:.3f}" for r in d["runs"])
    lines.append(f"| {d['name']} | {d.get('init', 'ade20k')} | {d.get('real_split', 'limited_real')} | {per_seed} |")

(T / "RUN_INDEX.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} runs indexed -> {T / 'RUN_INDEX.md'}")
