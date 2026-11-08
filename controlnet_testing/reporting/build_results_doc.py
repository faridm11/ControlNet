"""
Build RESULTS.docx — the raw results document for the report.

Only runs made after the pipeline fixes, scored with the judges used in the report (Mapillary, ADE20K) are included.
Superseded and bug-affected results are in DDPM/_archive_not_in_report/ (see its README); their history is in
EXPERIMENT_LOG.md. Everything here is read from the result files, so re-running the script after new jobs brings the
document up to date. New training runs only need a line in RUN_GROUPS below.

Run from anywhere:
    PYTHONPATH=/home/woody/iwi5/iwi5388h/software/private/reportdeps \
        python controlnet_testing/reporting/build_results_doc.py
"""

import csv
import glob
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches, Pt
from PIL import Image

T = Path(__file__).resolve().parents[1]          # controlnet_testing/
OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
DS = T / "step20_downstream_sanpo"

# ---------------------------------------------------------------------------
# Registry of training runs: (section title, intro, mIoU shown?, [(run dir, label, settings), ...])
# mIoU is hidden for runs whose training-time mIoU came from the old UNet++ judge (replaced, not used in the report).
# ---------------------------------------------------------------------------
RUN_GROUPS = [
    ("Hyperparameter screen — dataset 1 (10 epochs each, one seed)",
     "All four runs share one 10-epoch schedule, so they are comparable with each other. KID only: their training-time mIoU "
     "used the old judge. Result: LoRA on attention + feed-forward layers was kept for v2.", False,
     [("step11_hparam_screen/reference", "reference", "rank 16, attention + conditioning conv"),
      ("step11_hparam_screen/rank32", "rank 32", "rank 32, alpha 32"),
      ("step11_hparam_screen/attn_ff", "attn + ff", "LoRA also on transformer projections and feed-forward layers"),
      ("step11_hparam_screen/hflip", "flip", "joint horizontal flip of photo and mask")]),
    ("v1 (FINAL) — both datasets",
     "LR 1e-4, LoRA rank 16 on attention + conditioning conv, dropout 0, batch 4, checkpoint by KID; V100 fp16. "
     "Dataset 1: 20 epochs, object prompts. SANPO: 8 epochs, prompts with recorded conditions. Training-time mIoU used the old "
     "judge and is not shown; three-seed Mapillary scores are in the comparison section.", False,
     [("FINAL/sidewalk/run", "v1, dataset 1", "LR 1e-4, rank 16, 20 epochs, prompts_objects"),
      ("FINAL/sanpo/run", "v1, SANPO", "LR 1e-4, rank 16, 8 epochs, prompts_meta")]),
    ("v2 (B, no segmentation loss) — both datasets",
     "v1 + LoRA also on transformer projections and feed-forward layers + training-time prompt augmentation (each true "
     "condition/object phrase kept with p = 0.7, order shuffled). Training-time mIoU: Mapillary judge.", True,
     [("FINAL_v2/sidewalk/run", "v2, dataset 1", "LR 1e-4, rank 16, attn+ff, 20 epochs, prompts_objects, prompt keep 0.7"),
      ("FINAL_v2/sanpo/run", "B (v2), SANPO", "LR 1e-4, rank 16, attn+ff, 8 epochs, prompts_meta, prompt keep 0.7")]),
    ("SANPO: B + segmentation loss, four λ settings",
     "Same recipe as B, trained from the pretrained model, plus the segmentation loss: frozen Mapillary judge on the decoded "
     "x0 estimate (cross-entropy + Dice, white/void pixels ignored), timestep < 400, at most 2 samples per batch. "
     "λ 0.006 and 0.02 ramp from 0 over the first epoch, checkpoint by KID; the two whole-training ramps use the last epoch.", True,
     [("FINAL_v2_segloss/sanpo/run", "λ 0.006", "λ 0 → 0.006 (epoch 1), then constant"),
      ("FINAL_v2_segloss/sanpo_l0.02/run", "λ 0.02", "λ 0 → 0.02 (epoch 1), then constant"),
      ("FINAL_v2_segloss/sanpo_l0.025-0.045/run", "λ 0.025 → 0.045 (selected)", "λ rises linearly over all 8 epochs, last epoch"),
      ("FINAL_v2_segloss/sanpo_ramp/run", "λ 0.05 → 0.4", "λ rises linearly over all 8 epochs, last epoch")]),
    ("SANPO: generator of the final flow, trained on only 200 photos",
     "Selected recipe (λ 0.025 → 0.045) on the 200 photos of the final downstream test only: 3,000 steps (15 epochs of 200 "
     "steps), λ rising over all steps, last epoch. Used to generate the fakes of setup C.", True,
     [("step20_downstream_sanpo/generator_200/run", "generator, 200 photos", "λ 0.025 → 0.045, 3000 steps, last epoch")]),
]

# Chart style: fixed categorical order (never cycled), recessive axes, ink-coloured text
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run(rel):
    """Return dict with per-epoch rows, final eval, job id and sample grids for one run dir."""
    d = T / rel
    csvs = sorted(glob.glob(str(d / "logs" / "metrics_*.csv")))
    if not csvs:
        return None
    rows = [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(open(csvs[-1]))]
    if not rows:
        return None
    job = Path(csvs[-1]).stem.replace("metrics_", "")
    fj = d / "logs" / f"{job}_final_eval.json"
    final = json.load(open(fj))["results"][0] if fj.exists() else None
    grids = sorted(glob.glob(str(d / "samples" / "*" / "sample_*_concat.png")))
    return {"rows": rows, "final": final, "job": job, "grids": grids, "done": final is not None}


def evals(pattern, tag):
    """Results of all evaluation JSONs matching the glob whose tag (without _seedN) equals tag."""
    out = []
    for f in sorted(glob.glob(str(T / pattern))):
        d = json.load(open(f))
        if d["tag"].rsplit("_seed", 1)[0] == tag:
            out.append(d["results"][0])
    return out


def fmt(v, p=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


def msd(vals, p=3):
    """mean ± sample standard deviation (n-1)"""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return fmt(vals[0], p) if vals else "–"
    return f"{np.mean(vals):.{p}f} ± {np.std(vals, ddof=1):.{p}f}"


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(0.7))
doc.styles["Normal"].font.size = Pt(10)
PAGE_W = 9.6  # usable width in inches


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = str(h)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()
    return t


def picture(path, caption, width=PAGE_W):
    doc.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)


def shrink(path, max_w=2200):
    """Downscale a large image to keep the document small; returns the path to embed."""
    im = Image.open(path)
    if im.width <= max_w:
        return path
    out = FIG / ("small_" + "_".join(Path(path).parts[-4:]).replace("/", "_"))
    im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS).convert("RGB").save(out.with_suffix(".jpg"), quality=88)
    return out.with_suffix(".jpg")


def curves(runs, name, show_miou):
    """Small multiples (one axis each): KID, (mIoU), LPIPS, val loss vs epochs completed."""
    panels = [("kid", "KID (lower is better)"), ("lpips", "LPIPS to real image (lower is better)"), ("val_loss", "Validation loss")]
    if show_miou:
        panels.insert(1, ("miou", "Mapillary judge mIoU (higher is better)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.9))
    for ax, (key, title) in zip(axes, panels):
        for i, (label, run) in enumerate(runs):
            pts = [(r["epoch"], r[key]) for r in run["rows"] if r.get(key) is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epochs completed (0 = no training)")
        ax.margins(x=0.04)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
doc.add_heading("ControlNet fine-tuning with a segmentation loss — results", 0)
doc.add_paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} from the files in DDPM/controlnet_testing/. Only valid results are "
                  "included: runs after the pipeline fixes, scored with the two judges below. Superseded and bug-affected results "
                  "are in DDPM/_archive_not_in_report/ (README there); the full history is in EXPERIMENT_LOG.md.")

n = 1
doc.add_heading(f"{n}. Measurement protocol", 1); n += 1
table(["Item", "Value"], [
    ["Base model", "Stable Diffusion 1.5 + lllyasviel/control_v11p_sd15_seg; only a LoRA on the ControlNet is trained (rank 16)"],
    ["Data", "Dataset 1: segments/sidewalk-semantic, 950 train / 50 test. Dataset 2: SANPO, 3,172 train / 522 test human-annotated "
             "chest-camera frames. 512×512 centre crop."],
    ["Conditioning", "ADE20K colours; classes without an ADE20K equivalent (SANPO unlabeled, obstacle, inaccessible surface, bus stop) "
                     "are white. White pixels are ignored pixel by pixel in every loss and metric, never whole images "
                     "(verified: 87 test images, 6.0 % white, 0 white pixels scored, 0 images excluded)."],
    ["Generation", "DDIM 35 steps, guidance 6.0, control scale 1.0. Dataset 1: all 50 test images; SANPO: a fixed 150-image subset."],
    ["Seeds", "Three-seed results use generation seeds 42, 43, 44; '±' is the standard deviation over seeds (n−1)."],
    ["KID / FID", "Distance between generated and real image sets, lower is better; KID subset 50. FID is unreliable at 50–150 images."],
    ["LPIPS / CLIP", "LPIPS: perceptual distance to the paired real photo (lower is better). CLIP image / prompt: similarity to the paired "
                     "photo / to the prompt (higher is better)."],
    ["Judge 1 (training + scoring)", "Mask2Former Swin-L trained on Mapillary Vistas, never trained on our data; used inside the "
                                     "segmentation loss and for scoring. Input 1024 px for scoring."],
    ["Judge 2 (independent)", "SegFormer-B5 trained on ADE20K, never trained on our data and not used in training; 640 px."],
    ["mIoU", "Both judges' classes are mapped to 12 shared classes (road, sidewalk, building, fence/wall, person, vehicle, bicycle, pole, "
             "sign/light, vegetation, terrain, sky). The judge labels the generated image; its labels are compared with the "
             "conditioning mask. 'Real photos' = the same judge on the real photos, the realistic ceiling."],
])

# --- training runs
for title, intro, show_miou, runs in RUN_GROUPS:
    loaded = [(label, settings, load_run(rel)) for rel, label, settings in runs]
    loaded = [(l, s, r) for l, s, r in loaded if r]
    if not loaded:
        continue
    doc.add_heading(f"{n}. Training: {title}", 1); n += 1
    doc.add_paragraph(intro)
    table(["Run", "Job", "Settings"], [[l, r["job"], s] for l, s, r in loaded])
    head = ["Run", "Epoch kept", "KID ↓", "FID ↓", "LPIPS ↓", "CLIP prompt ↑", "CLIP image ↑"] + (["mIoU ↑"] if show_miou else [])
    rows = []
    for l, s, r in loaded:
        if not r["done"]:
            continue
        f = r["final"]
        ep = int(max(x["epoch"] for x in r["rows"]))
        kids = [(x["kid"], x["epoch"]) for x in r["rows"] if x.get("kid") is not None and abs(x["kid"] - f["kid_mean"]) < 1e-6]
        if kids:
            ep = int(kids[-1][1])
        rows.append([l, ep, fmt(f["kid_mean"], 4), fmt(f["fid"], 1), fmt(f["lpips"]), fmt(f["clip_prompt"]), fmt(f["clip_img"])]
                    + ([fmt(f.get("miou"))] if show_miou else []))
    b = loaded[0][2]["rows"][0]
    rows.insert(0, ["before training (epoch 0)", 0, fmt(b["kid"], 4), fmt(b["fid"], 1), fmt(b["lpips"]), fmt(b["clip_prompt"]),
                    fmt(b["clip_img"])] + ([fmt(b.get("miou"))] if show_miou else []))
    doc.add_paragraph("Final evaluation of the kept checkpoint (seed 42, during training):")
    table(head, rows)
    if any(len(r["rows"]) > 2 for _, _, r in loaded):
        picture(curves([(l, r) for l, _, r in loaded], f"curves_{n}", show_miou),
                "Generation metrics at each evaluation; validation loss every epoch. Epoch 0 is the untrained model.")
    doc.add_heading("All logged values", 2)
    for l, s, r in loaded:
        doc.add_paragraph(f"{l} (job {r['job']})").runs[0].bold = True
        cols = ["Epochs done", "Train loss", "Seg loss", "λ", "Val loss", "KID", "FID", "LPIPS", "CLIP prompt", "CLIP image"] + (["mIoU"] if show_miou else [])
        table(cols, [[int(x["epoch"]), fmt(x["train_loss"], 4), fmt(x.get("seg_loss"), 4), fmt(x.get("seg_weight"), 3), fmt(x["val_loss"], 4),
                      fmt(x["kid"], 4), fmt(x["fid"], 1), fmt(x["lpips"], 4), fmt(x["clip_prompt"], 4), fmt(x["clip_img"], 4)]
                     + ([fmt(x.get("miou"), 4)] if show_miou else [])
                     for x in r["rows"] if x["epoch"] == 0 or x.get("kid") is not None or x["epoch"] == r["rows"][-1]["epoch"]])
    for l, s, r in loaded:
        for g in r["grids"][:1]:
            picture(shrink(g), f"{l} (job {r['job']}): real photo | conditioning mask | four generated images (different noise).")

# --- three-seed comparison, Mapillary judge
MODELS = {  # dataset -> [(eval glob, tag, label)]
    "sidewalk": [("FINAL/sidewalk/eval_mapillary/*.json", "rescore_pretrained_sidewalk", "pretrained"),
                 ("FINAL/sidewalk/eval_mapillary/*.json", "rescore_final_sidewalk", "v1"),
                 ("FINAL_v2/sidewalk/eval/*.json", "v2_sidewalk", "v2")],
    "sanpo": [("FINAL/sanpo/eval_mapillary/*.json", "rescore_pretrained_sanpo", "pretrained"),
              ("FINAL/sanpo/eval_mapillary/*.json", "rescore_final_sanpo", "v1"),
              ("FINAL_v2/sanpo/eval/*.json", "v2_sanpo", "B (v2, no seg loss)"),
              ("FINAL_v2_segloss/sanpo/eval/*.json", "v2_segloss_sanpo", "B + seg λ 0.006"),
              ("FINAL_v2_segloss/sanpo_l0.02/eval/*.json", "v2_segloss_l002_sanpo", "B + seg λ 0.02"),
              ("FINAL_v2_segloss/sanpo_l0.025-0.045/eval/*.json", "v2_segloss_l0025_0045_sanpo", "B + seg λ 0.025 → 0.045 (selected)"),
              ("FINAL_v2_segloss/sanpo_ramp/eval/*.json", "v2_segloss_ramp_sanpo", "B + seg λ 0.05 → 0.4")],
}
doc.add_heading(f"{n}. Three-seed results — realism and mask adherence (Mapillary judge)", 1); n += 1
doc.add_paragraph("Same test images, prompts and seeds for every model. Dataset 1: all 50 test images; SANPO: the fixed 150-image subset.")
rows, per_class = [], {}
real = {}
for ds, label in (("sidewalk", "Dataset 1"), ("sanpo", "SANPO")):
    for pattern, tag, name in MODELS[ds]:
        rs = evals(pattern, tag)
        if not rs:
            continue
        real[ds] = np.mean([r["miou_real"] for r in rs])
        rows.append([label, name, len(rs), msd([r["kid_mean"] for r in rs], 4), msd([r["fid"] for r in rs], 1), msd([r["lpips"] for r in rs]),
                     msd([r["clip_img"] for r in rs]), msd([r["miou"] for r in rs]), msd([r["pixel_acc"] for r in rs])])
        if ds == "sanpo":
            per_class[name] = rs
    if ds in real:
        rows.append([label, "real photos (judge ceiling)", "–", "–", "–", "–", "–", f"{real[ds]:.3f}", "–"])
table(["Dataset", "Model", "Seeds", "KID ↓", "FID ↓", "LPIPS ↓", "CLIP image ↑", "mIoU ↑", "Pixel acc ↑"], rows)
if per_class:
    cls = list(next(iter(per_class.values()))[0]["iou_per_class"])
    doc.add_paragraph("SANPO per-class IoU, Mapillary judge, mean of three seeds:")
    table(["Model"] + cls, [[name] + [f"{np.mean([r['iou_per_class'].get(c, np.nan) for r in rs]):.2f}" for c in cls]
                            for name, rs in per_class.items()])
for path, cap in (("FINAL_v2/sidewalk/comparison/comparison_sheet.jpg", "Dataset 1: real photo | mask | pretrained | v1 | v2."),
                  ("FINAL_v2/sanpo/comparison/comparison_sheet.jpg", "SANPO: real photo | mask | pretrained | v1 | B."),
                  ("FINAL_v2_segloss/sanpo_l0.025-0.045/comparison/comparison_sheet.jpg",
                   "SANPO: real photo | mask | pretrained | B | λ 0.02 | λ 0.025 → 0.045 | λ 0.05 → 0.4."),
                  ("FINAL_v2_segloss/sanpo_ramp/comparison/ramp_gallery_epoch8.jpg", "SANPO, λ 0.05 → 0.4: real photo | mask | four generations.")):
    if (T / path).exists():
        picture(shrink(T / path, 2600), cap + " Same mask, prompt and noise per row.")

# --- second judge and white split
J2 = [("pretrained", "pretrained"), ("B", "B (no seg loss)"), ("seg0.006", "λ 0.006"), ("seg0.02", "λ 0.02"),
      ("seg0.025-0.045", "λ 0.025 → 0.045 (selected)"), ("seg0.05-0.4", "λ 0.05 → 0.4")]
j2_rows, j2_real = [], None
for tag, label in J2:
    fs = glob.glob(str(T / f"step18_second_judge/eval/eval_judge2_{tag}_seed42_*.json"))
    if fs:
        r = json.load(open(fs[0]))["results"][0]
        j2_real = r["miou_real"]
        j2_rows.append([label, f"{r['kid_mean']:.4f}", f"{r['fid']:.1f}", f"{r['clip_img']:.3f}", f"{r['miou']:.3f}", f"{r['pixel_acc']:.3f}"])
if j2_rows:
    doc.add_heading(f"{n}. SANPO — independent judge (SegFormer-B5 ADE20K)", 1); n += 1
    doc.add_paragraph(f"Seed 42, 150-image subset, 640 px. Judge on the real photos: mIoU {j2_real:.3f}. "
                      "SegFormer Cityscapes was tested as a second judge and rejected: it calls 83 % of SANPO sidewalk 'road'.")
    j2_rows.append(["real photos (judge ceiling)", "–", "–", "–", f"{j2_real:.3f}", "–"])
    table(["Model", "KID ↓", "FID ↓", "CLIP image ↑", "mIoU ADE20K ↑", "Pixel acc ↑"], j2_rows)
    sp = T / "step18_second_judge/split_by_white.json"
    if sp.exists():
        d = json.load(open(sp)); res = d["results"]
        doc.add_paragraph("Same 150 images split at the median white share of the mask "
                          f"({100*d['meta']['threshold_white']:.1f} %) into a low-white and a high-white half (75 images each); seed 42.")
        rows = []
        for tag, label in J2 + [("real photos", "real photos")]:
            lo, hi = res["low_white"].get(tag), res["high_white"].get(tag)
            if lo is None:
                continue
            g = lambda r, k, p=3: f"{r[k]:.{p}f}" if k in r else "–"
            rows.append([label, g(lo, "kid", 4), g(hi, "kid", 4), g(lo, "miou_mapillary"), g(hi, "miou_mapillary"),
                         g(lo, "miou_ade20k"), g(hi, "miou_ade20k")])
        table(["Model", "KID low-white", "KID high-white", "mIoU Mapillary low", "mIoU Mapillary high", "mIoU ADE20K low", "mIoU ADE20K high"], rows)
        for g_ in ("low_white", "high_white"):
            sheet = T / f"step18_second_judge/images_by_white/{g_}_sheet.jpg"
            if sheet.exists():
                picture(shrink(sheet, 2600), f"{g_.replace('_', '-')} half: real photo | mask | pretrained | B | λ 0.006 | λ 0.02 | λ 0.05 → 0.4.")
    gal = T / "FINAL_v2_segloss/white_regions_gallery.jpg"
    if gal.exists():
        picture(shrink(gal, 2600), "What each model draws in the white (void) mask areas, outlined in red.")

# --- judge mapping check
rm = T / "step19_judge_mapping/rescore_mappings.json"
if rm.exists():
    d = json.load(open(rm))
    doc.add_heading(f"{n}. Check of the judge class mapping", 1); n += 1
    doc.add_paragraph("The saved seed-42 SANPO images were rescored with an alternative mapping (v2: Mapillary 'bench'-type classes and "
                      "ADE20K 'floor/path' moved). The model ranking does not change; v1, the mapping used everywhere else, is kept.")
    models = list(d["mapillary"]["v1"]["all"])
    table(["Model", "Mapillary v1", "Mapillary v2", "ADE20K v1", "ADE20K v2"],
          [[dict(J2).get(m, m)] + [fmt(d[j][v]["all"].get(m, {}).get("miou")) for j in ("mapillary", "ade20k") for v in ("v1", "v2")]
           for m in models])

# --- downstream: does generated data help a segmentation model trained on few real photos?
DOWN = [("300 real photos; fakes from 1,000 other masks; generator trained on all 3,172 photos; ADE20K start",
         [("real_only", "A: real only"), ("real_plus_base", "B: + base ControlNet fakes"), ("real_plus_ours", "C: + seg-loss fakes")]),
        ("FINAL FLOW — 200 real photos; 4 fakes per mask of the same 200 masks; generator trained on only the 200 photos; ADE20K start",
         [("real_only_200", "A: real only"), ("real_plus_base_200", "B: + 800 base ControlNet fakes"), ("real_plus_ours_200", "C: + 800 seg-loss fakes")]),
        ("Same as the final flow, segmentation model starting from ImageNet weights (no segmentation or street-scene pretraining)",
         [("imagenet_real_only_200", "A: real only"), ("imagenet_real_plus_base_200", "B: + 800 base ControlNet fakes"),
          ("imagenet_real_plus_ours_200", "C: + 800 seg-loss fakes")])]
res = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(DS / "results/*.json"))}
if res:
    doc.add_heading(f"{n}. Downstream test: do the generated images help a segmentation model?", 1); n += 1
    doc.add_paragraph("SegFormer-B5 (new 12-class head) fine-tuned for 3,000 steps, batch 4, LR 6e-5, horizontal flip; with fakes, half of "
                      "each batch is drawn from them. White/void pixels ignored. Three training seeds (0, 1, 2). Test: all 522 real SANPO "
                      "test photos, never used in any training step. '±' is the standard deviation over seeds (n−1).")
    for title, setups in DOWN:
        have = [(k, l) for k, l in setups if k in res]
        if not have:
            continue
        doc.add_heading(title, 2)
        rows = []
        for k, l in have:
            runs = res[k]["runs"]
            rows.append([l, runs[0]["train_images"], " / ".join(f"{r['miou']:.3f}" for r in runs), msd([r["miou"] for r in runs]),
                         msd([r["pixel_acc"] for r in runs])])
        table(["Setup", "Training images", "mIoU per seed", "mIoU ↑", "Pixel acc ↑"], rows)
        cls = list(res[have[0][0]]["runs"][0]["iou_per_class"])
        table(["Setup"] + cls, [[l] + [f"{np.mean([r['iou_per_class'].get(c, np.nan) for r in res[k]['runs']]):.3f}" for c in cls]
                                for k, l in have])
    doc.add_paragraph("Notes: in the 300-photo test the generator had seen all 3,172 training photos, so setup C there is a best case. "
                      "In the final flow only the 200 photos and their masks are used anywhere, apart from downloadable pretrained models.")

# --- alignment overlays
doc.add_heading(f"{n}. Data check: image / mask alignment after the training crop", 1)
for g in sorted(glob.glob(str(T / "step1_cpu_checks/overlays/*.png")))[:3]:
    picture(g, f"{Path(g).name}: photo | conditioning image | blend", width=7.5)

out = OUT.parent / "RESULTS.docx"
doc.save(out)
print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
