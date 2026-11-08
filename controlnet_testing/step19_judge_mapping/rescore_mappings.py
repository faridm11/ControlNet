"""
Step 19 — re-score the saved SANPO images with judge mapping v1 and v2, side by side, no generation.

Images: step18_second_judge/images/<model>/ (seed 42, 150-image subset) and the real photos. Judges: Mapillary at
512 px and ADE20K at 640 px. v2 differs by: Mapillary Snow -> terrain; ADE20K floor, pier -> sidewalk, column -> pole.
Groups: all images and the low-white / high-white halves of split_by_white.json. Result: rescore_mappings.json + table.
RESCORE_MAX=<n> limits the images per group (quick test).
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from src import config, segmentor as S
from src.data.dataset import create_val_dataset

HERE = Path(__file__).resolve().parent
STEP18 = HERE.parent / "step18_second_judge"
MODELS = ["pretrained", "B", "seg0.006", "seg0.02", "seg0.05-0.4", "real photos"]
JUDGES = {"mapillary": (1.0, S.MAPILLARY_TO_JUDGE, S.MAPILLARY_TO_JUDGE_V2),
          "ade20k": (1.25, S.ADE20K_TO_JUDGE, S.ADE20K_TO_JUDGE_V2)}


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = create_val_dataset(config.RESOLUTION)
    items = [ds[i] for i in range(len(ds))]
    names = [Path(it["image_name"]).with_suffix(".png").name for it in items]
    split = json.load(open(STEP18 / "split_by_white.json"))["meta"]["groups"]
    pos = {n: i for i, n in enumerate(names)}
    groups = {"all": list(range(len(names))), "low_white": [pos[n] for n in split["low_white"]],
              "high_white": [pos[n] for n in split["high_white"]]}
    limit = int(os.environ.get("RESCORE_MAX", "0"))
    if limit:
        groups = {g: v[:limit] for g, v in groups.items()}
    used = sorted(set(sum(groups.values(), [])))

    def image(model, k):
        if model == "real photos":
            return (items[k]["image"] * 0.5 + 0.5).clamp(0, 1)
        return torch.from_numpy(np.array(Image.open(STEP18 / "images" / model / names[k]).convert("RGB"))).permute(2, 0, 1).float() / 255

    res = {}
    for jname, (scale, v1, v2) in JUDGES.items():
        os.environ["DDPM_SEG_INPUT_SCALE"] = str(scale)
        S.SEG_JUDGE = jname
        judge = S.load_segmentor(device)
        for ver, table in (("v1", v1), ("v2", v2)):
            judge.to_judge.copy_(torch.nn.functional.one_hot(torch.from_numpy(table), S.SEG_NUM_CLASSES).float().to(device))
            for m in MODELS:
                conf = {k: torch.zeros(S.SEG_NUM_CLASSES, S.SEG_NUM_CLASSES, dtype=torch.long) for k in used}
                for k in used:
                    pred = S.segmentor_logits(judge, image(m, k)[None].to(device)).argmax(1).cpu()
                    conf[k] = S.confusion_matrix(pred, items[k]["mask_seg"][None])
                for g, idx in groups.items():
                    miou, acc, per = S.miou_from_confusion(sum(conf[k] for k in idx))
                    res.setdefault(jname, {}).setdefault(ver, {}).setdefault(g, {})[m] = {"miou": miou, "pixel_acc": acc, "per_class": per}
                print(f"{jname} {ver} {m}: mIoU {res[jname][ver]['all'][m]['miou']:.3f}", flush=True)
        del judge
        if device == "cuda":
            torch.cuda.empty_cache()
    json.dump(res, open(HERE / ("rescore_test.json" if limit else "rescore_mappings.json"), "w"), indent=1)
    for jname in res:
        for g in groups:
            print(f"\n{jname}, {g}\n  {'model':14s}   v1 mIoU  v2 mIoU   v1 acc  v2 acc")
            for m in MODELS:
                a, b = res[jname]["v1"][g][m], res[jname]["v2"][g][m]
                print(f"  {m:14s}   {a['miou']:.3f}    {b['miou']:.3f}     {a['pixel_acc']:.3f}   {b['pixel_acc']:.3f}")


if __name__ == "__main__":
    main()
