"""
Step 20c — does generated data help a segmentation model when real photos are limited?

Fine-tunes SegFormer-B5 (ADE20K weights, new 12-class head) and tests it on ALL 522 REAL SANPO test photos.
    A  --name real_only                      limited real images
    B  --name real_plus_base --synthetic ..  limited real + base ControlNet images
    C  --name real_plus_ours --synthetic ..  limited real + seg-loss ControlNet images
Every setting: same steps, batch, LR, flip; 50 % of samples drawn from synthetic when present. White/void pixels are ignored.
Run from DDPM/ with DDPM_DATA_ROOT=.../data_sanpo, DDPM_SEG_JUDGE=ade20k, DDPM_VAL_SUBSET_SIZE unset.
"""
import argparse, json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation
from src import segmentor as S
from src.data import create_train_dataset, create_val_dataset

MEAN = torch.tensor(S._MEAN).view(3, 1, 1)
STD = torch.tensor(S._STD).view(3, 1, 1)
N_CLASSES = 12  # shared classes without "other", which never occurs in a target
INIT = {"ade20k": S.ADE20K_MODEL_ID,   # SegFormer-B5 already trained for segmentation on ADE20K (knows street scenes)
        "imagenet": "nvidia/mit-b5"}    # same encoder, ImageNet classification only: no segmentation, no street scenes


def norm(img01):
    return (img01 - MEAN) / STD


class Real(Dataset):
    def __init__(self, ds, indices, flip):
        self.ds, self.idx, self.flip = ds, indices, flip

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        it = self.ds[self.idx[i]]
        img, mask = norm((it["image"] * 0.5 + 0.5).clamp(0, 1)), it["mask_seg"].long()
        if self.flip and random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        return img, mask


class Synthetic(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted((self.root / "images").glob("*.jpg"))
        assert self.files, f"no images in {root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = norm(torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1).float() / 255)
        mask = torch.from_numpy(np.array(Image.open(self.root / "masks" / f"{f.stem}.png"))).long()
        if random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        return img, mask


def forward(model, img):
    return F.interpolate(model(pixel_values=img).logits, size=img.shape[-2:], mode="bilinear", align_corners=False)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(S.SEG_NUM_CLASSES, S.SEG_NUM_CLASSES, dtype=torch.long)
    for img, mask in loader:
        conf += S.confusion_matrix(forward(model, img.to(device)).argmax(1).cpu(), mask)
    model.train()
    return S.miou_from_confusion(conf)


def run(args, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = create_train_dataset(resolution=512)
    pos = {d["image_name"]: i for i, d in enumerate(base.data)}
    names = json.load(open(HERE / "splits.json"))[args.real_split]
    real = Real(base, [pos[n] for n in names], flip=True)
    parts = [real]
    sampler = None
    if args.synthetic:
        syn = Synthetic(args.synthetic)
        parts.append(syn)
        w = [0.5 / len(real)] * len(real) + [0.5 / len(syn)] * len(syn)
        sampler = WeightedRandomSampler(w, num_samples=sum(map(len, parts)), replacement=True)
    train = ConcatDataset(parts)
    loader = DataLoader(train, batch_size=args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0)
    val = create_val_dataset(resolution=512, full=True)
    test_set = Real(val, list(range(len(val)))[:args.max_test or None], flip=False)
    test = DataLoader(test_set, batch_size=4, num_workers=args.workers)
    model = SegformerForSemanticSegmentation.from_pretrained(INIT[args.init], num_labels=N_CLASSES, ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, history = 0, []
    model.train()
    while step < args.steps:
        for img, mask in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(forward(model, img.to(device)), mask.to(device), ignore_index=S.SEG_IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                miou, acc, _ = evaluate(model, test, device)
                history.append({"step": step, "miou": miou, "pixel_acc": acc})
                print(f"  seed {seed} step {step}: loss {loss.item():.3f} | test mIoU {miou:.4f} | pixel acc {acc:.4f}", flush=True)
            if step >= args.steps:
                break
    miou, acc, per = evaluate(model, test, device)
    return {"seed": seed, "train_images": len(train), "miou": miou, "pixel_acc": acc, "iou_per_class": per, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--real-split", default="limited_real", help="list of real photos in splits.json")
    ap.add_argument("--init", default="ade20k", choices=list(INIT), help="starting weights of the segmentation model")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=0, help="quick test only")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== {args.name}: init={args.init} synthetic={args.synthetic} steps={args.steps} batch={args.batch} device={device}", flush=True)
    runs = [run(args, s, device) for s in args.seeds]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "init": args.init, "real_split": args.real_split, "synthetic": args.synthetic, "steps": args.steps, "batch": args.batch, "lr": args.lr,
               "miou_mean": float(np.mean([r["miou"] for r in runs])), "miou_std": float(np.std([r["miou"] for r in runs])),
               "pixel_acc_mean": float(np.mean([r["pixel_acc"] for r in runs])), "runs": runs}
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(f"== {args.name}: test mIoU {summary['miou_mean']:.4f} ± {summary['miou_std']:.4f} over {len(runs)} seeds", flush=True)


if __name__ == "__main__":
    main()
