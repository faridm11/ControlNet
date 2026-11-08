"""
White (void) areas gallery for SANPO: every already-generated scene whose mask has white areas (SANPO
'obstacle' / 'unlabeled'), with those areas outlined in red on every image, for all models.

  A: scenes of the latest sample sheet (FINAL_v2_segloss/sanpo_l0.02/comparison/row_*.png, same noise per row)
  B: the 4 training-sample scenes of each run (4 generations each), from run/samples/

    python controlnet_testing/reporting/make_white_gallery.py   ->  FINAL_v2_segloss/white_regions_gallery.jpg
"""

import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

T = Path(__file__).resolve().parents[1]
SHEET_ROWS = T / "FINAL_v2_segloss/sanpo_l0.02/comparison"
SHEET_HEADS = ["real photo", "mask", "pretrained", "B (no seg loss)", "seg λ 0.006", "seg λ 0.02", "seg λ 0.05→0.4"]
RUNS = [("v1 final", "FINAL/sanpo"), ("B (no seg loss)", "FINAL_v2/sanpo"), ("seg λ 0.006", "FINAL_v2_segloss/sanpo"),
        ("seg λ 0.02", "FINAL_v2_segloss/sanpo_l0.02"), ("seg λ 0.05→0.4", "FINAL_v2_segloss/sanpo_ramp")]
S, LBL, NCOL = 384, 150, 7
try:
    FONT, BIG = ImageFont.truetype("DejaVuSans.ttf", 18), ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    FONT = BIG = ImageFont.load_default()


def outline(img, white):
    """Red outline (3 px) around the mask's white areas."""
    m = Image.fromarray((white * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
    edge = (np.array(m.filter(ImageFilter.MaxFilter(7))) > 0) & ~(np.array(m) > 0)
    a = np.array(img.resize((S, S)).convert("RGB"))
    a[edge] = (255, 0, 0)
    return Image.fromarray(a)


def main():
    blocks = []  # (title, column heads, [(row label, [images])])
    for f in sorted(SHEET_ROWS.glob("row_*.png")):
        im = Image.open(f).convert("RGB")
        w = im.width // len(SHEET_HEADS)
        panels = [im.crop((k * w, 0, (k + 1) * w, im.height)) for k in range(len(SHEET_HEADS))]
        white = np.array(panels[1]).min(axis=2) >= 250
        if white.mean() >= 0.01:
            blocks.append((f"{f.stem[4:]}  —  white (void) = {100*white.mean():.0f}% of the mask", SHEET_HEADS,
                           [("", [outline(p, white) for p in panels])]))
    for i in range(4):
        rows, white = [], None
        for label, d in RUNS:
            s = glob.glob(str(T / d / "run/samples/*"))[0] + "/"
            mask = Image.open(f"{s}sample_{i}_control.png").convert("RGB")
            white = np.array(mask).min(axis=2) >= 250
            if white.mean() < 0.01:
                break
            imgs = [Image.open(f"{s}sample_{i}_input.png"), mask] + [Image.open(f"{s}sample_{i}_output_{k}.png") for k in range(4)]
            rows.append((label, [outline(x, white) for x in imgs]))
        if rows:
            blocks.append((f"training-sample scene {i}  —  white (void) = {100*white.mean():.0f}% of the mask",
                           ["real photo", "mask", "generation 1", "generation 2", "generation 3", "generation 4"], rows))

    H = sum(30 + 28 + len(r) * S + 20 for _, _, r in blocks) + 70
    sheet = Image.new("RGB", (LBL + NCOL * S, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), "SANPO: what the models draw in the white (void) mask areas — outlined in red. "
           "White = SANPO 'obstacle' (snow piles, benches, bins, barriers) + 'unlabeled'.", fill="black", font=BIG)
    y = 60
    for title, heads, rows in blocks:
        d.text((10, y), title, fill="black", font=BIG)
        y += 30
        for k, h in enumerate(heads):
            d.text((LBL + k * S + 6, y), h, fill="black", font=FONT)
        y += 28
        for label, imgs in rows:
            d.text((8, y + S // 2 - 10), label, fill="black", font=FONT)
            for k, im in enumerate(imgs):
                sheet.paste(im, (LBL + k * S, y))
            y += S
        y += 20
    out = T / "FINAL_v2_segloss/white_regions_gallery.jpg"
    sheet.save(out, quality=90)
    print(out, sheet.size, len(blocks), "scenes")


if __name__ == "__main__":
    main()
