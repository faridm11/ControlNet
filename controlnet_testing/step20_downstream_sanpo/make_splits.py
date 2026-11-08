"""
Step 20a — fixed image lists for the SANPO low-data test (no GPU).

limited_real: N_REAL training frames, evenly spaced over the name-sorted list, so all sessions are represented.
other_masks:  N_SYNTH masks of OTHER training frames, evenly spaced; synthetic images are generated from these.
Writes splits.json. Run from DDPM/ with DDPM_DATA_ROOT=.../data_sanpo.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.data import create_train_dataset

N_REAL, N_SYNTH = int(os.environ.get("N_REAL", 300)), int(os.environ.get("N_SYNTH", 1000))
ds = create_train_dataset(resolution=512)
names = sorted(d["image_name"] for d in ds.data)
real_idx = set(np.linspace(0, len(names) - 1, N_REAL).round().astype(int).tolist())
rest = [n for i, n in enumerate(names) if i not in real_idx]
other = [rest[i] for i in np.linspace(0, len(rest) - 1, N_SYNTH).round().astype(int)]
real = [names[i] for i in sorted(real_idx)]
assert not set(real) & set(other)
sess = lambda l: len({n.split("sanpo_")[1].rsplit("_", 1)[0] for n in l})
print(f"{len(names)} training frames | limited real {len(real)} from {sess(real)} sessions | other masks {len(other)} from {sess(other)} sessions")
json.dump({"limited_real": real, "other_masks": other}, open(Path(__file__).parent / "splits.json", "w"), indent=1)
