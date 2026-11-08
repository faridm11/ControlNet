"""
Dataset 2 (SANPO-Real) class definitions.

Labels in data_sanpo/ are stored in SANPO's own 31 class IDs (no information lost).
Each class is conditioned with an ADE20K colour, because ADE20K's 150 classes are
what the seg ControlNet was trained on:

  exact   — ADE20K has the same class        (e.g. hand rail -> "bannister;handrail")
  closest — no ADE20K class; nearest one     (e.g. curb -> "sidewalk")
  void    — no sensible ADE20K class; white, the model fills it from context

Choices for unclear classes were made after looking at SANPO examples
(controlnet_testing/EXPERIMENT_LOG.md): "obstacle" is mostly snow piles plus
bins/barriers (no ADE20K class), "wall/fence" is mostly metal fences, "terrain" is
mostly lawns, "vegetation" mostly bushes and hedges, "bike rack" metal racks/docks.

Two extra mappings for code that works in the project's 35-class space:
  SANPO_TO_V1  — prompt generation and cross-dataset comparison
  segmentor targets are derived from SANPO_TO_V1 via src/segmentor.py
"""

import csv
from pathlib import Path

NUM_CLASSES = 31

CLASS_NAMES = {
    0: "unlabeled", 1: "road", 2: "curb", 3: "sidewalk", 4: "guard rail/road barrier", 5: "crosswalk",
    6: "paved trail", 7: "building", 8: "wall/fence", 9: "hand rail", 10: "opening-door", 11: "opening-gate",
    12: "pedestrian", 13: "rider", 14: "animal", 15: "stairs", 16: "water body", 17: "other walkable surface",
    18: "inaccessible surface", 19: "railway track", 20: "obstacle", 21: "vehicle", 22: "traffic sign",
    23: "traffic light", 24: "pole", 25: "bus stop", 26: "bike rack", 27: "sky", 28: "tree", 29: "vegetation",
    30: "terrain",
}
SANPO_CLASSES = CLASS_NAMES  # backward-compatible name

# SANPO id -> (ADE20K index in data_prep/ade20k_150_palette.csv, or None for void, kind)
SANPO_TO_ADE = {
    0: (None, "void"),       # unlabeled
    1: (7, "exact"),         # road            -> road;route
    2: (12, "closest"),      # curb            -> sidewalk;pavement   (ADE20K has no curb)
    3: (12, "exact"),        # sidewalk        -> sidewalk;pavement
    4: (39, "exact"),        # guard rail      -> railing;rail
    5: (7, "closest"),       # crosswalk       -> road;route          (ADE20K has no crosswalk)
    6: (53, "exact"),        # paved trail     -> path
    7: (2, "exact"),         # building        -> building;edifice
    8: (33, "exact"),        # wall/fence      -> fence;fencing       (mostly metal fences in SANPO)
    9: (96, "exact"),        # hand rail       -> bannister;banister;balustrade;balusters;handrail
    10: (15, "exact"),       # opening-door    -> door
    11: (33, "closest"),     # opening-gate    -> fence;fencing       (ADE20K has no gate)
    12: (13, "exact"),       # pedestrian      -> person
    13: (13, "closest"),     # rider           -> person              (ADE20K has no rider)
    14: (127, "exact"),      # animal          -> animal
    15: (54, "exact"),       # stairs          -> stairs;steps
    16: (22, "exact"),       # water body      -> water
    17: (12, "closest"),     # other walkable  -> sidewalk;pavement   (paved plazas)
    18: (None, "void"),      # inaccessible surface (mixed content)
    19: (7, "closest"),      # railway track   -> road;route          (ADE20K has no railway; 0.01 % of pixels)
    20: (None, "void"),      # obstacle        (mostly snow piles; ADE20K has no snow or generic obstacle)
    21: (21, "exact"),       # vehicle         -> car
    22: (44, "exact"),       # traffic sign    -> signboard;sign
    23: (137, "exact"),      # traffic light   -> traffic light
    24: (94, "exact"),       # pole            -> pole
    25: (None, "void"),      # bus stop        (no pixels in the selected frames)
    26: (39, "closest"),     # bike rack       -> railing;rail        (metal bars)
    27: (3, "exact"),        # sky             -> sky
    28: (5, "exact"),        # tree            -> tree
    29: (18, "exact"),       # vegetation      -> plant;flora (bushes, hedges)
    30: (10, "exact"),       # terrain         -> grass               (mostly lawns)
}

_ADE = {int(r["ade_idx"]): (r["name"], (int(r["r"]), int(r["g"]), int(r["b"])))
        for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "ade20k_150_palette.csv"))}

COLOR_PALETTE = {s: ((255, 255, 255) if a is None else _ADE[a][1]) for s, (a, _) in SANPO_TO_ADE.items()}
ADE_NAMES = {s: ("void" if a is None else _ADE[a][0].split(";")[0]) for s, (a, _) in SANPO_TO_ADE.items()}

# SANPO -> the project's 35-class ids (class_mapping.py); used for prompts and the segmentor target
SANPO_TO_V1 = {
    0: 0, 1: 1, 2: 7, 3: 2, 4: 21, 5: 3, 6: 2, 7: 18, 8: 21, 9: 21, 10: 19, 11: 21, 12: 8, 13: 9, 14: 32,
    15: 24, 16: 33, 17: 2, 18: 31, 19: 6, 20: 33, 21: 10, 22: 26, 23: 27, 24: 25, 25: 33, 26: 33, 27: 30,
    28: 28, 29: 28, 30: 29,
}

assert sorted(SANPO_TO_ADE) == sorted(SANPO_TO_V1) == list(range(NUM_CLASSES))
