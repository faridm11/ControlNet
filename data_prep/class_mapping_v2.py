"""
V2 class mapping: 21 classes, all ADE20K colors.

Strategy: the 10 new-color classes from V1 are merged into their nearest ADE20K
semantic neighbor. Result: zero new color learning — the pretrained ControlNet
already knows every color here. LoRA capacity goes entirely to domain shift
(street-level perspective) instead of learning new RGB→feature mappings.

Masks on disk stay as V1 IDs (0–34). REMAP_V1_TO_V2 is applied at load time
in dataset.py when CLASS_VERSION=v2.

V1 → V2 merges:
  crosswalk(3), cycling_lane(4), parking_driveway(5),
  railtrack(6), curb(7)                                → road(1)
  rider(9)                                             → person(3)
  tramtrain(13)                                        → bus(6)
  caravan(16), trailer(17)                             → truck(5)
  tunnel(23)                                           → wall(11)
  unlabeled(0), void_ground(31), dynamic_void(32),
  static_void(33), unclear_void(34)                    → void(0)
"""

NUM_CLASSES = 21

CLASS_NAMES = {
    0:  "void",
    1:  "road",
    2:  "sidewalk",
    3:  "person",
    4:  "car",
    5:  "truck",
    6:  "bus",
    7:  "motorbike",
    8:  "bicycle",
    9:  "building",
    10: "door",
    11: "wall",
    12: "fence",
    13: "bridge",
    14: "stairs",
    15: "pole",
    16: "sign",
    17: "traffic_light",
    18: "vegetation",
    19: "earth",
    20: "sky",
}

# All ADE20K colors — zero new colors vs pretrained ControlNet
COLOR_PALETTE = {
    0:  (255, 255, 255),   # void — white, neutral
    1:  (140, 140, 140),   # road (ADE20K)
    2:  (235, 255, 7),     # sidewalk (ADE20K)
    3:  (150, 5, 61),      # person (ADE20K)
    4:  (0, 102, 200),     # car (ADE20K)
    5:  (255, 0, 20),      # truck (ADE20K)
    6:  (255, 0, 245),     # bus (ADE20K)
    7:  (163, 0, 255),     # motorbike (ADE20K)
    8:  (255, 245, 0),     # bicycle (ADE20K)
    9:  (180, 120, 120),   # building (ADE20K)
    10: (8, 255, 51),      # door (ADE20K)
    11: (120, 120, 120),   # wall (ADE20K)
    12: (255, 184, 6),     # fence (ADE20K)
    13: (255, 82, 0),      # bridge (ADE20K)
    14: (255, 224, 0),     # stairs (ADE20K)
    15: (51, 0, 255),      # pole (ADE20K)
    16: (255, 5, 153),     # sign (ADE20K)
    17: (41, 0, 255),      # traffic_light (ADE20K)
    18: (4, 200, 3),       # vegetation — ADE20K tree color
    19: (120, 120, 70),    # earth (ADE20K)
    20: (6, 230, 230),     # sky (ADE20K)
}

# Maps original V1 IDs (0–34) → V2 IDs (0–20).
# Applied at load time in dataset.py; masks on disk are never modified.
REMAP_V1_TO_V2 = {
    0:  0,   # unlabeled       → void
    1:  1,   # road            → road
    2:  2,   # sidewalk        → sidewalk
    3:  1,   # crosswalk       → road
    4:  1,   # cycling_lane    → road
    5:  1,   # parking_driveway→ road
    6:  1,   # railtrack       → road
    7:  2,   # curb            → sidewalk
    8:  3,   # person          → person
    9:  3,   # rider           → person
    10: 4,   # car             → car
    11: 5,   # truck           → truck
    12: 6,   # bus             → bus
    13: 6,   # tramtrain       → bus
    14: 7,   # motorbike       → motorbike
    15: 8,   # bicycle         → bicycle
    16: 5,   # caravan         → truck
    17: 5,   # trailer         → truck
    18: 9,   # building        → building
    19: 10,  # door            → door
    20: 11,  # wall            → wall
    21: 12,  # fence           → fence
    22: 13,  # bridge          → bridge
    23: 11,  # tunnel          → wall
    24: 14,  # stairs          → stairs
    25: 15,  # pole            → pole
    26: 16,  # sign            → sign
    27: 17,  # traffic_light   → traffic_light
    28: 18,  # vegetation      → vegetation
    29: 19,  # earth           → earth
    30: 20,  # sky             → sky
    31: 0,   # void_ground     → void
    32: 0,   # dynamic_void    → void
    33: 0,   # static_void     → void
    34: 0,   # unclear_void    → void
}

# Only void(0) has no reliable ground-truth content in V2
VOID_CLASS_IDS = {0}
