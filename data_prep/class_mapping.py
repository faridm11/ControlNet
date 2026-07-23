"""
Class mapping for sidewalk segmentation dataset (segments/sidewalk-semantic, 35 classes, IDs 0-34).
Source: https://huggingface.co/datasets/segments/sidewalk-semantic

Palette strategy:
- 21 classes reuse ADE20K colors so the pretrained ControlNet seg weights apply directly.
- 13 classes use new unique colors (chosen far from ADE20K to avoid confusion).
- 5 void/unknown classes (0, 31-34) all use white (255,255,255) — one neutral color
  so the model fills those regions from context rather than learning spurious content.
"""

# Number of classes (0-34 = 35 total)
NUM_CLASSES = 35

# Class names for all 35 classes
CLASS_NAMES = {
    0: "unlabeled",
    1: "road",
    2: "sidewalk",
    3: "crosswalk",
    4: "cycling_lane",
    5: "parking_driveway",
    6: "railtrack",
    7: "curb",
    8: "person",
    9: "rider",
    10: "car",
    11: "truck",
    12: "bus",
    13: "tramtrain",    
    14: "motorbike",
    15: "bicycle",
    16: "caravan",
    17: "trailer",
    18: "building",
    19: "door",
    20: "wall",
    21: "fence",
    22: "bridge",
    23: "tunnel",
    24: "stairs",
    25: "pole",
    26: "sign",
    27: "traffic_light",
    28: "vegetation",
    29: "earth",
    30: "sky",
    31: "void_ground",
    32: "dynamic_void",
    33: "static_void",
    34: "unclear_void",
}

# ============================================================================
# COLOR PALETTE (35 classes total)
# 21 classes use ADE20K colors (pretrained knowledge)
# 13 classes use NEW unique colors (far from ADE20K to avoid confusion)
#  5 void classes (0, 31-34) use white (255,255,255) — neutral/context-fill
# ============================================================================

COLOR_PALETTE = {
    # Void classes
    0: (255, 255, 255),   # unlabeled
    
    # Flat surfaces - 1-2 are ADE20K, 3-7 are NEW
    1: (140, 140, 140),   # road (ADE20K)
    2: (235, 255, 7),     # sidewalk (ADE20K)
    3: (210, 210, 60),    # crosswalk - NEW COLOR
    4: (170, 210, 140),   # cycling_lane - NEW COLOR
    5: (150, 170, 210),   # parking_driveway - NEW COLOR
    6: (110, 110, 200),   # railtrack - NEW COLOR
    7: (200, 150, 255),   # curb - NEW COLOR
    
    # Humans - 8 is ADE20K, 9 is NEW
    8: (150, 5, 61),      # person (ADE20K)
    9: (255, 120, 120),   # rider - NEW COLOR
    
    # Vehicles - 10-12, 14-15 are ADE20K, 13, 16-17 are NEW
    10: (0, 102, 200),    # car (ADE20K)
    11: (255, 0, 20),     # truck (ADE20K)
    12: (255, 0, 245),    # bus (ADE20K)
    13: (0, 140, 170),    # tramtrain - NEW COLOR
    14: (163, 0, 255),    # motorbike (ADE20K)
    15: (255, 245, 0),    # bicycle (ADE20K)
    16: (170, 30, 200),    # caravan - NEW COLOR
    17: (255, 80, 80),    # trailer - NEW COLOR
    
    # Construction - 18-22, 24 are ADE20K, 23 is NEW
    18: (180, 120, 120),  # building (ADE20K)
    19: (8, 255, 51),     # door (ADE20K)
    20: (120, 120, 120),  # wall (ADE20K)
    21: (255, 184, 6),    # fence (ADE20K)    
    22: (255, 82, 0),     # bridge (ADE20K)
    23: (80, 100, 130),   # tunnel - NEW COLOR (blue-gray, distinct from wall/vegetation)
    24: (255, 224, 0),    # stairs (ADE20K)
    
    # Objects - all ADE20K
    25: (51, 0, 255),     # pole (ADE20K)
    26: (255, 5, 153),    # sign (ADE20K)
    27: (41, 0, 255),     # traffic_light (ADE20K)
    
    # Nature - all ADE20K
    28: (4, 200, 3),      # vegetation — ADE20K tree color (stronger prior than plant for outdoor scenes)
    29: (120, 120, 70),   # earth (ADE20K)
    30: (6, 230, 230),    # sky (ADE20K)

    # Void classes - all white so the model fills from context, not spurious content
    31: (255, 255, 255),  # void_ground  (dataset: void-ground)
    32: (255, 255, 255),  # dynamic_void (dataset: void-dynamic)
    33: (255, 255, 255),  # static_void  (dataset: void-static)
    34: (255, 255, 255),  # unclear_void (dataset: void-unclear)
}

# IDs to ignore in segmentation loss (Phase 2) and mIoU evaluation.
# These classes have no reliable ground-truth content — penalizing them
# would add noise to the semantic consistency signal.
VOID_CLASS_IDS = {0, 31, 32, 33, 34}
