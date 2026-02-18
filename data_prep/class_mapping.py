"""
Class mapping for sidewalk segmentation dataset.
Maps 35 original classes to simplified 8-class schema and defines class properties.
"""

import numpy as np

# Simplified 8-class schema for training
SIMPLIFIED_CLASSES = {
    0: "background",      # void, unlabeled, sky, buildings
    1: "road",           # road, rail track
    2: "walkable",       # sidewalk, crosswalk, bike lane, parking, curb
    3: "pedestrian",     # person, rider
    4: "vehicle",        # all vehicles
    5: "traffic_control", # signs, lights
    6: "obstacle",       # walls, fences, poles, stairs, doors, bridges, tunnel
    7: "environment",    # vegetation, terrain
}

# Mapping from 35 original classes to 8 simplified classes
CLASS_REMAPPING = {
    # Background/void (all mapped to 0)
    0: 0,   # unlabeled
    18: 0,  # building
    30: 0,  # sky
    31: 0,  # void-ground
    32: 0,  # void-dynamic
    33: 0,  # void-static
    34: 0,  # void-unclear
    
    # Road surfaces (mapped to 1)
    1: 1,   # road
    6: 1,   # rail track
    
    # Walkable surfaces (mapped to 2)
    2: 2,   # sidewalk
    3: 2,   # crosswalk
    4: 2,   # cycling lane
    5: 2,   # parking
    7: 2,   # curb
    
    # Pedestrians (mapped to 3)
    8: 3,   # person
    9: 3,   # rider
    
    # Vehicles (mapped to 4)
    10: 4,  # car
    11: 4,  # truck
    12: 4,  # bus
    13: 4,  # tram/train
    14: 4,  # motorcycle
    15: 4,  # bicycle
    16: 4,  # caravan
    17: 4,  # trailer
    
    # Traffic control (mapped to 5)
    26: 5,  # traffic sign
    27: 5,  # traffic light
    
    # Obstacles (mapped to 6)
    19: 6,  # door
    20: 6,  # wall
    21: 6,  # fence/guardrail
    22: 6,  # bridge
    23: 6,  # tunnel
    24: 6,  # stairs
    25: 6,  # pole
    
    # Environment (mapped to 7)
    28: 7,  # vegetation
    29: 7,  # terrain
}

def remap_mask(mask_array):
    """
    Remap 35-class mask to 8-class simplified schema.
    
    Args:
        mask_array: numpy array of shape (H, W) with original class IDs (0-34)
        
    Returns:
        numpy array of shape (H, W) with remapped class IDs (0-7)
    """
    remapped = np.zeros_like(mask_array)
    for old_id, new_id in CLASS_REMAPPING.items():
        remapped[mask_array == old_id] = new_id
    return remapped

# Original 35-class mapping from dataset (for reference)
ORIGINAL_CLASSES = {
    0: "unlabeled",
    1: "flat-road",
    2: "flat-sidewalk",
    3: "flat-crosswalk",
    4: "flat-cyclinglane",
    5: "flat-parkingdriveway",
    6: "flat-railtrack",
    7: "flat-curb",
    8: "human-person",
    9: "human-rider",
    10: "vehicle-car",
    11: "vehicle-truck",
    12: "vehicle-bus",
    13: "vehicle-tramtrain",
    14: "vehicle-motorcycle",
    15: "vehicle-bicycle",
    16: "vehicle-caravan",
    17: "vehicle-cartrailer",
    18: "construction-building",
    19: "construction-door",
    20: "construction-wall",
    21: "construction-fenceguardrail",
    22: "construction-bridge",
    23: "construction-tunnel",
    24: "construction-stairs",
    25: "object-pole",
    26: "object-trafficsign",
    27: "object-trafficlight",
    28: "nature-vegetation",
    29: "nature-terrain",
    30: "sky",
    31: "void-ground",
    32: "void-dynamic",
    33: "void-static",
    34: "void-unclear",
}

# Classes to exclude from prompts (not relevant for navigation)
EXCLUDED_CLASSES = {
    0,   # unlabeled
    30,  # sky
    31,  # void-ground
    32,  # void-dynamic
    33,  # void-static
    34,  # void-unclear
    18,  # construction-building (background, not obstacle)
}

# Simplified class groupings for prompt generation
CLASS_GROUPS = {
    "walkable_surfaces": {
        "ids": [2, 3, 4, 5, 7],  # sidewalk, crosswalk, cycling lane, parking, curb
        "names": ["sidewalk", "crosswalk", "cycling lane", "parking area", "curb"],
        "priority": "high"
    },
    "road_surfaces": {
        "ids": [1, 6],  # road, rail track
        "names": ["road", "rail track"],
        "priority": "medium"
    },
    "pedestrians": {
        "ids": [8, 9],  # person, rider
        "names": ["person", "rider"],
        "priority": "high"
    },
    "vehicles": {
        "ids": [10, 11, 12, 13, 14, 15, 16, 17],  # car, truck, bus, tram, motorcycle, bicycle, caravan, trailer
        "names": ["car", "truck", "bus", "tram", "motorcycle", "bicycle", "caravan", "trailer"],
        "priority": "high"
    },
    "traffic_control": {
        "ids": [26, 27],  # traffic sign, traffic light
        "names": ["traffic sign", "traffic light"],
        "priority": "high"
    },
    "obstacles": {
        "ids": [25, 19, 20, 21, 22, 24],  # pole, door, wall, fence, bridge, stairs
        "names": ["pole", "door", "wall", "fence", "bridge", "stairs"],
        "priority": "high"
    },
    "environment": {
        "ids": [28, 29],  # vegetation, terrain
        "names": ["vegetation", "terrain"],
        "priority": "low"
    }
}

# Human-readable names for prompt generation
PROMPT_FRIENDLY_NAMES = {
    1: "road",
    2: "sidewalk",
    3: "crosswalk",
    4: "bike lane",
    5: "parking area",
    6: "rail track",
    7: "curb",
    8: "pedestrian",
    9: "cyclist",
    10: "car",
    11: "truck",
    12: "bus",
    13: "tram",
    14: "motorcycle",
    15: "bicycle",
    16: "caravan",
    17: "trailer",
    19: "door",
    20: "wall",
    21: "fence",
    22: "bridge",
    24: "stairs",
    25: "pole",
    26: "traffic sign",
    27: "traffic light",
    28: "trees and vegetation",
    29: "terrain",
}

def get_present_classes(mask_array):
    """
    Extract unique class IDs from a mask, excluding void/sky classes.
    
    Args:
        mask_array: numpy array of shape (H, W) with class IDs
        
    Returns:
        list: sorted list of present class IDs (excluding EXCLUDED_CLASSES)
    """
    import numpy as np
    unique_classes = np.unique(mask_array)
    return sorted([int(c) for c in unique_classes if c not in EXCLUDED_CLASSES])

def get_class_names(class_ids):
    """
    Convert class IDs to prompt-friendly names.
    
    Args:
        class_ids: list of class IDs
        
    Returns:
        list: human-readable class names
    """
    return [PROMPT_FRIENDLY_NAMES.get(cid, ORIGINAL_CLASSES.get(cid, f"class_{cid}")) 
            for cid in class_ids]

def group_classes_by_category(class_ids):
    """
    Group class IDs into semantic categories for structured prompts.
    
    Args:
        class_ids: list of class IDs
        
    Returns:
        dict: {category_name: [class_names]}
    """
    grouped = {}
    
    for category, info in CLASS_GROUPS.items():
        present = [PROMPT_FRIENDLY_NAMES.get(cid, ORIGINAL_CLASSES[cid]) 
                   for cid in class_ids if cid in info["ids"]]
        if present:
            grouped[category] = present
    
    return grouped
