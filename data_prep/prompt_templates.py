"""
Text prompt templates for ControlNet fine-tuning.
Combines scene conditions with class-based object descriptions.
Full schema: scene + lighting + weather + sensor + objects + realism.

Class IDs follow class_mapping.py (35-class system, IDs 0-34).
"""

import random
import numpy as np

# Scene/Environment templates
SCENE_TEMPLATES = [
    "urban street scene",
    "suburban sidewalk",
    "campus walkway",
    "industrial district street",
    "city pedestrian zone",
    "residential neighborhood street",
    "commercial district sidewalk",
    "busy intersection",
    "quiet backstreet",
    "downtown corridor",
    "marketplace street",
    "transit hub area",
    "inner-city road",
    "narrow side street",
]

# Lighting conditions
LIGHTING_CONDITIONS = [
    "dawn lighting",
    "early morning golden light",
    "soft morning light",
    "noon bright sunlight",
    "harsh midday sun",
    "soft afternoon light",
    "dusk lighting",
    "evening twilight",
    "night with street lights",
    "night illumination",
    "HDR lighting",
    "flat overcast light",
    "diffuse overcast sky",
    "sunny conditions",
    "backlit scene",
    "mixed artificial and natural light",
    "low winter sun",
]

# Weather conditions
WEATHER_CONDITIONS = [
    "clear weather",
    "rain",
    "light rain",
    "heavy rain",
    "wet road surface",
    "after rain with wet surfaces",
    "drizzle",
    "fog",
    "light fog",
    "dense fog",
    "overcast sky",
    "puddles on ground",
    "humid haze",
]

# Camera/Sensor realism
SENSOR_REALISM = [
    "phone photo",
    "smartphone camera",
    "JPEG compression",
    "JPEG compression artifacts",
    "slight chroma noise",
    "ISO 1600",
    "ISO 800 with noise",
    "high ISO noise",
    "wide-angle lens",
    "action camera footage",
    "dashcam recording",
    "slight lens distortion",
    "slightly underexposed",
    "slightly overexposed",
    "consumer camera JPEG",
]

# Motion/Optics effects
MOTION_OPTICS = [
    "slight motion blur",
    "motion blur",
    "rolling shutter skew",
    "rolling shutter effect",
    "shallow depth of field",
    "shallow DOF",
    "slight blur",
    "camera shake",
]

# Realism anchors — photographic/documentary style, not generic filler
REALISM_ANCHORS = [
    "photorealistic",
    "documentary-style photograph",
    "authentic street photography",
    "candid urban photo",
    "pedestrian-level perspective",
    "handheld camera feel",
    "real-world capture",
    "unprocessed street photo",
    "ground-level urban view",
    "eye-level shot",
]

# ---------------------------------------------------------------------------
# Class → prompt-friendly name mapping (35-class system, matches class_mapping.py)
# None = skip this class in object descriptions (too minor or void)
# ---------------------------------------------------------------------------
CLASS_PROMPT_NAMES = {
    0: None,            # unlabeled — void, skip
    1: "road",
    2: "sidewalk",
    3: "crosswalk",
    4: "cycling lane",
    5: None,            # parking_driveway — too generic, skip
    6: "rail track",
    7: None,            # curb — too minor, skip
    8: "pedestrian",
    9: "cyclist",
    10: "car",
    11: "truck",
    12: "bus",
    13: "tram",
    14: "motorbike",
    15: "bicycle",
    16: "caravan",
    17: "trailer",
    18: "building",
    19: None,           # door — too minor, skip
    20: None,           # wall — background, skip
    21: "fence",
    22: "bridge",
    23: "tunnel",
    24: "stairs",
    25: "street pole",
    26: "sign",
    27: "traffic light",
    28: "vegetation",
    29: None,           # earth — too generic, skip
    30: None,           # sky — skip
    31: None,           # field — skip
    32: None,           # dynamic_void — void, skip
    33: None,           # static_void — void, skip
    34: None,           # unclear_void — void, skip
}

# Classes considered vehicles (for semantic context detection)
_VEHICLE_IDS = {10, 11, 12, 13, 14, 15, 16, 17}  # car, truck, bus, tram, motorbike, bicycle, caravan, trailer
# Classes considered persons (for semantic context detection)
_PERSON_IDS = {8, 9}                               # person, rider
# Classes considered walkable surfaces (for semantic context)
_WALKABLE_IDS = {2, 3, 4}                          # sidewalk, crosswalk, cycling_lane


def format_object_list(class_names, use_and=True):
    """
    Format a list of objects into natural language.

    Args:
        class_names: list of class name strings
        use_and: whether to join with 'and' before the last item

    Returns:
        str: e.g. "car, pedestrian, and traffic light"
    """
    if not class_names:
        return ""
    if len(class_names) == 1:
        return class_names[0]
    if len(class_names) == 2:
        return f"{class_names[0]} and {class_names[1]}" if use_and else f"{class_names[0]}, {class_names[1]}"
    if use_and:
        return ", ".join(class_names[:-1]) + f", and {class_names[-1]}"
    return ", ".join(class_names)


def get_present_objects(mask_array, max_objects=6):
    """
    Return prompt-friendly names for salient classes present in mask_array.

    Background classes (road, sidewalk, building, vegetation) are excluded from
    the object phrase — they are implied by the scene template and appear in
    nearly every image, so they add tokens without discriminative value.

    Priority order (front → back):
      1. People: pedestrian, cyclist
      2. Vehicles: car, truck, bus, tram, motorbike, bicycle, caravan, trailer
      3. Traffic control: traffic light, sign, street pole
      4. Structural features: crosswalk, cycling lane, rail track, fence, bridge, tunnel, stairs

    Args:
        mask_array: numpy array with class IDs 0-34
        max_objects: hard cap on the number of objects returned (default 6)

    Returns:
        list[str]: e.g. ["pedestrian", "tram", "traffic light"]
    """
    present_ids = set(np.unique(mask_array).tolist())

    priority_groups = [
        [8, 9],                             # people
        [10, 11, 12, 13, 14, 15, 16, 17],   # vehicles
        [27, 26, 25],                        # traffic control
        [3, 4, 6, 21, 22, 23, 24],           # structural features
        # road(1), sidewalk(2), building(18), vegetation(28) intentionally excluded:
        # they appear in nearly every image and are implied by the scene template
    ]

    ordered = []
    for group in priority_groups:
        for cid in group:
            if cid in present_ids and CLASS_PROMPT_NAMES.get(cid) is not None:
                ordered.append(CLASS_PROMPT_NAMES[cid])
                if len(ordered) >= max_objects:
                    return ordered

    return ordered


def is_valid_combo(lighting, weather):
    """
    Check if a lighting + weather combination is physically plausible.

    Invalid combinations:
    - Direct sunlight (noon / midday / sunny / harsh) + fog or overcast sky
    - Night lighting + any sunny weather descriptor
    - Heavy fog + clear or sunny lighting

    Args:
        lighting: string from LIGHTING_CONDITIONS
        weather:  string from WEATHER_CONDITIONS

    Returns:
        bool: True if combination is valid
    """
    l = lighting.lower()
    w = weather.lower()

    # Direct sunlight can't coexist with fog or overcast sky
    is_direct_sun = "sunny" in l or "noon" in l or "midday" in l or "harsh midday" in l
    if is_direct_sun and ("fog" in w or "overcast" in w):
        return False

    # Night + any sunny weather
    if "night" in l and "sunny" in w:
        return False

    # Dense fog incompatible with sunny or clear lighting
    if "dense fog" in w and ("sunny" in l or "noon" in l or "midday" in l or "clear" in l):
        return False

    # Backlit scene incompatible with fog (backlit requires visible sun behind subject)
    if "backlit" in l and "fog" in w:
        return False

    return True


def filter_anchor_by_scene(scene, anchor):
    """
    Ensure a semantic anchor is compatible with the scene type.

    Args:
        scene:  string from SCENE_TEMPLATES
        anchor: semantic anchor string (or empty string)

    Returns:
        str or None: anchor if valid, None if incompatible
    """
    if not anchor:
        return anchor

    scene_lower = scene.lower()

    # Campus walkways: only quiet/pedestrian anchors
    if "campus" in scene_lower:
        allowed = {
            "pedestrian street environment", "quiet urban environment",
            "pedestrian area", "walkway environment", "calm urban area",
        }
        return anchor if anchor in allowed else None

    # Suburban sidewalks: light traffic only
    if "suburban" in scene_lower and "sidewalk" in scene_lower:
        allowed = {
            "pedestrian street environment", "roadside activity",
            "quiet urban environment", "pedestrian area", "walkway environment",
            "calm urban area", "tranquil street setting",
        }
        return anchor if anchor in allowed else None

    return anchor


def get_semantic_context(mask_array):
    """
    Return a single semantic anchor phrase based on what is present in mask_array.

    Uses correct 35-class IDs from class_mapping.py:
      - person = 8, rider = 9
      - car = 10, truck = 11, bus = 12, tram = 13, motorbike = 14, bicycle = 15
      - sidewalk = 2, crosswalk = 3, cycling_lane = 4

    Args:
        mask_array: numpy array with class IDs 0-34

    Returns:
        str: single semantic anchor phrase, or "" if nothing notable
    """
    present = set(np.unique(mask_array).tolist())

    has_vehicles    = bool(present & _VEHICLE_IDS)
    has_pedestrians = bool(present & _PERSON_IDS)
    has_walkable    = bool(present & _WALKABLE_IDS)

    if has_vehicles and has_pedestrians:
        return random.choice([
            "busy urban street",
            "pedestrian street environment",
            "street-level urban activity",
            "mixed traffic environment",
            "urban activity zone",
            "shared road environment",
        ])
    elif has_vehicles:
        return random.choice([
            "urban street traffic",
            "roadside activity",
            "street-level urban movement",
            "active traffic corridor",
            "road activity",
            "active roadway",
            "vehicle-dominant street",
        ])
    elif has_pedestrians and has_walkable:
        return random.choice([
            "pedestrian area",
            "walkway environment",
            "foot traffic zone",
            "pedestrian-only zone",
        ])
    elif has_walkable:
        return random.choice([
            "quiet urban environment",
            "low traffic street",
            "residential street scene",
            "calm urban area",
            "tranquil street setting",
            "empty sidewalk scene",
        ])
    return ""


def generate_prompt_from_mask(mask_array, seed=None):
    """
    Generate a complete text prompt from a segmentation mask.

    Output format (strict ordering):
        {scene}, {lighting}, {weather}, {sensor}, {objects} visible, [{motion}], [{semantic anchor}], [{realism}]

    Features:
    - Object names extracted directly from mask classes (e.g. "tram and pedestrian visible")
    - Sensor always present (no None fallback)
    - Compatibility checks: no contradictory lighting/weather pairs
    - Scene-anchor matching prevents semantic mismatches
    - Probabilistic motion (10–60% depending on scene)
    - Optional realism anchor (40%)

    Args:
        mask_array: numpy array with class IDs 0-34
        seed: optional random seed for reproducibility

    Returns:
        str: complete prompt, e.g.
            "busy intersection, harsh midday sun, clear weather, dashcam recording,
             pedestrian and car and traffic light visible, slight motion blur,
             mixed traffic environment, candid urban photo"
    """
    if seed is not None:
        random.seed(seed)

    # 1. Scene
    scene = random.choice(SCENE_TEMPLATES)

    # 2. Lighting + weather with compatibility check
    # Default is guaranteed valid; only overwritten if a valid pair is found
    lighting, weather = "soft morning light", "clear weather"
    for _ in range(10):
        _l = random.choice(LIGHTING_CONDITIONS)
        _w = random.choice(WEATHER_CONDITIONS)
        if is_valid_combo(_l, _w):
            lighting, weather = _l, _w
            break

    # 3. Sensor — always present; HDR lighting biases toward technical artifacts
    if "HDR" in lighting:
        sensor = random.choice([
            "JPEG compression", "JPEG compression artifacts",
            "phone photo", "smartphone camera",
            "slight chroma noise", "ISO 1600", "ISO 800 with noise",
            "high ISO noise", "action camera footage",
        ])
    else:
        sensor = random.choice(SENSOR_REALISM)

    # 4. Salient objects from mask
    objects = get_present_objects(mask_array)
    object_phrase = format_object_list(objects) + " visible" if objects else ""

    # 5. Semantic anchor (scene-filtered)
    semantic_anchor = get_semantic_context(mask_array)
    semantic_anchor = filter_anchor_by_scene(scene, semantic_anchor)
    if semantic_anchor is None:
        present = set(np.unique(mask_array).tolist())
        if present & _WALKABLE_IDS:
            semantic_anchor = random.choice([
                "quiet urban environment", "pedestrian area",
                "walkway environment", "calm urban area",
            ])
        else:
            semantic_anchor = ""

    # 6. Motion blur (probability depends on scene busyness)
    busy_traffic = semantic_anchor in {
        "busy urban street", "active traffic corridor",
        "mixed traffic environment", "urban activity zone",
        "street-level urban activity", "shared road environment",
        "vehicle-dominant street",
    }
    is_walkway = semantic_anchor in {
        "walkway environment", "pedestrian area",
        "foot traffic zone", "pedestrian-only zone",
    } or "campus" in scene

    if is_walkway:
        motion = "slight blur" if random.random() < 0.1 else None
    elif busy_traffic:
        motion = random.choice(MOTION_OPTICS) if random.random() < 0.3 else None
    else:
        motion = random.choice(MOTION_OPTICS) if random.random() < 0.6 else None

    # HDR + rolling shutter is incoherent
    if "HDR" in lighting and motion in {"rolling shutter effect", "rolling shutter skew"}:
        motion = None

    # 7. Assemble prompt
    parts = [scene, lighting, weather, sensor]

    if object_phrase:
        parts.append(object_phrase)

    if motion:
        parts.append(motion)

    if semantic_anchor:
        parts.append(semantic_anchor)

    # Realism anchor (40%)
    if random.random() < 0.4:
        available = [r for r in REALISM_ANCHORS if r not in parts]
        if available:
            parts.append(random.choice(available))

    return ", ".join(parts)
