"""
Text prompt templates for ControlNet fine-tuning.
Combines scene conditions with class-based object descriptions.
Full schema as per project requirements + semantic context from 8-class masks.
"""

import random
import numpy as np
from pathlib import Path

# Scene/Environment templates
SCENE_TEMPLATES = [
    "urban street scene",
    "suburban sidewalk area",
    "campus walkway",
    "industrial district street",
    "city pedestrian zone",
    "residential neighborhood",
    "commercial district",
    "intersection area",
]

# Lighting conditions (full range as per spec)
LIGHTING_CONDITIONS = [
    "dawn lighting",
    "early morning light",
    "noon bright sunlight",
    "midday lighting",
    "dusk lighting",
    "evening twilight",
    "night illumination",
    "HDR lighting",
    "overcast lighting",
    "sunny conditions",
]




# Weather conditions (full range as per spec)
WEATHER_CONDITIONS = [
    "clear weather",
    "rain",
    "light rain",
    "wet road surface",
    "after rain with wet surfaces",
    "drizzle",
    "fog",
    "light fog",
    "overcast sky",
]

# Camera/Sensor realism (full range as per spec)
SENSOR_REALISM = [
    "phone photo",
    "smartphone camera",
    "JPEG compression",
    "JPEG compression artifacts",
    "slight chroma noise",
    "ISO 1600",
    "ISO 800 with noise",
    "high ISO noise",
]

# Motion/Optics effects (new - as per spec)
MOTION_OPTICS = [
    "slight motion blur",
    "motion blur",
    "rolling shutter skew",
    "rolling shutter effect",
    "shallow depth of field",
    "shallow DOF",
    "slight blur",
]

# Realism anchors (helps SD stay out of synthetic mode)
REALISM_ANCHORS = [
    "realistic photo",
    "natural lighting",
    "raw camera capture",
]

# Object relationship templates
OBJECT_RELATIONSHIP_TEMPLATES = [
    "{objects} visible in the scene",
    "scene containing {objects}",
    "street view with {objects}",
    "{objects} present in frame",
    "outdoor scene featuring {objects}",
    "navigation perspective showing {objects}",
    "pedestrian view of {objects}",
    "sidewalk environment with {objects}",
    "urban setting including {objects}",
    "street photograph capturing {objects}",
]

# Full prompt templates (simplified - no object details, just scene conditions)
FULL_PROMPT_TEMPLATES = [
    # Template 1: Scene + conditions
    "{scene}, {lighting}, {weather}, {sensor}",
    
    # Template 2: With motion
    "{scene}, {lighting}, {sensor}, {motion}",
    
    # Template 3: Weather emphasis
    "{weather}, {scene}, {lighting}, {sensor}",
    
    # Template 4: All conditions
    "{lighting} {scene}, {weather}, {sensor}, {motion}",
    
    # Template 5: Simple
    "{scene}, {weather}, {sensor}",
    
    # Template 6: Lighting first
    "{lighting} {scene}, {weather}, {motion}",
    
    # Template 7: Comprehensive
    "{scene} in {weather}, {lighting}, {sensor}, {motion}",
    
    # Template 8: Sensor emphasis
    "{sensor} of {scene}, {lighting}, {weather}",
    
    # Template 9: Motion + weather
    "{scene}, {weather}, {motion}, {sensor}",
    
    # Template 10: Complete
    "{lighting} {weather} {scene}, {sensor}, {motion}",
]


def format_object_list(class_names, use_and=True):
    """
    Format a list of objects into natural language.
    
    Args:
        class_names: list of class names (e.g., ['car', 'pedestrian', 'sidewalk'])
        use_and: whether to use 'and' before the last item
        
    Returns:
        str: formatted string (e.g., "car, pedestrian, and sidewalk")
    """
    if not class_names:
        return "various urban elements"
    
    if len(class_names) == 1:
        return class_names[0]
    
    if len(class_names) == 2:
        return f"{class_names[0]} and {class_names[1]}" if use_and else f"{class_names[0]}, {class_names[1]}"
    
    if use_and:
        return ", ".join(class_names[:-1]) + f", and {class_names[-1]}"
    else:
        return ", ".join(class_names)


def format_grouped_objects(grouped_classes):
    """
    Format grouped classes into natural language description.
    
    Args:
        grouped_classes: dict from group_classes_by_category()
        
    Returns:
        str: natural language object description
    """
    descriptions = []
    
    # Priority order for description
    priority_order = [
        "pedestrians",
        "vehicles", 
        "traffic_control",
        "walkable_surfaces",
        "obstacles",
        "road_surfaces",
        "environment"
    ]
    
    for category in priority_order:
        if category in grouped_classes:
            objects = grouped_classes[category]
            if category == "walkable_surfaces":
                descriptions.append(format_object_list(objects, use_and=False))
            elif category == "pedestrians":
                descriptions.append(format_object_list(objects, use_and=True))
            elif category == "vehicles":
                descriptions.append(format_object_list(objects, use_and=False))
            else:
                descriptions.append(format_object_list(objects, use_and=False))
    
    return format_object_list(descriptions, use_and=True) if descriptions else "urban street elements"


def is_valid_combo(lighting, weather):
    """
    Check if lighting and weather combination is physically plausible.
    Prevents contradictions like "sunny + overcast" or "night + sunlight".
    
    Args:
        lighting: lighting condition string
        weather: weather condition string
        
    Returns:
        bool: True if combination is valid
    """
    lighting_lower = lighting.lower()
    weather_lower = weather.lower()
    
    # Sunny conditions incompatible with overcast
    if "sunny" in lighting_lower and "overcast" in weather_lower:
        return False
    
    # Night incompatible with sunlight
    if "night" in lighting_lower and "sunlight" in lighting_lower:
        return False
    
    # HDR lighting incompatible with night
    if "hdr" in lighting_lower and "night" in lighting_lower:
        return False
    
    # Noon/midday incompatible with night
    if ("noon" in lighting_lower or "midday" in lighting_lower) and "night" in weather_lower:
        return False
    
    return True


def filter_anchor_by_scene(scene, anchor):
    """
    Ensure semantic anchor is compatible with scene type.
    Strict constraints to prevent semantic mismatches.
    
    Args:
        scene: scene type string
        anchor: semantic anchor string
        
    Returns:
        str or None: anchor if valid, None if incompatible
    """
    if not anchor:
        return anchor
    
    scene_lower = scene.lower()
    
    # Campus walkways: only pedestrian/quiet anchors
    if "campus" in scene_lower:
        allowed = ["pedestrian street environment", "quiet urban environment", 
                   "pedestrian area", "walkway environment", "calm urban area"]
        if anchor not in allowed:
            return None
    
    # Suburban sidewalks: pedestrian or light traffic only
    elif "suburban" in scene_lower and "sidewalk" in scene_lower:
        allowed = ["pedestrian street environment", "roadside activity", 
                   "quiet urban environment", "pedestrian area", "walkway environment",
                   "calm urban area", "tranquil street setting"]
        if anchor not in allowed:
            return None
    
    # Intersections and industrial: busy traffic allowed
    # (no restrictions for these)
    
    return anchor


def get_semantic_context(mask_array):
    """
    Extract semantic context from 8-class simplified mask.
    Returns ONE diversified semantic anchor (expanded vocabulary).
    
    Args:
        mask_array: numpy array with class IDs 0-7 (after remapping)
        
    Returns:
        str: single semantic anchor phrase (or empty string)
    """
    # Detect simplified classes (0-7)
    present_classes = set(np.unique(mask_array))
    
    has_walkable = 2 in present_classes
    has_pedestrians = 3 in present_classes
    has_vehicles = 4 in present_classes
    
    # Return ONE diversified semantic anchor (expanded vocabulary)
    if has_vehicles and has_pedestrians:
        # Busy urban streets with both
        return random.choice([
            "busy urban street",
            "pedestrian street environment",
            "street-level urban activity",
            "mixed traffic environment",
            "urban activity zone",
        ])
    elif has_vehicles:
        # Vehicles present (expanded vocabulary)
        return random.choice([
            "urban street traffic",
            "roadside activity",
            "street-level urban movement",
            "active traffic corridor",
            "road activity",
            "street-level road scene",
            "active roadway",
        ])
    elif has_pedestrians and has_walkable:
        # Pedestrians without vehicles
        return random.choice([
            "pedestrian area",
            "walkway environment",
            "foot traffic zone",
        ])
    elif has_walkable:
        # Few or no vehicles (expanded vocabulary)
        return random.choice([
            "quiet urban environment",
            "low traffic street",
            "residential street scene",
            "calm urban area",
            "tranquil street setting",
        ])
    else:
        return ""  # No semantic anchor needed


def generate_prompt_from_mask(mask_array, seed=None):
    """
    Generate complete prompt from mask using professor's schema + semantic context.
    STRICT ORDERING: [scene], [lighting], [weather], [sensor], [motion], [semantic anchor], [realism]
    
    Features:
    - Compatibility checks (no contradictory lighting/weather)
    - Scene-anchor matching (no mismatches)
    - Probabilistic motion (60%) and sensor noise (70%)
    - Optional realism anchor (40%)
    
    Args:
        mask_array: numpy array with remapped class IDs (0-7)
        seed: random seed for reproducibility
        
    Returns:
        str: complete prompt with strict ordering and no contradictions
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample scene first
    scene = random.choice(SCENE_TEMPLATES)
    
    # Sample lighting and weather with compatibility check
    max_attempts = 10
    for _ in range(max_attempts):
        lighting = random.choice(LIGHTING_CONDITIONS)
        weather = random.choice(WEATHER_CONDITIONS)
        if is_valid_combo(lighting, weather):
            break
    
    # Probabilistic sensor noise (70% chance)
    sensor = random.choice(SENSOR_REALISM) if random.random() < 0.7 else "natural lighting"
    
    # Probabilistic sensor noise (70% chance)
    # If HDR lighting, prefer technical artifacts (not natural lighting)
    if "HDR" in lighting:
        sensor = random.choice([
            "JPEG compression",
            "JPEG compression artifacts",
            "phone photo",
            "smartphone camera",
            "slight chroma noise",
            "ISO 1600",
            "ISO 800 with noise",
            "high ISO noise",
        ]) if random.random() < 0.7 else None
    else:
        sensor = random.choice(SENSOR_REALISM) if random.random() < 0.7 else "natural lighting"
    
    # Get semantic anchor and filter by scene compatibility
    semantic_anchor = get_semantic_context(mask_array)
    semantic_anchor = filter_anchor_by_scene(scene, semantic_anchor)
    
    # If filtered out, try again with pedestrian/quiet anchors only
    if semantic_anchor is None:
        present_classes = set(np.unique(mask_array))
        has_walkable = 2 in present_classes
        if has_walkable:
            semantic_anchor = random.choice([
                "quiet urban environment",
                "pedestrian area",
                "walkway environment",
                "calm urban area",
            ])
        else:
            semantic_anchor = ""
    
    # Probabilistic motion (reduced for busy traffic to preserve edges)
    # Motion blur hurts ControlNet edge consistency in busy scenes
    busy_traffic = semantic_anchor in [
        "busy urban street",
        "active traffic corridor",
        "mixed traffic environment",
        "urban activity zone",
        "street-level urban activity",
    ]
    
    # Check if this is a walkway/pedestrian dominant scene
    is_walkway_scene = semantic_anchor in [
        "walkway environment",
        "pedestrian area",
        "foot traffic zone",
    ] or "campus walkway" in scene
    
    if is_walkway_scene:
        # Walkway scenes: no motion blur (weakens curb alignment & pedestrian silhouettes)
        # Use slight blur at most (10% chance) or nothing
        motion = "slight blur" if random.random() < 0.1 else None
    elif busy_traffic:
        # Only 30% chance for busy traffic (preserves edges)
        motion = random.choice(MOTION_OPTICS) if random.random() < 0.3 else None
    else:
        # 60% chance for quiet/pedestrian scenes
        motion = random.choice(MOTION_OPTICS) if random.random() < 0.6 else None
    
    # HDR conflict fix: remove rolling shutter if HDR present
    if "HDR" in lighting and motion in ["rolling shutter effect", "rolling shutter skew"]:
        motion = None
    
    # Build prompt with STRICT ORDERING
    # Format: scene, lighting, weather, sensor, [motion], [semantic_anchor], [realism]
    parts = [scene, lighting, weather]
    
    # Add sensor if not None
    if sensor:
        parts.append(sensor)
    
    if motion:
        parts.append(motion)
    
    if semantic_anchor:
        parts.append(semantic_anchor)
    
    # Add realism anchor (40% chance)
    # BUT: not with HDR lighting, and not if "natural lighting" already present
    can_add_realism = (
        "HDR" not in lighting and
        sensor != "natural lighting" and
        random.random() < 0.4
    )
    
    if can_add_realism:
        # Choose realism anchor that's not already present
        # Also exclude "realistic photo" with HDR (causes warped cars/rubbery pedestrians)
        available_realism = [r for r in REALISM_ANCHORS if r not in parts]
        if "HDR" in lighting:
            available_realism = [r for r in available_realism if r != "realistic photo"]
        if available_realism:
            parts.append(random.choice(available_realism))
    
    prompt = ", ".join(parts)
    
    return prompt
