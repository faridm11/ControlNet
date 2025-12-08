"""
Text prompt templates for ControlNet fine-tuning.
Combines scene conditions with class-based object descriptions.
Full schema as per project requirements.
"""

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
