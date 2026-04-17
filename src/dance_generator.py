# Dance Generation Module
# AI-powered choreography creation from audio beats

import random
from typing import List, Dict, Any
import numpy as np

# Comprehensive Step Library with 30 dance steps
STEP_LIBRARY = [
    # --- BASIC FOOTWORK ---
    {
        "id": "step_1",
        "name": "Right Step Touch",
        "beats": 4,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["legs"],
        "description": "Step right and bring left foot to touch",
        "pose_hint": ["ankles", "knees"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_2",
        "name": "Left Step Touch",
        "beats": 4,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["legs"],
        "description": "Step left and bring right foot to touch",
        "pose_hint": ["ankles", "knees"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_3",
        "name": "March in Place",
        "beats": 8,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["legs"],
        "description": "Lift knees alternately like marching",
        "pose_hint": ["knees", "hips"],
        "tempo_range": ["slow", "medium", "fast"]
    },

    # --- HAND BASICS ---
    {
        "id": "step_4",
        "name": "Clap Forward",
        "beats": 4,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["arms"],
        "description": "Clap hands in front",
        "pose_hint": ["wrists", "elbows"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_5",
        "name": "Overhead Clap",
        "beats": 4,
        "difficulty": "easy",
        "energy": "medium",
        "body_parts": ["arms"],
        "description": "Raise hands and clap above head",
        "pose_hint": ["shoulders", "wrists"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_6",
        "name": "Side Arm Wave",
        "beats": 8,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["arms"],
        "description": "Wave arms side to side",
        "pose_hint": ["wrists", "elbows"],
        "tempo_range": ["slow", "medium", "fast"]
    },

    # --- COMBO BASICS ---
    {
        "id": "step_7",
        "name": "Step Clap Right",
        "beats": 8,
        "difficulty": "easy",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Step right and clap",
        "pose_hint": ["ankles", "wrists"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_8",
        "name": "Step Clap Left",
        "beats": 8,
        "difficulty": "easy",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Step left and clap",
        "pose_hint": ["ankles", "wrists"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_9",
        "name": "Basic Bounce",
        "beats": 8,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["full"],
        "description": "Light bounce on knees",
        "pose_hint": ["knees"],
        "tempo_range": ["slow", "medium", "fast"]
    },
    {
        "id": "step_10",
        "name": "Shoulder Bounce",
        "beats": 8,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["upper"],
        "description": "Bounce shoulders rhythmically",
        "pose_hint": ["shoulders"],
        "tempo_range": ["slow", "medium", "fast"]
    },

    # --- INTERMEDIATE FOOTWORK ---
    {
        "id": "step_11",
        "name": "Grapevine Right",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["legs"],
        "description": "Step right, cross left behind, step right",
        "pose_hint": ["ankles"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_12",
        "name": "Grapevine Left",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["legs"],
        "description": "Step left, cross right behind",
        "pose_hint": ["ankles"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_13",
        "name": "Side Kick",
        "beats": 4,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["legs"],
        "description": "Kick leg to the side",
        "pose_hint": ["knees", "hips"],
        "tempo_range": ["medium", "fast"]
    },

    # --- HAND MOVES ---
    {
        "id": "step_14",
        "name": "Arm Circle",
        "beats": 8,
        "difficulty": "medium",
        "energy": "low",
        "body_parts": ["arms"],
        "description": "Rotate arms in circles",
        "pose_hint": ["shoulders"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_15",
        "name": "Cross Punch",
        "beats": 8,
        "difficulty": "medium",
        "energy": "high",
        "body_parts": ["arms"],
        "description": "Punch arms across body",
        "pose_hint": ["elbows"],
        "tempo_range": ["medium", "fast"]
    },

    # --- BODY MOVES ---
    {
        "id": "step_16",
        "name": "Body Roll",
        "beats": 8,
        "difficulty": "medium",
        "energy": "low",
        "body_parts": ["torso"],
        "description": "Roll chest and torso smoothly",
        "pose_hint": ["hips", "shoulders"],
        "tempo_range": ["slow", "medium"]
    },
    {
        "id": "step_17",
        "name": "Hip Sway",
        "beats": 8,
        "difficulty": "medium",
        "energy": "low",
        "body_parts": ["torso"],
        "description": "Move hips side to side",
        "pose_hint": ["hips"],
        "tempo_range": ["slow", "medium", "fast"]
    },

    # --- COMBO ---
    {
        "id": "step_18",
        "name": "Step + Kick Combo",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Step and kick forward",
        "pose_hint": ["knees"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_19",
        "name": "Turn Half",
        "beats": 4,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Turn 180 degrees",
        "pose_hint": ["hips"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_20",
        "name": "Turn Full",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Turn 360 degrees",
        "pose_hint": ["hips"],
        "tempo_range": ["medium", "fast"]
    },

    # --- ADVANCED / HIGH ENERGY ---
    {
        "id": "step_21",
        "name": "Jump Step",
        "beats": 4,
        "difficulty": "hard",
        "energy": "high",
        "body_parts": ["legs"],
        "description": "Jump and land on both feet",
        "pose_hint": ["knees"],
        "tempo_range": ["fast"]
    },
    {
        "id": "step_22",
        "name": "High Knees",
        "beats": 8,
        "difficulty": "hard",
        "energy": "high",
        "body_parts": ["legs"],
        "description": "Lift knees high quickly",
        "pose_hint": ["knees", "hips"],
        "tempo_range": ["fast"]
    },

    # --- HAND + BODY ---
    {
        "id": "step_23",
        "name": "Wave Combo",
        "beats": 8,
        "difficulty": "hard",
        "energy": "medium",
        "body_parts": ["arms"],
        "description": "Full arm wave sequence",
        "pose_hint": ["wrists", "elbows"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_24",
        "name": "Chest Pop",
        "beats": 4,
        "difficulty": "hard",
        "energy": "medium",
        "body_parts": ["torso"],
        "description": "Sharp chest movement",
        "pose_hint": ["chest"],
        "tempo_range": ["medium", "fast"]
    },

    # --- COMBOS ---
    {
        "id": "step_25",
        "name": "Jump + Clap",
        "beats": 8,
        "difficulty": "hard",
        "energy": "high",
        "body_parts": ["full"],
        "description": "Jump and clap overhead",
        "pose_hint": ["wrists", "knees"],
        "tempo_range": ["fast"]
    },
    {
        "id": "step_26",
        "name": "Spin + Pose",
        "beats": 8,
        "difficulty": "hard",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Spin and hold pose",
        "pose_hint": ["hips"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_27",
        "name": "Slide Step",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["legs"],
        "description": "Slide foot smoothly",
        "pose_hint": ["ankles"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_28",
        "name": "Back Step Groove",
        "beats": 8,
        "difficulty": "medium",
        "energy": "medium",
        "body_parts": ["full"],
        "description": "Step back with groove",
        "pose_hint": ["hips"],
        "tempo_range": ["medium", "fast"]
    },
    {
        "id": "step_29",
        "name": "Freestyle Bounce",
        "beats": 8,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["full"],
        "description": "Loose freestyle bounce",
        "pose_hint": ["knees"],
        "tempo_range": ["slow", "medium", "fast"]
    },
    {
        "id": "step_30",
        "name": "Final Pose",
        "beats": 4,
        "difficulty": "easy",
        "energy": "low",
        "body_parts": ["full"],
        "description": "Hold a stylish ending pose",
        "pose_hint": ["full_body"],
        "tempo_range": ["slow", "medium", "fast"]
    }
]


def classify_tempo(bpm: float) -> str:
    """Classify tempo into slow/medium/fast categories."""
    if bpm < 90:
        return "slow"
    elif bpm < 130:
        return "medium"
    else:
        return "fast"


def filter_steps_by_tempo(tempo: str, difficulty: str = None, energy: str = None) -> List[Dict]:
    """Filter steps based on tempo, difficulty, and energy preferences."""
    filtered = [step for step in STEP_LIBRARY if tempo in step["tempo_range"]]

    if difficulty:
        filtered = [step for step in filtered if step["difficulty"] == difficulty]

    if energy:
        filtered = [step for step in filtered if step["energy"] == energy]

    return filtered


def generate_choreography(beats: List[float], tempo_bpm: float, difficulty: str = "easy",
                         energy_preference: str = None, max_steps: int = 20) -> List[Dict]:
    """
    Generate choreography sequence from beat timestamps.

    Args:
        beats: List of beat timestamps in seconds
        tempo_bpm: Estimated tempo in BPM
        difficulty: Target difficulty level ("easy", "medium", "hard")
        energy_preference: Energy level preference ("low", "medium", "high")
        max_steps: Maximum number of steps to generate

    Returns:
        List of step dictionaries with timing information
    """
    if len(beats) == 0:
        return []

    tempo_category = classify_tempo(tempo_bpm)

    # Get available steps for this tempo
    available_steps = filter_steps_by_tempo(tempo_category, difficulty, energy_preference)

    if not available_steps:
        # Fallback to basic steps if no matches
        available_steps = filter_steps_by_tempo(tempo_category, "easy")

    # Group beats into 8-count segments (common in dance)
    beat_intervals = np.diff(beats)
    avg_beat_interval = np.mean(beat_intervals) if len(beat_intervals) > 0 else 0.5

    # Estimate 8-count duration (8 beats)
    eight_count_duration = 8 * avg_beat_interval

    choreography = []
    current_time = 0
    step_count = 0

    while current_time < beats[-1] and step_count < max_steps:
        # Select step based on current context
        step = select_step_for_timing(available_steps, current_time, beats, tempo_category)

        # Calculate step duration in seconds
        step_duration = step["beats"] * avg_beat_interval

        # Add step to choreography
        choreography.append({
            **step,
            "start_time": current_time,
            "end_time": current_time + step_duration,
            "beat_start": find_nearest_beat(beats, current_time),
            "beat_end": find_nearest_beat(beats, current_time + step_duration)
        })

        current_time += step_duration
        step_count += 1

    return choreography


def select_step_for_timing(available_steps: List[Dict], current_time: float,
                          beats: List[float], tempo: str) -> Dict:
    """Select an appropriate step based on current timing and context."""

    # Weight steps by their natural fit for the tempo
    weights = []
    for step in available_steps:
        weight = 1.0

        # Prefer steps that match tempo well
        if tempo in step["tempo_range"]:
            weight *= 2.0

        # Prefer variety - avoid repeating the same step type recently
        # (This is a simple implementation - could be enhanced)

        weights.append(weight)

    # Select step based on weights
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(available_steps)

    pick = random.uniform(0, total_weight)
    current_weight = 0

    for step, weight in zip(available_steps, weights):
        current_weight += weight
        if pick <= current_weight:
            return step

    return available_steps[0]  # Fallback


def find_nearest_beat(beats: List[float], time: float) -> int:
    """Find the index of the nearest beat to a given time."""
    if len(beats) == 0:
        return 0

    beats_array = np.array(beats)
    idx = np.argmin(np.abs(beats_array - time))
    return idx


def get_step_by_id(step_id: str) -> Dict:
    """Get a step by its ID."""
    for step in STEP_LIBRARY:
        if step["id"] == step_id:
            return step
    return None


def get_steps_by_difficulty(difficulty: str) -> List[Dict]:
    """Get all steps of a specific difficulty level."""
    return [step for step in STEP_LIBRARY if step["difficulty"] == difficulty]


def get_steps_by_body_part(body_part: str) -> List[Dict]:
    """Get all steps that use a specific body part."""
    return [step for step in STEP_LIBRARY if body_part in step["body_parts"]]


def calculate_choreography_stats(choreography: List[Dict]) -> Dict:
    """Calculate statistics about a generated choreography."""
    if not choreography:
        return {}

    total_beats = sum(step["beats"] for step in choreography)
    total_duration = sum(step["end_time"] - step["start_time"] for step in choreography)

    difficulty_counts = {}
    energy_counts = {}
    body_part_counts = {}

    for step in choreography:
        difficulty_counts[step["difficulty"]] = difficulty_counts.get(step["difficulty"], 0) + 1
        energy_counts[step["energy"]] = energy_counts.get(step["energy"], 0) + 1

        for body_part in step["body_parts"]:
            body_part_counts[body_part] = body_part_counts.get(body_part, 0) + 1

    return {
        "total_steps": len(choreography),
        "total_beats": total_beats,
        "total_duration": total_duration,
        "difficulty_distribution": difficulty_counts,
        "energy_distribution": energy_counts,
        "body_part_distribution": body_part_counts,
        "avg_step_duration": total_duration / len(choreography) if choreography else 0
    }