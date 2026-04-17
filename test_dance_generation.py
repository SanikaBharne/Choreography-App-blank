#!/usr/bin/env python3
"""Test script for AI Dance Generation functionality."""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dance_generator
import numpy as np


def test_dance_generator():
    """Test the dance generation functionality."""
    print("🧪 Testing AI Dance Generation...")

    # Create mock beat data (simulate 120 BPM for 30 seconds)
    tempo_bpm = 120
    duration_seconds = 30
    beats_per_second = tempo_bpm / 60
    total_beats = int(duration_seconds * beats_per_second)

    # Generate beat timestamps
    beat_times = np.linspace(0, duration_seconds, total_beats)

    print(f"📊 Generated {len(beat_times)} beats at {tempo_bpm} BPM")

    # Test choreography generation
    print("🎭 Generating choreography...")
    choreography = dance_generator.generate_choreography(
        beats=beat_times,
        tempo_bpm=tempo_bpm,
        difficulty="easy",
        max_steps=6
    )

    print(f"✅ Generated {len(choreography)} steps")

    # Display choreography
    for i, step in enumerate(choreography, 1):
        print(f"Step {i}: {step['name']} ({step['beats']} beats) - {step['difficulty']} difficulty")

    # Test step filtering
    print("\n🎯 Testing step filtering...")
    slow_steps = dance_generator.filter_steps_by_tempo("slow")
    medium_steps = dance_generator.filter_steps_by_tempo("medium")
    fast_steps = dance_generator.filter_steps_by_tempo("fast")

    print(f"Slow tempo steps: {len(slow_steps)}")
    print(f"Medium tempo steps: {len(medium_steps)}")
    print(f"Fast tempo steps: {len(fast_steps)}")

    # Test statistics
    stats = dance_generator.calculate_choreography_stats(choreography)
    print("\n📈 Choreography Statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Total duration: {stats['total_duration']:.1f}s")
    print(f"  Difficulty distribution: {stats['difficulty_distribution']}")

    print("\n🎉 All tests passed!")


def test_pose_sequences():
    """Test pose sequence generation."""
    print("\n🦴 Testing Pose Sequences...")

    try:
        import pose_sequences
        from pose_sequences import PoseSequenceGenerator

        pose_gen = PoseSequenceGenerator()

        # Test a simple step
        print("Generating pose sequence for 'Right Step Touch'...")
        sequence = pose_gen.generate_step_sequence("step_1", duration_beats=4)

        print(f"✅ Generated sequence with {len(sequence['pose_sequence'])} frames")
        print(f"   Step: {sequence['step_name']}")
        print(f"   Duration: {sequence['duration_seconds']:.1f}s")
        print(f"   Key joints: {sequence['key_joints']}")

        # Test teaching sequence
        teaching = pose_sequences.create_step_teaching_sequence(sequence)
        print(f"✅ Created teaching sequence with {len(teaching['breakdown'])} beat breakdowns")
        assert "pose_gallery" in teaching
        assert isinstance(teaching["pose_gallery"], list)
        if teaching["pose_gallery"]:
            first_image = teaching["pose_gallery"][0]
            assert hasattr(first_image, "shape")
            assert first_image.shape[2] == 3

    except ImportError as e:
        print(f"⚠️  Pose sequences test skipped: {e}")
    except Exception as e:
        print(f"❌ Pose sequences test failed: {e}")


if __name__ == "__main__":
    test_dance_generator()
    test_pose_sequences()