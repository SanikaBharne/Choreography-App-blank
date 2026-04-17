import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import scratch_beat_detection


class TestScratchBeatDetection(unittest.TestCase):
    def test_frame_audio_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            scratch_beat_detection.frame_audio(np.array([]))

    def test_detect_beats_finds_synthetic_pulses(self):
        sr = 100
        audio = np.zeros(600, dtype=float)
        audio[100:110] = 1.0
        audio[250:260] = 1.0
        audio[400:410] = 1.0

        beats = scratch_beat_detection.detect_beats(
            audio,
            sr,
            frame_size=20,
            hop_size=10,
            threshold_scale=0.5,
            min_interval=0.2,
        )

        self.assertGreaterEqual(len(beats), 3)
        self.assertTrue(any(abs(beat - 1.0) < 0.25 for beat in beats))
        self.assertTrue(any(abs(beat - 2.5) < 0.25 for beat in beats))
        self.assertTrue(any(abs(beat - 4.0) < 0.25 for beat in beats))

    def test_pick_beats_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            scratch_beat_detection.pick_beats([0.1, 0.2], sr=22050, hop_size=512, threshold_scale=0)


if __name__ == "__main__":
    unittest.main()
