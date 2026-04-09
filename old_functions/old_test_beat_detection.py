import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import  hand_built_beat_detection


class TestBeatDetection(unittest.TestCase):
    def test_get_frame(self):
        """get_frame splits sample data into frames of the requested size."""
        frames = hand_built_beat_detection.get_frame([1, 2, 3, 4, 5], frame_size=2)
        self.assertEqual([[1, 2], [3, 4], [5]], frames)

    def test_get_frame_asserts_on_invalid_frame_size(self):
        """get_frame raises ValueError when frame_size is not positive."""
        with self.assertRaises(ValueError):
            hand_built_beat_detection.get_frame([1, 2, 3], frame_size=0)

    def test_rms(self):
        """RMS returns the root-mean-square value for a frame."""
        value = hand_built_beat_detection.RMS(np.array([3.0, 4.0]))
        self.assertAlmostEqual(np.sqrt(12.5), value)

    def test_rms_asserts_on_empty_frame(self):
        """RMS raises ValueError for an empty frame."""
        with self.assertRaises(ValueError):
            hand_built_beat_detection.RMS([])

    def test_get_rms_energy(self):
        """get_RMS_energy returns one RMS value for each frame."""
        frames = [np.array([3.0, 4.0]), np.array([0.0, 0.0])]
        values = hand_built_beat_detection.get_RMS_energy(frames)
        self.assertEqual(2, len(values))
        self.assertAlmostEqual(np.sqrt(12.5), values[0])
        self.assertEqual(0.0, values[1])

    def test_get_onset_strength(self):
        """get_onset_strength keeps only positive RMS increases."""
        dec_test = hand_built_beat_detection.get_onset_strength([100, 80, 60, 40, 20])
        test_list = [0, 0, 0, 0]
        self.assertEqual(test_list, dec_test)

        eq_test = hand_built_beat_detection.get_onset_strength([100, 100, 100, 100, 100])
        self.assertEqual(test_list, eq_test)

        spike_test = hand_built_beat_detection.get_onset_strength([10, 10, 1000, 10, 10])
        test_list = [0, 990, 0, 0]
        self.assertEqual(test_list, spike_test)

        empty_test = hand_built_beat_detection.get_onset_strength([])
        self.assertEqual([], empty_test)

        single_test = hand_built_beat_detection.get_onset_strength([100])
        self.assertEqual([], single_test)

    def test_get_onset_strength_asserts_on_non_numeric_values(self):
        """get_onset_strength raises TypeError for non-numeric RMS values."""
        with self.assertRaises(TypeError):
            hand_built_beat_detection.get_onset_strength([1, "bad", 3])

    def test_get_beats_with_explicit_threshold(self):
        """get_beats returns indices above an explicit threshold."""
        beats = hand_built_beat_detection.get_beats([0, 2, 1, 4, 3], threshold=2)
        self.assertEqual([3, 4], beats)

    def test_get_beats_with_mean_threshold(self):
        """get_beats uses the mean onset strength when no threshold is given."""
        beats = hand_built_beat_detection.get_beats([0, 2, 1, 4, 3])
        self.assertEqual([3, 4], beats)

    def test_get_beats_returns_empty_for_empty_input(self):
        """get_beats returns an empty list for empty onset input."""
        self.assertEqual([], hand_built_beat_detection.get_beats([]))

    def test_get_beats_asserts_on_non_numeric_threshold(self):
        """get_beats raises TypeError for a non-numeric threshold."""
        with self.assertRaises(TypeError):
            hand_built_beat_detection.get_beats([1, 2, 3], threshold="high")

    def test_get_timestamp(self):
        """get_timestamp converts beat indices into seconds."""
        timestamps = hand_built_beat_detection.get_timestamp([0, 2, 5], frame_size=1024, sample_rate=2048)
        self.assertEqual([0.0, 1.0, 2.5], timestamps)

    def test_get_timestamp_asserts_on_negative_index(self):
        """get_timestamp raises ValueError for a negative beat index."""
        with self.assertRaises(ValueError):
            hand_built_beat_detection.get_timestamp([-1], frame_size=1024, sample_rate=2048)

    def test_get_timestamp_asserts_on_invalid_sample_rate(self):
        """get_timestamp raises ValueError for a non-positive sample rate."""
        with self.assertRaises(ValueError):
            hand_built_beat_detection.get_timestamp([1], frame_size=1024, sample_rate=0)

if __name__ == "__main__":
    unittest.main()
