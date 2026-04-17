import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import app_services


class TestAppServices(unittest.TestCase):
    def test_is_video_file_detects_video_extensions(self):
        self.assertTrue(app_services.is_video_file("clip.mp4"))
        self.assertTrue(app_services.is_video_file("clip.MOV"))
        self.assertFalse(app_services.is_video_file("song.mp3"))

    def test_build_beat_rows_flattens_grouped_counts(self):
        grouped_counts = [
            [(1.0, True), (1.5, False)],
            [(2.0, True)],
        ]

        rows = app_services.build_beat_rows(grouped_counts)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["eight_count"], 1)
        self.assertEqual(rows[1]["kind"], "subdivision")
        self.assertEqual(rows[2]["timestamp_seconds"], 2.0)

    def test_build_timeline_rows_counts_beats_per_bucket(self):
        beats = [0.1, 0.2, 1.2, 2.9]

        rows = app_services.build_timeline_rows(beats, duration=3.0, bucket_count=3)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["beat_count"], 2)
        self.assertEqual(rows[1]["beat_count"], 1)
        self.assertEqual(rows[2]["beat_count"], 1)

    def test_summarize_eight_counts_formats_labels(self):
        grouped_counts = [[(1.0, True), (1.5, False)]]

        summaries = app_services.summarize_eight_counts(grouped_counts)

        self.assertEqual(summaries[0]["label"], "Eight-count 1")
        self.assertIn("1.50s (sub)", summaries[0]["values"])

    @patch("app_services.scratch_beat_detection.detect_beats", return_value=np.array([0.5, 1.0]))
    def test_detect_beats_for_method_uses_scratch_detector(self, mock_detect_beats):
        audio = np.array([0.0, 0.1, 0.0, 0.2])

        beats = app_services.detect_beats_for_method(audio, 22050, "scratch")

        self.assertEqual(beats.tolist(), [0.5, 1.0])
        mock_detect_beats.assert_called_once()

    def test_estimate_tempo_uses_median_interval(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5])

        tempo = app_services.estimate_tempo(beats, duration=2.0)

        self.assertEqual(tempo, 120.0)


if __name__ == "__main__":
    unittest.main()
