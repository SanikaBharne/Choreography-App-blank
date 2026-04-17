import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import app_services


class TestPoseAppServices(unittest.TestCase):
    @patch("app_services.pose_estimation.analyze_video_pose")
    def test_analyze_pose_builds_overview_and_rows(self, mock_analyze_video_pose):
        mock_analyze_video_pose.return_value = [
            {
                "timestamp": 0.0,
                "has_pose": True,
                "overlay_frame": "frame-a",
                "summary": {
                    "visible_landmarks": 20,
                    "left_arm_angle": 100.0,
                    "right_arm_angle": 110.0,
                    "left_leg_angle": 170.0,
                    "right_leg_angle": 168.0,
                    "shoulder_tilt": 3.0,
                    "hip_tilt": 1.5,
                    "visibility_score": 0.8,
                },
            },
            {
                "timestamp": 1.0,
                "has_pose": False,
                "overlay_frame": "frame-b",
                "summary": None,
            },
        ]

        results = app_services.analyze_pose("sample.mp4", sample_count=2)

        self.assertEqual(results["overview"]["sample_count"], 2)
        self.assertEqual(results["overview"]["detected_frames"], 1)
        self.assertEqual(results["overview"]["average_visibility"], 0.8)
        self.assertFalse(results["rows"][1]["pose_detected"])


if __name__ == "__main__":
    unittest.main()
