import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pose_estimation


def make_landmark(x, y, visibility=0.9):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


class TestPoseEstimation(unittest.TestCase):
    def test_calculate_angle_returns_expected_value(self):
        angle = pose_estimation._calculate_angle(
            point_a=pose_estimation.np.asarray([0.0, 1.0]),
            point_b=pose_estimation.np.asarray([0.0, 0.0]),
            point_c=pose_estimation.np.asarray([1.0, 0.0]),
        )

        self.assertEqual(angle, 90.0)

    def test_summarize_pose_landmarks_counts_visible_points(self):
        if not pose_estimation.LANDMARK_NAMES:
            self.skipTest("mediapipe landmark names are unavailable")

        landmarks = [make_landmark(0.5, 0.5, 0.9) for _ in pose_estimation.LANDMARK_NAMES]
        for name, coords in {
            "left_shoulder": (0.3, 0.4),
            "left_elbow": (0.4, 0.5),
            "left_wrist": (0.5, 0.6),
            "right_shoulder": (0.7, 0.4),
            "right_elbow": (0.6, 0.5),
            "right_wrist": (0.5, 0.6),
            "left_hip": (0.35, 0.7),
            "left_knee": (0.35, 0.85),
            "left_ankle": (0.35, 0.98),
            "right_hip": (0.65, 0.7),
            "right_knee": (0.65, 0.85),
            "right_ankle": (0.65, 0.98),
        }.items():
            index = pose_estimation.LANDMARK_NAMES.index(name)
            landmarks[index] = make_landmark(coords[0], coords[1], 0.95)

        summary = pose_estimation.summarize_pose_landmarks(landmarks)

        self.assertGreater(summary["visible_landmarks"], 0)
        self.assertIsNotNone(summary["left_arm_angle"])
        self.assertIsNotNone(summary["right_leg_angle"])


if __name__ == "__main__":
    unittest.main()
