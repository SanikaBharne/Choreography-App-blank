import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend.app.main import app
import app_services


class TestFastApiApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        app_services.ensure_workspace()

    def setUp(self):
        self.media_id = "test_api_sample.mp4"
        self.media_path = app_services.UPLOADS_DIR / self.media_id
        self.media_path.write_bytes(b"fake-video")

    def tearDown(self):
        if self.media_path.exists():
            self.media_path.unlink()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("backend.app.main.app_services.save_upload_bytes")
    def test_upload_media(self, mock_save_upload_bytes):
        saved_path = app_services.UPLOADS_DIR / "uploaded_song.mp3"
        mock_save_upload_bytes.return_value = saved_path

        response = self.client.post(
            "/api/media/upload",
            files={"file": ("song.mp3", b"audio-bytes", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["filename"].endswith("song.mp3"))
        self.assertEqual(payload["media_type"], "audio")

    @patch("backend.app.main.app_services.analyze_media_with_method")
    def test_analyze_media(self, mock_analyze_media):
        mock_analyze_media.return_value = {
            "audio_path": str(self.media_path),
            "media_type": "video",
            "beats": [0.5, 1.0],
            "grouped_counts": [[(0.5, True), (0.75, False)]],
            "duration": 3.0,
            "sample_rate": 22050,
            "tempo_estimate": 120.0,
            "beat_rows": [{"eight_count": 1, "position": 1, "timestamp_seconds": 0.5, "kind": "beat"}],
            "timeline_rows": [{"window_start": 0.0, "beat_count": 1}],
            "subdivisions": 2,
            "beat_method": "scratch",
        }

        response = self.client.post(
            f"/api/media/{self.media_id}/analysis",
            json={"beat_method": "scratch", "subdivisions": 2},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["beat_method"], "scratch")
        self.assertEqual(payload["beats"], [0.5, 1.0])

    @patch("backend.app.main.app_services.create_practice_media")
    def test_create_practice_clip(self, mock_create_practice_media):
        preview_path = app_services.ANALYSIS_DIR / "preview.mp4"
        preview_path.write_bytes(b"preview")
        self.addCleanup(lambda: preview_path.unlink() if preview_path.exists() else None)
        mock_create_practice_media.return_value = preview_path

        response = self.client.post(
            f"/api/media/{self.media_id}/practice-clip",
            json={"start_sec": 0.0, "end_sec": 4.0, "speed_multiplier": 0.75, "mirror": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["preview_url"], "/media/analysis/preview.mp4")

    @patch("backend.app.main.app_services.create_practice_media")
    def test_create_practice_clip_audio(self, mock_create_practice_media):
        audio_id = "test_audio_sample.mp3"
        audio_path = app_services.UPLOADS_DIR / audio_id
        audio_path.write_bytes(b"fake-audio")
        self.addCleanup(lambda: audio_path.unlink() if audio_path.exists() else None)

        preview_path = app_services.ANALYSIS_DIR / "preview.wav"
        preview_path.write_bytes(b"preview")
        self.addCleanup(lambda: preview_path.unlink() if preview_path.exists() else None)
        mock_create_practice_media.return_value = preview_path

        response = self.client.post(
            f"/api/media/{audio_id}/practice-clip",
            json={"start_sec": 0.0, "end_sec": 3.0, "speed_multiplier": 1.0, "mirror": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["preview_url"], "/media/analysis/preview.wav")

    @patch("backend.app.main.app_services.save_overlay_frame")
    @patch("backend.app.main.app_services.analyze_pose")
    def test_pose_endpoint(self, mock_analyze_pose, mock_save_overlay_frame):
        overlay_path = app_services.ANALYSIS_DIR / "pose_frame.png"
        overlay_path.write_bytes(b"png")
        self.addCleanup(lambda: overlay_path.unlink() if overlay_path.exists() else None)
        mock_save_overlay_frame.return_value = overlay_path
        mock_analyze_pose.return_value = {
            "overview": {"sample_count": 2, "detected_frames": 1, "average_visibility": 0.7},
            "rows": [{"timestamp": 0.0, "pose_detected": True}],
            "frames": [
                {"timestamp": 0.0, "has_pose": True, "overlay_frame": object(), "summary": {"visible_landmarks": 12}},
            ],
        }

        response = self.client.post(
            f"/api/media/{self.media_id}/pose",
            json={"sample_count": 2},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["overview"]["sample_count"], 2)
        self.assertEqual(payload["frames"][0]["overlay_url"], "/media/analysis/pose_frame.png")

    @patch("backend.app.main.app_services.generate_choreography_plan")
    def test_choreography_endpoint(self, mock_generate_choreography_plan):
        gallery_path = app_services.ANALYSIS_DIR / "test_step_pose_1.png"
        gallery_path.write_bytes(b"png")
        animation_path = app_services.ANALYSIS_DIR / "test_step_animation.gif"
        animation_path.write_bytes(b"gif")
        self.addCleanup(lambda: gallery_path.unlink() if gallery_path.exists() else None)
        self.addCleanup(lambda: animation_path.unlink() if animation_path.exists() else None)

        mock_generate_choreography_plan.return_value = {
            "analysis": {
                "audio_path": str(self.media_path),
                "media_type": "video",
                "beats": [0.5, 1.0],
                "grouped_counts": [[(0.5, True), (0.75, False)]],
                "duration": 3.0,
                "sample_rate": 22050,
                "tempo_estimate": 120.0,
                "beat_rows": [{"eight_count": 1, "position": 1, "timestamp_seconds": 0.5, "kind": "beat"}],
                "timeline_rows": [{"window_start": 0.0, "beat_count": 1}],
                "subdivisions": 2,
                "beat_method": "scratch",
            },
            "choreography": [
                {
                    "id": "step_1",
                    "name": "Right Step Touch",
                    "beats": 4,
                    "difficulty": "easy",
                    "energy": "low",
                    "body_parts": ["legs"],
                    "description": "Step right and bring left foot to touch",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "beat_start": 1,
                    "beat_end": 2,
                    "pose_gallery_paths": [str(gallery_path)],
                    "skeleton_animation_path": str(animation_path),
                }
            ],
            "choreography_stats": {
                "total_steps": 1,
                "total_beats": 4,
                "total_duration": 2.0,
                "difficulty_distribution": {"easy": 1},
                "energy_distribution": {"low": 1},
                "body_part_distribution": {"legs": 1},
                "avg_step_duration": 2.0,
            },
        }

        response = self.client.post(
            f"/api/media/{self.media_id}/choreography",
            json={
                "beat_method": "scratch",
                "subdivisions": 2,
                "difficulty": "easy",
                "energy_preference": "medium",
                "pose_preview_count": 6,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("choreography", payload)
        self.assertEqual(payload["choreography"][0]["pose_gallery_urls"], ["/media/analysis/test_step_pose_1.png"])
        self.assertEqual(payload["choreography"][0]["skeleton_animation_url"], "/media/analysis/test_step_animation.gif")


if __name__ == "__main__":
    unittest.main()
