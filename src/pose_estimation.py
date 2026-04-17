from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError:
    mp = None
    BaseOptions = None
    vision = None

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "workspace" / "models"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_CONNECTIONS = tuple() if vision is None else tuple(vision.PoseLandmarksConnections.POSE_LANDMARKS)
LANDMARK_NAMES = [] if vision is None else [landmark.name.lower() for landmark in vision.PoseLandmark]


def ensure_mediapipe():
    if mp is None or vision is None or BaseOptions is None:
        raise ImportError("mediapipe is not installed. Install it to enable pose estimation.")


def ensure_pose_model():
    ensure_mediapipe()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


def _normalize_point(landmarks, name):
    index = LANDMARK_NAMES.index(name)
    landmark = landmarks[index]
    return np.asarray([landmark.x, landmark.y], dtype=float), float(getattr(landmark, "visibility", 0.0))


def _calculate_angle(point_a, point_b, point_c):
    ba = point_a - point_b
    bc = point_c - point_b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return None
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 1)


def summarize_pose_landmarks(landmarks):
    ensure_mediapipe()
    visible_count = sum(1 for landmark in landmarks if getattr(landmark, "visibility", 0.0) >= 0.5)

    left_shoulder, left_shoulder_vis = _normalize_point(landmarks, "left_shoulder")
    left_elbow, left_elbow_vis = _normalize_point(landmarks, "left_elbow")
    left_wrist, left_wrist_vis = _normalize_point(landmarks, "left_wrist")
    right_shoulder, right_shoulder_vis = _normalize_point(landmarks, "right_shoulder")
    right_elbow, right_elbow_vis = _normalize_point(landmarks, "right_elbow")
    right_wrist, right_wrist_vis = _normalize_point(landmarks, "right_wrist")
    left_hip, left_hip_vis = _normalize_point(landmarks, "left_hip")
    left_knee, left_knee_vis = _normalize_point(landmarks, "left_knee")
    left_ankle, left_ankle_vis = _normalize_point(landmarks, "left_ankle")
    right_hip, right_hip_vis = _normalize_point(landmarks, "right_hip")
    right_knee, right_knee_vis = _normalize_point(landmarks, "right_knee")
    right_ankle, right_ankle_vis = _normalize_point(landmarks, "right_ankle")

    return {
        "visible_landmarks": visible_count,
        "left_arm_angle": _calculate_angle(left_shoulder, left_elbow, left_wrist),
        "right_arm_angle": _calculate_angle(right_shoulder, right_elbow, right_wrist),
        "left_leg_angle": _calculate_angle(left_hip, left_knee, left_ankle),
        "right_leg_angle": _calculate_angle(right_hip, right_knee, right_ankle),
        "shoulder_tilt": round(
            float(np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))),
            1,
        ),
        "hip_tilt": round(
            float(np.degrees(np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]))),
            1,
        ),
        "visibility_score": round(
            float(
                np.mean(
                    [
                        left_shoulder_vis,
                        left_elbow_vis,
                        left_wrist_vis,
                        right_shoulder_vis,
                        right_elbow_vis,
                        right_wrist_vis,
                        left_hip_vis,
                        left_knee_vis,
                        left_ankle_vis,
                        right_hip_vis,
                        right_knee_vis,
                        right_ankle_vis,
                    ]
                )
            ),
            3,
        ),
    }


def _draw_pose_overlay(frame_rgb, landmarks):
    height, width = frame_rgb.shape[:2]
    image = frame_rgb.copy()

    for connection in POSE_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        start_xy = (int(start.x * width), int(start.y * height))
        end_xy = (int(end.x * width), int(end.y * height))
        cv2.line(image, start_xy, end_xy, (237, 175, 144), 2)

    for landmark in landmarks:
        if getattr(landmark, "visibility", 0.0) < 0.35:
            continue
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(image, center, 4, (124, 167, 196), -1)

    return image


def _create_landmarker():
    model_path = ensure_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)



def analyze_video_pose(video_path, sample_count=4, timestamps=None):
    ensure_mediapipe()
    results = []
    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
        if timestamps is not None and len(timestamps) > 0:
            # Use provided timestamps, clamp to video duration
            timestamps_to_use = [min(max(0.0, float(t)), duration - 0.001) for t in timestamps]
        else:
            if sample_count <= 0:
                raise ValueError("sample_count must be positive")
            timestamps_to_use = np.linspace(0, max(duration - 0.001, 0), sample_count)
        with _create_landmarker() as landmarker:
            for timestamp in timestamps_to_use:
                frame = clip.get_frame(float(timestamp))
                rgb_frame = np.asarray(frame, dtype=np.uint8)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = landmarker.detect(mp_image)

                if not detection.pose_landmarks:
                    results.append(
                        {
                            "timestamp": round(float(timestamp), 3),
                            "has_pose": False,
                            "overlay_frame": rgb_frame,
                            "summary": None,
                        }
                    )
                    continue

                landmarks = detection.pose_landmarks[0]
                overlay_frame = _draw_pose_overlay(rgb_frame, landmarks)
                results.append(
                    {
                        "timestamp": round(float(timestamp), 3),
                        "has_pose": True,
                        "overlay_frame": overlay_frame,
                        "summary": summarize_pose_landmarks(landmarks),
                    }
                )
    return results
