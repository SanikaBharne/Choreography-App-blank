# Pose Sequence Generator for Dance Steps
# Creates skeleton animations and teaching sequences

import io
import numpy as np
from typing import List, Dict, Any, Tuple
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    # Check if we have the old API (mediapipe.solutions)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        MP_AVAILABLE = True
        USE_OLD_API = True
    # Check if we have the new API (mediapipe.tasks)
    elif hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision'):
        MP_AVAILABLE = True
        USE_OLD_API = False
    else:
        MP_AVAILABLE = False
        USE_OLD_API = False
except ImportError:
    MP_AVAILABLE = False
    USE_OLD_API = False
    mp = None

# MediaPipe-style skeleton connections for drawing poses
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32)
]

def _draw_pose_with_pillow(landmarks, image_size, background_color):
    from PIL import Image, ImageDraw

    width, height = image_size
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        if landmarks[start_idx][2] < 0.1 or landmarks[end_idx][2] < 0.1:
            continue
        draw.line(
            (
                landmarks[start_idx][0],
                landmarks[start_idx][1],
                landmarks[end_idx][0],
                landmarks[end_idx][1],
            ),
            fill=(220, 190, 120),
            width=2,
        )

    for x, y, visibility in landmarks:
        if visibility < 0.1:
            continue
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(236, 240, 255), outline=(64, 130, 255), width=1)

    return np.array(image, dtype=np.uint8)


def _pose_to_image(pose: np.ndarray, image_size: Tuple[int, int] = (360, 360),
                   background_color: Tuple[int, int, int] = (18, 18, 36)) -> np.ndarray:
    """Render a single pose as a skeleton image."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Validate pose input
        if pose is None:
            logger.error("Pose is None")
            raise ValueError("Pose cannot be None")
        
        if not isinstance(pose, np.ndarray):
            logger.error(f"Pose is not a numpy array: {type(pose)}")
            raise ValueError(f"Pose must be numpy array, got {type(pose)}")
        
        if pose.shape != (33, 4):
            logger.error(f"Pose shape is {pose.shape}, expected (33, 4)")
            raise ValueError(f"Pose must have shape (33, 4), got {pose.shape}")
        
        width, height = image_size
        canvas = np.full((height, width, 3), background_color, dtype=np.uint8)

        landmarks = []
        for i, lm in enumerate(pose):
            try:
                x = float(np.clip(lm[0], 0.0, 1.0) * (width - 1))
                y = float(np.clip(lm[1], 0.0, 1.0) * (height - 1))
                visibility = float(np.clip(lm[3], 0.0, 1.0))
                landmarks.append((int(x), int(y), visibility))
            except (IndexError, TypeError) as e:
                logger.error(f"Error processing landmark {i}: {lm}, error: {e}")
                landmarks.append((width//2, height//2, 0.0))

        if not CV2_AVAILABLE:
            return _draw_pose_with_pillow(landmarks, image_size, background_color)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            if landmarks[start_idx][2] < 0.1 or landmarks[end_idx][2] < 0.1:
                continue
            cv2.line(
                canvas,
                (landmarks[start_idx][0], landmarks[start_idx][1]),
                (landmarks[end_idx][0], landmarks[end_idx][1]),
                (220, 190, 120),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        for x, y, visibility in landmarks:
            if visibility < 0.1:
                continue
            cv2.circle(canvas, (x, y), 5, (236, 240, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 8, (64, 130, 255), thickness=1, lineType=cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        logger.error(f"Error in _pose_to_image: {e}")
        width, height = image_size
        if CV2_AVAILABLE:
            canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
            cv2.putText(canvas, "Pose Error", (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        from PIL import Image, ImageDraw
        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        draw.text((width * 0.25, height * 0.45), "Pose Error", fill=(255, 0, 0))
        return np.array(image, dtype=np.uint8)


def generate_pose_gallery(step_sequence: Dict[str, Any], image_count: int = 8) -> List[np.ndarray]:
    """Generate a gallery of representative pose images for the step."""
    frames = step_sequence.get("pose_sequence", [])
    if not frames:
        return []

    count = min(image_count, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [_pose_to_image(frames[idx]) for idx in indices]


def create_skeleton_animation(step_sequence: Dict[str, Any], fps: int = 10,
                              max_frames: int = 60) -> bytes | None:
    """Render a skeleton animation GIF for a pose sequence."""
    import logging
    logger = logging.getLogger(__name__)
    
    frames = step_sequence.get("pose_sequence", [])
    if not frames:
        logger.warning("No pose_sequence frames found in step_sequence")
        return None

    logger.info(f"Generating animation with {len(frames)} frames")

    sampled = frames
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        sampled = [frames[idx] for idx in indices]
        logger.info(f"Sampled {len(sampled)} frames for animation")

    try:
        images = [_pose_to_image(frame) for frame in sampled]
        logger.info(f"Generated {len(images)} images from poses")
    except Exception as e:
        logger.error(f"Failed to convert poses to images: {e}")
        return None

    # Try imageio first
    try:
        import imageio
        gif_buffer = io.BytesIO()
        imageio.mimsave(gif_buffer, images, format="GIF", fps=fps)
        logger.info("Successfully created GIF with imageio")
        return gif_buffer.getvalue()
    except ImportError:
        logger.warning("imageio not available, trying PIL")
    except Exception as e:
        logger.error(f"imageio GIF creation failed: {e}")

    # Fallback to PIL
    try:
        from PIL import Image
        pil_images = [Image.fromarray(image) for image in images]
        gif_buffer = io.BytesIO()
        pil_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=pil_images[1:],
            loop=0,
            duration=int(1000 / fps),
        )
        logger.info("Successfully created GIF with PIL")
        return gif_buffer.getvalue()
    except ImportError:
        logger.warning("PIL not available")
    except Exception as e:
        logger.error(f"PIL GIF creation failed: {e}")

    logger.error("All GIF creation methods failed")
    return None


class PoseSequenceGenerator:
    """Generates pose sequences for dance steps using MediaPipe format."""

    def __init__(self):
        if not MP_AVAILABLE:
            self.pose = None
            return

        if USE_OLD_API:
            # Old MediaPipe API (pre-0.10.x)
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # New MediaPipe API (0.10.x+)
            # For now, we'll create a mock pose object since the new API is more complex
            self.pose = None  # We'll handle this in the methods

    def generate_step_sequence(self, step_id: str, duration_beats: int = 8,
                              fps: int = 30) -> Dict[str, Any]:
        """
        Generate a pose sequence for a dance step.

        Args:
            step_id: ID of the step from STEP_LIBRARY
            duration_beats: How many beats this step lasts
            fps: Frames per second for the animation

        Returns:
            Dictionary with pose keypoints sequence and metadata
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Import here to avoid circular imports
        from dance_generator import get_step_by_id

        step = get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")

        # Calculate sequence parameters
        duration_seconds = (duration_beats / 8) * 4.0  # Assume 120 BPM baseline
        total_frames = int(duration_seconds * fps)
        
        logger.info(f"Generating step sequence for {step_id} ({step['name']}): {duration_beats} beats, {total_frames} frames at {fps} fps")

        # Generate pose sequence based on step type
        if not MP_AVAILABLE:
            logger.info("MediaPipe not available, using mock pose sequence")
            # Return mock pose sequence when MediaPipe is not available
            pose_sequence = self._generate_mock_pose_sequence(total_frames)
        else:
            logger.info(f"MediaPipe available, generating pose for step type")
            pose_sequence = self._generate_pose_for_step(step, total_frames)
        
        logger.info(f"Generated {len(pose_sequence) if pose_sequence else 0} pose frames")

        return {
            "step_id": step_id,
            "step_name": step["name"],
            "duration_beats": duration_beats,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "total_frames": total_frames,
            "pose_sequence": pose_sequence,  # List of pose landmarks per frame
            "key_joints": step["pose_hint"],
            "body_parts": step["body_parts"],
            "difficulty": step["difficulty"],
            "energy": step["energy"]
        }

    def _generate_pose_for_step(self, step: Dict, total_frames: int) -> List[np.ndarray]:
        """Generate pose keypoints sequence for a specific step."""
        if not MP_AVAILABLE:
            return self._generate_mock_pose_sequence(total_frames)

        pose_sequence = []

        # Create base pose (standing neutral)
        base_pose = self._create_base_pose()

        # Modify pose based on step type
        if step["id"] == "step_1":  # Right Step Touch
            pose_sequence = self._generate_step_touch_sequence(base_pose, total_frames, "right")
        elif step["id"] == "step_2":  # Left Step Touch
            pose_sequence = self._generate_step_touch_sequence(base_pose, total_frames, "left")
        elif step["id"] == "step_3":  # March in Place
            pose_sequence = self._generate_march_sequence(base_pose, total_frames)
        elif step["id"] == "step_4":  # Clap Forward
            pose_sequence = self._generate_clap_sequence(base_pose, total_frames, "forward")
        elif step["id"] == "step_5":  # Overhead Clap
            pose_sequence = self._generate_clap_sequence(base_pose, total_frames, "overhead")
        elif step["id"] == "step_6":  # Side Arm Wave
            pose_sequence = self._generate_arm_wave_sequence(base_pose, total_frames)
        elif step["id"] == "step_7":  # Step Clap Right
            pose_sequence = self._generate_step_clap_sequence(base_pose, total_frames, "right")
        elif step["id"] == "step_8":  # Step Clap Left
            pose_sequence = self._generate_step_clap_sequence(base_pose, total_frames, "left")
        elif step["id"] == "step_9":  # Basic Bounce
            pose_sequence = self._generate_bounce_sequence(base_pose, total_frames, "basic")
        elif step["id"] == "step_10":  # Shoulder Bounce
            pose_sequence = self._generate_shoulder_bounce_sequence(base_pose, total_frames)
        elif step["id"] == "step_11":  # Grapevine Right
            pose_sequence = self._generate_grapevine_sequence(base_pose, total_frames, "right")
        elif step["id"] == "step_12":  # Grapevine Left
            pose_sequence = self._generate_grapevine_sequence(base_pose, total_frames, "left")
        elif step["id"] == "step_13":  # Side Kick
            pose_sequence = self._generate_kick_sequence(base_pose, total_frames, "side")
        elif step["id"] == "step_14":  # Arm Circle
            pose_sequence = self._generate_arm_circle_sequence(base_pose, total_frames)
        elif step["id"] == "step_15":  # Cross Punch
            pose_sequence = self._generate_punch_sequence(base_pose, total_frames, "cross")
        elif step["id"] == "step_16":  # Body Roll
            pose_sequence = self._generate_body_roll_sequence(base_pose, total_frames)
        elif step["id"] == "step_17":  # Hip Sway
            pose_sequence = self._generate_hip_sway_sequence(base_pose, total_frames)
        elif step["id"] == "step_18":  # Step + Kick Combo
            pose_sequence = self._generate_step_kick_sequence(base_pose, total_frames)
        elif step["id"] == "step_19":  # Turn Half
            pose_sequence = self._generate_turn_sequence(base_pose, total_frames, 180)
        elif step["id"] == "step_20":  # Turn Full
            pose_sequence = self._generate_turn_sequence(base_pose, total_frames, 360)
        elif step["id"] == "step_21":  # Jump Step
            pose_sequence = self._generate_jump_sequence(base_pose, total_frames)
        elif step["id"] == "step_22":  # High Knees
            pose_sequence = self._generate_high_knees_sequence(base_pose, total_frames)
        elif step["id"] == "step_23":  # Wave Combo
            pose_sequence = self._generate_wave_sequence(base_pose, total_frames)
        elif step["id"] == "step_24":  # Chest Pop
            pose_sequence = self._generate_chest_pop_sequence(base_pose, total_frames)
        elif step["id"] == "step_25":  # Jump + Clap
            pose_sequence = self._generate_jump_clap_sequence(base_pose, total_frames)
        elif step["id"] == "step_26":  # Spin + Pose
            pose_sequence = self._generate_spin_pose_sequence(base_pose, total_frames)
        elif step["id"] == "step_27":  # Slide Step
            pose_sequence = self._generate_slide_sequence(base_pose, total_frames)
        elif step["id"] == "step_28":  # Back Step Groove
            pose_sequence = self._generate_back_step_sequence(base_pose, total_frames)
        elif step["id"] == "step_29":  # Freestyle Bounce
            pose_sequence = self._generate_bounce_sequence(base_pose, total_frames, "freestyle")
        elif step["id"] == "step_30":  # Final Pose
            pose_sequence = self._generate_final_pose_sequence(base_pose, total_frames)
        else:
            # Default: hold base pose
            pose_sequence = [base_pose.copy() for _ in range(total_frames)]

        return pose_sequence

    def _generate_mock_pose_sequence(self, total_frames: int) -> List[np.ndarray]:
        """Generate a mock pose sequence when MediaPipe is not available."""
        # Create a simple mock pose that just holds a neutral standing position
        base_pose = self._create_base_pose()
        return [base_pose.copy() for _ in range(total_frames)]

    def _create_base_pose(self) -> np.ndarray:
        """Create a neutral standing pose in MediaPipe format."""
        # 33 landmarks (x, y, z, visibility) - MediaPipe Pose format
        pose = np.zeros((33, 4))

        # Basic standing pose coordinates (normalized 0-1)
        # Nose
        pose[0] = [0.5, 0.1, 0, 1.0]

        # Eyes, ears
        pose[1] = [0.48, 0.08, 0, 1.0]  # Left eye
        pose[2] = [0.52, 0.08, 0, 1.0]  # Right eye
        pose[3] = [0.46, 0.06, 0, 1.0]  # Left ear
        pose[4] = [0.54, 0.06, 0, 1.0]  # Right ear

        # Shoulders
        pose[11] = [0.4, 0.2, 0, 1.0]   # Left shoulder
        pose[12] = [0.6, 0.2, 0, 1.0]   # Right shoulder

        # Elbows
        pose[13] = [0.35, 0.3, 0, 1.0]  # Left elbow
        pose[14] = [0.65, 0.3, 0, 1.0]  # Right elbow

        # Wrists
        pose[15] = [0.3, 0.4, 0, 1.0]   # Left wrist
        pose[16] = [0.7, 0.4, 0, 1.0]   # Right wrist

        # Hips
        pose[23] = [0.45, 0.5, 0, 1.0]  # Left hip
        pose[24] = [0.55, 0.5, 0, 1.0]  # Right hip

        # Knees
        pose[25] = [0.43, 0.7, 0, 1.0]  # Left knee
        pose[26] = [0.57, 0.7, 0, 1.0]  # Right knee

        # Ankles
        pose[27] = [0.42, 0.9, 0, 1.0]  # Left ankle
        pose[28] = [0.58, 0.9, 0, 1.0]  # Right ankle

        # Feet
        pose[29] = [0.4, 0.95, 0, 1.0]  # Left heel
        pose[30] = [0.44, 0.95, 0, 1.0]  # Left foot index
        pose[31] = [0.56, 0.95, 0, 1.0]  # Right heel
        pose[32] = [0.6, 0.95, 0, 1.0]   # Right foot index

        return pose

    def _generate_step_touch_sequence(self, base_pose: np.ndarray, frames: int, direction: str) -> List[np.ndarray]:
        """Generate step touch sequence."""
        sequence = []
        offset = 0.05 if direction == "right" else -0.05

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            if progress < 0.5:  # Step out
                pose[28 if direction == "right" else 27][0] += offset * (progress * 2)  # Move foot
            else:  # Bring back
                pose[28 if direction == "right" else 27][0] += offset * (1 - (progress - 0.5) * 2)

            sequence.append(pose)

        return sequence

    def _generate_march_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate marching in place sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 march cycles

            # Alternate knee lifts
            left_knee_lift = 0.05 * np.sin(progress * np.pi)
            right_knee_lift = 0.05 * np.sin((progress + 2) * np.pi)

            pose[25][1] -= left_knee_lift   # Left knee up
            pose[26][1] -= right_knee_lift  # Right knee up

            sequence.append(pose)

        return sequence

    def _generate_clap_sequence(self, base_pose: np.ndarray, frames: int, style: str) -> List[np.ndarray]:
        """Generate clap sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            if style == "forward":
                # Hands come together at chest level
                hand_y = 0.35 + 0.05 * np.sin(progress * np.pi * 2)
                pose[15][0] = 0.45 + 0.05 * np.sin(progress * np.pi * 2)  # Left hand
                pose[16][0] = 0.55 - 0.05 * np.sin(progress * np.pi * 2)  # Right hand
                pose[15][1] = pose[16][1] = hand_y
            elif style == "overhead":
                # Hands go up and clap
                hand_y = 0.2 - 0.15 * np.sin(progress * np.pi * 2)
                pose[15][0] = 0.45 + 0.05 * np.sin(progress * np.pi * 2)
                pose[16][0] = 0.55 - 0.05 * np.sin(progress * np.pi * 2)
                pose[15][1] = pose[16][1] = hand_y

            sequence.append(pose)

        return sequence

    def _generate_arm_wave_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate arm wave sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 wave cycles

            # Side to side arm movement
            arm_offset = 0.1 * np.sin(progress * np.pi)
            pose[15][0] = 0.3 + arm_offset  # Left hand
            pose[16][0] = 0.7 + arm_offset  # Right hand

            sequence.append(pose)

        return sequence

    def _generate_step_clap_sequence(self, base_pose: np.ndarray, frames: int, direction: str) -> List[np.ndarray]:
        """Generate step and clap combo."""
        sequence = []
        step_offset = 0.05 if direction == "right" else -0.05

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Step movement
            if progress < 0.6:
                pose[28 if direction == "right" else 27][0] += step_offset * (progress / 0.6)

            # Clap movement
            clap_progress = max(0, (progress - 0.3) / 0.4)
            if clap_progress > 0:
                hand_y = 0.35 + 0.05 * np.sin(clap_progress * np.pi * 2)
                pose[15][0] = 0.45 + 0.05 * np.sin(clap_progress * np.pi * 2)
                pose[16][0] = 0.55 - 0.05 * np.sin(clap_progress * np.pi * 2)
                pose[15][1] = pose[16][1] = hand_y

            sequence.append(pose)

        return sequence

    def _generate_bounce_sequence(self, base_pose: np.ndarray, frames: int, style: str) -> List[np.ndarray]:
        """Generate bounce sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 8  # 8 bounce cycles

            # Vertical bounce
            bounce = 0.02 * np.sin(progress * np.pi)
            for j in range(23, 33):  # Lower body
                pose[j][1] += bounce

            if style == "freestyle":
                # Add some arm movement
                arm_bounce = 0.01 * np.sin(progress * np.pi * 1.5)
                pose[15][1] += arm_bounce
                pose[16][1] += arm_bounce

            sequence.append(pose)

        return sequence

    def _generate_shoulder_bounce_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate shoulder bounce sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 8  # 8 bounce cycles

            # Shoulder bounce
            shoulder_bounce = 0.03 * np.sin(progress * np.pi)
            pose[11][1] += shoulder_bounce  # Left shoulder
            pose[12][1] += shoulder_bounce  # Right shoulder

            sequence.append(pose)

        return sequence

    def _generate_grapevine_sequence(self, base_pose: np.ndarray, frames: int, direction: str) -> List[np.ndarray]:
        """Generate grapevine sequence."""
        sequence = []
        step_size = 0.03

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 steps

            # Complex footwork pattern
            cycle = progress % 1
            if cycle < 0.25:  # Step side
                offset = step_size * (cycle / 0.25)
                pose[28 if direction == "right" else 27][0] += offset if direction == "right" else -offset
            elif cycle < 0.5:  # Cross behind
                pose[27 if direction == "right" else 28][0] += step_size * ((cycle - 0.25) / 0.25)
            elif cycle < 0.75:  # Step side again
                offset = step_size * ((cycle - 0.5) / 0.25)
                pose[28 if direction == "right" else 27][0] += offset if direction == "right" else -offset

            sequence.append(pose)

        return sequence

    def _generate_kick_sequence(self, base_pose: np.ndarray, frames: int, direction: str) -> List[np.ndarray]:
        """Generate kick sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Quick kick motion
            if progress < 0.3:
                kick_height = 0.1 * (progress / 0.3)
                pose[26 if direction == "right" else 25][1] -= kick_height  # Kick leg up
            elif progress < 0.6:
                kick_height = 0.1 * (1 - (progress - 0.3) / 0.3)
                pose[26 if direction == "right" else 25][1] -= kick_height

            sequence.append(pose)

        return sequence

    def _generate_arm_circle_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate arm circle sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 circle rotations

            # Circular arm movement
            angle = progress * 2 * np.pi
            radius = 0.1

            pose[15][0] = 0.4 + radius * np.cos(angle)      # Left hand x
            pose[15][1] = 0.3 + radius * np.sin(angle)      # Left hand y
            pose[16][0] = 0.6 + radius * np.cos(angle + np.pi)  # Right hand x
            pose[16][1] = 0.3 + radius * np.sin(angle + np.pi)  # Right hand y

            sequence.append(pose)

        return sequence

    def _generate_punch_sequence(self, base_pose: np.ndarray, frames: int, style: str) -> List[np.ndarray]:
        """Generate punch sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 punch cycles

            # Alternating punches
            left_punch = 0.1 * np.sin(progress * np.pi)
            right_punch = 0.1 * np.sin((progress + 2) * np.pi)

            pose[15][0] += left_punch   # Left arm punch
            pose[16][0] -= right_punch  # Right arm punch

            sequence.append(pose)

        return sequence

    def _generate_body_roll_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate body roll sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 roll cycles

            # Smooth torso rolling motion
            roll_offset = 0.02 * np.sin(progress * np.pi)

            # Upper body roll
            for j in [11, 12, 13, 14, 15, 16]:  # Shoulders, arms
                pose[j][1] += roll_offset

            sequence.append(pose)

        return sequence

    def _generate_hip_sway_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate hip sway sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 sway cycles

            # Side to side hip movement
            sway = 0.03 * np.sin(progress * np.pi)
            pose[23][0] += sway  # Left hip
            pose[24][0] += sway  # Right hip

            sequence.append(pose)

        return sequence

    def _generate_step_kick_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate step and kick combo."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            if progress < 0.5:  # Step
                step_progress = progress / 0.5
                pose[28][0] += 0.05 * step_progress  # Right foot step
            else:  # Kick
                kick_progress = (progress - 0.5) / 0.5
                kick_height = 0.08 * np.sin(kick_progress * np.pi)
                pose[26][1] -= kick_height  # Right leg kick

            sequence.append(pose)

        return sequence

    def _generate_turn_sequence(self, base_pose: np.ndarray, frames: int, degrees: int) -> List[np.ndarray]:
        """Generate turn sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Rotational movement (simplified)
            rotation = (degrees / 360) * 2 * np.pi * progress

            # Apply rotation to upper body
            center_x, center_y = 0.5, 0.3
            for j in [11, 12, 13, 14, 15, 16]:  # Upper body
                x, y = pose[j][0] - center_x, pose[j][1] - center_y
                new_x = x * np.cos(rotation) - y * np.sin(rotation)
                new_y = x * np.sin(rotation) + y * np.cos(rotation)
                pose[j][0] = center_x + new_x
                pose[j][1] = center_y + new_y

            sequence.append(pose)

        return sequence

    def _generate_jump_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate jump sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Jump arc
            jump_height = 0.05 * np.sin(progress * np.pi)

            # Lift entire body
            for j in range(23, 33):  # Lower body
                pose[j][1] -= jump_height

            sequence.append(pose)

        return sequence

    def _generate_high_knees_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate high knees sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 8  # 8 knee lift cycles

            # Alternating high knee lifts
            left_knee = 0.08 * np.sin(progress * np.pi)
            right_knee = 0.08 * np.sin((progress + 4) * np.pi)

            pose[25][1] -= left_knee   # Left knee
            pose[26][1] -= right_knee  # Right knee

            sequence.append(pose)

        return sequence

    def _generate_wave_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate wave sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 2  # 2 wave cycles

            # Complex arm wave pattern
            wave1 = 0.05 * np.sin(progress * np.pi)
            wave2 = 0.05 * np.sin(progress * np.pi * 2)

            pose[15][0] += wave1  # Left arm
            pose[15][1] += wave2
            pose[16][0] -= wave1  # Right arm
            pose[16][1] += wave2

            sequence.append(pose)

        return sequence

    def _generate_chest_pop_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate chest pop sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 8  # 8 pop cycles

            # Sharp chest movements
            pop = 0.02 * np.sin(progress * np.pi * 4)  # Quick pops

            pose[11][1] += pop  # Shoulders
            pose[12][1] += pop

            sequence.append(pose)

        return sequence

    def _generate_jump_clap_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate jump and clap combo."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Jump
            jump_height = 0.05 * np.sin(progress * np.pi)
            for j in range(23, 33):  # Lower body
                pose[j][1] -= jump_height

            # Clap at peak of jump
            if 0.4 < progress < 0.6:
                clap_progress = (progress - 0.4) / 0.2
                hand_y = 0.2 - 0.1 * np.sin(clap_progress * np.pi)
                pose[15][0] = 0.45 + 0.05 * np.sin(clap_progress * np.pi)
                pose[16][0] = 0.55 - 0.05 * np.sin(clap_progress * np.pi)
                pose[15][1] = pose[16][1] = hand_y

            sequence.append(pose)

        return sequence

    def _generate_spin_pose_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate spin and pose sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            if progress < 0.7:  # Spinning
                rotation = progress / 0.7 * 4 * np.pi  # Multiple rotations

                center_x, center_y = 0.5, 0.3
                for j in range(33):  # All landmarks
                    x, y = pose[j][0] - center_x, pose[j][1] - center_y
                    new_x = x * np.cos(rotation) - y * np.sin(rotation)
                    new_y = x * np.sin(rotation) + y * np.cos(rotation)
                    pose[j][0] = center_x + new_x
                    pose[j][1] = center_y + new_y
            # Hold final pose for last 30%

            sequence.append(pose)

        return sequence

    def _generate_slide_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate slide step sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 2  # 2 slide cycles

            # Smooth sliding motion
            slide = 0.04 * np.sin(progress * np.pi)
            pose[27][0] += slide  # Left foot
            pose[28][0] += slide  # Right foot

            sequence.append(pose)

        return sequence

    def _generate_back_step_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate back step with groove."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = (i / frames) * 4  # 4 step cycles

            # Back step motion
            step_back = 0.03 * np.sin(progress * np.pi)
            pose[27][0] -= step_back  # Left foot back
            pose[28][0] -= step_back  # Right foot back

            # Add groove with hip movement
            groove = 0.02 * np.sin(progress * np.pi * 2)
            pose[23][0] += groove  # Left hip
            pose[24][0] += groove  # Right hip

            sequence.append(pose)

        return sequence

    def _generate_final_pose_sequence(self, base_pose: np.ndarray, frames: int) -> List[np.ndarray]:
        """Generate final pose sequence."""
        sequence = []

        for i in range(frames):
            pose = base_pose.copy()
            progress = i / frames

            # Build up to final pose
            if progress > 0.5:
                pose_progress = (progress - 0.5) / 0.5

                # Dramatic pose: arms up, one leg back
                pose[15][1] = 0.2 - 0.1 * pose_progress  # Left arm up
                pose[16][1] = 0.2 - 0.1 * pose_progress  # Right arm up
                pose[15][0] = 0.35 + 0.05 * pose_progress  # Left arm out
                pose[16][0] = 0.65 - 0.05 * pose_progress  # Right arm out

                # Leg back
                pose[26][0] += 0.05 * pose_progress  # Right leg back

            sequence.append(pose)

        return sequence


def create_step_teaching_sequence(step_sequence: Dict) -> Dict[str, Any]:
    """
    Create a teaching sequence with different speeds and breakdowns.

    Args:
        step_sequence: Pose sequence from generate_step_sequence

    Returns:
        Teaching data with full, slow, and breakdown versions
    """
    fps = step_sequence["fps"]
    pose_sequence = step_sequence["pose_sequence"]

    # Full speed (normal)
    full_speed = {
        "frames": pose_sequence,
        "fps": fps,
        "duration": len(pose_sequence) / fps
    }

    # Half speed
    slow_frames = []
    for i in range(0, len(pose_sequence), 2):  # Every other frame
        slow_frames.extend([pose_sequence[i]] * 2)  # Hold each frame longer
    slow_speed = {
        "frames": slow_frames,
        "fps": fps // 2,
        "duration": len(slow_frames) / (fps // 2)
    }

    # Count breakdown (8 counts)
    beats = step_sequence["duration_beats"]
    frames_per_beat = len(pose_sequence) // beats

    breakdown = []
    for beat in range(beats):
        start_frame = beat * frames_per_beat
        end_frame = min((beat + 1) * frames_per_beat, len(pose_sequence))

        beat_frames = pose_sequence[start_frame:end_frame]
        breakdown.append({
            "beat_number": beat + 1,
            "frames": beat_frames,
            "description": f"Count {beat + 1}"
        })

    pose_gallery = generate_pose_gallery(step_sequence, image_count=min(8, len(pose_sequence)))
    skeleton_animation = create_skeleton_animation(step_sequence, fps=min(10, step_sequence.get("fps", 10)))

    for beat, beat_data in enumerate(breakdown, start=1):
        beat_frames = beat_data["frames"]
        if beat_frames:
            beat_data["image"] = _pose_to_image(beat_frames[0])
        else:
            beat_data["image"] = None

    return {
        "step_info": step_sequence,
        "full_speed": full_speed,
        "slow_speed": slow_speed,
        "breakdown": breakdown,
        "pose_gallery": pose_gallery,
        "skeleton_animation": skeleton_animation,
        "key_joints": step_sequence["key_joints"],
        "teaching_tips": generate_teaching_tips(step_sequence)
    }


def generate_teaching_tips(step_sequence: Dict) -> List[str]:
    """Generate teaching tips for a step."""
    tips = []

    step_name = step_sequence["step_name"]
    body_parts = step_sequence["body_parts"]
    difficulty = step_sequence["difficulty"]
    energy = step_sequence["energy"]

    # Basic tips
    tips.append(f"Focus on your {', '.join(body_parts)} during this step")
    tips.append(f"This is a {difficulty} difficulty step")

    if energy == "high":
        tips.append("Use lots of energy and power!")
    elif energy == "low":
        tips.append("Keep it smooth and controlled")

    # Step-specific tips
    if "step" in step_sequence["step_id"] and "touch" in step_sequence["step_name"].lower():
        tips.append("Keep your supporting leg strong")
        tips.append("Touch lightly and quickly")

    elif "march" in step_sequence["step_name"].lower():
        tips.append("Lift your knees high but keep your upper body steady")
        tips.append("March in rhythm with the beat")

    elif "clap" in step_sequence["step_name"].lower():
        tips.append("Clap at the right moment in the music")
        tips.append("Keep your arms relaxed")

    elif "bounce" in step_sequence["step_name"].lower():
        tips.append("Bounce from your knees, not your ankles")
        tips.append("Stay loose and have fun!")

    return tips


def create_full_routine_animation(choreography: List[Dict], fps: int = 10) -> bytes | None:
    """
    Create a single animation showing the complete dance routine from pose 1 to pose 8.

    Args:
        choreography: List of step dictionaries from dance_generator
        fps: Frames per second for the animation

    Returns:
        GIF bytes of the full routine animation, or None if generation fails
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not choreography:
        logger.warning("Empty choreography provided")
        return None

    logger.info(f"Creating full routine animation for {len(choreography)} steps")
    
    pose_gen = PoseSequenceGenerator()
    full_pose_sequence = []

    # Generate pose sequence for each step and concatenate
    for step in choreography:
        try:
            step_sequence = pose_gen.generate_step_sequence(
                step_id=step["id"],
                duration_beats=step["beats"],
                fps=fps
            )
            pose_frames = step_sequence.get("pose_sequence", [])
            logger.info(f"Step {step['id']} generated {len(pose_frames)} pose frames")
            full_pose_sequence.extend(pose_frames)
        except Exception as e:
            logger.error(f"Could not generate sequence for step {step['id']}: {e}")
            continue

    if not full_pose_sequence:
        logger.error("No pose frames generated for full routine")
        return None

    logger.info(f"Total pose frames: {len(full_pose_sequence)}")

    # Limit total frames to prevent huge animations (max ~10 seconds at 10fps)
    max_frames = 100
    if len(full_pose_sequence) > max_frames:
        indices = np.linspace(0, len(full_pose_sequence) - 1, max_frames, dtype=int)
        sampled_sequence = [full_pose_sequence[idx] for idx in indices]
        logger.info(f"Sampled {len(sampled_sequence)} frames")
    else:
        sampled_sequence = full_pose_sequence

    # Generate images for animation
    try:
        images = [_pose_to_image(frame) for frame in sampled_sequence]
        logger.info(f"Generated {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to convert poses to images: {e}")
        return None

    # Create GIF - try imageio first
    try:
        import imageio
        gif_buffer = io.BytesIO()
        imageio.mimsave(gif_buffer, images, format="GIF", fps=fps)
        logger.info("Successfully created full routine GIF with imageio")
        return gif_buffer.getvalue()
    except ImportError:
        logger.warning("imageio not available, trying PIL")
    except Exception as e:
        logger.error(f"imageio GIF creation failed: {e}")

    # Fallback to PIL
    try:
        from PIL import Image
        pil_images = [Image.fromarray(image) for image in images]
        gif_buffer = io.BytesIO()
        pil_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=pil_images[1:],
            loop=0,
            duration=int(1000 / fps),
        )
        logger.info("Successfully created full routine GIF with PIL")
        return gif_buffer.getvalue()
    except ImportError:
        logger.warning("PIL not available")
    except Exception as e:
        logger.error(f"PIL GIF creation failed: {e}")

    logger.error("All GIF creation methods failed for full routine")
    return None