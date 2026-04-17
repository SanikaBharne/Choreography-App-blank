import os

try:
    from moviepy import VideoFileClip, vfx
except ImportError:
    from moviepy.editor import VideoFileClip, vfx

USES_CLASS_BASED_EFFECTS = not hasattr(vfx, "speedx")
if USES_CLASS_BASED_EFFECTS and hasattr(vfx, "MultiplySpeed"):
    vfx.speedx = vfx.MultiplySpeed
if USES_CLASS_BASED_EFFECTS and hasattr(vfx, "MirrorX"):
    vfx.mirror_x = vfx.MirrorX


def _is_real_moviepy_clip(clip):
    return clip.__class__.__module__.startswith("moviepy")

def load_video(filepath):
    """Load a video file from disk.

    Args:
        filepath: Path to the video file to open.

    Returns:
        A ``VideoFileClip`` instance for the requested file.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at: {filepath}")
    return VideoFileClip(filepath)

def change_speed(clip, speed_multiplier):
    """Return a clip with adjusted playback speed.

    Args:
        clip: The MoviePy clip to modify.
        speed_multiplier: Factor used to speed up or slow down playback.

    Returns:
        A new clip with the speed effect applied.

    Raises:
        ValueError: If ``speed_multiplier`` is less than or equal to zero.
    """
    if speed_multiplier <= 0:
        raise ValueError("Speed must be greater than 0")
    if USES_CLASS_BASED_EFFECTS and _is_real_moviepy_clip(clip):
        return clip.with_effects([vfx.MultiplySpeed(speed_factor=speed_multiplier)])
    return clip.fx(vfx.speedx, speed_multiplier)

def loop_section(clip, start_sec, end_sec):
    """Extract a section of a clip for looping.

    Args:
        clip: The source MoviePy clip.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.

    Returns:
        A subclip covering the requested time range.

    Raises:
        ValueError: If either loop point is out of bounds or the range is invalid.
    """
    if start_sec < 0 or end_sec > clip.duration:
        raise ValueError(f"Loop points must be within video duration (0 to {clip.duration:.2f}s)")
    if start_sec >= end_sec:
        raise ValueError("Start must be before end")
    if hasattr(clip, "subclipped") and _is_real_moviepy_clip(clip):
        return clip.subclipped(start_sec, end_sec)
    return clip.subclip(start_sec, end_sec)

def get_frame(clip, timestamp_sec):
    """Fetch a single frame from the clip.

    Args:
        clip: The MoviePy clip to sample.
        timestamp_sec: Timestamp in seconds for the desired frame.

    Returns:
        The frame data returned by MoviePy for the given timestamp.

    Raises:
        ValueError: If ``timestamp_sec`` falls outside the clip duration.
    """
    if timestamp_sec < 0 or timestamp_sec > clip.duration:
        raise ValueError(f"Timestamp out of range (0 to {clip.duration:.2f}s)")
    return clip.get_frame(timestamp_sec)

def step_frames(clip, current_sec, direction="forward"):
    """Move one frame forward or backward from the current time.

    Args:
        clip: The MoviePy clip whose frame rate defines the step size.
        current_sec: Current playback time in seconds.
        direction: Either ``"forward"`` or ``"backward"``.

    Returns:
        The new playback time after moving one frame, clamped to valid bounds.

    Raises:
        ValueError: If ``direction`` is not ``"forward"`` or ``"backward"``.
    """
    frame_duration = 1 / clip.fps
    if direction == "forward":
        return min(current_sec + frame_duration, clip.duration)
    elif direction == "backward":
        return max(current_sec - frame_duration, 0)
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

def mirror_video(clip):
    """Flip a video horizontally.

    Args:
        clip: The MoviePy clip to mirror.

    Returns:
        A horizontally mirrored version of the clip.
    """
    if USES_CLASS_BASED_EFFECTS and _is_real_moviepy_clip(clip):
        return clip.with_effects([vfx.MirrorX()])
    return clip.fx(vfx.mirror_x)
