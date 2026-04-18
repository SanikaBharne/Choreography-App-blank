from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from pydub import AudioSegment

import audio_utils
import beat_detection
import pose_estimation
import scratch_beat_detection
import source_separation
import video_controls

from dance_generator import generate_choreography, calculate_choreography_stats
from pose_sequences import PoseSequenceGenerator, create_step_teaching_sequence

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
ANALYSIS_DIR = WORKSPACE_DIR / "analysis"
SEPARATED_DIR = WORKSPACE_DIR / "separated"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def ensure_workspace():
    for directory in (WORKSPACE_DIR, UPLOADS_DIR, ANALYSIS_DIR, SEPARATED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file):
    ensure_workspace()
    destination = UPLOADS_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def save_upload_bytes(filename, content):
    ensure_workspace()
    destination = UPLOADS_DIR / filename
    destination.write_bytes(content)
    return destination


def is_video_file(file_path):
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path):
    ensure_workspace()
    output_path = ANALYSIS_DIR / f"{Path(video_path).stem}_audio.wav"
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError("The uploaded video does not contain an audio track.")
        clip.audio.write_audiofile(str(output_path), logger=None)
    return output_path


def prepare_audio_source(file_path):
    path = Path(file_path)
    if path.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if is_video_file(path):
        audio_path = extract_audio_from_video(path)
        media_type = "video"
    else:
        audio_path = path
        media_type = "audio"
    return audio_path, media_type


def analyze_media(file_path, subdivisions=2):
    return analyze_media_with_method(file_path, beat_method="librosa", subdivisions=subdivisions)


def analyze_media_with_method(file_path, beat_method="librosa", subdivisions=2):
    audio_path, media_type = prepare_audio_source(file_path)
    audio, sr = audio_utils.load_audio(str(audio_path))
    beats = detect_beats_for_method(audio, sr, beat_method)
    grouped_counts = beat_detection.eight_count_grouping(beats, subdivisions=subdivisions)
    duration = float(librosa.get_duration(y=audio, sr=sr))

    return build_analysis_result(
        audio_path=audio_path,
        media_type=media_type,
        beats=beats,
        grouped_counts=grouped_counts,
        duration=duration,
        sr=sr,
        beat_method=beat_method,
        subdivisions=subdivisions,
    )


def build_analysis_result(audio_path, media_type, beats, grouped_counts, duration, sr, beat_method, subdivisions):
    return {
        "audio_path": str(audio_path),
        "media_type": media_type,
        "beats": beats,
        "grouped_counts": grouped_counts,
        "duration": duration,
        "sample_rate": sr,
        "tempo_estimate": estimate_tempo(beats, duration),
        "beat_rows": build_beat_rows(grouped_counts),
        "timeline_rows": build_timeline_rows(beats, duration),
        "subdivisions": subdivisions,
        "beat_method": beat_method,
    }


def detect_beats_for_method(audio, sr, beat_method):
    method = beat_method.lower()
    if method == "librosa":
        return beat_detection.detect_beats(audio, sr)
    if method == "scratch":
        return scratch_beat_detection.detect_beats(audio, sr)
    raise ValueError(f"Unsupported beat method: {beat_method}")


def estimate_tempo(beats, duration):
    if duration <= 0 or len(beats) < 2:
        return None
    intervals = np.diff(np.asarray(beats, dtype=float))
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        return None
    return round(60.0 / float(np.median(intervals)), 1)


def build_beat_rows(grouped_counts):
    rows = []
    for group_index, group in enumerate(grouped_counts, start=1):
        for position, (timestamp, is_beat) in enumerate(group, start=1):
            rows.append(
                {
                    "eight_count": group_index,
                    "position": position,
                    "timestamp_seconds": round(float(timestamp), 3),
                    "kind": "beat" if is_beat else "subdivision",
                }
            )
    return rows


def build_timeline_rows(beats, duration, bucket_count=48):
    if duration <= 0:
        return []
    buckets = np.linspace(0, duration, bucket_count + 1)
    counts, edges = np.histogram(beats, bins=buckets)
    rows = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "window_start": round(float(edges[index]), 2),
                "beat_count": int(count),
            }
        )
    return rows


def save_pose_image(image, output_name):
    ensure_workspace()
    output_path = ANALYSIS_DIR / output_name
    Image.fromarray(image).save(output_path)
    return output_path


def save_pose_animation(animation_bytes, output_name):
    ensure_workspace()
    output_path = ANALYSIS_DIR / output_name
    with open(output_path, "wb") as fp:
        fp.write(animation_bytes)
    return output_path


def generate_choreography_plan(file_path, beat_method="librosa", subdivisions=2,
                                difficulty="easy", energy_preference=None,
                                pose_preview_count=6, max_steps=10):
    analysis = analyze_media_with_method(file_path, beat_method=beat_method, subdivisions=subdivisions)
    tempo = analysis["tempo_estimate"] if analysis["tempo_estimate"] is not None else 100.0
    choreography = generate_choreography(
        analysis["beats"],
        tempo_bpm=tempo,
        difficulty=difficulty,
        energy_preference=energy_preference,
        max_steps=max_steps,
    )

    pose_generator = PoseSequenceGenerator()
    enriched_steps = []
    media_id = Path(file_path).stem

    for index, step in enumerate(choreography, start=1):
        step_sequence = pose_generator.generate_step_sequence(step_id=step["id"], duration_beats=step["beats"])
        teaching = create_step_teaching_sequence(step_sequence)

        pose_gallery_paths = []
        for image_index, pose_image in enumerate(teaching.get("pose_gallery", []), start=1):
            filename = f"{media_id}_{step['id']}_pose_{image_index}.png"
            pose_gallery_paths.append(save_pose_image(pose_image, filename))

        animation_path = None
        if teaching.get("skeleton_animation"):
            animation_name = f"{media_id}_{step['id']}_animation.gif"
            animation_path = save_pose_animation(teaching["skeleton_animation"], animation_name)

        enriched_steps.append({
            **step,
            "pose_gallery_paths": pose_gallery_paths,
            "skeleton_animation_path": animation_path,
        })

    return {
        "analysis": analysis,
        "choreography": enriched_steps,
        "choreography_stats": calculate_choreography_stats(choreography),
    }


def separate_stems(file_path):
    ensure_workspace()
    audio_path, _ = prepare_audio_source(file_path)
    stems = source_separation.separate(str(audio_path), output_dir=str(SEPARATED_DIR))
    required = ("vocals", "drums", "bass", "other")
    missing = [stem for stem in required if stem not in stems]
    if missing:
        raise RuntimeError(
            "Stem separation did not produce all required tracks: "
            + ", ".join(missing)
        )
    instrumental_path = source_separation.get_instrumental(
        stems,
        output_path=str(SEPARATED_DIR / f"{Path(file_path).stem}_instrumental.wav"),
    )
    ordered_stems = {stem: stems[stem] for stem in required}
    return ordered_stems, instrumental_path


def change_audio_speed(audio_segment, speed_multiplier):
    if speed_multiplier <= 0:
        raise ValueError("Speed multiplier must be positive")
    if speed_multiplier == 1.0:
        return audio_segment

    new_frame_rate = int(audio_segment.frame_rate * speed_multiplier)
    sped_audio = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": new_frame_rate})
    return sped_audio.set_frame_rate(audio_segment.frame_rate)


def create_practice_audio(file_path, start_sec, end_sec, speed_multiplier=1.0):
    ensure_workspace()
    source = AudioSegment.from_file(str(file_path))
    duration_seconds = len(source) / 1000.0

    if start_sec < 0 or end_sec > duration_seconds:
        raise ValueError(f"Loop points must be within audio duration (0 to {duration_seconds:.2f}s)")
    if start_sec >= end_sec:
        raise ValueError("Start must be before end")

    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    clip = source[start_ms:end_ms]
    clip = change_audio_speed(clip, speed_multiplier)

    output_name = (
        f"{Path(file_path).stem}_practice_"
        f"{int(start_sec * 1000)}_{int(end_sec * 1000)}_"
        f"{int(speed_multiplier * 100)}.wav"
    )
    output_path = ANALYSIS_DIR / output_name
    clip.export(str(output_path), format="wav")
    return output_path


def create_practice_media(file_path, start_sec, end_sec, speed_multiplier=1.0, mirror=False):
    if is_video_file(file_path):
        return create_practice_video(file_path, start_sec, end_sec, speed_multiplier=speed_multiplier, mirror=mirror)
    return create_practice_audio(file_path, start_sec, end_sec, speed_multiplier=speed_multiplier)


def get_video_metadata(file_path):
    with VideoFileClip(str(file_path)) as clip:
        return {
            "duration": float(clip.duration),
            "fps": float(clip.fps),
            "size": clip.size,
        }


def create_practice_video(file_path, start_sec, end_sec, speed_multiplier=1.0, mirror=False):
    ensure_workspace()
    output_name = (
        f"{Path(file_path).stem}_practice_"
        f"{int(start_sec * 1000)}_{int(end_sec * 1000)}_"
        f"{int(speed_multiplier * 100)}_{'mirror' if mirror else 'plain'}.mp4"
    )
    output_path = ANALYSIS_DIR / output_name

    with VideoFileClip(str(file_path)) as clip:
        section = video_controls.loop_section(clip, start_sec, end_sec)
        adjusted = video_controls.change_speed(section, speed_multiplier)
        if mirror:
            adjusted = video_controls.mirror_video(adjusted)
        adjusted.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

    return output_path


def extract_frame_for_preview(file_path, frame_index):
    with VideoFileClip(str(file_path)) as clip:
        fps = float(clip.fps)
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        max_index = max(0, int(np.floor(clip.duration * fps)))
        safe_index = min(max(frame_index, 0), max_index)
        timestamp = min(safe_index / fps, clip.duration)
        frame = video_controls.get_frame(clip, timestamp)
        return {
            "frame": frame,
            "timestamp": round(timestamp, 3),
            "frame_index": safe_index,
            "fps": fps,
            "max_index": max_index,
        }



def analyze_pose(video_path, sample_count=4, timestamps=None):
    if timestamps is not None and len(timestamps) > 0:
        pose_results = pose_estimation.analyze_video_pose(video_path, timestamps=timestamps)
    else:
        pose_results = pose_estimation.analyze_video_pose(video_path, sample_count=sample_count)
    detected = [result for result in pose_results if result["has_pose"]]
    summaries = [result["summary"] for result in detected]

    pose_rows = []
    for result in pose_results:
        if result["summary"] is None:
            pose_rows.append(
                {
                    "timestamp": result["timestamp"],
                    "pose_detected": False,
                    "visible_landmarks": 0,
                    "visibility_score": 0.0,
                    "left_arm_angle": None,
                    "right_arm_angle": None,
                    "left_leg_angle": None,
                    "right_leg_angle": None,
                    "shoulder_tilt": None,
                    "hip_tilt": None,
                }
            )
            continue
        pose_rows.append(
            {
                "timestamp": result["timestamp"],
                "pose_detected": True,
                **result["summary"],
            }
        )

    overview = {
        "sample_count": len(pose_results),
        "detected_frames": len(detected),
        "average_visibility": round(
            float(np.mean([summary["visibility_score"] for summary in summaries])),
            3,
        ) if summaries else 0.0,
        "average_shoulder_tilt": round(
            float(np.mean([summary["shoulder_tilt"] for summary in summaries])),
            1,
        ) if summaries else None,
        "average_hip_tilt": round(
            float(np.mean([summary["hip_tilt"] for summary in summaries])),
            1,
        ) if summaries else None,
    }

    return {
        "frames": pose_results,
        "rows": pose_rows,
        "overview": overview,
    }


def save_overlay_frame(frame, output_name):
    ensure_workspace()
    output_path = ANALYSIS_DIR / output_name
    Image.fromarray(frame).save(output_path)
    return output_path


def format_timestamp(seconds):
    """Convert seconds to M:SS format."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

def summarize_eight_counts(grouped_counts, limit=8):
    summaries = []
    for index, group in enumerate(grouped_counts[:limit], start=1):
        formatted = []
        for timestamp, is_beat in group:
            ts = float(timestamp)
            label = f"{format_timestamp(ts)} ({ts:.2f}s)"
            if not is_beat:
                label += " (sub)"
            formatted.append(label)
        summaries.append(
            {
                "label": f"Eight-count {index}",
                "values": ", ".join(formatted),
            }
        )
    return summaries
