from pathlib import Path

import app_services


def workspace_url(file_path):
    path = Path(file_path).resolve()
    relative = path.relative_to(app_services.WORKSPACE_DIR.resolve()).as_posix()
    return f"/media/{relative}"


def serialize_media_asset(media_id, file_path):
    path = Path(file_path)
    return {
        "media_id": media_id,
        "filename": path.name,
        "media_type": "video" if app_services.is_video_file(path) else "audio",
        "original_url": workspace_url(path),
    }


def serialize_analysis_result(result):
    grouped_counts = []
    for group_index, group in enumerate(result["grouped_counts"], start=1):
        grouped_counts.append(
            {
                "eight_count": group_index,
                "items": [
                    {
                        "timestamp_seconds": round(float(timestamp), 3),
                        "kind": "beat" if is_beat else "subdivision",
                    }
                    for timestamp, is_beat in group
                ],
            }
        )

    payload = {
        "audio_url": workspace_url(result["audio_path"]),
        "media_type": result["media_type"],
        "duration": result["duration"],
        "sample_rate": result["sample_rate"],
        "tempo_estimate": result["tempo_estimate"],
        "beat_method": result["beat_method"],
        "subdivisions": result["subdivisions"],
        "beats": [round(float(timestamp), 3) for timestamp in result["beats"]],
        "beat_rows": result["beat_rows"],
        "timeline_rows": result["timeline_rows"],
        "grouped_counts": grouped_counts,
        "eight_count_summaries": app_services.summarize_eight_counts(result["grouped_counts"]),
    }
    return payload


def _maybe_workspace_url(path):
    return workspace_url(path) if path else None


def serialize_choreography_step(step):
    return {
        "id": step["id"],
        "name": step["name"],
        "beats": step["beats"],
        "difficulty": step["difficulty"],
        "energy": step["energy"],
        "body_parts": step["body_parts"],
        "description": step["description"],
        "start_time": round(float(step["start_time"]), 3),
        "end_time": round(float(step["end_time"]), 3),
        "beat_start": step["beat_start"],
        "beat_end": step["beat_end"],
        "pose_gallery_urls": [workspace_url(path) for path in step.get("pose_gallery_paths", [])],
        "skeleton_animation_url": _maybe_workspace_url(step.get("skeleton_animation_path")),
    }


def serialize_choreography_result(media_id, result):
    payload = serialize_analysis_result(result["analysis"])
    payload["choreography"] = [serialize_choreography_step(step) for step in result["choreography"]]
    payload["choreography_stats"] = result.get("choreography_stats", {})
    return payload


def serialize_pose_result(media_id, pose_result):
    frame_payload = []
    for index, frame in enumerate(pose_result["frames"], start=1):
        overlay_path = app_services.save_overlay_frame(
            frame["overlay_frame"],
            f"{media_id}_pose_{index}.png",
        )
        frame_payload.append(
            {
                "timestamp": frame["timestamp"],
                "has_pose": frame["has_pose"],
                "overlay_url": workspace_url(overlay_path),
                "summary": frame["summary"],
            }
        )

    return {
        "overview": pose_result["overview"],
        "rows": pose_result["rows"],
        "frames": frame_payload,
    }
