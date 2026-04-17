from pathlib import Path
import sys
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app_services

from backend.app.schemas import AnalysisRequest, ChoreographyRequest, PoseRequest, PracticeClipRequest
from backend.app.serializers import (
    serialize_analysis_result,
    serialize_choreography_result,
    serialize_media_asset,
    serialize_pose_result,
    workspace_url,
)


app = FastAPI(title="Choreo App API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_services.ensure_workspace()
app.mount("/media", StaticFiles(directory=app_services.WORKSPACE_DIR), name="media")


def _build_media_filename(original_name):
    safe_name = Path(original_name).name
    return f"{uuid4().hex}_{safe_name}"


def _resolve_media_path(media_id):
    candidate = app_services.UPLOADS_DIR / media_id
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Media not found")
    return candidate


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/media/upload")
async def upload_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename = _build_media_filename(file.filename)
    content = await file.read()
    saved_path = app_services.save_upload_bytes(filename, content)
    return serialize_media_asset(filename, saved_path)


@app.get("/api/media/{media_id}")
def get_media(media_id: str):
    media_path = _resolve_media_path(media_id)
    return serialize_media_asset(media_id, media_path)


@app.post("/api/media/{media_id}/analysis")
def analyze_media(media_id: str, request: AnalysisRequest):
    media_path = _resolve_media_path(media_id)
    try:
        result = app_services.analyze_media_with_method(
            media_path,
            beat_method=request.beat_method,
            subdivisions=request.subdivisions,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_analysis_result(result)


@app.get("/api/media/{media_id}/metadata")
def get_media_metadata(media_id: str):
    media_path = _resolve_media_path(media_id)
    payload = serialize_media_asset(media_id, media_path)
    if app_services.is_video_file(media_path):
        payload["video_metadata"] = app_services.get_video_metadata(media_path)
    return payload


@app.post("/api/media/{media_id}/practice-clip")
def create_practice_clip(media_id: str, request: PracticeClipRequest):
    media_path = _resolve_media_path(media_id)

    if request.end_sec <= request.start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")

    try:
        preview_path = app_services.create_practice_media(
            media_path,
            start_sec=request.start_sec,
            end_sec=request.end_sec,
            speed_multiplier=request.speed_multiplier,
            mirror=request.mirror,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "preview_url": workspace_url(preview_path),
        "filename": Path(preview_path).name,
    }


@app.post("/api/media/{media_id}/pose")

def analyze_pose(media_id: str, request: PoseRequest):
    media_path = _resolve_media_path(media_id)
    if not app_services.is_video_file(media_path):
        raise HTTPException(status_code=400, detail="Pose estimation is only available for video uploads")

    try:
        if request.timestamps is not None and len(request.timestamps) > 0:
            pose_result = app_services.analyze_pose(media_path, timestamps=request.timestamps)
        else:
            pose_result = app_services.analyze_pose(media_path, sample_count=request.sample_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return serialize_pose_result(media_id, pose_result)


@app.post("/api/media/{media_id}/choreography")
def generate_choreography(media_id: str, request: ChoreographyRequest):
    media_path = _resolve_media_path(media_id)
    try:
        choreography_result = app_services.generate_choreography_plan(
            media_path,
            beat_method=request.beat_method,
            subdivisions=request.subdivisions,
            difficulty=request.difficulty,
            energy_preference=request.energy_preference,
            pose_preview_count=request.pose_preview_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return serialize_choreography_result(media_id, choreography_result)
