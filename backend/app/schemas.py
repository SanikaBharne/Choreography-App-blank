from typing import Literal

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    beat_method: Literal["librosa", "scratch"] = "librosa"
    subdivisions: int = Field(default=2, ge=1, le=4)


class PracticeClipRequest(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    speed_multiplier: float = Field(default=1.0, gt=0)
    mirror: bool = False



from typing import List, Optional

class PoseRequest(BaseModel):
    sample_count: int = Field(default=4, ge=1, le=12)
    timestamps: Optional[List[float]] = None  # List of seconds for pose estimation


class ChoreographyRequest(BaseModel):
    beat_method: Literal["librosa", "scratch"] = "librosa"
    subdivisions: int = Field(default=2, ge=1, le=4)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    energy_preference: Literal["low", "medium", "high"] | None = None
    pose_preview_count: int = Field(default=6, ge=1, le=12)
