# Choreo App

A tool for learning/creating choreography from video or songs. Instead of endlessly rewinding, this app gives you precise controls to slow down, loop, and analyze movement in any video file.

## Features
- **🤖 AI Dance Generation**: Upload a song and get AI-generated choreography automatically
- **📊 Instant Beat Analysis**: Get tempo, beat timing, and rhythm breakdowns
- **🎵 Step-by-Step Learning**: Master each dance step with guided animations and tips
- **⏱️ Practice Loops**: Loop over custom start/stop points for both video and audio uploads
- **🎬 Frame-by-Frame Stepping**: Analyze movement details in uploaded videos
- **🧘 Pose Estimation**: Detect body landmarks and posture angles
- **🎙️ Audio Stem Separation**: Isolate vocals, drums, bass, and other tracks
- **🔄 Playback Speed Control**: Practice at custom speeds (0.25x to 2x)
- Soft light Streamlit UI for local upload, playback, and rhythm review
- FastAPI backend for uploads, analysis, practice clips, and pose endpoints
- Flutter frontend scaffold for mobile/desktop API-driven playback and review

## How It Works

1. **Upload** a song or dance video
2. **Beat Analysis** runs automatically
3. **AI Generates** a custom choreography routine
4. **Learn Steps** with guided animations (full speed, slow motion, count breakdown)
5. **Practice** with loop controls and frame stepping

## Step Library

The AI uses a comprehensive library of 30 dance steps across 3 difficulty levels:

### Basic (Easy)
- Right/Left Step Touch, March in Place, Clap variations, Basic Bounce, Shoulder Bounce

### Intermediate (Medium) 
- Grapevine steps, Side Kicks, Arm Circles, Cross Punches, Body Rolls, Hip Sways

### Advanced (Hard)
- Jump Steps, High Knees, Wave Combos, Chest Pops, Jump + Clap, Spin + Pose

Each step includes:
- Duration in beats
- Energy level (low/medium/high)
- Body parts involved
- Teaching descriptions and pose hints

## Tech Stack
- Python + FastAPI (backend)
- Flutter (mobile/desktop frontend, coming later)
- Librosa, MediaPipe, PyTorch (ML features)
- Custom rule-based choreography generation
- MediaPipe pose sequences for step animations

## Setup
```bash
# clone the repo
git clone https://github.com/Neel-Karkhanis/choreo-app.git
cd choreo-app

# create and activate virtual environment
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run the Streamlit app
streamlit run streamlit_app.py

# run the FastAPI backend
uvicorn backend.app.main:app --reload
```

## System Prerequisites
- `ffmpeg` must be installed and available on `PATH` for audio/video processing.
- Demucs downloads model weights on first use, so internet access is required the first time source separation runs.
- MediaPipe requires camera permissions for pose estimation (webcam access)
- The current tests expect local sample media under `test_files/`, which is not included in this repository.

## AI Dance Generation Details

### Step Library Architecture
- **30 Steps** across 3 difficulty levels (10 basic, 10 intermediate, 10 advanced)
- Each step has metadata: beats, difficulty, energy, body parts, description, pose hints
- Tempo-aware selection (slow/medium/fast tempo ranges)

### Choreography Generation
- Rule-based algorithm maps beats to appropriate steps
- Considers tempo, difficulty preference, and energy level
- Generates 6-20 step routines based on song length
- Maintains variety and natural flow

### Pose Sequence Animation
- MediaPipe skeleton format for all animations
- 30+ pose generation functions for different step types
- Teaching sequences: full speed, slow motion, count breakdown
- Key joint highlighting for learning focus

### Learning Interface
- **Full Demo**: Watch complete step at normal speed
- **Slow Motion**: 0.5x speed for following along
- **Count Breakdown**: 8-count step-by-step guidance
- **Tips**: Contextual teaching advice based on step type

## App Workflow
- Upload an audio file or dance video in the Streamlit UI.
- Choose either the `librosa` beat engine or the scratch-built detector.
- Review the media preview, beat density chart, and eight-count grouping.
- For videos, generate slowed loop previews and step through frames.
- Optionally sample video frames for pose estimation and posture overlays.
- Optionally run stem separation to generate vocals, drums, bass, other, and instrumental previews.

## API
- `POST /api/media/upload` uploads audio or video and returns a `media_id`.
- `POST /api/media/{media_id}/analysis` runs beat analysis with `librosa` or `scratch`.
- `POST /api/media/{media_id}/practice-clip` generates a slowed loop clip for videos.
- `POST /api/media/{media_id}/pose` runs sampled pose estimation for videos.
- `POST /api/media/{media_id}/choreography` generates step-by-step AI choreography with skeleton GIFs and pose preview images.
- `GET /media/...` serves uploaded and generated files from the workspace.

## Flutter Client
- Flutter client files live under `frontend/flutter_app/`.
- Update the base URL in the app to point at your FastAPI server, then run it with the normal Flutter workflow on a machine that has `flutter` installed.

## Project Structure
```
choreo-app/
├── video_controls.py   # speed, looping, frame stepping
├── README.md
└── requirements.txt
```

## Roadmap
- [x] Video controls (speed, loop, frame stepping)
- [x] Beat detection from scratch
- [x] Streamlit UI
- [x] Pose estimation
- [x] FastAPI backend
- [x] Flutter frontend

