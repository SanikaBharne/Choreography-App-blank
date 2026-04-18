from pathlib import Path
import sys
import base64
import mimetypes

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app_services
import video_controls
import dance_generator
import pose_sequences


st.set_page_config(
    page_title="Choreo App",
    page_icon="C",
    layout="wide",
)


def apply_soft_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #fff8f3 0%, transparent 34%),
                radial-gradient(circle at top right, #eef8ff 0%, transparent 28%),
                linear-gradient(180deg, #fdfbf8 0%, #f7f3ee 100%);
            color: #43342f;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .hero-card, .soft-card {
            background: rgba(255, 252, 248, 0.82);
            border: 1px solid rgba(199, 178, 167, 0.35);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 40px rgba(109, 88, 77, 0.08);
            backdrop-filter: blur(8px);
        }
        .hero-title {
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0 0 0.45rem 0;
            color: #3a2320;
            font-weight: 700;
            text-shadow: 0 1px 4px #fff8f3;
        }
        .hero-copy {
            color: #3a2320;
            font-size: 1.08rem;
            margin: 0;
            text-shadow: 0 1px 4px #fff8f3;
        }
        .section-label {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b4f45;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            background: #fff6f0;
            padding: 0.2em 0.7em;
            border-radius: 8px;
            box-shadow: 0 1px 4px #f7e6d8;
        }
        div[data-testid="stMetric"] {
            background: #fff6f0;
            border: 1px solid #e0cfc2;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            color: #3a2320;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fffaf6 0%, #f7efe8 100%);
            border-right: 1px solid rgba(201, 185, 174, 0.45);
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid #d2b4a8;
            background: linear-gradient(180deg, #f3d8cf 0%, #eccabf 100%);
            color: #3a2320;
            font-weight: 600;
            box-shadow: 0 2px 8px #f7e6d8;
        }
        .stButton.big-button > button {
            height: 48px;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .stDownloadButton > button {
            border-radius: 999px;
        }
        .beat-summary {
            background: #fff6f0;
            border: 1px solid #e0cfc2;
            border-radius: 16px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: #3a2320;
        }
        .beat-stat {
            font-size: 1.25rem;
            font-weight: 700;
            color: #3a2320;
        }
        .beat-stat-label {
            font-size: 0.85rem;
            color: #6b4f45;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        /* Ensure all text is visible on light backgrounds */
        .stApp, .block-container, .hero-card, .soft-card, .beat-summary, .stButton > button, .stMetric, .section-label {
            color: #3a2320 !important;
        }
        /* Fix for Streamlit's default text color on some widgets */
        .stTextInput > div > input, .stSelectbox > div > div, .stSlider > div, .stRadio > div, .stCheckbox > label, .stCaption, .stDataFrame, .stExpanderHeader {
            color: #3a2320 !important;
        }
        /* Increase font size for better readability */
        body, .stApp, .block-container {
            font-size: 1.08rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Choreography Practice</div>
            <h1 class="hero-title">Choreo App</h1>
            <p class="hero-copy">
                Upload a song or dance video, get instant beat analysis, practice loops at custom speeds,
                and let AI generate choreography automatically. Master choreography with precision control.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### Choreography Settings")
    with st.sidebar.expander("🎵 Beat detection", expanded=True):
        beat_method = st.radio(
            "Pick your beat engine",
            options=["librosa", "scratch"],
            help="Librosa = more accurate library, Scratch = lightweight custom detector.",
        )
        subdivisions = st.slider("Beats per section", min_value=1, max_value=4, value=2, help="1 = only main beats, 2+ = add subdivisions between beats.")
    with st.sidebar.expander("⚙️ Advanced options", expanded=False):
        run_separation = st.checkbox("🎙️ Isolate audio stems", value=False, help="Split audio into vocals, drums, bass, and other tracks.")
        run_pose_estimation = st.checkbox("🧘 Track body position", value=False, help="Detect shoulders, elbows, hips, and posture angles.")
        if run_pose_estimation:
            pose_sample_count = st.slider("Frames to analyze", min_value=2, max_value=12, value=4, help="More frames = better detail, slower processing.")
            pose_timing = st.selectbox(
                "When to sample",
                options=["Evenly spaced", "At detected beats"],
                help="Evenly = spread throughout video, At beats = capture key rhythm moments.",
            )
            st.caption("💡 Longer videos? Use 8+ frames. Short clips? 2-4 is fine.")
        else:
            pose_sample_count = 4
            pose_timing = "Evenly spaced"
    return beat_method, subdivisions, run_separation, run_pose_estimation, pose_sample_count, pose_timing


def render_upload_help():
    st.markdown(
        """
        <div class="soft-card">
            <div class="section-label">Your practice studio</div>
            <b>Upload</b> → Beat analysis starts automatically → <b>AI generates choreography</b> → Learn step-by-step → Practice and perfect
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_media_preview(results, original_file):
    st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
    if results["media_type"] == "video":
        left, right = st.columns([1.15, 0.85])
        with left:
            st.video(str(original_file))
        with right:
            st.audio(str(results["audio_path"]))
            st.caption("Extracted audio track used for beat analysis.")
    else:
        st.audio(str(results["audio_path"]))


def render_metrics(results):
    columns = st.columns(4)
    columns[0].metric("Duration", f'{results["duration"]:.1f}s')
    columns[1].metric("Detected beats", str(len(results["beats"])))
    columns[2].metric("Estimated tempo", f'{results["tempo_estimate"]} BPM' if results["tempo_estimate"] else "N/A")
    columns[3].metric("Beat engine", results["beat_method"].title())


def render_beat_summary(results):
    """Display key beat metrics in an easy-to-scan summary format."""
    st.markdown('<div class="section-label">Beat Summary</div>', unsafe_allow_html=True)
    
    # Key stats in a grid
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.markdown(
            f"""
            <div class="beat-summary">
                <div class="beat-stat">{len(results['beats'])} beats</div>
                <div class="beat-stat-label">Found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with stat_col2:
        tempo = f"{results['tempo_estimate']} BPM" if results["tempo_estimate"] else "N/A"
        st.markdown(
            f"""
            <div class="beat-summary">
                <div class="beat-stat">{tempo}</div>
                <div class="beat-stat-label">Tempo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with stat_col3:
        st.markdown(
            f"""
            <div class="beat-summary">
                <div class="beat-stat">{results['duration']:.1f}s</div>
                <div class="beat-stat-label">Duration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Beat density visualization
    st.markdown("#### Beat distribution")
    st.bar_chart(results["timeline_rows"], x="window_start", y="beat_count", use_container_width=True)


def render_analysis(results):
    st.markdown('<div class="section-label">Beat Analysis</div>', unsafe_allow_html=True)
    st.caption("🎵 Quick look at your clip's rhythm and timing.")
    render_beat_summary(results)
    
    with st.expander("📊 Detailed breakdown", expanded=False):
        st.markdown("#### Beat timeline")
        st.dataframe(results["beat_rows"], use_container_width=True, hide_index=True)
        
        st.markdown("#### Eight-count breakdown")
        summary_columns = st.columns(2)
        summaries = app_services.summarize_eight_counts(results["grouped_counts"], limit=8)
        for index, summary in enumerate(summaries):
            with summary_columns[index % 2]:
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="section-label">{summary["label"]}</div>
                        {summary["values"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_stems(original_file):
    st.markdown('<div class="section-label">Stem Separation</div>', unsafe_allow_html=True)
    st.caption("🎙️ Separate audio into vocals, drums, bass, and other instruments.")
    try:
        stems, instrumental_path = app_services.separate_stems(original_file)
    except Exception as exc:
        st.warning(
            "Stem separation could not complete in this environment. "
            f"Details: {exc}"
        )
        return

    stem_columns = st.columns(2)
    stem_items = list(stems.items()) + [("instrumental", instrumental_path)]
    for index, (stem_name, stem_path) in enumerate(stem_items):
        with stem_columns[index % 2]:
            st.markdown(f"#### {stem_name.title()}")
            st.audio(stem_path)
            st.caption(stem_path)


def render_pose_estimation(original_file, pose_results):
    st.markdown('<div class="section-label">Pose Estimation</div>', unsafe_allow_html=True)
    st.caption("🧘 Detected body position and posture landmarks in your video.")
    overview = pose_results["overview"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Sampled frames", str(overview["sample_count"]))
    metric_columns[1].metric("Pose detections", str(overview["detected_frames"]))
    metric_columns[2].metric("Avg visibility", str(overview["average_visibility"]))
    metric_columns[3].metric(
        "Avg shoulder tilt",
        f'{overview["average_shoulder_tilt"]} deg' if overview["average_shoulder_tilt"] is not None else "N/A",
    )

    st.dataframe(pose_results["rows"], use_container_width=True, hide_index=True)

    overlay_columns = st.columns(2)
    for index, frame_result in enumerate(pose_results["frames"]):
        with overlay_columns[index % 2]:
            st.markdown(f"#### {frame_result['timestamp']}s")
            st.image(frame_result["overlay_frame"], use_container_width=True, clamp=True)
            if frame_result["summary"] is None:
                st.caption("No pose detected in this frame.")
            else:
                summary = frame_result["summary"]
                st.caption(
                    f"Visible landmarks: {summary['visible_landmarks']} | "
                    f"Left arm: {summary['left_arm_angle']} deg | "
                    f"Right arm: {summary['right_arm_angle']} deg"
                )


def render_dance_generation(results, original_file, metadata=None):
    """Render AI dance generation interface."""
    st.markdown('<div class="section-label">🤖 AI Dance Generation</div>', unsafe_allow_html=True)
    st.caption("Generate choreography automatically from your song's beat analysis!")

    file_key = Path(original_file).name
    choreography_key = f"choreography::{file_key}"
    learning_key = f"learning_step::{file_key}"

    if st.button("🎵 Generate Dance Routine", type="primary", use_container_width=True):
        with st.spinner("🎭 Creating your custom choreography..."):
            try:
                choreography = dance_generator.generate_choreography(
                    beats=results["beats"],
                    tempo_bpm=results.get("tempo_estimate", 120),
                    difficulty="easy",
                    max_steps=8,
                )
                st.session_state[choreography_key] = choreography
                if learning_key in st.session_state:
                    del st.session_state[learning_key]
                st.success(f"✅ Generated {len(choreography)} steps for your routine!")
            except Exception as exc:
                st.error(f"Could not generate choreography: {exc}")

    choreography = st.session_state.get(choreography_key)
    if choreography:
        render_choreography_summary(
            choreography,
            key_prefix=file_key,
            learning_state_key=learning_key,
        )
        if learning_key in st.session_state:
            step = st.session_state[learning_key]
            render_single_step_learning(step)
            if st.button("⬅️ Back to Routine", key=f"back_from_learning::{file_key}"):
                del st.session_state[learning_key]
                st.rerun()


def render_choreography_summary(choreography, key_prefix="default", learning_state_key=None):
    """Display summary of generated choreography."""
    if not choreography:
        return

    st.markdown("#### Your Custom Routine")
    stats = dance_generator.calculate_choreography_stats(choreography)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Steps", stats["total_steps"])
    with col2:
        st.metric("Duration", f"{stats['total_duration']:.1f}s")
    with col3:
        st.metric("Beats", stats["total_beats"])
    with col4:
        st.metric("Avg Step", f"{stats['avg_step_duration']:.1f}s")

    st.markdown("#### 🎬 Complete Routine Animation")
    routine_signature = tuple(
        (
            step.get("id"),
            step.get("beats"),
            step.get("start_time"),
            step.get("end_time"),
        )
        for step in choreography
    )
    animation_cache_key = f"routine_animation::{key_prefix}"
    cached_animation = st.session_state.get(animation_cache_key)
    try:
        if not cached_animation or cached_animation.get("signature") != routine_signature:
            full_animation = pose_sequences.create_full_routine_animation(choreography, fps=8)
            st.session_state[animation_cache_key] = {
                "signature": routine_signature,
                "data": full_animation,
            }
        else:
            full_animation = cached_animation.get("data")
    except Exception as exc:
        full_animation = None
        st.error(f"Could not generate full routine animation: {exc}")

    try:
        if full_animation:
            st.image(full_animation, use_container_width=True)
            st.caption("Watch your complete dance routine from start to finish!")
        else:
            st.info("🎬 Full routine animation could not be generated.")
    except Exception as exc:
        st.error(f"Could not generate full routine animation: {exc}")

    st.markdown("#### Step Sequence")
    for i, step in enumerate(choreography, 1):
        with st.expander(f"Step {i}: {step['name']}", ex
(Content truncated due to size limit. Use line ranges to read remaining content)
