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
        with st.expander(f"Step {i}: {step['name']}", expanded=False):
            details_col, action_col = st.columns([1.75, 1], gap="large")
            with details_col:
                st.write(f"**Duration:** {step['beats']} beats")
                st.write(f"**Difficulty:** {step['difficulty'].title()}")
                st.write(f"**Energy:** {step['energy'].title()}")
                st.write(f"**Body Parts:** {', '.join(step['body_parts']).title()}")
                st.write(f"**Description:** {step['description']}")
            with action_col:
                st.markdown("**Let's learn this step**")
                st.caption("Open full demo, slow motion, count breakdown, and tips.")
                if st.button("🎓 Learn this step", key=f"{key_prefix}_learn_step_{i}"):
                    target_key = learning_state_key or f"learning_step_{key_prefix}"
                    st.session_state[target_key] = step
                    st.rerun()


def render_single_step_learning(step):
    """Render learning interface for a single step."""
    st.markdown(f"## 🎭 Learn: {step['name']}")
    st.markdown(f"*{step['description']}*")
    
    # Generate pose sequence
    try:
        pose_gen = pose_sequences.PoseSequenceGenerator()
        step_sequence = pose_gen.generate_step_sequence(
            step_id=step["id"],
            duration_beats=step["beats"]
        )
        
        teaching_data = pose_sequences.create_step_teaching_sequence(step_sequence)
        
        # Teaching tabs
        tab1, tab2, tab3, tab4 = st.tabs(["👀 Full Demo", "🐢 Slow Motion", "🔢 Count Breakdown", "💡 Tips"])
        
        with tab1:
            st.markdown("**Watch the full step as a skeleton animation:**")
            if teaching_data.get("skeleton_animation"):
                st.image(teaching_data["skeleton_animation"], use_container_width=True)
            else:
                st.info("🎬 Skeleton animation is not available in this environment. Showing pose gallery instead.")

            st.caption(f"Duration: {teaching_data['full_speed']['duration']:.1f} seconds")
            st.markdown("**Pose images from the dance step:**")
            pose_gallery = teaching_data.get("pose_gallery", [])
            if pose_gallery:
                cols = st.columns(min(4, len(pose_gallery)))
                for idx, pose_image in enumerate(pose_gallery):
                    with cols[idx % len(cols)]:
                        st.image(pose_image, caption=f"Pose {idx + 1}", use_container_width=True)
            else:
                st.info("No pose preview images were generated.")
        
        with tab2:
            st.markdown("**Follow the move in slow motion:**")
            st.info("🐌 Slow motion preview is available in the teaching breakdown below.")
            st.caption(f"Duration: {teaching_data['slow_speed']['duration']:.1f} seconds")
        
        with tab3:
            st.markdown("**Break it down by counts:**")
            for beat_data in teaching_data["breakdown"]:
                st.markdown(f"**Count {beat_data['beat_number']}:** {beat_data['description']}")
                if beat_data.get("image") is not None:
                    st.image(beat_data["image"], caption=f"Count {beat_data['beat_number']} pose", use_container_width=True)
                else:
                    st.info(f"Frame preview for count {beat_data['beat_number']} is unavailable.")
        
        with tab4:
            st.markdown("**Teaching Tips:**")
            for tip in teaching_data["teaching_tips"]:
                st.markdown(f"• {tip}")
    
    except Exception as exc:
        st.error(f"Could not generate step animation: {exc}")


def render_step_learning_interface(choreography, original_file, metadata=None):
    """Render the complete step learning interface."""
    st.markdown("## 📚 Step-by-Step Learning")
    st.caption("Master each step in your routine with guided animations and tips")
    
    # Step navigation
    step_names = [f"Step {i+1}: {step['name']}" for i, step in enumerate(choreography)]
    selected_step_name = st.selectbox("Choose a step to learn:", step_names)
    
    # Find selected step
    step_index = step_names.index(selected_step_name)
    selected_step = choreography[step_index]
    
    render_single_step_learning(selected_step)
    
    # Frame stepping (only available for video files)
    if metadata is not None:
        st.markdown('<div class="section-label">Frame stepping</div>', unsafe_allow_html=True)
        st.caption("⏩ Move through the video frame-by-frame to analyze movement details.")
        frame_time_key = f"frame_time::{Path(original_file).name}"
        frame_slider_key = f"frame_slider::{Path(original_file).name}"
        prev_button_key = f"prev_frame::{Path(original_file).name}"
        next_button_key = f"next_frame::{Path(original_file).name}"

        if frame_time_key not in st.session_state:
            st.session_state[frame_time_key] = 0.0

        class _Meta:
            pass

        clip_meta = _Meta()
        clip_meta.fps = metadata["fps"]
        clip_meta.duration = metadata["duration"]

        button_col1, button_col2, button_col3 = st.columns([1, 1, 0.3])
        with button_col1:
            if st.button("⬅️ Previous frame", key=prev_button_key, use_container_width=True, help="Step backward one frame"):
                st.session_state[frame_time_key] = video_controls.step_frames(
                    clip_meta,
                    st.session_state[frame_time_key],
                    direction="backward",
                )
        with button_col2:
            if st.button("➡️ Next frame", key=next_button_key, use_container_width=True, help="Step forward one frame"):
                st.session_state[frame_time_key] = video_controls.step_frames(
                    clip_meta,
                    st.session_state[frame_time_key],
                    direction="forward",
                )

        max_frame = max(1, int(metadata["duration"] * metadata["fps"]))
        current_frame_index = min(max(0, int(round(st.session_state[frame_time_key] * metadata["fps"]))), max_frame)
        selected_frame = st.slider(
            "Choose frame",
            min_value=0,
            max_value=max_frame,
            value=current_frame_index,
            step=1,
            key=frame_slider_key,
            help="Drag to jump to any frame in the video.",
        )
        st.session_state[frame_time_key] = selected_frame / metadata["fps"]

        try:
            frame_data = app_services.extract_frame_for_preview(original_file, selected_frame)
            st.image(frame_data["frame"], use_container_width=True, clamp=True)
            st.caption(
                f'Frame {frame_data["frame_index"]} at {frame_data["timestamp"]}s '
                f'({frame_data["fps"]:.2f} fps)'
            )
        except Exception as exc:
            st.error(f"Could not load frame preview: {exc}")
    else:
        st.info("🎵 Frame stepping is only available for video files. Upload a dance video to analyze movement frame-by-frame.")


def render_frame_stepping(original_file, metadata):
    st.markdown('<div class="section-label">Frame stepping</div>', unsafe_allow_html=True)
    st.caption("⏩ Move through the video frame-by-frame to analyze movement details.")
    frame_time_key = f"frame_time_main::{Path(original_file).name}"
    frame_slider_key = f"frame_slider_main::{Path(original_file).name}"
    prev_button_key = f"prev_frame_main::{Path(original_file).name}"
    next_button_key = f"next_frame_main::{Path(original_file).name}"

    if frame_time_key not in st.session_state:
        st.session_state[frame_time_key] = 0.0

    class _Meta:
        pass

    clip_meta = _Meta()
    clip_meta.fps = metadata["fps"]
    clip_meta.duration = metadata["duration"]

    button_col1, button_col2, _ = st.columns([1, 1, 0.3])
    with button_col1:
        if st.button("⬅️ Previous frame", key=prev_button_key, use_container_width=True, help="Step backward one frame"):
            st.session_state[frame_time_key] = video_controls.step_frames(
                clip_meta,
                st.session_state[frame_time_key],
                direction="backward",
            )
    with button_col2:
        if st.button("➡️ Next frame", key=next_button_key, use_container_width=True, help="Step forward one frame"):
            st.session_state[frame_time_key] = video_controls.step_frames(
                clip_meta,
                st.session_state[frame_time_key],
                direction="forward",
            )

    max_frame = max(1, int(metadata["duration"] * metadata["fps"]))
    current_frame_index = min(max(0, int(round(st.session_state[frame_time_key] * metadata["fps"]))), max_frame)
    selected_frame = st.slider(
        "Choose frame",
        min_value=0,
        max_value=max_frame,
        value=current_frame_index,
        step=1,
        key=frame_slider_key,
        help="Drag to jump to any frame in the video.",
    )
    st.session_state[frame_time_key] = selected_frame / metadata["fps"]

    try:
        frame_data = app_services.extract_frame_for_preview(original_file, selected_frame)
        st.image(frame_data["frame"], use_container_width=True, clamp=True)
        st.caption(
            f'Frame {frame_data["frame_index"]} at {frame_data["timestamp"]}s '
            f'({frame_data["fps"]:.2f} fps)'
        )
    except Exception as exc:
        st.error(f"Could not load frame preview: {exc}")


class MediaTooLargeError(Exception):
    """Raised when media file is too large for base64 encoding in HTML components."""
    pass


def _media_path_to_data_url(media_path, default_mime, max_size_mb=8):
    path = Path(media_path)
    file_size = path.stat().st_size
    if file_size > max_size_mb * 1024 * 1024:
        raise MediaTooLargeError(f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds limit for custom player.")

    payload = path.read_bytes()
    detected_mime, _ = mimetypes.guess_type(str(media_path))
    mime_type = detected_mime or default_mime
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}", mime_type


def render_interval_loop_player(preview_path, media_type, loop_interval_seconds=3.0, player_key="default"):
    default_mime = "video/mp4" if media_type == "video" else "audio/wav"
    data_url, mime_type = _media_path_to_data_url(preview_path, default_mime)
    element_tag = "video" if media_type == "video" else "audio"
    player_height = 420 if media_type == "video" else 120
    player_style = "width: 100%; border-radius: 12px;" if media_type == "video" else "width: 100%;"
    element_id = f"loop-player-{player_key}".replace(".", "_").replace(" ", "_")

    html = f"""
    <div style="width:100%;">
      <{element_tag} id="{element_id}" controls preload="auto" style="{player_style}">
        <source src="{data_url}" type="{mime_type}">
      </{element_tag}>
    </div>
    <script>
      const player = document.getElementById("{element_id}");
      const pauseMs = {int(loop_interval_seconds * 1000)};
      let waiting = false;
      if (player) {{
        player.addEventListener("ended", () => {{
          if (waiting) return;
          waiting = true;
          setTimeout(() => {{
            player.currentTime = 0;
            const playPromise = player.play();
            if (playPromise !== undefined) {{
              playPromise.catch(() => {{}});
            }}
            waiting = false;
          }}, pauseMs);
        }});
      }}
    </script>
    """
    components.html(html, height=player_height, scrolling=False)


def render_loop_controls(original_file, metadata, media_type):
    st.markdown('<div class="section-label">⏱️ Practice Loop Generator</div>', unsafe_allow_html=True)
    st.caption("Set your loop by entering start time, end time, and speed. The loop restarts automatically after 3 seconds.")

    duration = float(metadata["duration"])
    min_loop_length = 0.1
    if duration <= min_loop_length:
        st.warning("Media is too short for loop practice.")
        return

    file_key = Path(original_file).name
    start_key = f"loop_start::{file_key}"
    end_key = f"loop_end::{file_key}"
    speed_key = f"loop_speed::{file_key}"
    mirror_key = f"loop_mirror::{file_key}"

    max_start = max(0.0, duration - min_loop_length)
    if start_key not in st.session_state:
        st.session_state[start_key] = 0.0
    st.session_state[start_key] = float(min(max(0.0, st.session_state[start_key]), max_start))
    min_end = min(duration, st.session_state[start_key] + min_loop_length)

    if end_key not in st.session_state:
        st.session_state[end_key] = min(duration, max(min_end, 4.0))
    st.session_state[end_key] = float(min(max(min_end, st.session_state[end_key]), duration))

    if speed_key not in st.session_state:
        st.session_state[speed_key] = 0.75
    st.session_state[speed_key] = float(min(max(0.25, st.session_state[speed_key]), 2.0))

    col1, col2, col3 = st.columns(3)
    with col1:
        start_sec = st.number_input(
            "Enter loop start time (seconds)",
            min_value=0.0,
            max_value=max_start,
            step=0.1,
            format="%.2f",
            key=start_key,
            help="Manually enter where the loop should start.",
        )
    with col2:
        min_end = min(duration, start_sec + min_loop_length)
        st.session_state[end_key] = float(min(max(min_end, st.session_state.get(end_key, min_end)), duration))
        end_sec = st.number_input(
            "Enter loop end time (seconds)",
            min_value=min_end,
            max_value=duration,
            step=0.1,
            format="%.2f",
            key=end_key,
            help="Manually enter where the loop should end.",
        )
    with col3:
        speed_multiplier = st.number_input(
            "Enter loop speed (x)",
            min_value=0.25,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            key=speed_key,
            help="0.5 = half speed, 1.0 = normal speed, 2.0 = double speed.",
        )
        st.metric("Current speed", f"{speed_multiplier:.2f}x")

    mirror = False
    if media_type == "video":
        with st.expander("🎬 Video options", expanded=False):
            mirror = st.checkbox(
                "Mirror video",
                value=st.session_state.get(mirror_key, False),
                key=mirror_key,
                help="Flip the video horizontally.",
            )

    preview_key = f"practice_preview_path::{file_key}"
    params_key = f"practice_preview_params::{file_key}"
    current_params = (round(float(start_sec), 2), round(float(end_sec), 2), round(float(speed_multiplier), 2), bool(mirror))
    previous_params = st.session_state.get(params_key)

    if previous_params != current_params or preview_key not in st.session_state:
        try:
            preview_path = app_services.create_practice_media(
                original_file,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                speed_multiplier=float(speed_multiplier),
                mirror=mirror,
            )
            st.session_state[preview_key] = str(preview_path)
            st.session_state[params_key] = current_params
        except Exception as exc:
            st.error(f"Could not generate preview: {exc}")
            preview_path = None
    else:
        preview_path = st.session_state.get(preview_key)

    st.markdown('<div class="section-label">Your practice clip</div>', unsafe_allow_html=True)
    if preview_path:
        # Provide a toggle for the custom player vs standard player
        use_custom = st.checkbox("Enable interval loop (3s pause between repeats)", value=True, help="Custom player that adds a pause. If the video doesn't show up, uncheck this.")
        
        preview_path_str = str(preview_path)
        
        if use_custom:
            try:
                render_interval_loop_player(
                    preview_path_str,
                    media_type=media_type,
                    loop_interval_seconds=3.0,
                    player_key=file_key,
                )
                st.success("✅ Loop preview ready with 3s pause.")
            except MediaTooLargeError as exc:
                st.warning(f"⚠️ {exc} Falling back to standard player.")
                if media_type == "video":
                    st.video(preview_path_str)
                else:
                    st.audio(preview_path_str)
            except Exception as exc:
                st.warning(f"⚠️ Custom player failed: {exc}. Falling back to standard player.")
                if media_type == "video":
                    st.video(preview_path_str)
                else:
                    st.audio(preview_path_str)
        else:
            if media_type == "video":
                st.video(preview_path_str)
            else:
                st.audio(preview_path_str)
            st.info("ℹ️ Using standard player (no pause between loops).")
    else:
        st.info("⏳ Generating preview...")


def main():
    apply_soft_theme()
    render_header()
    beat_method, subdivisions, run_separation, run_pose_estimation, pose_sample_count, pose_timing = render_sidebar()
    render_upload_help()

    uploaded_file = st.file_uploader(
        "📁 Upload audio or video",
        type=["mp3", "wav", "ogg", "flac", "m4a", "mp4", "mov", "m4v", "avi", "mkv"],
        label_visibility="visible",
        help="Drag and drop a file here, or click to browse. Supported: MP3, WAV, MP4, MOV, AVI, MKV.",
    )

    if uploaded_file is None:
        st.info("📋 Upload a media file to get started with beat analysis and practice controls.")
        return

    try:
        original_file = app_services.save_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Could not save the uploaded file: {exc}")
        return

    with st.spinner("🎵 Analyzing your clip..."):
        try:
            results = app_services.analyze_media_with_method(
                original_file,
                beat_method=beat_method,
                subdivisions=subdivisions,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    # --- FULL WIDTH LAYOUT: Stack all main content vertically ---
    render_media_preview(results, original_file)

    if results["media_type"] == "video":
        try:
            metadata = app_services.get_video_metadata(original_file)
            render_frame_stepping(original_file, metadata)
        except Exception as exc:
            st.warning(f"Frame stepping unavailable: {exc}")

    render_analysis(results)

    # AI Dance Generation
    if results["media_type"] == "video":
        try:
            metadata = app_services.get_video_metadata(original_file)
            render_dance_generation(results, original_file, metadata)
        except Exception as exc:
            st.warning(f"Dance generation unavailable: {exc}")
            render_dance_generation(results, original_file)
    else:
        render_dance_generation(results, original_file)

    if results["media_type"] == "video":
        try:
            metadata = app_services.get_video_metadata(original_file)
            render_loop_controls(original_file, metadata, "video")
        except Exception as exc:
            st.warning(f"Video controls are unavailable for this file: {exc}")
    else:
        render_loop_controls(original_file, results, "audio")
        
    # Pose estimation with timestamps support
    if run_pose_estimation and results["media_type"] == "video":
        st.divider()
        with st.spinner("🧘 Running pose estimation..."):
            try:
                # Determine timestamps based on timing option
                pose_timestamps = None
                if pose_timing == "At detected beats" and "beats" in results:
                    # Use beat timestamps (limit to pose_sample_count)
                    beats = results["beats"]
                    if len(beats) > pose_sample_count:
                        # Sample beats evenly
                        indices = np.linspace(0, len(beats) - 1, pose_sample_count, dtype=int)
                        pose_timestamps = [float(beats[i]) for i in indices]
                    else:
                        pose_timestamps = [float(b) for b in beats]
                
                if pose_timestamps:
                    st.info(f"📍 Analyzing pose at {len(pose_timestamps)} beat timestamps...")
                    pose_results = app_services.analyze_pose(original_file, timestamps=pose_timestamps)
                else:
                    st.info(f"📍 Analyzing pose at {pose_sample_count} evenly spaced points...")
                    pose_results = app_services.analyze_pose(original_file, sample_count=pose_sample_count)
                render_pose_estimation(original_file, pose_results)
            except Exception as exc:
                st.warning(f"Pose estimation could not complete: {exc}")
    
    # Stem separation (full width at bottom)
    if run_separation:
        st.divider()
        with st.spinner("🎙️ Separating audio stems..."):
            try:
                render_stems(original_file)
            except Exception as exc:
                st.warning(f"Stem separation could not complete: {exc}")


if __name__ == "__main__":
    main()
