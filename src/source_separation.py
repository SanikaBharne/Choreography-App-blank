import os
import demucs.separate
from pydub import AudioSegment


def separate(audio_path, output_dir="separated"):
    """Run Demucs 4-stem separation on an audio file.

    Args:
        audio_path: Path to the audio file.
        output_dir: Directory to save separated stems.

    Returns:
        A dictionary mapping stem names to their file paths.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
        FileNotFoundError: If audio_path does not exist.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    output_root = os.path.abspath(output_dir)
    os.makedirs(output_root, exist_ok=True)

    try:
        demucs.separate.main([
            "-n", "htdemucs",
            "-o", output_root,
            audio_path
        ])
    except SystemExit as exc:
        raise RuntimeError(f"Demucs separation failed for {audio_path}") from exc

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_root, "htdemucs", track_name)

    stems = {}
    for stem in ["vocals", "drums", "bass", "other"]:
        path = os.path.join(stem_dir, f"{stem}.wav")
        if os.path.exists(path):
            stems[stem] = path
    missing = [stem for stem in ["vocals", "drums", "bass", "other"] if stem not in stems]
    if missing:
        raise RuntimeError(
            "Demucs did not produce required stems: "
            + ", ".join(missing)
        )

    return stems


def get_stem(stems, name):
    """Get the path to a specific stem.

    Args:
        stems: Dictionary returned by separate().
        name: Stem name (vocals, drums, bass, other).

    Returns:
        The file path to the requested stem.

    Raises:
        KeyError: If the stem name is not found.
    """
    if name not in stems:
        raise KeyError(f"Stem '{name}' not found. Available: {list(stems.keys())}")
    return stems[name]


def get_vocals(stems):
    return get_stem(stems, "vocals")


def get_drums(stems):
    return get_stem(stems, "drums")


def get_bass(stems):
    return get_stem(stems, "bass")


def get_other(stems):
    return get_stem(stems, "other")


def get_instrumental(stems, output_path="instrumental.wav"):
    """Combine drums, bass, and other into an instrumental track.

    Args:
        stems: Dictionary returned by separate().
        output_path: Where to save the combined file.

    Returns:
        The output file path.
    """
    drums = AudioSegment.from_wav(get_drums(stems))
    bass = AudioSegment.from_wav(get_bass(stems))
    other = AudioSegment.from_wav(get_other(stems))

    instrumental = drums.overlay(bass).overlay(other)
    instrumental.export(output_path, format="wav")
    return output_path