import numpy as np
import librosa


def detect_beats(audio_path, sr=None):
    """Detect beat timestamps in an audio file.

    Args:
        audio_path: Path to the audio file to analyze.
        sr: Target sample rate. If None, the file's native sample rate is used.

    Returns:
        A NumPy array of beat timestamps in seconds.

    Raises:
        TypeError: If audio_path is not a string or sr is not numeric.
        ValueError: If audio_path is empty or sr is not positive.
        FileNotFoundError: If audio_path does not exist.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if sr is not None:
        if not isinstance(sr, (int, float)):
            raise TypeError("sr must be numeric")
        if sr <= 0:
            raise ValueError("sr must be positive")

    y, sr = librosa.load(audio_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beats, sr=sr)


def merge_beats(drum_beats, mix_beats, gap_threshold=1.0):
    """Merge two beat arrays, keeping mix beats that are far enough from drum beats.

    Args:
        drum_beats: NumPy array of beat timestamps from the drum stem, in seconds.
        mix_beats: NumPy array of beat timestamps from the full mix, in seconds.
        gap_threshold: Minimum distance in seconds a mix beat must be from all
            drum beats to be included in the result.

    Returns:
        A sorted NumPy array of merged beat timestamps in seconds.

    Raises:
        TypeError: If drum_beats or mix_beats are not array-like, or
            gap_threshold is not numeric.
        ValueError: If gap_threshold is not positive.
    """
    drum_beats = np.asarray(drum_beats)
    mix_beats = np.asarray(mix_beats)

    if not np.issubdtype(drum_beats.dtype, np.number):
        raise TypeError("drum_beats must contain numeric values")
    if not np.issubdtype(mix_beats.dtype, np.number):
        raise TypeError("mix_beats must contain numeric values")
    if not isinstance(gap_threshold, (int, float)):
        raise TypeError("gap_threshold must be numeric")
    if gap_threshold <= 0:
        raise ValueError("gap_threshold must be positive")

    if len(drum_beats) == 0:
        return mix_beats

    merged = list(drum_beats)
    for t in mix_beats:
        if min(abs(drum_beats - t)) >= gap_threshold:
            merged.append(t)

    return np.sort(merged)
