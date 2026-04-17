import numpy as np


def _to_mono(audio):
    array = np.asarray(audio, dtype=float)
    if array.ndim == 0:
        raise ValueError("audio must not be scalar")
    if array.size == 0:
        raise ValueError("audio must not be empty")
    if array.ndim == 1:
        return array
    return np.mean(array, axis=0)


def frame_audio(audio, frame_size=1024, hop_size=512):
    mono = _to_mono(audio)
    if not isinstance(frame_size, int) or not isinstance(hop_size, int):
        raise TypeError("frame_size and hop_size must be integers")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if mono.size < frame_size:
        return np.array([np.pad(mono, (0, frame_size - mono.size))])

    frames = []
    for start in range(0, mono.size - frame_size + 1, hop_size):
        frames.append(mono[start:start + frame_size])
    return np.asarray(frames)


def rms_energy(frames):
    frame_array = np.asarray(frames, dtype=float)
    if frame_array.size == 0:
        raise ValueError("frames must not be empty")
    return np.sqrt(np.mean(np.square(frame_array), axis=1))


def onset_strength(energy):
    energy_array = np.asarray(energy, dtype=float)
    if energy_array.size < 2:
        return np.array([], dtype=float)
    deltas = np.diff(energy_array)
    return np.maximum(deltas, 0.0)


def pick_beats(onset, sr, hop_size, threshold_scale=1.5, min_interval=0.25):
    onset_array = np.asarray(onset, dtype=float)
    if onset_array.size == 0:
        return np.array([], dtype=float)
    if sr <= 0:
        raise ValueError("sr must be positive")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if threshold_scale <= 0 or min_interval <= 0:
        raise ValueError("threshold_scale and min_interval must be positive")

    baseline = np.median(onset_array)
    spread = np.std(onset_array)
    threshold = baseline + (spread * threshold_scale)
    min_gap_frames = max(1, int((min_interval * sr) / hop_size))

    peaks = []
    last_index = -min_gap_frames
    for index, value in enumerate(onset_array):
        if value < threshold:
            continue
        left = onset_array[index - 1] if index > 0 else value
        right = onset_array[index + 1] if index < len(onset_array) - 1 else value
        if value < left or value < right:
            continue
        if index - last_index < min_gap_frames:
            if peaks and value > onset_array[peaks[-1]]:
                peaks[-1] = index
                last_index = index
            continue
        peaks.append(index)
        last_index = index

    return np.asarray([(index * hop_size) / sr for index in peaks], dtype=float)


def detect_beats(audio, sr, frame_size=1024, hop_size=512, threshold_scale=1.5, min_interval=0.25):
    if not isinstance(sr, (int, float)):
        raise TypeError("sr must be numeric")
    if sr <= 0:
        raise ValueError("sr must be positive")

    frames = frame_audio(audio, frame_size=frame_size, hop_size=hop_size)
    energy = rms_energy(frames)
    onset = onset_strength(energy)
    return pick_beats(
        onset,
        sr=sr,
        hop_size=hop_size,
        threshold_scale=threshold_scale,
        min_interval=min_interval,
    )
