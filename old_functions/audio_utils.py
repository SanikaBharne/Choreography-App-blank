from pydub import AudioSegment, utils
import numpy as np

def _to_mono(audio):
    """Convert multichannel audio to mono when needed.

    Args:
        audio: A `pydub.AudioSegment` to convert.

    Returns:
        A mono `AudioSegment` if `audio` has at least two channels; otherwise the
        original `AudioSegment`.

    Raises:
        TypeError: If `audio` is not an `AudioSegment`.
    """
    if not isinstance(audio, AudioSegment):
        raise TypeError("audio must be an AudioSegment")
    if audio.channels >= 2:
        return audio.set_channels(1)
    
    return audio


def _to_numpy(audio):
    """Convert an audio segment into a NumPy array.

    Args:
        audio: A `pydub.AudioSegment` whose samples should be converted.

    Returns:
        A NumPy array containing the samples from `audio`.

    Raises:
        TypeError: If `audio` is not an `AudioSegment`.
    """
    if not isinstance(audio, AudioSegment):
        raise TypeError("audio must be an AudioSegment")
    data_type = utils.get_array_type(audio.sample_width * 8)
    audio = audio.get_array_of_samples()
    return np.array(audio, dtype=data_type)


def load_audio(filepath, mono=True):
    """Load an audio file and return its samples with sample rate.

    Args:
        filepath: Path to the audio file to load.
        mono: Whether the loaded audio should be converted to mono.

    Returns:
        A tuple of `(sample_rate, samples)`.

    Raises:
        TypeError: If `filepath` is not a string or `mono` is not a boolean.
        ValueError: If `filepath` is empty.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    if not filepath:
        raise ValueError("filepath must not be empty")
    if not isinstance(mono, bool):
        raise TypeError("mono must be a boolean")
    audio = AudioSegment.from_file(filepath)
    if mono:
        audio  = _to_mono(audio)

    return audio.frame_rate, _to_numpy(audio)
