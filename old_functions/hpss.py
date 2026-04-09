import numpy as np
from scipy.ndimage import median_filter

def stft(audio, window_size=2048, hop_length=512):
    """Compute the short-time Fourier transform of an audio signal.

    Args:
        audio: One-dimensional audio sample data.
        window_size: Number of samples per analysis window.
        hop_length: Number of samples between consecutive windows.

    Returns:
        A 2-D NumPy array containing the transposed real FFT of each windowed frame.

    Raises:
        ValueError: If ``audio`` is empty, not one-dimensional, or if either size
            argument is not positive.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional array")
    if len(audio) == 0:
        raise ValueError("audio must not be empty")

    window = []
    for i in range(0, len(audio), hop_length):
        chunk = np.array(audio[i : i + window_size], dtype=float, copy=True)
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        chunk *= np.hanning(window_size)
        chunk = np.fft.rfft(chunk)

        window.append(chunk)
    window = np.array(window)
    window = window.T
    return window

def compute_masks(fourier_transform, harm_size=31, perc_size=31):
    """Build median-filtered harmonic and percussive magnitude estimates.

    Args:
        fourier_transform: A 2-D spectrogram produced by the STFT.
        harm_size: Median filter width used across time for harmonic content.
        perc_size: Median filter height used across frequency for percussive content.

    Returns:
        A tuple of ``(horizontal, vertical)`` filtered magnitude arrays.

    Raises:
        ValueError: If ``fourier_transform`` is empty, not two-dimensional, or if
            either filter size is not positive.
    """
    if harm_size <= 0:
        raise ValueError("harm_size must be greater than 0")
    if perc_size <= 0:
        raise ValueError("perc_size must be greater than 0")

    fourier_transform = np.asarray(fourier_transform)
    if fourier_transform.ndim != 2:
        raise ValueError("fourier_transform must be a two-dimensional array")
    if fourier_transform.size == 0:
        raise ValueError("fourier_transform must not be empty")

    mag = np.abs(fourier_transform)

    horizontal = median_filter(mag, size=(1, harm_size))
    vertical = median_filter(mag, size=(perc_size, 1))

    return horizontal, vertical

def build_masks(horizontal, vertical):
    """Convert filtered magnitudes into soft harmonic and percussive masks.

    Args:
        horizontal: Harmonic energy estimate array.
        vertical: Percussive energy estimate array.

    Returns:
        A tuple of ``(harmonic_mask, percussive_mask)`` ratio masks.

    Raises:
        ValueError: If either input is empty or the input shapes do not match.
    """
    horizontal = np.asarray(horizontal)
    vertical = np.asarray(vertical)
    if horizontal.size == 0 or vertical.size == 0:
        raise ValueError("horizontal and vertical masks must not be empty")
    if horizontal.shape != vertical.shape:
        raise ValueError("horizontal and vertical masks must have matching shapes")

    harmonic_mask = horizontal / (horizontal + vertical + 1e-8)
    percussive_mask = vertical / (vertical + horizontal + 1e-8)
    return harmonic_mask, percussive_mask

def apply_masks(harmonic_mask, percussive_mask, fourier_transform):
    """Apply harmonic and percussive masks to a spectrogram.

    Args:
        harmonic_mask: Soft mask emphasizing harmonic components.
        percussive_mask: Soft mask emphasizing percussive components.
        fourier_transform: The original spectrogram to separate.

    Returns:
        A tuple of ``(harmonic_spectrogram, percussive_spectrogram)``.

    Raises:
        ValueError: If any input is empty or if their shapes do not all match.
    """
    harmonic_mask = np.asarray(harmonic_mask)
    percussive_mask = np.asarray(percussive_mask)
    fourier_transform = np.asarray(fourier_transform)

    if harmonic_mask.size == 0 or percussive_mask.size == 0 or fourier_transform.size == 0:
        raise ValueError("masks and fourier_transform must not be empty")
    if harmonic_mask.shape != percussive_mask.shape:
        raise ValueError("harmonic_mask and percussive_mask must have matching shapes")
    if harmonic_mask.shape != fourier_transform.shape:
        raise ValueError("mask shapes must match fourier_transform")

    harmonic_spectrogram = harmonic_mask * fourier_transform
    percussive_spectrogram = percussive_mask * fourier_transform
    return harmonic_spectrogram, percussive_spectrogram

def istft(fourier_transform, window_size=2048, hop_length=512):
    """Reconstruct audio samples from a spectrogram using overlap-add.

    Args:
        fourier_transform: A 2-D spectrogram whose rows are inverse transformed.
        window_size: Number of samples in each reconstructed frame.
        hop_length: Number of samples between consecutive output frames.

    Returns:
        A one-dimensional NumPy array containing the reconstructed audio signal.

    Raises:
        ValueError: If ``fourier_transform`` is empty, not two-dimensional, or if
            either size argument is not positive.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    fourier_transform = np.asarray(fourier_transform)
    if fourier_transform.ndim != 2:
        raise ValueError("fourier_transform must be a two-dimensional array")
    if fourier_transform.size == 0:
        raise ValueError("fourier_transform must not be empty")

    fourier_transform = fourier_transform.T
    inverse_array = np.zeros((len(fourier_transform) - 1) * hop_length + window_size)

    window = np.hanning(window_size)
    norm = np.zeros_like(inverse_array)

    for i, time in enumerate(fourier_transform):
        chunk = np.fft.irfft(time, n=window_size)
        chunk *= window
        inverse_array[i * hop_length : i * hop_length + window_size] += chunk
        norm[i * hop_length : i * hop_length + window_size] += window ** 2

    inverse_array[norm > 1e-8] /= norm[norm > 1e-8]
    return inverse_array


        

