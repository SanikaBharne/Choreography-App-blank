import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter


os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import blueprints.hpss as hpss


class TestHPSS(unittest.TestCase):
    def test_stft_returns_windowed_rfft_columns(self):
        """stft applies a Hann window, pads the final chunk, and returns transposed bins."""
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = hpss.stft(audio.copy(), window_size=4, hop_length=4)

        expected_chunks = []
        for start in (0, 4):
            chunk = audio[start : start + 4]
            if len(chunk) < 4:
                chunk = np.pad(chunk, (0, 4 - len(chunk)))
            expected_chunks.append(np.fft.rfft(chunk * np.hanning(4)))
        expected = np.array(expected_chunks).T

        self.assertEqual((3, 2), result.shape)
        np.testing.assert_allclose(result, expected)

    def test_stft_raises_for_invalid_input(self):
        """stft rejects empty audio and non-positive frame parameters."""
        with self.assertRaises(ValueError):
            hpss.stft([], window_size=4, hop_length=2)

        with self.assertRaises(ValueError):
            hpss.stft([1.0, 2.0], window_size=0, hop_length=2)

        with self.assertRaises(ValueError):
            hpss.stft([1.0, 2.0], window_size=4, hop_length=0)

    def test_compute_masks_uses_magnitude_and_axis_specific_medians(self):
        """compute_masks applies median filtering to the magnitude spectrogram."""
        fourier_transform = np.array(
            [
                [1 + 1j, 10 + 0j, 3 + 4j],
                [2 + 0j, 8 + 6j, 4 + 0j],
                [5 + 12j, 7 + 0j, 6 + 8j],
            ]
        )

        horizontal, vertical = hpss.compute_masks(
            fourier_transform, harm_size=3, perc_size=3
        )

        magnitude = np.abs(fourier_transform)
        expected_horizontal = median_filter(magnitude, size=(1, 3))
        expected_vertical = median_filter(magnitude, size=(3, 1))

        np.testing.assert_allclose(horizontal, expected_horizontal)
        np.testing.assert_allclose(vertical, expected_vertical)

    def test_compute_masks_raises_for_invalid_input(self):
        """compute_masks rejects empty spectrograms and non-positive filter sizes."""
        with self.assertRaises(ValueError):
            hpss.compute_masks(np.array([]), harm_size=3, perc_size=3)

        with self.assertRaises(ValueError):
            hpss.compute_masks(np.ones((2, 2)), harm_size=0, perc_size=3)

        with self.assertRaises(ValueError):
            hpss.compute_masks(np.ones((2, 2)), harm_size=3, perc_size=0)

    def test_build_masks_returns_complementary_ratios(self):
        """build_masks divides each mask by the combined energy plus epsilon."""
        horizontal = np.array([[2.0, 0.0], [3.0, 4.0]])
        vertical = np.array([[1.0, 0.0], [1.0, 4.0]])

        harmonic_mask, percussive_mask = hpss.build_masks(horizontal, vertical)

        expected_harmonic = horizontal / (horizontal + vertical + 1e-8)
        expected_percussive = vertical / (horizontal + vertical + 1e-8)

        np.testing.assert_allclose(harmonic_mask, expected_harmonic)
        np.testing.assert_allclose(percussive_mask, expected_percussive)
        self.assertEqual(0.0, harmonic_mask[0, 1])
        self.assertEqual(0.0, percussive_mask[0, 1])

    def test_build_masks_raises_for_mismatched_shapes(self):
        """build_masks rejects arrays with different shapes."""
        with self.assertRaises(ValueError):
            hpss.build_masks(np.ones((2, 2)), np.ones((2, 3)))

    def test_apply_masks_multiplies_each_mask_with_spectrogram(self):
        """apply_masks produces masked harmonic and percussive spectrograms."""
        harmonic_mask = np.array([[1.0, 0.25], [0.5, 0.0]])
        percussive_mask = np.array([[0.0, 0.75], [0.5, 1.0]])
        fourier_transform = np.array([[2 + 1j, 4 - 2j], [6 + 0j, 8 + 3j]])

        harmonic_spectrogram, percussive_spectrogram = hpss.apply_masks(
            harmonic_mask, percussive_mask, fourier_transform
        )

        np.testing.assert_allclose(harmonic_spectrogram, harmonic_mask * fourier_transform)
        np.testing.assert_allclose(
            percussive_spectrogram, percussive_mask * fourier_transform
        )

    def test_apply_masks_raises_for_mismatched_shapes(self):
        """apply_masks rejects masks that do not match the spectrogram shape."""
        with self.assertRaises(ValueError):
            hpss.apply_masks(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 3)))

    def test_istft_overlap_adds_inverse_transforms_with_hann_window(self):
        """istft overlap-adds each inverse-transformed row into the output array."""
        fourier_transform = np.array(
            [
                [1.0 + 0j, 2.0 + 0j],
                [3.0 + 0j, 4.0 + 0j],
                [5.0 + 0j, 6.0 + 0j],
            ]
        )

        result = hpss.istft(fourier_transform, window_size=4, hop_length=2)

        transposed_transform = fourier_transform.T
        expected = np.zeros((len(transposed_transform) - 1) * 2 + 4)
        for index, row in enumerate(transposed_transform):
            chunk = np.fft.irfft(row, n=4) * np.hanning(4)
            start = index * 2
            expected[start : start + len(chunk)] += chunk

        np.testing.assert_allclose(result, expected)

    def test_istft_raises_for_invalid_input(self):
        """istft rejects empty transforms and non-positive synthesis parameters."""
        with self.assertRaises(ValueError):
            hpss.istft(np.array([]), window_size=4, hop_length=2)

        with self.assertRaises(ValueError):
            hpss.istft(np.ones((2, 2)), window_size=0, hop_length=2)

        with self.assertRaises(ValueError):
            hpss.istft(np.ones((2, 2)), window_size=4, hop_length=0)


if __name__ == "__main__":
    unittest.main()
