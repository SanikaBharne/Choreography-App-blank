import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import old_functions.audio_utils as audio_utils


class TestAudioUtils(unittest.TestCase):
    def test_to_mono_returns_original_mono_segment(self):
        """_to_mono returns the original segment when audio is already mono."""
        audio = AudioSegment.silent(duration=10, frame_rate=8000).set_channels(1)

        result = audio_utils._to_mono(audio)

        self.assertIs(audio, result)
        self.assertEqual(1, result.channels)

    def test_to_mono_converts_multichannel_audio(self):
        """_to_mono collapses multichannel audio down to one channel."""
        audio = AudioSegment.silent(duration=10, frame_rate=8000).set_channels(2)

        result = audio_utils._to_mono(audio)

        self.assertEqual(2, audio.channels)
        self.assertEqual(1, result.channels)
        self.assertEqual(audio.frame_rate, result.frame_rate)

    def test_to_mono_rejects_non_audiosegment_input(self):
        """_to_mono raises TypeError for non-AudioSegment values."""
        with self.assertRaises(TypeError):
            audio_utils._to_mono("not audio")

    def test_to_numpy_returns_sample_array(self):
        """_to_numpy returns a NumPy array with the segment samples."""
        audio = AudioSegment(
            data=(1).to_bytes(2, byteorder="little", signed=True)
            + (-2).to_bytes(2, byteorder="little", signed=True)
            + (3).to_bytes(2, byteorder="little", signed=True),
            sample_width=2,
            frame_rate=8000,
            channels=1,
        )

        result = audio_utils._to_numpy(audio)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, -2, 3], dtype=np.int16))

    def test_to_numpy_rejects_non_audiosegment_input(self):
        """_to_numpy raises TypeError for non-AudioSegment values."""
        with self.assertRaises(TypeError):
            audio_utils._to_numpy(123)

    def test_load_audio_rejects_invalid_arguments(self):
        """load_audio validates filepath and mono flag types before loading."""
        with self.assertRaises(TypeError):
            audio_utils.load_audio(123)

        with self.assertRaises(ValueError):
            audio_utils.load_audio("")

        with self.assertRaises(TypeError):
            audio_utils.load_audio("song.wav", mono="yes")

    def test_load_audio_converts_to_mono_by_default(self):
        """load_audio converts audio to mono before extracting samples by default."""
        original_audio = AudioSegment.silent(duration=10, frame_rate=44100).set_channels(2)
        converted_audio = original_audio.set_channels(1)
        converted_samples = np.array([1, 2, 3], dtype=np.int16)

        with patch.object(audio_utils.AudioSegment, "from_file", return_value=original_audio) as mock_from_file:
            with patch.object(audio_utils, "_to_mono", return_value=converted_audio) as mock_to_mono:
                with patch.object(audio_utils, "_to_numpy", return_value=converted_samples) as mock_to_numpy:
                    result = audio_utils.load_audio("song.wav")

        mock_from_file.assert_called_once_with("song.wav")
        mock_to_mono.assert_called_once_with(original_audio)
        mock_to_numpy.assert_called_once_with(converted_audio)
        self.assertEqual((converted_audio.frame_rate, converted_samples), result)

    def test_load_audio_skips_mono_conversion_when_disabled(self):
        """load_audio skips mono conversion when the mono flag is False."""
        audio = AudioSegment.silent(duration=10, frame_rate=22050).set_channels(2)
        samples = np.array([4, 5, 6], dtype=np.int16)

        with patch.object(audio_utils.AudioSegment, "from_file", return_value=audio) as mock_from_file:
            with patch.object(audio_utils, "_to_mono") as mock_to_mono:
                with patch.object(audio_utils, "_to_numpy", return_value=samples) as mock_to_numpy:
                    result = audio_utils.load_audio("song.wav", mono=False)

        mock_from_file.assert_called_once_with("song.wav")
        mock_to_mono.assert_not_called()
        mock_to_numpy.assert_called_once_with(audio)
        self.assertEqual((audio.frame_rate, samples), result)


if __name__ == "__main__":
    unittest.main()
