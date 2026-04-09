import unittest
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from source_separation import (
    separate, get_stem, get_vocals, get_drums,
    get_bass, get_other, get_instrumental
)


class TestSeparate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stems = separate("test_files/TBH.mp3", output_dir="separated")

    def test_returns_dict(self):
        self.assertIsInstance(self.stems, dict)

    def test_contains_all_stems(self):
        for stem in ["vocals", "drums", "bass", "other"]:
            self.assertIn(stem, self.stems)

    def test_stem_files_exist(self):
        for path in self.stems.values():
            self.assertTrue(os.path.exists(path))

    def test_stem_files_are_wav(self):
        for path in self.stems.values():
            self.assertTrue(path.endswith(".wav"))

    def test_type_error_on_non_string(self):
        with self.assertRaises(TypeError):
            separate(123)

    def test_value_error_on_empty_string(self):
        with self.assertRaises(ValueError):
            separate("")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            separate("nonexistent.mp3")


class TestGetStem(unittest.TestCase):
    def setUp(self):
        self.stems = {
            "vocals": "separated/htdemucs/TBH/vocals.wav",
            "drums": "separated/htdemucs/TBH/drums.wav",
            "bass": "separated/htdemucs/TBH/bass.wav",
            "other": "separated/htdemucs/TBH/other.wav",
        }

    def test_returns_correct_path(self):
        self.assertEqual(get_stem(self.stems, "drums"), self.stems["drums"])

    def test_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_stem(self.stems, "snare")


class TestConvenienceFunctions(unittest.TestCase):
    def setUp(self):
        self.stems = {
            "vocals": "separated/htdemucs/TBH/vocals.wav",
            "drums": "separated/htdemucs/TBH/drums.wav",
            "bass": "separated/htdemucs/TBH/bass.wav",
            "other": "separated/htdemucs/TBH/other.wav",
        }

    def test_get_vocals(self):
        self.assertEqual(get_vocals(self.stems), self.stems["vocals"])

    def test_get_drums(self):
        self.assertEqual(get_drums(self.stems), self.stems["drums"])

    def test_get_bass(self):
        self.assertEqual(get_bass(self.stems), self.stems["bass"])

    def test_get_other(self):
        self.assertEqual(get_other(self.stems), self.stems["other"])


class TestGetInstrumental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stems = separate("test_files/TBH.mp3", output_dir="separated")

    def test_creates_file(self):
        path = get_instrumental(self.stems, output_path="test_instrumental.wav")
        self.assertTrue(os.path.exists(path))

    def test_file_is_nonempty(self):
        path = get_instrumental(self.stems, output_path="test_instrumental.wav")
        self.assertGreater(os.path.getsize(path), 0)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test_instrumental.wav"):
            os.remove("test_instrumental.wav")


if __name__ == "__main__":
    unittest.main()