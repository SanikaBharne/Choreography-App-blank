import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

def load_audio(filepath):
    sample_rate, data = wavfile.read(filepath)
    if data.ndim == 2:
        data = np.mean(data, 1)
    return sample_rate, data/32767

def stft(audio, window_size=2048, hop_length=512):
    window = []

    for i in range(0, len(audio), hop_length):
        window.append(audio[i:i+window_size])



