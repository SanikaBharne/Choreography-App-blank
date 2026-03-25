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

    # separating the data into managable chunks
    window = []
    for i in range(0, len(audio), hop_length):
        chunk = audio[i:i+window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        # hanning each chunk to smoothen out
        chunk *= np.hanning(window_size)
        # getting the Real Fast Fourier Transform on each chunk
        chunk = np.fft.rfft(chunk)

        window.append(chunk)
    window = np.array(window)
    window = window.T
    return window


 




        



