# -*- coding: utf-8 -*-
"""
Realtime viewer for click detection based on Detector-solorun.py
"""

import time
import numpy as np
import soundfile as sf
import importlib.util, sys, os
import matplotlib.pyplot as plt

# Dynamically import Detector-solorun.py
detector_path = os.path.join(os.path.dirname(__file__), "Detector-solorun.py")
spec = importlib.util.spec_from_file_location("Detector_solorun", detector_path)
Detector_solorun = importlib.util.module_from_spec(spec)
sys.modules["Detector_solorun"] = Detector_solorun
spec.loader.exec_module(Detector_solorun)

from Detector_solorun import SimpleAutoClicksDetector
from clicks_processing.ClickUtils import butter_pass_filter, TeagerKaiser_operator

def stream_and_display(file_path, chunk_duration=0.2, threshold=0.001, highpass_freq=50_000):
    """
    Streams a WAV file chunk by chunk and displays detected clicks in real time
    using the same detection functions from Detector-solorun.py.
    """
    # === Load the audio ===
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    chunk_size = int(sr * chunk_duration)

    # === Set up live plot ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], color="tab:blue", lw=1)
    scat = ax.scatter([], [], color="tab:orange", s=10)
    ax.set_xlim(0, chunk_duration)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.title("Real-time Click Detection")

    # === Streaming loop ===
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start+chunk_size]
        t = np.arange(start, start + len(chunk)) / sr

        # Filter + TK operator
        filtered = butter_pass_filter(chunk, highpass_freq, sr)
        tk = TeagerKaiser_operator(filtered)

        # Peak detection
        peaks = np.where(tk > threshold)[0]

        # Update display
        line.set_data(t, filtered)
        scat.set_offsets(np.c_[t[peaks], filtered[peaks]])
        ax.set_xlim(t[0], t[-1])
        plt.pause(0.05)  # adjust for smoother display

    plt.ioff()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "Audio_data/SCW1807_20200711_082500.wav"
    stream_and_display(file_path, chunk_duration=0.2, threshold=0.001, highpass_freq=50_000)
