import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import fft

# Streamlit app
st.title("Audio Signal Visualizer")

# Upload audio file
uploaded_file = st.file_uploader("new.wav", type=["wav"])

if uploaded_file is not None:
    # Load the audio file
    audio_data, sample_rate = sf.read(uploaded_file)
    
    # Handle multi-channel audio (convert to mono)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Display audio player
    st.audio(uploaded_file)

    # Plot waveform
    st.subheader("Waveform (Amplitude vs. Time)")
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    fig_waveform, ax_waveform = plt.subplots()
    ax_waveform.plot(time, audio_data)
    ax_waveform.set_xlabel("Time (seconds)")
    ax_waveform.set_ylabel("Amplitude")
    ax_waveform.set_title("Waveform")
    st.pyplot(fig_waveform)

    # Plot frequency spectrum
    st.subheader("Frequency Spectrum (FFT)")
    n = len(audio_data)
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    fft_magnitude = np.abs(fft(audio_data))[:len(freq)]
    fig_spectrum, ax_spectrum = plt.subplots()
    ax_spectrum.plot(freq, fft_magnitude)
    ax_spectrum.set_xlabel("Frequency (Hz)")
    ax_spectrum.set_ylabel("Magnitude")
    ax_spectrum.set_title("Frequency Spectrum")
    st.pyplot(fig_spectrum)

    # Display basic audio stats
    st.write(f"Sample rate: {sample_rate} Hz")
    st.write(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
