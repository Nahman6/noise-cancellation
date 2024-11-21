import noisereduce as nr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

try:
    # Load the audio file
    print("Loading audio file...")
    input_file = "new.wav"
    audio_data, sample_rate = sf.read(input_file)
    print(f"Audio data shape: {audio_data.shape}, Sample rate: {sample_rate}")

    # Handle multi-channel audio
    if len(audio_data.shape) > 1:
        print("Converting to mono...")
        audio_data = audio_data.mean(axis=1)

    # Plot original waveform
    print("Plotting original audio...")
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_data, label="Original Audio")
    plt.title("Waveform: Original Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot original frequency spectrum
    print("Plotting original frequency spectrum...")
    n = len(audio_data)
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    fft_original = np.abs(np.fft.rfft(audio_data))
    plt.figure(figsize=(12, 4))
    plt.plot(freq, fft_original, label="Original Audio Spectrum")
    plt.title("Frequency Spectrum: Original Audio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

    # Get noise profile from first 1 second
    print("Extracting noise profile...")
    noise_clip = audio_data[:min(len(audio_data), sample_rate)]

    # Perform noise reduction
    print("Reducing noise...")
    reduced_noise = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        y_noise=noise_clip,
        prop_decrease=0.8
    )

    # Save the processed audio
    print("Saving processed audio...")
    output_file = "cleaned_audio.wav"
    sf.write(output_file, reduced_noise, sample_rate)

    # Plot reduced waveform
    print("Plotting noise-reduced audio...")
    plt.figure(figsize=(12, 4))
    plt.plot(time, reduced_noise, label="Noise-Reduced Audio", color='orange')
    plt.title("Waveform: Noise-Reduced Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot reduced frequency spectrum
    print("Plotting noise-reduced frequency spectrum...")
    fft_reduced = np.abs(np.fft.rfft(reduced_noise))
    plt.figure(figsize=(12, 4))
    plt.plot(freq, fft_reduced, label="Noise-Reduced Audio Spectrum", color='orange')
    plt.title("Frequency Spectrum: Noise-Reduced Audio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Success! Cleaned audio saved as {output_file}")

except Exception as e:
    print(f"An error occurred: {str(e)}")