import noisereduce as nr
import soundfile as sf

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

    print(f"Success! Cleaned audio saved as {output_file}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
