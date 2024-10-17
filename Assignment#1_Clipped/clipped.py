import wave
import numpy as np
import struct
import sounddevice as sd

# Wave header format

N_CHANNELS = 1  # Mono
SAMPLE_FORMAT = 32767   # 16-bit signed (values in the range -32767 to 32767
AMPLITUDE = 8192  # 1/4 of maximum 16-bit signed integer amplitude for sine.wav file
CLIPPED_AMPLITUDE = 16384  # 1/2 of maximum 16-bit signed integer amplitude for clipped.wav file
DURATION = 1  # Duration of the sine wave in seconds
FREQUENCY = 440  # 440 cycles per second
SAMPLE_RATE = 48000  # Samples per second
SAMPLE_WIDTH = 2  # In  wave library, the SAMPLE_WIDTH parameter specifies the number of bytes used to store each sample
# in the audio file, in this case 16-bit/8-bit = 2.


# Function to write to WAV file, using wave library
def write_wave_file(filename, data, sample_rate, num_channels, sample_width):
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        for sample in data:
            wav_file.writeframes(struct.pack('<h', sample))

# Generate sine wave samples
t_array = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)  # Time array
sine_wave = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * t_array)  # Sine wave calculation

# Convert the sine wave to 16-bit PCM format
sine_wave_pcm = (sine_wave.astype(np.int16)).tolist()

# Write sine.wav
write_wave_file("sine.wav", sine_wave_pcm, SAMPLE_RATE, N_CHANNELS, SAMPLE_WIDTH)

# Generate clipped sine wave samples
clipped_wave = CLIPPED_AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * t_array)
clipped_wave = np.clip(clipped_wave, -AMPLITUDE, AMPLITUDE)  # Clip to range [-8192, 8192]
clipped_wave_pcm = (clipped_wave.astype(np.int16)).tolist()

# Write clipped.wav
write_wave_file("clipped.wav", clipped_wave_pcm, SAMPLE_RATE, N_CHANNELS, SAMPLE_WIDTH)

# Play the clipped.wav
sd.play(clipped_wave.astype(np.int16), SAMPLE_RATE)
sd.wait()  # Wait the sound finishes playing