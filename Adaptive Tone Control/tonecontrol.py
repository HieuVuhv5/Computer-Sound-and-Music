import argparse
from scipy import io, signal
import numpy as np
import sounddevice as sd
import sys

# Constants
BUFFER_SIZE = 2048
BANDS = [(0, 300), (300, 2000), (2000, 8000)]  # Low, mid, high frequency bands

def read_wav(filename):
    """Read a WAV file and return sample rate and data."""
    rate, data = io.wavfile.read(filename)
    assert data.dtype == np.int16
    data = data.astype(np.float64) / 32768
    return rate, data

def write_wav(filename, rate, data):
    """Write a WAV file."""
    data = np.clip(data, -1, 1)
    data = (data * 32767).astype(np.int16)
    io.wavfile.write(filename, rate, data)

def play(rate, wav):
    """Play an audio buffer."""
    channels = 1 if wav.ndim == 1 else wav.shape[1]
    stream = sd.RawOutputStream(samplerate=rate, blocksize=BUFFER_SIZE, channels=channels, dtype='float32')
    stream.start()
    indices = np.arange(BUFFER_SIZE, wav.shape[0], BUFFER_SIZE)
    samples = np.ascontiguousarray(wav, dtype=np.float32)
    for buffer in np.array_split(samples, indices):
        stream.write(buffer)
    stream.stop()
    stream.close()

def fft_band_energy(data, rate, bands):
    """Compute energy in each frequency band using FFT."""
    fft_size = BUFFER_SIZE
    freqs = np.fft.rfftfreq(fft_size, 1 / rate)
    fft_vals = np.abs(np.fft.rfft(data, n=fft_size))

    band_energy = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy.append(np.sum(fft_vals[mask]))
    return band_energy

def dynamic_equalization(data, rate, bands, volume=1.0):
    """Dynamically equalize audio to balance band energies and adjust volume."""
    num_samples = len(data)
    output = np.zeros_like(data)

    for start in range(0, num_samples, BUFFER_SIZE):
        end = min(start + BUFFER_SIZE, num_samples)
        window = data[start:end]

        # FFT to measure band energy
        energy = fft_band_energy(window, rate, bands)
        average_energy = np.mean(energy)

        # Calculate gain for each band
        gains = [average_energy / (e + 1e-6) for e in energy]  # Avoid division by zero

        # Apply filters dynamically
        coeffs_low = signal.iirfilter(10, bands[0][1] / (rate / 2), btype='lowpass', output='sos')
        coeffs_mid = signal.iirfilter(10, [bands[1][0] / (rate / 2), bands[1][1] / (rate / 2)], btype='bandpass', output='sos')
        coeffs_high = signal.iirfilter(10, bands[2][0] / (rate / 2), btype='highpass', output='sos')

        low = gains[0] * signal.sosfilt(coeffs_low, window)
        mid = gains[1] * signal.sosfilt(coeffs_mid, window)
        high = gains[2] * signal.sosfilt(coeffs_high, window)

        output[start:end] = low[:len(window)] + mid[:len(window)] + high[:len(window)]

    # Normalize and apply volume adjustment
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output /= max_val
    output *= volume
    return output

def process_low_band(rate, data, volume):
    """Process the low-frequency band."""
    coeffs = signal.iirfilter(10, 0.05, btype='lowpass', output='sos')
    out_data = signal.sosfilt(coeffs, data)
    out_data *= volume
    return out_data

def process_mid_band(rate, data, volume):
    """Process the mid-frequency band."""
    low_cutoff = 300 / (rate / 2)
    high_cutoff = 2000 / (rate / 2)
    coeffs = signal.iirfilter(6, [low_cutoff, high_cutoff], btype='bandpass', ftype='butter', output='sos')
    out_data = signal.sosfilt(coeffs, data)
    max_val = np.max(np.abs(out_data))
    if max_val > 0:
        out_data /= max_val
    out_data *= volume
    return out_data

def process_high_band(rate, data, volume):
    """Process the high-frequency band."""
    coeffs = signal.iirfilter(10, 0.02, btype='highpass', output='sos')
    out_data = signal.sosfilt(coeffs, data)
    out_data *= volume
    return out_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Tone Equalizer")
    parser.add_argument("wav", help="Input WAV file")
    parser.add_argument("--out", help="Output WAV file", default=None)
    parser.add_argument("--band", choices=["low", "mid", "high", "average"], default="average", help="Frequency band to process")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume adjustment factor")
    args = parser.parse_args()

    # Read input file
    rate, data = read_wav(args.wav)

    # Process based on selected band
    if args.band == "low":
        processed_data = process_low_band(rate, data, args.volume)
    elif args.band == "mid":
        processed_data = process_mid_band(rate, data, args.volume)
    elif args.band == "high":
        processed_data = process_high_band(rate, data, args.volume)
    else:  # average
        processed_data = dynamic_equalization(data, rate, BANDS, args.volume)

    # Save or play the result
    if args.out:
        write_wav(args.out, rate, processed_data)
    else:
        play(rate, processed_data)
