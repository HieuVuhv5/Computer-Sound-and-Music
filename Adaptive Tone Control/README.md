# HW2 Tone Adaptive
**Author**: Hieu Vu

This is a Python-based Tone Adaptive Equalizer, built for dynamically adjusting the volume of different frequency bands in an audio file. The script uses FFT (Fast Fourier Transform) and IIR filters from SciPy to process low, mid, and high-frequency bands, allowing real-time audio equalization.

### Overview
The program reads an input WAV file, analyzes its frequency content using FFT, and applies dynamic gain adjustments to different frequency bands based on their energy. IIR filters are applied for low-pass, band-pass, and high-pass filtering to target the low, mid, and high-frequency bands respectively.

### Key Features
- **FFT Band Energy**: Calculates the energy in specific frequency bands (low, mid, high) using FFT.
- **Dynamic Equalization**: Adjusts the gain dynamically for each frequency band based on its energy, resulting in a balanced audio output.
- **Scipy IIR Filters**: Uses `scipy.signal.iirfilter` to create filters for processing the low, mid, and high-frequency bands.
  
### Frequency Bands
The equalizer processes the following frequency bands:
- **Low**: 0 Hz to 300 Hz
- **Mid**: 300 Hz to 2000 Hz
- **High**: 2000 Hz to 8000 Hz

### Dependencies
You need the following Python libraries to run the script:
- `scipy`
- `numpy`
- `sounddevice`

### How to Run
The script supports four modes of operation:
 -   Low Frequency Band Processing (--band low) :python tone.py gc.wav --band low --volume 2.0 
 -   Mid Frequency Band Processing (--band mid) : python tone.py gc.wav --band mid --volume 2.0 
 -   High Frequency Band Processing (--band high): python tone.py gc.wav --band high --volume 2.0 
 -   Average Band Energy Processing (--band average), where dynamic equalization is applied across all bands: python tone.py gc.wav --band average --volume 1.2