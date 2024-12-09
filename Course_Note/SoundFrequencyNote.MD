
# SoundFrequencyNote.MD

## Summary:

In signal processing, the **time domain** represents signals as a function of time, while the **frequency domain** shows how a signal is composed of various frequency components. The **Fourier Transform** is a mathematical tool that transforms a signal from the time domain to the frequency domain, revealing the frequency components that make up the signal.

**Euler's Formula** connects complex exponentials with trigonometric functions, and the **Discrete Fourier Transform (DFT)** is a discrete version of the Fourier Transform, used for analyzing signals in digital form.

## Python Code Example:

The following Python code demonstrates the concepts of the time domain, frequency domain, and the DFT:

```python
import numpy as np
import matplotlib.pyplot as plt

# Signal Parameters
Fs = 500  # Sampling frequency in Hz
T = 1 / Fs  # Sampling period
t = np.arange(0, 1, T)  # Time vector of 1 second
f1, f2 = 50, 120  # Frequencies of sine waves in Hz

# Time-domain Signal (sum of two sine waves)
x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# DFT Calculation (Discrete Fourier Transform)
N = len(x)
X = np.zeros(N, dtype=complex)
for k in range(N):
    X[k] = sum(x[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))

# Frequency Axis
frequencies = np.fft.fftfreq(N, T)

# Plot Time Domain Signal
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot Frequency Domain (Magnitude of DFT)
plt.subplot(2, 1, 2)
plt.plot(frequencies[:N//2], np.abs(X)[:N//2])  # Only positive frequencies
plt.title('Frequency Domain (Magnitude of DFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
```

## Explanation:

1. **Time Domain**: The first plot shows the sum of two sine waves with frequencies 50 Hz and 120 Hz, which is the time-domain representation of the signal.
2. **Frequency Domain**: The second plot shows the magnitude of the DFT of the signal, representing the signal's frequency components. Peaks at 50 Hz and 120 Hz indicate the presence of those frequencies in the signal.
3. **Euler's Formula**: The DFT calculation uses complex exponentials, as represented by Euler's formula \( e^{-i2\pi rac{k}{N}} \), to decompose the signal into frequency components.

This code demonstrates the **Fourier Transform** process from the time domain to the frequency domain using the DFT.
