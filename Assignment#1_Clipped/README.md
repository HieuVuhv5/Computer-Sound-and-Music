
# Homework #1 Assignment

**Author**: Hieu Vu  
**Title**: WAV Files

## Introduction
Build a program in Python that writes a sine wave to a WAV file named `sine.wav` in the current directory. The specifications for your sine wave are as follows:
- **Channels per frame**: 1 (mono)
- **Sample format**: 16-bit signed (values in the range -32767 to 32767)
- **Amplitude**: ¼ of the maximum possible 16-bit amplitude (values in the range -8192 to 8192)
- **Duration**: 1 second
- **Frequency**: 440Hz (440 cycles per second)
- **Sample rate**: 48000 samples per second

Generate the samples yourself using Python's `sin()` function.

### Clipped Wave
Extend your program to also write a WAV file named `clipped.wav` in the current directory. The clipped sine wave should have:
- ½ of the maximum amplitude (values in the range -16384 to 16384),
- But samples greater than ¼ maximum amplitude (8192) should be clipped to 8192,
- And samples less than ¼ minimum amplitude (-8192) should be clipped to -8192.

## How to Run

### Setup Environment:
This project uses the `numpy`, `wave`, and `sounddevice` Python libraries. To install them, run the following command in your terminal:
```bash
pip install numpy wave sounddevice
```

### Run the Program:
To run this project, execute the following command in your terminal:
```bash
python clipped.py
```
It will export 2 WAV files (`sine.wav` and `clipped.wav`) and play `clipped.wav`.
