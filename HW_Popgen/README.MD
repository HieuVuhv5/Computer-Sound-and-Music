# HW Popgen

**Author**: Hieu Vu

## Introduction

This project focuses on sound generation and processing for musical applications. The program can generate sound using both sine waveforms and custom wavetable (e.g., `cat.wav`). The code includes functionalities for generating melodies and bass lines, processing them through different types of sounds, and supporting user-defined chord progressions.

Key features include:
- **Waveform Generation**: The program can generate sound using sine waves or a custom waveform loaded from a `.wav` file (like `cat.wav`).
- **Sound Envelope**: The program simulates a basic envelope for the "piano" type sound, including Attack, Sustain, and Release stages, allowing for more expressive sound design.
- **Chord Loop**: The user can define a chord progression to create a repeating cycle of chords for melody and bass generation.

This program provides flexibility for experimenting with different sound synthesis techniques and musical structures.

## Features

- **Waveform Options**:
  - **Sine Wave**: Default sound generation using pure sine waves.
  - **Custom Waveform**: Optionally load a custom waveform from a `.wav` file (such as `cat.wav`) to generate sound.
  
- **Sound Envelope with `--type piano`**: 
  - When the `--type` is set to `piano`, the sound is processed with an envelope that simulates the dynamics of a piano note: Attack, Sustain, and Release. The envelope allows for smooth transitions in volume, replicating how real piano notes behave.
  
- **Chord Loop (`--chord-loop`)**: 
  - The program allows users to specify a chord progression through the `--chord-loop` argument. The sequence is parsed and used to generate both the melody and bass lines. Chords are picked based on the scale degrees and mapped to specific notes for musical structure.

## How to Run
### 1. **Piano Sound**
Run the script to generate piano-like notes with ADSR envelopes:

```bash
python popgen.py --type piano
```

### 2. **MIDI Sound**
Use MIDI note generation for a pure sine wave:

```bash
python popgen.py --type MIDI
```
### 3. **Custom Waveform**
Enable a custom waveform for note generation:

```bash
python popgen.py --waveform yes
```
### 4. **Chord Loop**
Define your own chord progression:

```bash
python popgen.py --type piano --chord-loop "1,5,6,4"
```