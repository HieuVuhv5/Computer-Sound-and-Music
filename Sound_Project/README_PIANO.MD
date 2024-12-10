# README_Piano.MD

## Small Course Project
Author: Hieu Vu

### Heavily Based On:
Rhosy ( [https://github.com/pdx-cs-sound/rhosy](https://github.com/pdx-cs-sound/rhosy) )

---

## Introduction:
This project aims to simulate the sound and behavior of a hammer piano using a computer keyboard. It provides an interactive and dynamic audio synthesis experience by replicating the piano’s keypress-triggered hammer mechanism.

Key features include:
- **Sound Envelope Simulation**: Implements an attack-decay-sustain-release (ADSR) envelope to mimic the dynamics of a piano’s sound. The envelope creates a realistic fade-in and fade-out effect.
- **Auto Cutoff Mechanism**: Automatically stops sound playback if a key is pressed and held for too long, simulating the behavior of a real piano.

### Key Mapping:
The keyboard keys from **z** to **p** are mapped to MIDI notes, starting from **A3** (note 57) for `z` and ascending chromatically to **E7** (note 82) for `p`. Each key press triggers the corresponding MIDI note.

- Holding down a key sustains the note as long as it’s pressed.
- The sound automatically fades out after a specific duration, even if the key is held down, to emulate real piano behavior.

---

## How to Play:
1. **Run the Program:**
   ```bash
   python keyboard_piano.py
   ```
2. **Play Notes:**
   - Press keys from `z` to `p` to play notes, starting from **C3** to **E7**.
   - Hold keys for sustained notes, or release them for shorter sounds.
3. **Control Sustain:**
   - Press the `space` key to toggle the sustain pedal effect.
   - Sustained notes will continue to play even after you release the keys.
4. **Exit the Program:**
   - Press `Esc` to stop the program.

---

### Additional Notes:
- **Sound Design:** The synthesized piano sound is created using harmonics and an ADSR envelope to replicate the warm and dynamic tonal qualities of a hammer piano.
- **Dependencies:** Make sure to have the necessary Python libraries installed, including `sounddevice`, `numpy`, and `keyboard`.

Enjoy playing your virtual piano!

