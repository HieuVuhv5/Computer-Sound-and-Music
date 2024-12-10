# Course Project 1: Keyboard Organ

Author: Hieu Vu

This project is heavily based on [rhosy](https://github.com/pdx-cs-sound/rhosy). 
## Introduction
This program is a virtual keyboard organ that maps computer keyboard keys to musical notes, allowing you to play sounds in real-time. It supports three main wave types:

- **MIDI**: Generates sine waves to mimic basic instrument tones.
- **Harmonica**: Simulates a harmonica sound by summing harmonics with an envelope.
- **Cat Meow (WAV file)**: Uses a `cat.wav` file as the source waveform, with additional processing to mimic a cat meowing.

The program also includes an optional **auto-wah effect**, which dynamically modulates the frequency of a filter applied to the sound. This effect works exceptionally well with the "Cat Meow" wave type because it emphasizes the harmonic complexity and variation present in the waveform.

Once you finish playing, pressing the `ESC` key stops the program and saves the generated audio to a `.wav` file if the `--output` argument is provided.

## Keyboard Mapping
The keys from `z` to `p` are mapped to MIDI notes, starting from `C3` (note 57) for `z` and ascending chromatically to `E7` (note 82) for `p`. Each key press triggers the corresponding note, and notes can be sustained by holding the key.

## Features
- Real-time audio synthesis and playback.
- Support for multiple wave types (`midi`, `harmonica`, `catmeow`).
- Auto-wah effect for dynamic modulation (particularly suited for `catmeow`).
- Option to save the generated audio to a `.wav` file.

## How to Play
### Basic Commands
1. **Play with Cat Meow and Auto-Wah**
   ```bash
   python keyboard_organ --type catmeow --effect wah
   ```
   - **Explanation**: The auto-wah effect modulates the filter frequency over time, creating a "wah-wah" sound. This is especially effective for the `catmeow` wave type because the `cat.wav` file contains rich harmonic content, and the modulation highlights these harmonics, creating a dynamic and expressive sound.

2. **Play with MIDI Wave Type**
   ```bash
   python keyboard_organ --type midi
   ```
   - **Explanation**: This uses a basic sine wave to produce simple tones for each note.

3. **Play with Harmonica Wave Type**
   ```bash
   python keyboard_organ --type harmonica
   ```
   - **Explanation**: Simulates a harmonica by summing harmonics and applying an envelope to mimic breathiness.

4. **Play Cat Meow Without Effects**
   ```bash
   python keyboard_organ --type catmeow
   ```
   - **Explanation**: Plays the cat meow sound as-is, without any additional effects.

### Exporting Audio to a File
You can save your performance to a `.wav` file by providing the `--output` argument. For example:
```bash
python keyboard_organ --type harmonica --output demo.wav
```
- **Explanation**: This will save the generated harmonica sound to `demo.wav` after you press `ESC` to stop the program.

## Additional Notes on Auto-Wah
The auto-wah effect in this program is implemented using a dynamic filter that sweeps its cutoff frequency based on a sinusoidal contour. When applied to the `catmeow` wave type, derived from the `cat.wav` file, the effect enhances the natural texture of the waveform. The harmonic richness of the cat meow sound interacts with the filter modulation, producing a lively and captivating output.


Enjoy experimenting with the keyboard organ and creating your own sounds!

