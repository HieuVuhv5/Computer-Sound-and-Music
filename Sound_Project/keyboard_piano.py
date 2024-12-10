import re, math, queue, sounddevice, wave
import numpy as np
import keyboard
from types import SimpleNamespace
import argparse
import time

# Default wave shape
wave_shape = "piano"

# Debugging flags
log_notes = True
log_envelope = True

# Constants
sample_rate = 48000
blocksize = 64

# MIDI note mapping for keyboard keys
# Updated MIDI note mapping based on your requirement
key_to_note = {
    'z': 57,  # A3
    'x': 58,  # B3
    'c': 59,  # C4
    'v': 60,  # D4
    'b': 61,  # E4
    'n': 62,  # F4
    'm': 63,  # G4
    'a': 64,  # A4
    's': 65,  # B4
    'd': 66,  # C5
    'f': 67,  # D5
    'g': 68,  # E5
    'h': 69,  # F5
    'j': 70,  # G5
    'k': 71,  # A5
    'l': 72,  # B5
    'q': 73,  # C6
    'w': 74,  # D6
    'e': 75,  # E6
    'r': 76,  # F6
    't': 77,  # G6
    'y': 78,  # A6
    'u': 79,  # B6
    'i': 80,  # C7
    'o': 81,  # D7
    'p': 82,  # E7
}


# Global variables
wavetable = None
wave_frequency = 349
notes = []
current_notes = dict()
command_queue = queue.SimpleQueue()
sustaining = False


# Wave generation functions
def make_sin(f):
    period = sample_rate / f
    ncycles = math.ceil(blocksize / period)
    nsin = round(ncycles * period)
    t_period = np.linspace(0, ncycles * (2 * np.pi), nsin, dtype=np.float32)
    return 0.125 * np.sin(t_period)


def make_piano(f):
    # Sample block duration and time vector
    block_duration = blocksize / sample_rate
    t = np.linspace(0, block_duration, blocksize, endpoint=False)

    # Number of harmonics to generate (you can change this if necessary)
    num_harmonics = 12
    signal = np.zeros_like(t)

    # Generate the harmonics for the piano sound using sine waves
    for i in range(1, num_harmonics + 1):
        harmonic_freq = f * i

        # Attenuate harmonics for lower notes and boost for higher ones
        # Adjust the amplitude scaling to make lower notes warmer
        amplitude = 1 / (i + 1) * (f / 440)  # Scale amplitude by the fundamental frequency (A4 = 440 Hz)

        # Apply a frequency-dependent factor to make lower notes warmer (less bright)
        # For lower frequencies, reduce the higher harmonics more
        if f < 220:  # Below a certain threshold, dampen higher harmonics more significantly
            amplitude *= 0.5  # Lower harmonics will be stronger, higher harmonics weaker

        harmonic_wave = amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        signal += harmonic_wave  # Add the harmonic wave to the signal

    # Apply ADSR envelope (Attack, Decay, Sustain, Release)
    attack_time = 0.01
    decay_time = 0.1
    sustain_level = 0.5
    release_time = 0.2

    total_time = attack_time + decay_time + release_time
    attack_samples = int((attack_time / total_time) * blocksize)
    decay_samples = int((decay_time / total_time) * blocksize)
    release_samples = int((release_time / total_time) * blocksize)

    envelope = np.zeros(blocksize, dtype=np.float32)

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    decay_start = attack_samples
    decay_end = decay_start + decay_samples
    if decay_samples > 0:
        envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_samples)

    sustain_start = decay_end
    sustain_end = blocksize - release_samples
    if sustain_end > sustain_start:
        envelope[sustain_start:sustain_end] = sustain_level

    if release_samples > 0:
        envelope[sustain_end:] = np.linspace(sustain_level, 0, release_samples)

    # Apply the envelope to the signal
    signal *= envelope
    signal = np.clip(signal, -1, 1)

    return signal





wave_types = {
    "sine": make_sin,
    "piano": make_piano,
}

# Argument parsing for --type
parser = argparse.ArgumentParser(description="Generate audio with different wave types.")
parser.add_argument("--type", choices=wave_types.keys(), default="piano", help="Type of wave to generate.")
args = parser.parse_args()

# Set wave shape from the argument
wave_shape = args.type
# Preload notes
for note in range(128):
    f = 440 * 2 ** ((note - 69) / 12)
    notes.append(wave_types[wave_shape](f))


class Note:
    def __init__(self, key):
        self.t = 0
        self.key = key
        self.release_rate = None
        attack_samples = 10 * sample_rate / 1000
        self.attack_rate = 1.0 / attack_samples
        self.attack_amplitude = 0
        self.wave_table = notes[key]
        self.held = False
        self.start_time = time.time()  # Track when the note starts
        self.max_duration = 2  # Maximum duration in seconds for a note
        self.released = False  # Track if the note has been automatically released

    def play(self, frame_count):
        # Check if the note should be stopped after max duration
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration and not self.released:
            # Automatically release the note after the threshold
            if log_envelope:
                print("Note held too long, auto-releasing:", self.key)
            self.release()  # Release the note
            self.released = True  # Mark the note as released
            return None  # Stop the sound

        wave_table = self.wave_table
        t_output = self.t
        nwave_table = len(wave_table)
        t_start = t_output % nwave_table
        t_end = (t_output + frame_count) % nwave_table
        if t_start < t_end:
            output = wave_table[t_start:t_end]
        else:
            output = np.append(wave_table[t_start:], wave_table[:t_end])

        # Handle the release phase (if applicable)
        if self.release_rate and not self.held:
            if self.release_amplitude <= 0:
                if log_envelope:
                    print("finishing note", self.key, self.t)
                return None
            end_amplitude = self.release_amplitude - frame_count * self.release_rate
            scale = np.linspace(self.release_amplitude, end_amplitude, frame_count).clip(0, 1)
            output = output * scale
            self.release_amplitude = max(end_amplitude, 0)

        # Handle the attack phase
        if self.attack_rate:
            end_amplitude = self.attack_amplitude + frame_count * self.attack_rate
            scale = np.linspace(self.attack_amplitude, end_amplitude, frame_count).clip(0, 1)
            output = output * scale
            if end_amplitude >= 1:
                if log_envelope:
                    print("finishing attack", self.key, self.t)
                self.attack_rate = None
            else:
                self.attack_amplitude = end_amplitude

        self.t += frame_count
        return output

    def release(self):
        if log_envelope:
            print("releasing note", self.key, self.t)
        release_samples = 100 * sample_rate / 1000
        self.release_rate = 1.0 / release_samples
        self.release_amplitude = 1.0
        if self.attack_rate:
            self.release_amplitude = self.attack_amplitude
            self.attack_rate = None

    def hold(self):
        self.held = True

    def unhold(self):
        self.held = False


# Audio output callback
def output_callback(out_data, frame_count, time_info, status):
    global current_notes, command_queue, sustaining

    if status:
        print("output callback:", status)

    while not command_queue.empty():
        mesg_type, mesg = command_queue.get_nowait()
        if mesg_type == 'note_on':
            key = mesg.note
            new_note = Note(key)
            if sustaining:
                new_note.held = True
            current_notes[key] = new_note
        elif mesg_type == 'note_off':
            key = mesg.note
            if key in current_notes:
                current_notes[key].release()
            else:
                print(f"Warning: Tried to release a non-existent note {key}")
        elif mesg_type == 'sustain_pedal':
            sustaining = mesg.value > 0
            for note in current_notes.values():
                if sustaining:
                    note.hold()
                else:
                    note.unhold()
        else:
            print(f"Unrecognized command: {mesg_type} {mesg}")

    output = np.zeros(frame_count, dtype=np.float32)
    finished_keys = []
    for key, note in current_notes.items():
        sound = note.play(frame_count)
        if sound is None:
            finished_keys.append(key)
        else:
            output += sound

    for key in finished_keys:
        del current_notes[key]

    out_data[:] = output.reshape(frame_count, 1)


# Global variable to track key state

key_held = {}  # Track if a key is currently held down
key_press_time = {}  # Track the press time of each key
key_released = {}  # Track if the key has been released after sound turned off

def get_keyboard_event():
    global sustaining, key_press_time, key_held, key_released
    event = keyboard.read_event(suppress=True)
    key = event.name
    current_time = time.time()

    # Check if the key has been pressed for too long
    if key in key_press_time and event.event_type == 'down':
        press_duration = current_time - key_press_time[key]
        if press_duration >= 0.02 and not key_released.get(key, False):
            # If the key has been held for more than 0.1 second, stop the sound
            print(f"Key {key} has been held for {press_duration:.2f} seconds. Stopping sound.")
            command_queue.put(('note_off', SimpleNamespace(note=key_to_note[key], velocity=0)))
            key_released[key] = True  # Mark that the key's sound has been turned off

    if key in key_to_note:
        note = key_to_note[key]
        if event.event_type == 'down':
            # If the key is not held down already and hasn't had its sound turned off
            if not key_held.get(key, False) and not key_released.get(key, False):
                key_press_time[key] = current_time  # Record the press time for the key
                key_held[key] = True  # Mark key as held down
                command_queue.put(('note_on', SimpleNamespace(note=note, velocity=127)))  # Full velocity
        elif event.event_type == 'up':
            # When the key is released, turn off the note
            if key_held.get(key, False):
                command_queue.put(('note_off', SimpleNamespace(note=note, velocity=0)))
                key_held[key] = False  # Mark the key as no longer held down
                key_released[key] = False  # Reset the key released state
                key_press_time.pop(key, None)  # Remove the key from tracking

    elif key == 'space':
        if event.event_type == 'down':
            sustaining = not sustaining
            for note in current_notes.values():
                if sustaining:
                    note.hold()
                else:
                    note.unhold()
    elif key == 'esc':
        return False
    return True





# Start audio stream
output_stream = sounddevice.OutputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=blocksize,
    callback=output_callback,
)
output_stream.start()

# Main event loop
try:
    while get_keyboard_event():
        pass
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    output_stream.stop()
    output_stream.close()
    print("Audio stream stopped.")
