import re, math, queue, sounddevice, wave
import numpy as np
import keyboard
from types import SimpleNamespace
import argparse
import time
import scipy.signal as signal  # Add this line
# Default wave shape
wave_shape = "piano"

# Debugging flags
log_notes = True
log_envelope = True

# Constants
sample_rate = 48000
blocksize = 64

# MIDI note mapping for keyboard keys
key_to_note = {
    # Row 1: QWERTY row for C3 to D4
    'q': 48,  # C3
    'w': 49,  # D3
    'e': 50,  # E3
    'r': 51,  # F3
    't': 52,  # G3
    'y': 53,  # A3
    'u': 54,  # B3
    'i': 55,  # C4
    'o': 56,  # D4
    'p': 57,  # E4

    # Row 2: ASDF row for C4 to D5
    'a': 60,  # C4
    's': 61,  # D4
    'd': 62,  # E4
    'f': 63,  # F4
    'g': 64,  # G4
    'h': 65,  # A4
    'j': 66,  # B4
    'k': 67,  # C5
    'l': 68,  # D5
}

# Global variables
wavetable = None
wave_frequency = 349
notes = []
current_notes = dict()
command_queue = queue.SimpleQueue()
sustaining = False

# Wave generation functions
def make_midi(f):
    period = sample_rate / f
    # Need enough cycles to be able to wrap around when
    # generating a block.
    ncycles = math.ceil(blocksize / period)
    nsin = round(ncycles * period)
    t_period = np.linspace(0, ncycles * (2 * np.pi), nsin, dtype=np.float32)
    # Allow for eight notes before clipping.
    return 0.125 * np.sin(t_period)


# Return an ascending saw wave of frequency f.
# XXX Currently broken.
def make_harmonica(f):
    period = round(sample_rate / f)
    ncycles = math.ceil(blocksize / period)
    nsaw = ncycles * period
    t = np.linspace(0, ncycles, nsaw, dtype=np.float32)

    # Generate a harmonica-like sound by summing harmonics with decreasing amplitude
    harmonics = 8  # Number of harmonics to include
    harmonica_wave = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        harmonic_freq = f * n
        if harmonic_freq > sample_rate / 2:  # Nyquist limit
            break
        amplitude = 0.125 / n  # Reduce amplitude of higher harmonics
        harmonica_wave += amplitude * np.sin(2 * np.pi * harmonic_freq * t / sample_rate)

    # Apply a simple envelope to mimic the breathiness of a harmonica
    envelope = np.exp(-5 * t / sample_rate)  # Exponential decay
    harmonica_wave *= envelope

    return harmonica_wave


# Return a wave table built from a sound.
# XXX WAV file currently hardwired.
wavetable = None
# XXX WAV frequency currently hardwired.
wave_frequency = 349

def make_catmeow(f):
    global wavetable
    if wavetable is None:
        wav = wave.open("cat.wav", "rb")
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 48000
        assert wav.getsampwidth() == 2
        nframes = wav.getnframes()
        w = wav.readframes(nframes)
        wavetable = np.frombuffer(w, dtype=np.int16).astype(np.float32) / 32768
    nframes_in = len(wavetable)
    nframes_out = int(nframes_in * wave_frequency / f)
    xs_out = np.linspace(0, nframes_in, nframes_out)
    xs_in = np.linspace(0, nframes_in, nframes_in)
    waves = np.interp(xs_out, xs_in, wavetable)
    return 0.125 * waves


def make_piano(f):
    # Define block size in seconds
    block_duration = blocksize / sample_rate

    # Time array for the block
    t = np.linspace(0, block_duration, blocksize, endpoint=False)

    # Piano harmonics: rich harmonic structure
    num_harmonics = 12
    signal = np.zeros_like(t)

    # Generate the harmonics
    for i in range(1, num_harmonics + 1):
        harmonic_freq = f * i
        amplitude = 1 / (i + 1)  # Decreasing amplitude for higher harmonics
        signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t)

    # ADSR envelope
    attack_time = 0.01  # 10ms
    decay_time = 0.1   # 100ms
    sustain_level = 0.5  # Sustain amplitude
    release_time = 0.2  # 200ms

    # Proportional number of samples for each phase (constrained by blocksize)
    total_time = attack_time + decay_time + release_time
    attack_samples = int((attack_time / total_time) * blocksize)
    decay_samples = int((decay_time / total_time) * blocksize)
    release_samples = int((release_time / total_time) * blocksize)

    # Envelope generation for the current block
    envelope = np.zeros(blocksize, dtype=np.float32)

    # Attack phase
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay phase
    decay_start = attack_samples
    decay_end = decay_start + decay_samples
    if decay_samples > 0:
        envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_samples)

    # Sustain phase
    sustain_start = decay_end
    sustain_end = blocksize - release_samples
    if sustain_end > sustain_start:
        envelope[sustain_start:sustain_end] = sustain_level

    # Release phase
    if release_samples > 0:
        envelope[sustain_end:] = np.linspace(sustain_level, 0, release_samples)

    # Apply the envelope
    signal *= envelope

    # Normalize the signal to avoid clipping
    signal = np.clip(signal, -1, 1)

    return signal




wave_types = {
    "midi": make_midi,
    "harmonica": make_harmonica,
    "catmeow": make_catmeow,
    "piano": make_piano,
}
# Argument parsing for --type
parser = argparse.ArgumentParser(description="Generate audio with different wave types.")
parser.add_argument("--type", choices=wave_types.keys(), default="piano", help="Type of wave to generate.")
parser.add_argument("--effect", choices=["none", "wah"], default="none", help="Effect to apply (none/wah).")
args = parser.parse_args()

# Set wave shape from the argument
wave_shape = args.type
effect_type = args.effect
# Preload notes
for note in range(128):
    f = 440 * 2 ** ((note - 69) / 12)
    notes.append(wave_types[wave_shape](f))

# Note class
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

    def play(self, frame_count):
        wave_table = self.wave_table
        t_output = self.t
        nwave_table = len(wave_table)
        t_start = t_output % nwave_table
        t_end = (t_output + frame_count) % nwave_table
        if t_start < t_end:
            output = wave_table[t_start:t_end]
        else:
            output = np.append(wave_table[t_start:], wave_table[:t_end])
        if self.release_rate and not self.held:
            if self.release_amplitude <= 0:
                if log_envelope:
                    print("finishing note", self.key, self.t)
                return None
            end_amplitude = self.release_amplitude - frame_count * self.release_rate
            scale = np.linspace(
                self.release_amplitude,
                end_amplitude,
                frame_count,
            ).clip(0, 1)
            output = output * scale
            self.release_amplitude = max(end_amplitude, 0)
        if self.attack_rate:
            end_amplitude = self.attack_amplitude + frame_count * self.attack_rate
            scale = np.linspace(
                self.attack_amplitude,
                end_amplitude,
                frame_count,
            ).clip(0, 1)
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

# Wah effect generation


# Wah effect parameters
wah_rate = 2.0  # Wah rate (how fast the wah filter modulates)
wah_depth = 0.2  # Wah depth (the minimum cutoff frequency)
wah_height = 0.9  # Wah height (the maximum cutoff frequency)

# Initialize wah filter
wah_filter = None


def generate_wah_filter():
    global wah_filter
    wah_rate_ = wah_rate  # Adjust this based on how fast you want the wah filter to change
    ncontour = int(wah_rate_ * sample_rate / blocksize)
    scale = wah_height - wah_depth
    cspace = np.linspace(0, np.pi, ncontour)
    contour = scale * np.log2(2 - np.sin(cspace)) + wah_depth
    wah_filter = [signal.firwin(128, c, window=('kaiser', 0.5)) for c in contour]
    print("Wah filter generated.")


def apply_wah_effect(signal_data):
    global wah_filter
    if wah_filter is None:
        print("Wah filter is None. Generating filter.")
        generate_wah_filter()

    ncontour = len(wah_filter)
    icontour = 0
    processed_signal = np.zeros_like(signal_data)
    for i in range(0, len(signal_data), blocksize):
        end = min(i + blocksize, len(signal_data))
        block = signal_data[i:end]
        block = signal.convolve(block, wah_filter[icontour], mode='same')
        processed_signal[i:end] += block
        icontour = (icontour + 1) % ncontour
    return processed_signal


# Audio output callback
def output_callback(out_data, frame_count, time_info, status):
    global current_notes, command_queue, sustaining, effect_type  # Declare effect_type as global

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

        # Apply the wah effect if enabled and wave type is "table"
        if effect_type == "wah" and wave_shape == "table":
            print("Applying Wah effect...")
            output = apply_wah_effect(output)
            print("Auto Wah effect active.")

    for key in finished_keys:
        del current_notes[key]

    out_data[:] = output.reshape(frame_count, 1)


# Keyboard event processing
# Track when keys are pressed down and held
key_hold_time = {}
key_press_threshold = 0.5  # Threshold in seconds for a long press



# Track when keys are pressed down and held
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
        if press_duration >= 5.0 and not key_released.get(key, False):
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
