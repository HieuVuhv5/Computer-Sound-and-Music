import argparse, random, re, wave
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
# Define note names and functions
names = [ "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
note_names = { s : i for i, s in enumerate(names) }
note_name_re = re.compile(r"([A-G]b?)(\[([0-8])\])?")
def parse_note(s):
    m = note_name_re.fullmatch(s)
    if m is None:
        raise ValueError
    s = m[1]
    s = s[0].upper() + s[1:]
    q = 4
    if m[3] is not None:
        q = int(m[3])
    return note_names[s] + 12 * q

def parse_log_knob(k, db_at_zero=-40):
    v = float(k)
    if v < 0 or v > 10:
        raise ValueError
    if v < 0.1:
        return 0
    if v > 9.9:
        return 10
    return 10**(-db_at_zero * (v - 10) / 200)

def parse_linear_knob(k):
    v = float(k)
    if v < 0 or v > 10:
        raise ValueError
    return v / 10

def parse_db(d):
    v = float(d)
    if v > 0:
        raise ValueError
    return 10**(v / 20)

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('--bpm', type=int, default=90)
ap.add_argument('--samplerate', type=int, default=48_000)
ap.add_argument('--root', type=parse_note, default="C[5]")
ap.add_argument('--bass-octave', type=int, default=2)
ap.add_argument('--balance', type=parse_linear_knob, default="5")
ap.add_argument('--gain', type=parse_db, default="-3")
ap.add_argument('--type', type=str, default='MIDI', choices=['MIDI', 'piano'],
                help="Specify the note generation type. Default is MIDI.")
ap.add_argument('--chord-loop', type=str, default="1,5,6,4",
                help="Enter chord loop as a comma-separated list of chord progressive (e.g., '1,4,5,1').")
ap.add_argument('--waveform', type=str, choices=['yes', 'no'], default='no',
                help="Use 'yes' to enable a custom waveform (cat.wav). Default is 'no'.")

ap.add_argument('--output')
ap.add_argument("--test", action="store_true", help=argparse.SUPPRESS)
args = ap.parse_args()

bpm = args.bpm
samplerate = args.samplerate
beat_samples = int(np.round(samplerate / (bpm / 60)))

major_scale = [0, 2, 4, 5, 7, 9, 11]
major_chord = [1, 3, 5]



def note_to_key_offset(note):
    scale_degree = note % 7
    return note // 7 * 12 + major_scale[scale_degree]

def chord_to_note_offset(posn):
    chord_posn = posn % 3
    return posn // 3 * 7 + major_chord[chord_posn] - 1

# Load wav form from input file (.wav)
def load_waveform(filename):
    samplerate, data = wavfile.read(filename)
    # Normalize the waveform to fit within [-1, 1]
    waveform = data / np.max(np.abs(data))
    # Use only a single cycle or a short section for looping
    single_cycle = waveform[:samplerate // 440]  # Approximately one cycle for 440 Hz
    return single_cycle

# Load custom waveform if specified
custom_waveform = None
if args.waveform == 'yes':
    try:
        custom_waveform = load_waveform('cat.wav')
        print("Custom waveform (cat.wav) loaded.")
    except Exception as e:
        print(f"Error loading custom waveform: {e}. Defaulting to sine wave.")


melody_root = args.root
bass_root = melody_root - 12 * args.bass_octave
# Parse the custom chord loop
try:
    chord_loop = list(map(int, args.chord_loop.split(',')))
    if not all(1 <= chord <= 7 for chord in chord_loop):
        raise ValueError
except ValueError:
    raise ValueError("Invalid chord loop format. Use a comma-separated list of scale degrees (1-7).")

# Function to pick notes for the chord loop
position = 0
def pick_notes(chord_root, n=4):
    global position
    p = position
    notes = []
    for _ in range(n):
        chord_note_offset = chord_to_note_offset(p)
        chord_note = note_to_key_offset(chord_root + chord_note_offset)
        notes.append(chord_note)

        if random.random() > 0.5:
            p = p + 1
        else:
            p = p - 1
    position = p
    return notes

def make_note(key, n=1):
    f = 440 * 2 ** ((key - 69) / 12)
    b = beat_samples * n
    cycles = 2 * np.pi * f * b / samplerate
    t = np.linspace(0, cycles, b)
    return np.sin(t)

def make_piano_note(key, n=1, sustain_duration=0.5, attack_duration=0.05, release_duration=0.2):
    f = 440 * 2 ** ((key - 69) / 12)
    b = beat_samples * n
    t = np.linspace(0, n, int(b))

    sound = np.zeros_like(t)
    for i in range(1, 5):
        sound += np.sin(2 * np.pi * f * i * t) / i
    # Sound ADSR
    envelope = np.zeros_like(t)
    attack_samples = int(samplerate * attack_duration)
    sustain_samples = int(samplerate * sustain_duration)
    release_samples = int(samplerate * release_duration)

    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:attack_samples + sustain_samples] = 1
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    sound *= envelope
    return sound


def load_wavetable(filename="cat.wav"):
    """Loads the wavetable (cat.wav file) for waveform."""
    # need to fix, the sound is not correct with the tone.
    try:
        with wave.open(filename, "rb") as wav:
            assert wav.getnchannels() == 1, "Only mono waveforms are supported."
            assert wav.getframerate() == samplerate, "Mismatched sample rate."
            assert wav.getsampwidth() == 2, "Only 16-bit samples are supported."
            nframes = wav.getnframes()
            w = wav.readframes(nframes)
            wavetable = np.frombuffer(w, dtype=np.int16).astype(np.float32) / 32768  # Normalize to [-1, 1]
    except Exception as e:
        raise RuntimeError(f"Failed to load wavetable: {e}")
    return wavetable


def make_custom_wave(key, n=1, wavetable=None):
    """Generates a sound using the custom wavetable (cat.wav)."""
    f = 440 * 2 ** ((key - 69) / 12)  # Frequency based on MIDI note
    b = beat_samples * n
    if wavetable is None:
        wavetable = load_wavetable()  # Load wavetable if not provided

    # Resample wavetable to match the frequency of the note
    wavetable_len = len(wavetable)
    num_samples = int(b)
    samples_per_cycle = int(samplerate / f)

    # Create the wave by repeating the wavetable and then slicing to match the note duration
    num_cycles = num_samples // samples_per_cycle
    extended_wave = np.tile(wavetable, num_cycles)

    return extended_wave[:num_samples]



def play(sound):
    sd.play(sound, samplerate=samplerate, blocking=True)

# Playing the melody and bass
sound = np.array([], dtype=np.float64)
for c in chord_loop:
    notes = pick_notes(c - 1)

    if args.waveform == 'yes':
        wavetable = load_wavetable()  # Load the wavetable (cat.wav)
        melody = np.concatenate([make_custom_wave(i + melody_root, wavetable=wavetable) for i in notes])


        melody = np.concatenate([make_custom_wave(i + melody_root) for i in notes])
    else:
        if args.type == 'MIDI':
            melody = np.concatenate([make_note(i + melody_root) for i in notes])
        elif args.type == 'piano':
            melody = np.concatenate([make_piano_note(i + melody_root) for i in notes])

    bass_note = note_to_key_offset(c - 1)
    if args.waveform == 'yes':
        bass = make_note(bass_note + bass_root, n=4)
    else:
        if args.type == 'MIDI':
            bass = make_note(bass_note + bass_root, n=4)
        elif args.type == 'piano':
            bass = make_note(bass_note + bass_root, n=4)

    melody_gain = args.balance
    bass_gain = 1 - melody_gain

    sound = np.append(sound, melody_gain * melody + bass_gain * bass)

# Output to file or play directly
if args.output:
    output = wave.open(args.output, "wb")
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(samplerate)
    output.setnframes(len(sound))

    data = args.gain * 32767 * sound.clip(-1, 1)
    output.writeframesraw(data.astype(np.int16))

    output.close()
else:
    play(args.gain * sound)
