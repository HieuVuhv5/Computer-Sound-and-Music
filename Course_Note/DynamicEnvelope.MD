# Dynamic Envelope

## Summary

A **sound envelope** defines how the amplitude of a sound changes over time. It typically includes stages like **attack**, **decay**, **sustain**, and **release** (ADSR), which shape the dynamics of a sound. This concept is crucial for synthesizers and sound design, as it gives instruments their expressive qualities.

## Python Code Example

This code demonstrates the basic implementation of an envelope in a software synthesizer, responding to MIDI input to generate sounds with controlled attack and release phases.

```python
# Note class with envelope handling
class Note:
    def __init__(self, key):
        self.t = 0
        self.key = key
        self.attack_rate = 1.0 / (10 * sample_rate / 1000)  # attack in samples
        self.attack_amplitude = 0
        self.wave_table = make_sin(440 * 2 ** ((key - 69) / 12))  # A4 note
        self.held = False

    def play(self, frame_count):
        # Attack phase
        if self.attack_rate:
            end_amplitude = self.attack_amplitude + frame_count * self.attack_rate
            scale = np.linspace(self.attack_amplitude, end_amplitude, frame_count).clip(0, 1)
            output = self.wave_table[:frame_count] * scale
            if end_amplitude >= 1:
                self.attack_rate = None  # end attack
            self.attack_amplitude = end_amplitude
        return output

```

#### Attack: The note gradually reaches its maximum amplitude, determined by attack_rate.
```python
def release(self):
    if log_envelope:
        print("releasing note", self.key, self.t)
    # Hardcode release time to 100ms
    release_samples = 100 * sample_rate / 1000
    self.release_rate = 1.0 / release_samples
    self.release_amplitude = 1.0
    if self.attack_rate:
        self.release_amplitude = self.attack_amplitude
        self.attack_rate = None

```

#### Release: A dummy release phase can be triggered once the note is "released".



