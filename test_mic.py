import sounddevice as sd
import numpy as np

duration = 3  # seconds
fs = 16000

print("Speak now...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

print("Recording finished.")
print("Max volume:", np.max(np.abs(recording)))
