import wavio
import sounddevice as sd

fs = 44100
seconds = 5
print("Speak now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording finished.")

# Save in project folder instead of Temp
wavio.write("my_recording.wav", recording, fs, sampwidth=2)
