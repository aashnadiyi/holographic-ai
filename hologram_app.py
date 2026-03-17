import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie
import pyttsx3

# Initialize text-to-speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# PyQt5 app
app = QApplication(sys.argv)
label = QLabel()
label.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
label.setAttribute(Qt.WA_TranslucentBackground)
label.setAlignment(Qt.AlignCenter)

# Load GIF
movie = QMovie("avatar.gif")
label.setMovie(movie)
movie.start()

# Show the window
label.resize(400, 400)  # Adjust size
label.show()

# Example: Make the AI “speak” after 2 seconds
def demo_response():
    speak("Hello Aashna! I am your holographic AI assistant.")

QTimer.singleShot(2000, demo_response)

sys.exit(app.exec_())
