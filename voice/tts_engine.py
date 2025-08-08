# voice/tts_engine.py

import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    print(f"🗣️ Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

# Predefined examples
def say_intro():
    speak("Hello, I’m HICS")

def say_emotion(emotion):
    speak(f"I see you're {emotion}")

def say_movement(gesture):
    speak(f"Moving {gesture} now")

# Testing
if __name__ == "__main__":
    say_intro()
    say_emotion("happy")
    say_movement("forward")

