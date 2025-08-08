import sys
import os
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2

# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.data_store import gesture, emotion, update_values
from voice.tts_engine import speak  # Import your TTS speak function
from communication.mqtt_publisher import publish_gesture  # Import your MQTT publisher

current_gesture = "None"
current_emotion = "None"
running = True  # Initialize running state

def speak_button():
    # Speak the current emotion and gesture
    speak(f"I see you are {emotion}")
    speak(f"Gesture detected: {gesture}")

def send_mqtt_button():
    # Send the current gesture over MQTT
    publish_gesture(gesture)

def stop_button():
    global running
    running = False
    print("🛑 Stopping camera...")

def update_frame():
    global running, current_gesture, current_emotion
    ret, frame = cap.read()
    if ret:
        # Convert BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((400, 300))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update current gesture and emotion from data_store
        current_gesture = gesture
        current_emotion = emotion

        # Update GUI labels
        gesture_label.config(text=f"Gesture: {current_gesture}")
        emotion_label.config(text=f"Emotion: {current_emotion}")

    if running:
        video_label.after(30, update_frame)  # ~30 FPS update

def run_gui():
    global root, video_label, gesture_label, emotion_label, cap, running
    cap = cv2.VideoCapture(0)
    running = True

    root = tk.Tk()
    root.title("🤖 HICS Control Panel")

    video_label = Label(root)
    video_label.pack()

    gesture_label = Label(root, text="Gesture: ...", font=("Helvetica", 14))
    gesture_label.pack()

    emotion_label = Label(root, text="Emotion: ...", font=("Helvetica", 14))
    emotion_label.pack()

    Button(root, text="🗣️ Speak", command=speak_button).pack(pady=5)
    Button(root, text="📡 Send MQTT", command=send_mqtt_button).pack(pady=5)
    Button(root, text="🛑 Stop", command=stop_button).pack(pady=5)

    update_frame()
    root.protocol("WM_DELETE_WINDOW", stop_button)  # Cleanup on close
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui()
