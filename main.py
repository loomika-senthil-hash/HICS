# main.py 🧠

import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# ✅ Import all modules
from vision.gesture_recognizer import detect_gesture
from vision.emotion_detector import detect_emotion
from voice.tts_engine import say_intro, say_emotion, say_movement
from communication.mqtt_publisher import publish_gesture
from communication.websocket_server import update_data
from locomotion.balance_simulator import check_posture
from gui.data_store import update_values  # Optional for GUI

# 🧠 MASTER FUNCTION
def run_robot_brain():
    say_intro()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed failed.")
            break

        # 🤖 STEP 1: Detect gesture & emotion
        gesture = detect_gesture(frame)
        emotion = detect_emotion(frame)

        print(f"[INFO] Gesture: {gesture} | Emotion: {emotion}")

        # 📡 STEP 2: Send to MQTT & WebSocket
        publish_gesture(gesture)
        update_data(gesture, emotion)

        # 🧠 STEP 3: Update GUI data store
        update_values(gesture, emotion)

        # 🗣️ STEP 4: Speak
        say_emotion(emotion)
        say_movement(gesture)

        # ⚖️ STEP 5: Simulate posture/balance
        pose_data = {
            "left_shoulder_y": 250,
            "right_shoulder_y": 290,
            "left_hip_y": 430,
            "right_hip_y": 420
        }
        check_posture(pose_data)

        # 🎥 STEP 6: Show camera feed (optional)
        cv2.imshow("Robot Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🚀 ENTRY POINT
if __name__ == "__main__":
    run_robot_brain()
