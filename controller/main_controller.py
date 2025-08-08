import cv2
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from communication.mqtt_publisher import publish_gesture
from communication.websocket_server import update_data
from vision.gesture_recognizer import detect_gesture
from vision.emotion_detector import detect_emotion
from voice.tts_engine import say_intro, say_emotion, say_movement
from locomotion.balance_simulator import check_posture  # ✅ Import this

def main():
    say_intro()

    cap = cv2.VideoCapture(0)

    last_speak_time = 0
    speak_interval = 5  # seconds cooldown between speeches

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture = detect_gesture(frame)
        emotion = detect_emotion(frame)

        print(f"Detected: Gesture = {gesture}, Emotion = {emotion}")

        # 🔽 Communication
        publish_gesture(gesture)
        update_data(gesture, emotion)

        # 🧍 Simulated Posture for Balance
        pose_data = {
            "left_shoulder_y": 240,
            "right_shoulder_y": 270,
            "left_hip_y": 410,
            "right_hip_y": 380
        }
        check_posture(pose_data)

        # 🗣️ Robot speaks only if cooldown passed
        current_time = time.time()
        if current_time - last_speak_time > speak_interval:
            say_emotion(emotion)
            say_movement(gesture)
            last_speak_time = current_time

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
