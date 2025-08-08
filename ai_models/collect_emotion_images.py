import cv2
import os

# 1. Get the emotion label from user
label = int(input("Enter Emotion Label (0:Happy, 1:Sad, 2:Angry): "))

# 2. Path where images will be saved (absolute path)
base_dir = os.path.dirname(os.path.abspath(__file__))  # Folder where this script is
save_path = os.path.join(base_dir, "emotion_data", str(label))
os.makedirs(save_path, exist_ok=True)

# 3. Start the webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera. Exiting.")
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    cv2.putText(frame, f"Capturing Emotion {label} | Count: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Emotion Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save the image to the correct folder
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved: {os.path.abspath(img_path)}")  # Show full path
        count += 1

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
