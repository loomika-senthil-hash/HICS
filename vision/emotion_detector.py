import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Step 1: Define the same model class
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# Step 2: Initialize model and load weights
MODEL_PATH = "ai_models/emotion_model.pth"
device = torch.device("cpu")

model = EmotionCNN().to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

_loaded = torch.load(MODEL_PATH, map_location=device)

if isinstance(_loaded, dict):
    try:
        model.load_state_dict(_loaded)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict into EmotionCNN: {e}")
else:
    model = _loaded.to(device)

model.eval()

# Step 3: Define transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Step 4: Define emotion labels
EMOTIONS = ["happy", "sad", "angry"]

# Step 5: Updated detect_emotion to handle both file paths and NumPy arrays
def detect_emotion(img_input):
    """
    img_input: path to image file (str) or NumPy array (OpenCV frame)
    returns: predicted emotion string (one of EMOTIONS)
    """
    try:
        # If input is a webcam frame (NumPy array)
        if isinstance(img_input, np.ndarray):
            import cv2
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_input).convert("L")
        else:
            # If input is a file path
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found: {img_input}")
            image = Image.open(img_input).convert("L")

        # Transform & prepare tensor
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            idx = int(predicted.item())
            if idx < 0 or idx >= len(EMOTIONS):
                return "unknown"
            return EMOTIONS[idx]

    except Exception as e:
        print("detect_emotion ERROR:", e)
        return "error"

# ✅ Example usage
if __name__ == "__main__":
    import cv2
    # Test with file path
    path = r"C:\Users\SENTHIL\Desktop\Humanoid_HICS\ai_models\emotion_data\0\27.jpg"
    print("From file:", detect_emotion(path))

    # Test with webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print("From webcam:", detect_emotion(frame))
    cap.release()
