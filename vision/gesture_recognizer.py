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

# Step 2: Initialize model and load weights (robustly)
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

# Step 4: Define gesture labels
GESTURE = ["Thumbs up", "peace", "stop"]

# Step 5: Updated detect_gesture to handle both file paths and NumPy arrays
def detect_gesture(img_input):
    """
    img_input: path to image file (str) or NumPy array (OpenCV frame)
    returns: predicted gesture string
    """
    try:
        # Handle NumPy array (frame from OpenCV)
        if isinstance(img_input, np.ndarray):
            # Convert BGR (OpenCV) → RGB (PIL expects RGB)
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_input).convert("L")
        else:
            # Handle file path
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found: {img_input}")
            image = Image.open(img_input).convert("L")

        # Transform and prepare tensor
        tensor = transform(image).unsqueeze(0).to(device)  # shape (1,1,48,48)

        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            idx = int(predicted.item())
            if idx < 0 or idx >= len(GESTURE):
                return "unknown"
            return GESTURE[idx]

    except Exception as e:
        print("detect_gesture ERROR:", e)
        return "error"

# ✅ Example usage
if __name__ == "__main__":
    import cv2  # Only needed for testing webcam
    # Test with image path
    path = r"C:\Users\SENTHIL\Desktop\Humanoid_HICS\gesture_data\4\39.jpg"
    print("From file:", detect_gesture(path))

    # Test with webcam frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print("From webcam:", detect_gesture(frame))
    cap.release()
