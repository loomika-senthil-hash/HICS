import time
from gui.data_store import update_values
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


gestures = ["Thumbs up", "peace", "stop"]
emotions = ["happy", "sad", "neutral"]

while True:
    for g, e in zip(gestures, emotions):
        update_values(g, e)
        print(f"Updated gesture={g}, emotion={e}")
        time.sleep(5)
