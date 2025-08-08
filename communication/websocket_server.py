import asyncio
import websockets

gesture = "none"
emotion = "none"

def update_data(new_gesture, new_emotion):
    global gesture, emotion
    gesture = new_gesture
    emotion = new_emotion

async def send_data(websocket):
    while True:
        data = f"Gesture: {gesture}, Emotion: {emotion}"
        await websocket.send(data)
        await asyncio.sleep(1)

async def start_server():
    print("🌐 Starting WebSocket Server at ws://localhost:5678")
    async with websockets.serve(send_data, "localhost", 5678):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(start_server())
