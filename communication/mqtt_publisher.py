import paho.mqtt.client as mqtt

# MQTT Config
broker = "broker.hivemq.com"
port = 1883
topic = "humanoid/gesture"

# Connect to MQTT Broker
client = mqtt.Client()
client.connect(broker, port)

# ✅ Function to publish gesture
def publish_gesture(gesture_result):
    message = str(gesture_result)
    client.publish(topic, message)
    print(f"📤 MQTT Sent: {message} to {topic}")
