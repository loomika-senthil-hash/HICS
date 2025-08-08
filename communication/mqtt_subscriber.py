import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print(f"📥 Received from MQTT: {message.payload.decode()}")

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)
client.subscribe("humanoid/gesture")
client.on_message = on_message

print("👂 Listening to MQTT Topic: humanoid/gesture ...")
client.loop_forever()
