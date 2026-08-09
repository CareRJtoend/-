import time
import random
import json
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "device/energy/data"


def generate_sensor_data():
    """生成模拟传感器数据"""
    return {
        "timestamp": datetime.now().isoformat(),
        "power_kw": round(random.uniform(50, 200), 1),
        "runtime_hours": round(random.uniform(0, 24), 1),
        "temperature_c": round(random.uniform(20, 85), 1),
        "load_percent": round(random.uniform(30, 100), 1)
    }


def calculate_energy(data):
    """计算能耗"""
    power = data["power_kw"]
    runtime = data["runtime_hours"]
    temp = data["temperature_c"]
    load = data["load_percent"]
    energy = power * runtime + temp * 0.5 + load * 2 + random.uniform(-10, 10)
    return round(energy, 1)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ 已连接到 MQTT Broker")
    else:
        print(f"❌ 连接失败，返回码: {rc}")


if __name__ == "__main__":
    # 创建 MQTT 客户端
    client = mqtt.Client()
    client.on_connect = on_connect

    # 连接 Broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print("🚀 MQTT 数据采集发布器启动...")
    print(f"📡 发布主题: {MQTT_TOPIC}")
    print("=" * 50)

    count = 0
    try:
        while True:
            # 生成数据
            sensor_data = generate_sensor_data()
            sensor_data["energy_kwh"] = calculate_energy(sensor_data)

            # 发布到 MQTT
            payload = json.dumps(sensor_data)
            client.publish(MQTT_TOPIC, payload)

            count += 1
            print(f"[{count}] 发布: 功率={sensor_data['power_kw']}kW, 能耗={sensor_data['energy_kwh']}kWh")

            time.sleep(3)
    except KeyboardInterrupt:
        print("\n🛑 发布器已停止")
        client.loop_stop()
        client.disconnect()