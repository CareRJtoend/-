import time
import json
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime

# MQTT 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "device/energy/data"

# 读取真实数据
print("📂 读取真实工业数据...")
df = pd.read_csv('gearbox_all_data.csv')
print(f"✅ 总数据量: {len(df)} 条")
print(f"📊 标签分布:\n{df['label'].value_counts()}")

# 连接 MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print(f"🚀 开始回放真实数据到 MQTT...")
print(f"📡 主题: {MQTT_TOPIC}")
print("=" * 50)

# 按顺序发送每条数据
for idx, row in df.iterrows():
    # 构造 MQTT 消息
    data = {
        'timestamp': datetime.now().isoformat(),
        'power_kw': row['sensor1'],
        'runtime_hours': row['sensor2'],
        'temperature_c': row['sensor3'],
        'load_percent': row['sensor4'],
        'energy_kwh': row['sensor4'],  # 用 sensor4 作为能耗
        'true_label': row['label']  # 真实标签，用于验证
    }

    payload = json.dumps(data)
    client.publish(MQTT_TOPIC, payload)

    print(f"[{idx + 1}] 发送: sensor4={row['sensor4']:.2f}, 真实状态={row['label']}")

    # 每 1 秒发送一条（可调整速度）
    time.sleep(1)

print("✅ 数据回放完成！")
client.loop_stop()
client.disconnect()