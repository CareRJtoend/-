import json
import pymysql
import paho.mqtt.client as mqtt
from datetime import datetime
from model_predictor import predictor

# MySQL 配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'energy_monitor',
    'charset': 'utf8mb4'
}

# MQTT 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "device/energy/data"

MAX_HISTORY = 150
MAX_TRASH = 150


def manage_data_limit():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) as count FROM history_data")
        count = cursor.fetchone()[0]
        if count > MAX_HISTORY:
            overflow_count = count - MAX_HISTORY
            cursor.execute("""
                INSERT INTO trash_data (timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh)
                SELECT timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh
                FROM history_data ORDER BY id ASC LIMIT %s
            """, (overflow_count,))
            cursor.execute("DELETE FROM history_data ORDER BY id ASC LIMIT %s", (overflow_count,))
            cursor.execute("SELECT COUNT(*) as count FROM trash_data")
            trash_count = cursor.fetchone()[0]
            if trash_count > MAX_TRASH:
                delete_count = trash_count - MAX_TRASH
                cursor.execute("DELETE FROM trash_data ORDER BY id ASC LIMIT %s", (delete_count,))
            conn.commit()
            print(f"📦 已移动 {overflow_count} 条数据到垃圾桶")
    except Exception as e:
        print(f"⚠️ 数据上限管理失败: {e}")
    finally:
        conn.close()


def save_to_mysql(data):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """INSERT INTO history_data (timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (
            data['timestamp'], data['power_kw'], data['runtime_hours'],
            data['temperature_c'], data['load_percent'], data['energy_kwh']
        ))
        conn.commit()
        conn.close()
        manage_data_limit()
        return True
    except Exception as e:
        print(f"❌ 数据库写入失败: {e}")
        return False


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ 已连接到 MQTT Broker")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 订阅主题: {MQTT_TOPIC}")
    else:
        print(f"❌ 连接失败，返回码: {rc}")


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        sensor1 = data.get('power_kw', 0)
        sensor2 = data.get('runtime_hours', 0)
        sensor3 = data.get('temperature_c', 0)
        sensor4 = data.get('energy_kwh', 0)

        print(f"📨 收到: 功率={sensor1:.4f}kW, 能耗={sensor4:.4f}kWh")

        # 添加到预测器缓冲区（关键代码）
        predictor.add_data_point(sensor1, sensor2, sensor3, sensor4)

        # 检查缓冲区状态
        if predictor.is_ready():
            status, confidence = predictor.predict()
            print(f"🔮 预测结果: {status} (置信度: {confidence:.1f}%)")
        else:
            remaining = 10 - len(predictor.buffers['sensor1'])
            print(f"📊 还需 {remaining} 条数据才能预测")

        # 保存到数据库
        db_data = {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'power_kw': sensor1,
            'runtime_hours': sensor2,
            'temperature_c': sensor3,
            'load_percent': data.get('load_percent', 50),
            'energy_kwh': sensor4
        }
        save_to_mysql(db_data)

    except Exception as e:
        print(f"❌ 处理消息失败: {e}")


if __name__ == "__main__":
    print("🚀 MQTT 订阅器启动（含模型预测）...")
    print("=" * 50)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()