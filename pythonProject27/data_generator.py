import time
import random
from datetime import datetime
import pymysql

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',  # ⚠️ 改成你的密码
    'database': 'energy_monitor',
    'charset': 'utf8mb4'
}


def get_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        print("✅ 数据库连接成功")
        return conn
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return None


def generate_sensor_data():
    return {
        'timestamp': datetime.now(),
        'power_kw': round(random.uniform(50, 200), 1),
        'runtime_hours': round(random.uniform(0, 24), 1),
        'temperature_c': round(random.uniform(20, 85), 1),
        'load_percent': round(random.uniform(30, 100), 1)
    }


def calculate_energy(data):
    power = data['power_kw']
    runtime = data['runtime_hours']
    temp = data['temperature_c']
    load = data['load_percent']
    energy = power * runtime + temp * 0.5 + load * 2 + random.uniform(-10, 10)
    return round(energy, 1)


def save_to_db(data):
    conn = get_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        sql = """INSERT INTO history_data (timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (
            data['timestamp'], data['power_kw'], data['runtime_hours'],
            data['temperature_c'], data['load_percent'], data['energy_kwh']
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ 插入数据失败: {e}")
        return False


if __name__ == '__main__':
    print("🚀 数据采集模拟器启动（MySQL版本）...")
    print("=" * 50)
    count = 0

    while True:
        sensor_data = generate_sensor_data()
        sensor_data['energy_kwh'] = calculate_energy(sensor_data)

        if save_to_db(sensor_data):
            count += 1
            print(f"[{count}] ✅ 采集成功: 功率={sensor_data['power_kw']}kW, 能耗={sensor_data['energy_kwh']}kWh")
        else:
            print(f"[{count + 1}] ❌ 写入失败")

        time.sleep(3)