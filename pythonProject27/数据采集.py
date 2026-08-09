import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 设置随机种子，保证结果可重复
np.random.seed(42)

print("开始生成工业设备能耗模拟数据...")
print("正在加载库...")

# 生成时间序列（2024年全年，每小时一条数据）
# 注意：将 'H' 改为 'h'（小写）
dates = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='h')

print(f"生成 {len(dates)} 条数据记录...")

# 模拟数据
power_kw = np.random.uniform(50, 200, len(dates))      # 功率 50-200 kW
runtime_hours = np.random.uniform(0, 24, len(dates))   # 运行时长 0-24 小时
temperature_c = np.random.uniform(20, 85, len(dates))  # 温度 20-85°C
load_percent = np.random.uniform(30, 100, len(dates))  # 负载率 30-100%

# 构造能耗目标值（线性关系 + 噪声）
# 能耗 = 功率 × 运行时长 + 温度影响 + 负载影响 + 噪声
energy_kwh = (power_kw * runtime_hours +
              temperature_c * 0.5 +
              load_percent * 2 +
              np.random.normal(0, 15, len(dates)))

# 确保能耗不为负数
energy_kwh = np.abs(energy_kwh)

# 创建DataFrame
data = pd.DataFrame({
    'timestamp': dates,
    'power_kw': power_kw,
    'runtime_hours': runtime_hours,
    'temperature_c': temperature_c,
    'load_percent': load_percent,
    'energy_kwh': energy_kwh
})

# 显示前几行数据
print("\n数据前5行预览：")
print(data.head())

# 显示数据基本信息
print("\n数据基本信息：")
print(data.info())

print("\n数据统计描述：")
print(data.describe())

# 保存数据到CSV文件
os.makedirs('data/raw', exist_ok=True)  # 创建文件夹（如果不存在）
data.to_csv('data/raw/equipment_energy.csv', index=False)

print("\n✅ 数据采集完成！")
print(f"文件已保存到：data/raw/equipment_energy.csv")
print(f"数据总量：{len(data)} 条记录")
print(f"时间范围：{dates[0]} 至 {dates[-1]}")

# 显示文件保存的完整路径
full_path = os.path.abspath('data/raw/equipment_energy.csv')
print(f"完整路径：{full_path}")