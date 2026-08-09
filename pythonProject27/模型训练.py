import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("="*50)
print("工业设备能耗管理系统 - 模型训练")
print("="*50)

# 1. 读取数据
print("\n1. 读取数据...")
data = pd.read_csv('data/raw/equipment_energy.csv')
print(f"✅ 成功读取 {len(data)} 条数据")
print(f"   字段：{list(data.columns)}")

# 2. 数据预处理
print("\n2. 数据预处理...")

# 检查缺失值
print(f"   缺失值统计：\n{data.isnull().sum()}")

# 特征选择（X）和目标变量（y）
features = ['power_kw', 'runtime_hours', 'temperature_c', 'load_percent']
target = 'energy_kwh'

X = data[features]
y = data[target]

print(f"   特征变量：{features}")
print(f"   目标变量：{target}")

# 3. 划分训练集和测试集（80% 训练，20% 测试）
print("\n3. 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   训练集大小：{len(X_train)} 条")
print(f"   测试集大小：{len(X_test)} 条")

# 4. 特征标准化（可选，线性回归不一定需要，但有助于理解）
print("\n4. 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ 标准化完成（均值≈0，标准差≈1）")

# 5. 训练线性回归模型
print("\n5. 训练线性回归模型...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(f"✅ 模型训练完成")
print(f"   模型系数：")
for feat, coef in zip(features, model.coef_):
    print(f"     {feat}: {coef:.4f}")
print(f"   截距（Intercept）: {model.intercept_:.2f}")

# 6. 模型评估
print("\n6. 模型评估...")
y_pred = model.predict(X_test_scaled)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"   均方误差 (MSE): {mse:.2f}")
print(f"   均方根误差 (RMSE): {rmse:.2f} kWh")
print(f"   R² 分数: {r2:.4f}")

# 7. 保存模型和标准化器
print("\n7. 保存模型...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/energy_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print(f"✅ 模型已保存到：models/energy_model.pkl")
print(f"✅ 标准化器已保存到：models/scaler.pkl")

# 8. 显示预测示例
print("\n8. 预测示例（前5个测试样本）：")
sample_results = pd.DataFrame({
    '实际能耗': y_test[:5].values,
    '预测能耗': y_pred[:5],
    '误差': np.abs(y_test[:5].values - y_pred[:5])
})
print(sample_results)

print("\n" + "="*50)
print("✅ 模型训练完成！")
print("="*50)