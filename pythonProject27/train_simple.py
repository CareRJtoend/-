import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 读取数据
df = pd.read_csv('gearbox_all_data.csv')
print(f"总数据量: {len(df)}")

# 添加滑动窗口统计特征
window_size = 10
feature_cols = []

for sensor in ['sensor1', 'sensor2', 'sensor3', 'sensor4']:
    df[f'{sensor}_mean'] = df[sensor].rolling(window=window_size).mean()
    df[f'{sensor}_std'] = df[sensor].rolling(window=window_size).std()
    df[f'{sensor}_max'] = df[sensor].rolling(window=window_size).max()
    df[f'{sensor}_min'] = df[sensor].rolling(window=window_size).min()
    feature_cols.extend([f'{sensor}_mean', f'{sensor}_std', f'{sensor}_max', f'{sensor}_min'])

# 删除 NaN 行
df = df.dropna()
print(f"处理后数据量: {len(df)}")

# 特征和目标
X = df[feature_cols]
y = df['label']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
print(f"特征数量: {len(feature_cols)}")

# 训练模型
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 评估
y_test_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {acc:.4f} ({acc*100:.2f}%)")

# 保存
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_simple.pkl')
joblib.dump(label_encoder, 'models/label_encoder_simple.pkl')
joblib.dump(scaler, 'models/scaler_simple.pkl')
joblib.dump(feature_cols, 'models/feature_cols_simple.pkl')

print("✅ 模型已保存")