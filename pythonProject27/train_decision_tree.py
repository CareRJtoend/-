import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ========== 配置 ==========
CSV_FILE = 'gearbox_all_data.csv'
WINDOW_SIZE = 10  # 滑动窗口大小

# ========== 1. 读取数据 ==========
print("📂 读取合并后的数据...")
df = pd.read_csv(CSV_FILE)
print(f"✅ 总数据量: {len(df)} 行")

# ========== 2. 按标签分组，添加时序特征 ==========
print(f"\n📊 添加时序特征（窗口大小={WINDOW_SIZE}）...")

feature_cols = []
all_rolling_features = []

for label in df['label'].unique():
    label_df = df[df['label'] == label].copy()

    # 对每个传感器添加滑动窗口统计
    for sensor in ['sensor1', 'sensor2', 'sensor3', 'sensor4']:
        # 均值
        rolling_mean = label_df[sensor].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        rolling_std = label_df[sensor].rolling(window=WINDOW_SIZE, min_periods=1).std()
        rolling_max = label_df[sensor].rolling(window=WINDOW_SIZE, min_periods=1).max()
        rolling_min = label_df[sensor].rolling(window=WINDOW_SIZE, min_periods=1).min()

        label_df[f'{sensor}_mean'] = rolling_mean
        label_df[f'{sensor}_std'] = rolling_std.fillna(0)
        label_df[f'{sensor}_max'] = rolling_max
        label_df[f'{sensor}_min'] = rolling_min

        feature_cols.extend([f'{sensor}_mean', f'{sensor}_std', f'{sensor}_max', f'{sensor}_min'])

    all_rolling_features.append(label_df)

# 合并所有标签的数据
df_processed = pd.concat(all_rolling_features, ignore_index=True)

print(f"✅ 原始特征: 4 个")
print(f"✅ 时序特征: {len(feature_cols)} 个")
print(f"✅ 总特征数: {len(feature_cols)} 个")

# ========== 3. 特征和目标 ==========
X = df_processed[feature_cols]
y = df_processed['label']

# 处理 NaN 值
X = X.fillna(0)

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n📊 标签映射:")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {label} -> {i}")

# ========== 4. 划分数据集 (8:1:1) ==========
print("\n📊 划分数据集...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✅ 训练集: {len(X_train)} 条 (80%)")
print(f"✅ 验证集: {len(X_val)} 条 (10%)")
print(f"✅ 测试集: {len(X_test)} 条 (10%)")

# ========== 5. 训练随机森林模型 ==========
print("\n🌲 训练随机森林分类模型...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ========== 6. 验证集评估 ==========
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"\n📊 验证集结果:")
print(f"   准确率 = {val_acc:.4f}")
print(f"   加权精确率 = {val_precision:.4f}")
print(f"   加权召回率 = {val_recall:.4f}")
print(f"   加权F1分数 = {val_f1:.4f}")

# ========== 7. 测试集评估 ==========
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n📊 测试集结果:")
print(f"   准确率 = {test_acc:.4f}")
print(f"   加权精确率 = {test_precision:.4f}")
print(f"   加权召回率 = {test_recall:.4f}")
print(f"   加权F1分数 = {test_f1:.4f}")

# ========== 8. 混淆矩阵 ==========
print(f"\n📊 测试集混淆矩阵:")
cm = confusion_matrix(y_test, y_test_pred)
print("       预测")
print("       ", "  ".join([f"{i:>4}" for i in range(len(label_encoder.classes_))]))
for i in range(len(label_encoder.classes_)):
    print(f"  实际 {i}: ", "  ".join([f"{cm[i][j]:>4}" for j in range(len(label_encoder.classes_))]))

# ========== 9. 特征重要性（Top 20）==========
print(f"\n📊 特征重要性 Top 20:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
for i, idx in enumerate(indices):
    print(f"   {i + 1}. {feature_cols[idx]}: {importances[idx]:.4f}")

# ========== 10. 保存模型 ==========
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest_temporal.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n✅ 模型已保存到 models/random_forest_temporal.pkl")
print("✅ 标签编码器已保存到 models/label_encoder.pkl")
print("✅ 标准化器已保存到 models/scaler.pkl")

# ========== 11. 保存模型信息 ==========
model_info = {
    'model_type': 'RandomForestClassifier',
    'task': 'multiclass_classification',
    'window_size': WINDOW_SIZE,
    'n_estimators': 100,
    'max_depth': 20,
    'split_ratio': '8:1:1',
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'val_accuracy': round(val_acc, 4),
    'val_precision': round(val_precision, 4),
    'val_recall': round(val_recall, 4),
    'val_f1': round(val_f1, 4),
    'test_accuracy': round(test_acc, 4),
    'test_precision': round(test_precision, 4),
    'test_recall': round(test_recall, 4),
    'test_f1': round(test_f1, 4),
    'feature_names': feature_cols,
    'target_classes': label_encoder.classes_.tolist()
}

with open('models/random_forest_temporal_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# ========== 12. 显示总结 ==========
print("\n" + "=" * 50)
print("🎉 时序特征 + 随机森林模型训练完成！")
print("=" * 50)
print(f"📊 数据总量: {len(df)} 条")
print(f"📊 类别数量: {len(label_encoder.classes_)} 种")
print(f"📊 原始特征: 4 个")
print(f"📊 时序特征: {len(feature_cols)} 个")
print(f"📊 划分比例: 8:1:1")
print(f"📊 测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")