import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置后端为 Agg（不显示窗口，只保存图片）
import matplotlib
matplotlib.use('Agg')

print("="*50)
print("Industrial Equipment Energy Management System - Visualization")
print("="*50)

# 1. Read data
print("\n1. Reading data...")
data = pd.read_csv('data/raw/equipment_energy.csv')
print(f"Successfully read {len(data)} records")

# 2. Load model
print("\n2. Loading model...")
model = joblib.load('models/energy_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print(f"Model loaded successfully")

# 3. Prepare prediction data
features = ['power_kw', 'runtime_hours', 'temperature_c', 'load_percent']
target = 'energy_kwh'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# 4. Generate charts (save to files, no popup windows)
print("\n3. Generating visualizations...")

# Figure 1: Actual vs Predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Energy (kWh)', fontsize=12)
plt.ylabel('Predicted Energy (kWh)', fontsize=12)
plt.title('Actual vs Predicted Energy Consumption', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reports/prediction_scatter.png', dpi=100)
plt.close()  # 关闭图片，释放内存
print("   Figure 1 saved: reports/prediction_scatter.png")

# Figure 2: Residual distribution
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Prediction Error (kWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Prediction Error Distribution', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reports/residual_histogram.png', dpi=100)
plt.close()
print("   Figure 2 saved: reports/residual_histogram.png")

# Figure 3: Feature vs Energy relationship (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features_name = ['Power (kW)', 'Runtime (hours)', 'Temperature (C)', 'Load Rate (%)']

for i, (feat, feat_name) in enumerate(zip(features, features_name)):
    row, col = i//2, i%2
    axes[row, col].scatter(data[feat], data[target], alpha=0.3, c='orange', s=1)
    axes[row, col].set_xlabel(feat_name, fontsize=12)
    axes[row, col].set_ylabel('Energy (kWh)', fontsize=12)
    axes[row, col].set_title(f'{feat_name} vs Energy', fontsize=12)
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/feature_analysis.png', dpi=100)
plt.close()
print("   Figure 3 saved: reports/feature_analysis.png")

# Figure 4: Feature importance (coefficients)
plt.figure(figsize=(8, 6))
coef_values = model.coef_
colors = ['red', 'blue', 'green', 'purple']
plt.barh(features_name, coef_values, color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Feature Importance (Model Coefficients)', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(coef_values):
    plt.text(v, i, f' {v:.2f}', va='center')
plt.savefig('reports/feature_importance.png', dpi=100)
plt.close()
print("   Figure 4 saved: reports/feature_importance.png")

print("\n" + "="*50)
print("Visualization completed! All charts saved to reports/ folder")
print("="*50)

# Show statistics
print("\n📊 Model Performance Statistics:")
print(f"   R² Score: {model.score(scaler.transform(X_test), y_test):.4f}")
print(f"   Mean Absolute Error: {np.mean(np.abs(residuals)):.2f} kWh")
print(f"   Max Error: {np.max(np.abs(residuals)):.2f} kWh")

# 显示保存的文件路径
import os
print("\n📁 Saved files:")
print(f"   {os.path.abspath('reports/prediction_scatter.png')}")
print(f"   {os.path.abspath('reports/residual_histogram.png')}")
print(f"   {os.path.abspath('reports/feature_analysis.png')}")
print(f"   {os.path.abspath('reports/feature_importance.png')}")