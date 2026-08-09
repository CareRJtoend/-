import joblib
import numpy as np
from collections import deque


class ModelPredictor:
    def __init__(self):
        # 加载模型和编码器
        self.model = joblib.load('models/rf_simple.pkl')
        self.label_encoder = joblib.load('models/label_encoder_simple.pkl')
        self.scaler = joblib.load('models/scaler_simple.pkl')
        self.feature_cols = joblib.load('models/feature_cols_simple.pkl')

        # 滑动窗口缓冲区（每个传感器保留最近10个值）
        self.window_size = 10
        self.buffers = {
            'sensor1': deque(maxlen=self.window_size),
            'sensor2': deque(maxlen=self.window_size),
            'sensor3': deque(maxlen=self.window_size),
            'sensor4': deque(maxlen=self.window_size)
        }

        print("✅ 模型加载成功")
        print(f"   窗口大小: {self.window_size}")
        print(f"   特征数量: {len(self.feature_cols)}")

    def add_data_point(self, sensor1, sensor2, sensor3, sensor4):
        """添加一个数据点到缓冲区"""
        self.buffers['sensor1'].append(sensor1)
        self.buffers['sensor2'].append(sensor2)
        self.buffers['sensor3'].append(sensor3)
        self.buffers['sensor4'].append(sensor4)
        print(
            f"📊 缓冲区: s1={len(self.buffers['sensor1'])}, s2={len(self.buffers['sensor2'])}, s3={len(self.buffers['sensor3'])}, s4={len(self.buffers['sensor4'])}")

    def extract_features(self):
        """从缓冲区提取特征（每个传感器：均值、标准差、最大值、最小值）"""
        features = []
        for sensor in ['sensor1', 'sensor2', 'sensor3', 'sensor4']:
            data = list(self.buffers[sensor])
            # 如果数据不足，用0填充
            while len(data) < self.window_size:
                data.append(0)

            features.append(np.mean(data))
            features.append(np.std(data))
            features.append(np.max(data))
            features.append(np.min(data))

        return np.array(features).reshape(1, -1)

    def is_ready(self):
        """检查缓冲区是否已满（可以开始预测）"""
        ready = len(self.buffers['sensor1']) == self.window_size
        if not ready:
            print(f"⏳ 未就绪: s1={len(self.buffers['sensor1'])}/{self.window_size}")
        return ready

    def predict(self):
        """预测当前设备状态"""
        if not self.is_ready():
            return None, 0

        X = self.extract_features()
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]

        label = self.label_encoder.inverse_transform([pred])[0]
        confidence = np.max(pred_proba) * 100

        # 中文映射
        label_map = {
            'gearbox00': '正常',
            'gearbox10': '故障1',
            'gearbox20': '故障2',
            'gearbox30': '故障3',
            'gearbox40': '故障4'
        }

        return label_map.get(label, label), confidence


# 创建全局实例
predictor = ModelPredictor()