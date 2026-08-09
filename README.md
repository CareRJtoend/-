# 智能监控系统

这是一个用于采集能源数据、训练模型并进行预测的 Python 项目。

## 运行环境
- Python 3.8+
- MySQL 数据库
- MQTT 服务

## 依赖安装
```bash
pip install paho-mqtt pymysql scikit-learn
```

## 启动方式
1. 先启动 MQTT 服务
2. 运行 `数据采集.py`
3. 运行 `模型训练.py`
4. 运行 `能源预测系统.py`
