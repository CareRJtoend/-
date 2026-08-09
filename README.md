# 智能监控系统

## 启动步骤（严格按照顺序）

**第一步：启动 MQTT 服务**  
双击运行 `启动MQTT.bat`

**第二步：打开 Navicat 连接数据库**  
连接本地 MySQL，确保数据库配置与 `user_emails.json` 一致

**第三步：运行发送端**  
`python 数据采集.py`

**第四步：运行接收端**  
`python 可视化文件.py`

**第五步：运行模型训练**  
`python 模型训练.py`

**第六步：运行 Web 界面**  
`python 能源预测系统.py`
