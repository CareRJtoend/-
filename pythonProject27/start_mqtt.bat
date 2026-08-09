@echo off
chcp 65001 > nul
title MQTT 服务启动器

cd /d C:\Users\任金涛\PycharmProjects\pythonProject27

echo 启动 MQTT 发布器...
start "MQTT 发布器" cmd /k "python mqtt_publisher_real.py"

timeout /t 1 /nobreak > nul

echo 启动 MQTT 订阅器...
start "MQTT 订阅器" cmd /k "python mqtt_subscriber.py"

echo.
echo 启动完成！
echo 关闭窗口可停止服务
pause