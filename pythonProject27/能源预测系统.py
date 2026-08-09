import joblib
import numpy as np
import os

print("=" * 60)
print("工业设备能耗预测与预警系统")
print("=" * 60)

# 加载训练好的模型和标准化器
print("\n正在加载模型...")
model = joblib.load('models/energy_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("✅ 模型加载成功！")


def predict_energy(power, runtime, temperature, load):
    """预测能耗"""
    # 输入数据标准化
    input_data = np.array([[power, runtime, temperature, load]])
    input_scaled = scaler.transform(input_data)

    # 预测
    predicted = model.predict(input_scaled)[0]
    return predicted


def check_energy_status(predicted, actual=None, threshold_ratio=0.2):
    """检查能耗状态"""
    if actual is None:
        return "ℹ️ 无实际值对比，无法判断异常"

    error_ratio = abs(predicted - actual) / actual

    if error_ratio > threshold_ratio:
        return f"⚠️ 警告：能耗异常！预测值 {predicted:.1f} kWh，实际值 {actual:.1f} kWh，偏差 {error_ratio * 100:.1f}%"
    else:
        return f"✅ 能耗正常：预测值 {predicted:.1f} kWh，实际值 {actual:.1f} kWh，偏差 {error_ratio * 100:.1f}%"


print("\n" + "-" * 60)
print("使用说明：")
print("1. 输入设备参数进行能耗预测")
print("2. 可选输入实际能耗进行异常检测")
print("3. 输入 q 退出系统")
print("-" * 60)

while True:
    print("\n" + "=" * 60)
    print("请输入设备运行参数：")

    try:
        # 输入参数
        power_input = input("  功率 (kW) [50-200]: ").strip()
        if power_input.lower() == 'q':
            print("\n感谢使用，再见！")
            break

        runtime_input = input("  运行时长 (小时) [0-24]: ").strip()
        if runtime_input.lower() == 'q':
            print("\n感谢使用，再见！")
            break

        temp_input = input("  温度 (°C) [20-85]: ").strip()
        if temp_input.lower() == 'q':
            print("\n感谢使用，再见！")
            break

        load_input = input("  负载率 (%) [30-100]: ").strip()
        if load_input.lower() == 'q':
            print("\n感谢使用，再见！")
            break

        # 转换为浮点数
        power = float(power_input)
        runtime = float(runtime_input)
        temperature = float(temp_input)
        load = float(load_input)

        # 验证输入范围
        if power < 0 or power > 300:
            print("⚠️ 功率超出合理范围，建议 50-200 kW")
        if runtime < 0 or runtime > 24:
            print("⚠️ 运行时长超出合理范围，建议 0-24 小时")
        if temperature < 0 or temperature > 120:
            print("⚠️ 温度超出合理范围，建议 20-85°C")
        if load < 0 or load > 100:
            print("⚠️ 负载率超出合理范围，建议 30-100%")

        # 预测能耗
        predicted = predict_energy(power, runtime, temperature, load)
        print(f"\n📊 预测结果：")
        print(f"   预测能耗 = {predicted:.2f} kWh")

        # 可选：输入实际能耗进行对比
        actual_input = input(f"\n是否输入实际能耗进行对比？(y/n): ").strip().lower()
        if actual_input == 'y':
            actual = float(input("   请输入实际能耗 (kWh): "))
            status = check_energy_status(predicted, actual)
            print(f"\n   {status}")

        # 显示设备状态建议
        print(f"\n💡 建议：")
        if predicted > 4000:
            print("   ⚠️ 能耗较高，建议检查设备运行效率")
        elif predicted > 2500:
            print("   📊 能耗中等，运行状态正常")
        else:
            print("   ✅ 能耗较低，设备运行状态良好")

        # 显示影响最大的因素
        print(f"\n📈 能耗影响因素分析：")
        print(f"   • 功率每增加1kW，能耗增加约 {model.coef_[0]:.2f} kWh")
        print(f"   • 运行时间每增加1小时，能耗增加约 {model.coef_[1]:.2f} kWh")

    except ValueError:
        print("❌ 输入错误，请输入有效的数字！")
    except KeyboardInterrupt:
        print("\n\n系统退出，再见！")
        break

print("\n" + "=" * 60)
