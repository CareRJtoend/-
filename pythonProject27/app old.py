from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'test_key_123'

# 加载你训练好的模型
model = joblib.load('models/energy_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# 简单账号
USERS = {'admin': '123456', '任金涛': '123'}


@app.route('/')
def home():
    if 'user' in session:
        return redirect('/predict_page')
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('username')
        pwd = request.form.get('password')
        if name in USERS and USERS[name] == pwd:
            session['user'] = name
            return redirect('/')
        return '<h2>登录失败</h2><a href="/login">重试</a>'

    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>登录</title>
        <style>
            body { font-family: Arial; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; display: flex; justify-content: center; align-items: center; }
            .box { background: white; padding: 40px; border-radius: 20px; width: 350px; text-align: center; }
            input { width: 100%; padding: 12px; margin: 10px 0; border: 2px solid #e0e0e0; border-radius: 8px; }
            button { width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; }
            button:hover { background: #764ba2; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🏭 能耗预测系统</h1>
            <form method="post">
                <input type="text" name="username" placeholder="用户名" required><br>
                <input type="password" name="password" placeholder="密码" required><br>
                <button type="submit">登录</button>
            </form>
            <p style="margin-top:20px;color:#999;">测试账号：admin / 123456</p>
        </div>
    </body>
    </html>
    '''


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')


@app.route('/predict_page')
def predict_page():
    if 'user' not in session:
        return redirect('/login')
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>能耗预测系统</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #f0f0f0; }
            .logout-btn { background: #ff4757; color: white; padding: 5px 15px; border-radius: 20px; text-decoration: none; font-size: 14px; }
            h1 { text-align: center; color: #333; margin-bottom: 10px; }
            .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }
            .input-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; color: #555; font-weight: 500; }
            input { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 16px; }
            input:focus { outline: none; border-color: #667eea; }
            button { width: 100%; padding: 14px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; }
            button:hover { transform: translateY(-2px); }
            .result { margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); border-radius: 15px; text-align: center; }
            .result-value { font-size: 48px; font-weight: bold; color: #667eea; }
            .result-unit { font-size: 20px; color: #666; }
            .suggestion { margin-top: 15px; padding: 10px; background: white; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span>👋 欢迎，<strong>''' + session['user'] + '''</strong></span>
                <a href="/logout" class="logout-btn">退出登录</a>
            </div>
            <h1>🏭 工业设备能耗预测系统</h1>
            <div class="subtitle">基于线性回归的智能能耗预测</div>
            <div class="input-group"><label>⚡ 功率 (kW)</label><input type="number" id="power" value="120"></div>
            <div class="input-group"><label>⏰ 运行时长 (小时)</label><input type="number" id="runtime" value="8"></div>
            <div class="input-group"><label>🌡️ 温度 (°C)</label><input type="number" id="temperature" value="50"></div>
            <div class="input-group"><label>📈 负载率 (%)</label><input type="number" id="load" value="70"></div>
            <button onclick="predict()">🔮 预测能耗</button>
            <div class="result" id="result" style="display: none;">
                <div><span class="result-value" id="resultValue">---</span><span class="result-unit"> kWh</span></div>
                <div class="suggestion" id="suggestion"></div>
            </div>
        </div>
        <script>
            async function predict() {
                const power = document.getElementById('power').value;
                const runtime = document.getElementById('runtime').value;
                const temperature = document.getElementById('temperature').value;
                const load = document.getElementById('load').value;
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({power, runtime, temperature, load})
                });
                const data = await response.json();
                document.getElementById('result').style.display = 'block';
                document.getElementById('resultValue').textContent = data.predicted;
                const val = data.predicted;
                let sug = '';
                if (val > 4000) sug = '⚠️ 能耗较高，建议检查设备';
                else if (val > 2500) sug = '📊 能耗中等，运行正常';
                else sug = '✅ 能耗较低，状态良好';
                document.getElementById('suggestion').textContent = sug;
            }
        </script>
    </body>
    </html>
    '''


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user' not in session:
        return jsonify({'error': '未登录'}), 401
    data = request.get_json()
    power = float(data['power'])
    runtime = float(data['runtime'])
    temperature = float(data['temperature'])
    load = float(data['load'])

    # 使用真实模型预测
    input_data = np.array([[power, runtime, temperature, load]])
    input_scaled = scaler.transform(input_data)
    predicted = model.predict(input_scaled)[0]

    return jsonify({'predicted': round(predicted, 2)})


if __name__ == '__main__':
    app.run(debug=True)