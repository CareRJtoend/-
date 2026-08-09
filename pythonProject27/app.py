from flask import Flask, render_template, request, session, redirect, url_for, jsonify, make_response
import json
import pandas as pd
import os
import hashlib
import secrets
from datetime import datetime
from decimal import Decimal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pymysql
from pymysql.cursors import DictCursor
from model_predictor import predictor

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'energy_monitor',
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}


def get_db():
    return pymysql.connect(**DB_CONFIG)


# ========== 用户数据 ==========
def init_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM users")
    result = cursor.fetchone()
    if result['count'] == 0:
        cursor.execute("INSERT INTO users (username, password, created_at) VALUES (%s, SHA2(%s, 256), %s)",
                       ('admin', '123456', datetime.now().isoformat()))
        conn.commit()
    conn.close()


init_users()


def verify_password(username, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = SHA2(%s, 256)", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None


def get_user_email_config(username):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_emails WHERE username = %s", (username,))
    config = cursor.fetchone()
    conn.close()
    return config or {}


def save_user_email(username, email, smtp_server, smtp_port, sender_email, sender_password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_emails (username, email, smtp_server, smtp_port, sender_email, sender_password, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        email = VALUES(email), smtp_server = VALUES(smtp_server), smtp_port = VALUES(smtp_port),
        sender_email = VALUES(sender_email), sender_password = VALUES(sender_password), updated_at = VALUES(updated_at)
    """, (username, email, smtp_server, smtp_port, sender_email, sender_password, datetime.now()))
    conn.commit()
    conn.close()


def send_fault_alert_email(username, fault_status, confidence, threshold):
    """发送故障报警邮件（针对具体故障类型）"""
    config = get_user_email_config(username)
    if not config or not config.get('email'):
        return False

    def send():
        try:
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['email']
            msg['Subject'] = f'⚠️ 设备故障报警 - {fault_status}'

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            body = f"""
            <html>
            <body>
                <h2 style="color: #ff4757;">⚠️ 设备故障报警</h2>
                <hr>
                <p><strong>故障类型:</strong> <span style="color: red; font-size: 20px;">{fault_status}</span></p>
                <p><strong>故障置信度:</strong> {confidence}%</p>
                <p><strong>报警阈值:</strong> {threshold}%</p>
                <p><strong>报警时间:</strong> {current_time}</p>
                <hr>
                <p style="color: #888;">请及时检查设备运行状态！</p>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, 'html', 'utf-8'))

            server = smtplib.SMTP(config['smtp_server'], int(config['smtp_port']))
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
            server.quit()

            print(f"✅ 故障报警邮件已发送到 {config['email']} - 故障: {fault_status} (置信度: {confidence}%)")
        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")

    threading.Thread(target=send).start()
    return True


# ========== 训练历史存储 ==========
def save_training_history(model_type, r2, rmse, train_samples):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INT PRIMARY KEY AUTO_INCREMENT,
            model_type VARCHAR(50),
            r2 DECIMAL(10,4),
            rmse DECIMAL(10,2),
            train_samples INT,
            created_at DATETIME
        )
    """)
    cursor.execute("""
        INSERT INTO training_history (model_type, r2, rmse, train_samples, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (model_type, r2, rmse, train_samples, datetime.now()))
    conn.commit()
    conn.close()


def get_training_history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM training_history ORDER BY id DESC LIMIT 50
    """)
    history = cursor.fetchall()
    conn.close()
    return history


# ========== 用户管理 ==========
def get_all_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    return users


def add_user(username, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    if result['count'] > 0:
        conn.close()
        return False
    cursor.execute("INSERT INTO users (username, password, created_at) VALUES (%s, SHA2(%s, 256), %s)",
                   (username, password, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return True


def delete_user(username):
    if username == 'admin':
        return False
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    conn.close()
    return True


# ========== 报警记录 ==========
def save_alert(username, status, confidence):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_records (
            id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(50),
            alert_status VARCHAR(50),
            confidence DECIMAL(10,2),
            created_at DATETIME
        )
    """)
    cursor.execute("""
        INSERT INTO alert_records (username, alert_status, confidence, created_at)
        VALUES (%s, %s, %s, %s)
    """, (username, status, confidence, datetime.now()))
    conn.commit()
    conn.close()


def get_alert_records(username, limit=100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alert_records 
        WHERE username = %s 
        ORDER BY id DESC LIMIT %s
    """, (username, limit))
    records = cursor.fetchall()
    conn.close()
    return records


# ========== 数据管理函数 ==========
MAX_HISTORY = 150
MAX_TRASH = 150


def manage_data_limit():
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) as count FROM history_data")
        count = cursor.fetchone()['count']
        if count > MAX_HISTORY:
            overflow_count = count - MAX_HISTORY
            cursor.execute("""
                INSERT INTO trash_data (timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh)
                SELECT timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh
                FROM history_data ORDER BY id ASC LIMIT %s
            """, (overflow_count,))
            cursor.execute("DELETE FROM history_data ORDER BY id ASC LIMIT %s", (overflow_count,))
            cursor.execute("SELECT COUNT(*) as count FROM trash_data")
            trash_count = cursor.fetchone()['count']
            if trash_count > MAX_TRASH:
                delete_count = trash_count - MAX_TRASH
                cursor.execute("DELETE FROM trash_data ORDER BY id ASC LIMIT %s", (delete_count,))
            conn.commit()
            print(f"📦 已移动 {overflow_count} 条数据到垃圾桶")
    except Exception as e:
        print(f"⚠️ 数据上限管理失败: {e}")
    finally:
        conn.close()


def append_to_history(data):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history_data (timestamp, power_kw, runtime_hours, temperature_c, load_percent, energy_kwh)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (data['timestamp'], data['power_kw'], data['runtime_hours'], data['temperature_c'], data['load_percent'],
          data['energy_kwh']))
    conn.commit()
    conn.close()
    manage_data_limit()


# ========== 登录 ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if verify_password(username, password):
            session['username'] = username
            session['login_time'] = datetime.now().isoformat()
            session['alert_threshold'] = 3000
            session['alert_enabled'] = True
            session['last_alert_time'] = None
            session['confidence_threshold'] = 70
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('monitor.html', username=session['username'])


# ========== API 接口 ==========
@app.route('/api/latest')
def get_latest():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_data ORDER BY id DESC LIMIT 1")
        data = cursor.fetchone()
        conn.close()
        if data:
            for key in data:
                if isinstance(data[key], Decimal):
                    data[key] = float(data[key])
            return jsonify(data)
        return jsonify({'error': '暂无数据'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/history')
def get_history():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 5
        offset = (page - 1) * per_page

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM history_data")
        total = cursor.fetchone()['total']
        cursor.execute("""
            SELECT * FROM history_data 
            ORDER BY id DESC 
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        data = cursor.fetchall()
        conn.close()

        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        for row in data:
            for key in row:
                if isinstance(row[key], Decimal):
                    row[key] = float(row[key])

        return jsonify({
            'data': data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'data_full': total >= MAX_HISTORY,
            'max_limit': MAX_HISTORY
        })
    except Exception as e:
        return jsonify({'error': str(e), 'data': [], 'total': 0, 'page': 1, 'total_pages': 1})


@app.route('/api/trash')
def get_trash():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 5
        offset = (page - 1) * per_page

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM trash_data")
        total = cursor.fetchone()['total']
        cursor.execute("""
            SELECT * FROM trash_data 
            ORDER BY id DESC 
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        data = cursor.fetchall()
        conn.close()

        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        return jsonify({
            'data': data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'max_limit': MAX_TRASH
        })
    except Exception as e:
        return jsonify({'error': str(e), 'data': [], 'total': 0, 'page': 1, 'total_pages': 1})


@app.route('/api/clear-trash', methods=['POST'])
def clear_trash():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trash_data")
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '垃圾桶已清空'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/set-threshold', methods=['POST'])
def set_threshold():
    data = request.get_json()
    session['alert_threshold'] = float(data.get('threshold', 3000))
    session['alert_enabled'] = data.get('enabled', True)
    return jsonify({'success': True, 'threshold': session['alert_threshold'], 'enabled': session['alert_enabled']})


@app.route('/api/get-threshold')
def get_threshold():
    return jsonify({
        'threshold': session.get('alert_threshold', 3000),
        'enabled': session.get('alert_enabled', True)
    })


@app.route('/api/set-confidence-threshold', methods=['POST'])
def set_confidence_threshold():
    """设置故障置信度报警阈值"""
    data = request.get_json()
    confidence_threshold = float(data.get('confidence_threshold', 70))
    session['confidence_threshold'] = confidence_threshold
    return jsonify({'success': True, 'confidence_threshold': confidence_threshold})


@app.route('/api/get-confidence-threshold')
def get_confidence_threshold():
    """获取故障置信度报警阈值"""
    return jsonify({
        'confidence_threshold': session.get('confidence_threshold', 70)
    })


@app.route('/api/bind-email', methods=['POST'])
def bind_email():
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': '未登录'})

    data = request.get_json()
    email = data.get('email', '')
    smtp_server = data.get('smtp_server', 'smtp.qq.com')
    smtp_port = data.get('smtp_port', 587)
    sender_email = data.get('sender_email', '')
    sender_password = data.get('sender_password', '')

    if not email or not sender_email or not sender_password:
        return jsonify({'success': False, 'error': '请填写完整信息'})

    save_user_email(username, email, smtp_server, smtp_port, sender_email, sender_password)
    return jsonify({'success': True, 'message': '邮箱绑定成功'})


@app.route('/api/get-email-config')
def get_email_config():
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': '未登录'})

    config = get_user_email_config(username)
    return jsonify({
        'success': True,
        'email': config.get('email', ''),
        'smtp_server': config.get('smtp_server', 'smtp.qq.com'),
        'smtp_port': config.get('smtp_port', 587),
        'sender_email': config.get('sender_email', '')
    })


@app.route('/api/test-email', methods=['POST'])
def test_email():
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': '未登录'})

    config = get_user_email_config(username)
    if not config or not config.get('email'):
        return jsonify({'success': False, 'error': '请先绑定邮箱'})

    try:
        send_fault_alert_email(username, "测试故障", 99.5, 70)
        return jsonify({'success': True, 'message': f'测试邮件已发送到 {config["email"]}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ========== 预测接口 ==========
@app.route('/api/prediction')
def get_prediction():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT power_kw, runtime_hours, temperature_c, energy_kwh FROM history_data ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return jsonify({'success': False, 'remaining': 10 - len(rows)})

        predictor.buffers = {'sensor1': [], 'sensor2': [], 'sensor3': [], 'sensor4': []}
        for row in rows[::-1]:
            predictor.add_data_point(
                float(row['power_kw']), float(row['runtime_hours']),
                float(row['temperature_c']), float(row['energy_kwh'])
            )

        X = predictor.extract_features()
        X_scaled = predictor.scaler.transform(X)

        pred_proba = predictor.model.predict_proba(X_scaled)[0]
        pred = predictor.model.predict(X_scaled)[0]

        label_map = {
            'gearbox00': '正常',
            'gearbox10': '故障1',
            'gearbox20': '故障2',
            'gearbox30': '故障3',
            'gearbox40': '故障4'
        }

        # 构建投票分布
        votes = {}
        for i, label in enumerate(predictor.label_encoder.classes_):
            status_name = label_map.get(label, label)
            votes[status_name] = round(pred_proba[i] * 100, 1)

        status_cn = label_map.get(predictor.label_encoder.inverse_transform([pred])[0], '未知')
        confidence = max(pred_proba) * 100

        confidence_threshold = session.get('confidence_threshold', 70)

        # 报警逻辑：找出置信度最高的故障（排除正常），如果其置信度 >= 阈值则报警
        max_fault_status = None
        max_fault_confidence = 0

        for status_name, conf in votes.items():
            if status_name != '正常' and conf > max_fault_confidence:
                max_fault_confidence = conf
                max_fault_status = status_name

        alert_triggered = False
        alert_status = None

        if max_fault_status is not None and max_fault_confidence >= confidence_threshold:
            alert_triggered = True
            alert_status = max_fault_status
            save_alert(session.get('username'), max_fault_status, max_fault_confidence)
            send_fault_alert_email(session.get('username'), max_fault_status, max_fault_confidence,
                                   confidence_threshold)

        return jsonify({
            'success': True,
            'status': status_cn,
            'confidence': round(confidence, 1),
            'votes': votes,
            'threshold': confidence_threshold,
            'alert_triggered': alert_triggered,
            'alert_status': alert_status
        })
    except Exception as e:
        print(f"预测错误: {e}")
        return jsonify({'success': False, 'remaining': 10})


@app.route('/api/alert-records')
def get_alert_records_api():
    username = session.get('username')
    if not username:
        return jsonify({'records': []})
    records = get_alert_records(username)
    for r in records:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].isoformat()
    return jsonify({'records': records})


@app.route('/api/training-history')
def get_training_history_api():
    history = get_training_history()
    for h in history:
        if isinstance(h.get('created_at'), datetime):
            h['created_at'] = h['created_at'].isoformat()
    return jsonify({'history': history})


# ========== 用户管理 API ==========
@app.route('/api/users')
def get_users():
    if 'username' not in session:
        return jsonify({'users': []})
    users = get_all_users()
    for u in users:
        if isinstance(u.get('created_at'), datetime):
            u['created_at'] = u['created_at'].isoformat()
    return jsonify({'users': users})


@app.route('/api/add-user', methods=['POST'])
def api_add_user():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'})
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'error': '用户名和密码不能为空'})
    if add_user(username, password):
        return jsonify({'success': True, 'message': '用户添加成功'})
    return jsonify({'success': False, 'error': '用户名已存在'})


@app.route('/api/delete-user', methods=['POST'])
def api_delete_user():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'})
    data = request.get_json()
    username = data.get('username')
    if username == session.get('username'):
        return jsonify({'success': False, 'error': '不能删除当前登录用户'})
    if delete_user(username):
        return jsonify({'success': True, 'message': '用户已删除'})
    return jsonify({'success': False, 'error': '删除失败'})


# ========== Matplotlib 图表生成 ==========
@app.route('/api/matplotlib-charts')
def get_matplotlib_charts():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_data ORDER BY id DESC LIMIT 500")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 5:
            return jsonify({'success': False, 'error': '数据量不足，至少需要5条数据'})

        data_list = []
        for row in rows:
            data_list.append({
                'power_kw': float(row['power_kw']) if row['power_kw'] else 0,
                'runtime_hours': float(row['runtime_hours']) if row['runtime_hours'] else 0,
                'temperature_c': float(row['temperature_c']) if row['temperature_c'] else 0,
                'energy_kwh': float(row['energy_kwh']) if row['energy_kwh'] else 0
            })

        df = pd.DataFrame(data_list)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        charts = {}

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(df['power_kw'], df['runtime_hours'], alpha=0.5, color='#409eff', s=10)
        ax1.set_xlabel('振动位移1 (mm)', fontsize=12)
        ax1.set_ylabel('振动位移2 (mm)', fontsize=12)
        ax1.set_title('振动位移1 vs 振动位移2 相关性分析', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#fafafa')
        fig1.patch.set_facecolor('white')

        buffer1 = BytesIO()
        fig1.savefig(buffer1, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer1.seek(0)
        charts['scatter'] = base64.b64encode(buffer1.getvalue()).decode('utf-8')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(df['energy_kwh'], bins=30, color='#409eff', edgecolor='white', alpha=0.7)
        ax2.set_xlabel('振动加速度 (m/s²)', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.set_title('振动加速度分布直方图', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#fafafa')
        fig2.patch.set_facecolor('white')

        buffer2 = BytesIO()
        fig2.savefig(buffer2, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer2.seek(0)
        charts['histogram'] = base64.b64encode(buffer2.getvalue()).decode('utf-8')
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sensor_data = [df['power_kw'], df['runtime_hours'], df['temperature_c'], df['energy_kwh']]
        labels = ['位移1', '位移2', '位移3', '加速度']
        bp = ax3.boxplot(sensor_data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#409eff')
            patch.set_alpha(0.7)
        ax3.set_ylabel('幅值', fontsize=12)
        ax3.set_title('振动传感器数据分布对比', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#fafafa')
        fig3.patch.set_facecolor('white')

        buffer3 = BytesIO()
        fig3.savefig(buffer3, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer3.seek(0)
        charts['boxplot'] = base64.b64encode(buffer3.getvalue()).decode('utf-8')
        plt.close(fig3)

        return jsonify({'success': True, 'charts': charts})
    except Exception as e:
        print(f"Matplotlib 错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ========== 机器学习 ==========
MODEL_FILE = 'models/energy_model.pkl'


def train_model_with_data():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_data ORDER BY id DESC LIMIT 1000")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {'success': False, 'error': f'数据量不足（当前{len(rows)}条），至少需要10条'}

        data_list = []
        for row in rows:
            data_list.append({
                'power_kw': float(row['power_kw']) if row['power_kw'] else 0,
                'runtime_hours': float(row['runtime_hours']) if row['runtime_hours'] else 0,
                'temperature_c': float(row['temperature_c']) if row['temperature_c'] else 0,
                'load_percent': float(row['load_percent']) if row['load_percent'] else 0,
                'energy_kwh': float(row['energy_kwh']) if row['energy_kwh'] else 0
            })

        df = pd.DataFrame(data_list)
        features = ['power_kw', 'runtime_hours', 'temperature_c', 'load_percent']
        X = df[features]
        y = df['energy_kwh']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_FILE)

        save_training_history('线性回归', r2, rmse, len(X_train))

        return {'success': True, 'r2_score': round(r2, 4), 'rmse': round(rmse, 2), 'train_samples': len(X_train)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def train_all_models():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_data ORDER BY id DESC LIMIT 1000")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {'success': False, 'error': f'数据量不足（当前{len(rows)}条），至少需要10条'}

        data_list = []
        for row in rows:
            data_list.append({
                'power_kw': float(row['power_kw']) if row['power_kw'] else 0,
                'runtime_hours': float(row['runtime_hours']) if row['runtime_hours'] else 0,
                'temperature_c': float(row['temperature_c']) if row['temperature_c'] else 0,
                'load_percent': float(row['load_percent']) if row['load_percent'] else 0,
                'energy_kwh': float(row['energy_kwh']) if row['energy_kwh'] else 0
            })

        df = pd.DataFrame(data_list)
        features = ['power_kw', 'runtime_hours', 'temperature_c', 'load_percent']
        X = df[features]
        y = df['energy_kwh']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_score = r2_score(y_test, lr.predict(X_test))
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))

        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        dt_score = r2_score(y_test, dt.predict(X_test))
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt.predict(X_test)))

        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = r2_score(y_test, rf.predict(X_test))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))

        scores = {'线性回归': lr_score, '决策树': dt_score, '随机森林': rf_score}
        best_name = max(scores, key=scores.get)

        if best_name == '线性回归':
            best_model = lr
        elif best_name == '决策树':
            best_model = dt
        else:
            best_model = rf

        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, MODEL_FILE)

        return {
            'success': True,
            'comparison': {
                '线性回归': {'r2': round(lr_score, 4), 'rmse': round(lr_rmse, 2)},
                '决策树': {'r2': round(dt_score, 4), 'rmse': round(dt_rmse, 2)},
                '随机森林': {'r2': round(rf_score, 4), 'rmse': round(rf_rmse, 2)}
            },
            'best_model': best_name,
            'best_r2': round(scores[best_name], 4)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None


@app.route('/api/train', methods=['POST'])
def api_train():
    result = train_model_with_data()
    return jsonify(result)


@app.route('/api/train-all', methods=['POST'])
def api_train_all():
    result = train_all_models()
    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        model = load_model()
        if model is None:
            return jsonify({'success': False, 'error': '模型未训练，请先点击"训练模型"'})

        data = request.get_json()
        power = float(data.get('power', 0))
        runtime = float(data.get('runtime', 0))
        temperature = float(data.get('temperature', 0))
        load = float(data.get('load', 0))

        input_data = np.array([[power, runtime, temperature, load]])
        predicted = model.predict(input_data)[0]

        return jsonify({'success': True, 'predicted': round(predicted, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/model-info')
def api_model_info():
    if os.path.exists('models/model_info.json'):
        with open('models/model_info.json', 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})


if __name__ == '__main__':
    app.run(debug=True, port=5000)