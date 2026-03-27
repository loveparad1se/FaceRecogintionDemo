import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from FaceRecognition import FaceRecognition   # 你的原文件
from database import AttendanceDB

# ---------- Flask 初始化 ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- 初始化数据库 ----------
db = AttendanceDB('face_attendance.db')

# ---------- 初始化人脸识别器 ----------
# 注意：原类 __init__ 需要 username 和 img_path，但我们并不使用它的 load_test_images 方法。
# 因此随便传入空值，然后手动清空其内部字典，后续我们自己加载数据库中的所有员工特征。
recognizer = FaceRecognition(
    yolo26_weights_path='./runs/detect/Facenet/exp17/weights/best.pt',   # 你的模型路径
    username='',          # 占位
    img_path=''           # 占位
)
# 清空原类中可能因为传入空路径而自动加载的内容（防止干扰）
recognizer.face_features_db = {}

def load_all_employees_to_recognizer():
    """从数据库加载所有员工特征到识别器的字典中（键为工号）"""
    employees = db.get_all_employees()
    recognizer.face_features_db = {emp['emp_id']: emp['face_feature'] for emp in employees}
    print(f"✅ 已加载 {len(employees)} 名员工特征")

# 启动时加载一次
load_all_employees_to_recognizer()

# ---------- 路由 ----------
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    """员工注册：接收 base64 图像和表单信息"""
    try:
        emp_id = request.form.get('emp_id')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        image_data = request.form.get('image_data')

        if not all([emp_id, name, image_data]):
            return jsonify({'success': False, 'message': '请填写完整信息并拍照'})

        # 解析 base64 图像
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 使用你的 FaceRecognition 类检测人脸（原类中的 detect 方法我们复用）
        # 注意：原类中没有专门的 detect_faces 方法，但我们可以直接调用 predict 得到 boxes
        # 为了不修改原类，我们直接使用 recognizer.yolo26_model 进行检测
        results = recognizer.yolo26_model.predict(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'message': '未检测到人脸，请重新拍照'})

        # 取第一张人脸
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face_img = frame[y1:y2, x1:x2]

        # 提取特征（调用原类方法）
        face_tensor = recognizer.preprocess_face_img(face_img)
        face_feature = recognizer.extract_face_feature(face_tensor)

        # 保存到数据库
        success = db.register_employee(emp_id, name, age, gender, face_feature)
        if not success:
            return jsonify({'success': False, 'message': f'工号 {emp_id} 已存在'})

        # 重新加载特征到识别器
        load_all_employees_to_recognizer()

        return jsonify({'success': True, 'message': f'员工 {name} 注册成功'})
    except Exception as e:
        print(f"注册错误: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/attendance', methods=['POST'])
def attendance():
    """打卡：接收 base64 图像，返回匹配的员工信息"""
    try:
        image_data = request.form.get('image_data')
        check_type = request.form.get('check_type', 'check_in')   # 'check_in' 或 'check_out'

        if not image_data:
            return jsonify({'success': False, 'message': '请拍照'})

        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 检测人脸
        results = recognizer.yolo26_model.predict(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'message': '未检测到人脸，请重新拍照'})

        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face_img = frame[y1:y2, x1:x2]

        # 提取特征
        face_tensor = recognizer.preprocess_face_img(face_img)
        face_feature = recognizer.extract_face_feature(face_tensor)

        # 调用原类的 is_same_person 方法进行比对
        # 注意：is_same_person 返回匹配的字典键（这里键是工号）
        matched_emp_id = recognizer.is_same_person(face_feature, threshold=0.6)

        if matched_emp_id == "unknown":
            return jsonify({'success': False, 'message': '未识别到员工，请确认是否已注册'})

        # 查询员工详细信息
        emp_info = db.get_employee_by_id(matched_emp_id)
        if not emp_info:
            return jsonify({'success': False, 'message': '员工信息异常'})

        # 记录打卡（计算相似度最大值，在 is_same_person 中已计算，但未返回，这里临时再算一次）
        # 为了简洁，我们重新计算最高相似度（可以优化）
        max_sim = -1.0
        for name, db_feat in recognizer.face_features_db.items():
            sim = np.dot(face_feature, db_feat) / (np.linalg.norm(face_feature) * np.linalg.norm(db_feat) + 1e-8)
            if sim > max_sim:
                max_sim = sim
        db.add_attendance(matched_emp_id, max_sim, check_type)

        return jsonify({
            'success': True,
            'message': f'{emp_info["name"]} 打卡成功',
            'employee': emp_info,
            'confidence': f'{max_sim:.2%}'
        })
    except Exception as e:
        print(f"打卡错误: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/employees', methods=['GET'])
def employees():
    """获取所有员工列表"""
    employees = db.get_all_employees()
    result = [{'emp_id': e['emp_id'], 'name': e['name'], 'age': e['age'], 'gender': e['gender']} for e in employees]
    return jsonify(result)

@app.route('/records/today', methods=['GET'])
def today_records():
    """今日打卡记录"""
    rows = db.get_today_attendance()
    result = [{
        'emp_id': r[1],
        'name': r[2],
        'check_time': r[3],
        'check_type': '上班' if r[4] == 'check_in' else '下班',
        'confidence': f"{r[5]:.2%}" if r[5] else 'N/A'
    } for r in rows]
    return jsonify(result)

@app.route('/records/all', methods=['GET'])
def all_records():
    """全部打卡记录"""
    rows = db.get_all_attendance()
    result = [{
        'emp_id': r[1],
        'name': r[2],
        'check_time': r[3],
        'check_type': '上班' if r[4] == 'check_in' else '下班',
        'confidence': f"{r[5]:.2%}" if r[5] else 'N/A'
    } for r in rows]
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 启动人脸识别打卡系统...")
    print("📱 访问地址: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)