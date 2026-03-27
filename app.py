import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from FaceRecognition import FaceRecognition
from database import AttendanceDB

# ---------- Flask 初始化 ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- 初始化数据库 ----------
db = AttendanceDB('face_attendance.db')

# ---------- 初始化人脸识别器 ----------
recognizer = FaceRecognition(
    yolo26_weights_path='./runs/detect/Facenet/exp17/weights/best.pt',
    username='',
    img_path='',
    auto_load=False,
)
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

        # 检测人脸
        results = recognizer.yolo26_model.predict(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'message': '未检测到人脸，请重新拍照'})

        # 取第一张人脸
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face_img = frame[y1:y2, x1:x2]

        # 提取特征
        face_tensor = recognizer.preprocess_face_img(face_img)
        face_feature = recognizer.extract_face_feature(face_tensor).flatten()

        # 检查人脸是否已注册
        matched_emp_id = recognizer.is_same_person(face_feature, threshold=0.6)
        if matched_emp_id != "unknown":
            existing_emp = db.get_employee_by_id(matched_emp_id)
            if existing_emp:
                max_sim = 0.0
                for db_feat in recognizer.face_features_db.values():
                    db_feat_1d = db_feat.flatten()
                    sim = np.dot(face_feature, db_feat_1d) / (
                        np.linalg.norm(face_feature) * np.linalg.norm(db_feat_1d) + 1e-8
                    )
                    if sim > max_sim:
                        max_sim = sim
                return jsonify({
                    'success': False,
                    'message': f'该人脸已注册（匹配用户：{existing_emp["name"]}，相似度：{max_sim:.2f}），请勿重复注册'
                })

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
        check_type = request.form.get('check_type', 'check_in')

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

        # 取第一张人脸进行识别
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face_img = frame[y1:y2, x1:x2]
        face_tensor = recognizer.preprocess_face_img(face_img)
        face_feature = recognizer.extract_face_feature(face_tensor).flatten()
        matched_emp_id = recognizer.is_same_person(face_feature, threshold=0.6)
        
        if matched_emp_id == "unknown":
            return jsonify({'success': False, 'message': '未识别到员工，请确认是否已注册'})

        # 查询员工详细信息
        emp_info = db.get_employee_by_id(matched_emp_id)
        if not emp_info:
            return jsonify({'success': False, 'message': '员工信息异常'})

        # 计算实际相似度
        max_sim = 0.0
        for name, db_feat in recognizer.face_features_db.items():
            db_feat_1d = db_feat.flatten()
            sim = np.dot(face_feature, db_feat_1d) / (
                np.linalg.norm(face_feature) * np.linalg.norm(db_feat_1d) + 1e-8
            )
            if sim > max_sim:
                max_sim = sim

        # 记录打卡
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


@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    """删除打卡记录"""
    try:
        record_id = request.form.get('record_id')
        if not record_id:
            return jsonify({'success': False, 'message': '请提供记录ID'})
        
        success = db.delete_attendance(record_id)
        if success:
            return jsonify({'success': True, 'message': '打卡记录已删除'})
        else:
            return jsonify({'success': False, 'message': '未找到该记录'})
    except Exception as e:
        print(f"删除打卡记录错误: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/delete_employee', methods=['POST'])
def delete_employee():
    """删除员工"""
    try:
        emp_id = request.form.get('emp_id')
        
        if not emp_id:
            return jsonify({'success': False, 'message': '请提供工号'})
        
        success = db.delete_employee(emp_id)
        
        if success:
            load_all_employees_to_recognizer()
            return jsonify({'success': True, 'message': f'员工 {emp_id} 已删除'})
        else:
            return jsonify({'success': False, 'message': f'未找到工号 {emp_id} 的员工'})
    except Exception as e:
        print(f"删除错误: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/employees', methods=['GET'])
def employees():
    """获取所有员工列表"""
    employees = db.get_all_employees()
    result = [{'emp_id': e['emp_id'], 'name': e['name'], 'age': e['age'], 'gender': e['gender']} for e in employees]
    return jsonify(result)


@app.route('/records/all', methods=['GET'])
def all_records():
    """获取所有打卡记录"""
    rows = db.get_all_attendance()
    result = [{
        'id': r[0],
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