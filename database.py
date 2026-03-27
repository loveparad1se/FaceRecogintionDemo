import sqlite3
import pickle
import numpy as np
from datetime import datetime

class AttendanceDB:
    """SQLite 数据库操作类（轻量级，无需安装）"""
    def __init__(self, db_path='face_attendance.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 员工表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    age INTEGER,
                    gender VARCHAR(10),
                    face_feature BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 打卡记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id VARCHAR(50) NOT NULL,
                    check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_type VARCHAR(20) DEFAULT 'check_in',
                    confidence REAL,
                    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
                )
            ''')
            conn.commit()
        print("✅ 数据库初始化完成")

    def register_employee(self, emp_id, name, age, gender, face_feature):
        """注册员工，保存特征（numpy数组 -> 二进制）"""
        feature_blob = pickle.dumps(face_feature)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO employees (emp_id, name, age, gender, face_feature)
                    VALUES (?, ?, ?, ?, ?)
                ''', (emp_id, name, age, gender, feature_blob))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                print(f"❌ 工号 {emp_id} 已存在")
                return False

    def get_all_employees(self):
        """获取所有员工特征（用于加载到人脸识别器）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT emp_id, name, age, gender, face_feature FROM employees')
            employees = []
            for row in cursor.fetchall():
                emp_id, name, age, gender, blob = row
                feature = pickle.loads(blob)
                employees.append({
                    'emp_id': emp_id,
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'face_feature': feature
                })
            return employees

    def get_employee_by_id(self, emp_id):
        """根据工号查询员工信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT emp_id, name, age, gender FROM employees WHERE emp_id = ?', (emp_id,))
            row = cursor.fetchone()
            if row:
                return {'emp_id': row[0], 'name': row[1], 'age': row[2], 'gender': row[3]}
            return None

    def add_attendance(self, emp_id, confidence, check_type='check_in'):
        """记录打卡"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO attendance (emp_id, check_type, confidence)
                VALUES (?, ?, ?)
            ''', (emp_id, check_type, confidence))
            conn.commit()

    def get_today_attendance(self):
        """获取今日所有打卡记录"""
        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, a.emp_id, e.name, a.check_time, a.check_type, a.confidence
                FROM attendance a
                JOIN employees e ON a.emp_id = e.emp_id
                WHERE DATE(a.check_time) = ?
                ORDER BY a.check_time DESC
            ''', (today,))
            return cursor.fetchall()

    def get_all_attendance(self, limit=100):
        """获取所有打卡记录（最近100条）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, a.emp_id, e.name, a.check_time, a.check_type, a.confidence
                FROM attendance a
                JOIN employees e ON a.emp_id = e.emp_id
                ORDER BY a.check_time DESC
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

    # ========== 新增删除功能 ==========
    
    def delete_employee(self, emp_id):
        """删除员工及其打卡记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 先检查员工是否存在
            cursor.execute('SELECT emp_id FROM employees WHERE emp_id = ?', (emp_id,))
            if not cursor.fetchone():
                return False
            # 删除打卡记录（由于设置了 ON DELETE CASCADE，也可以不手动删除）
            cursor.execute('DELETE FROM attendance WHERE emp_id = ?', (emp_id,))
            # 删除员工
            cursor.execute('DELETE FROM employees WHERE emp_id = ?', (emp_id,))
            conn.commit()
            return True

    def delete_all_attendance(self):
        """删除所有打卡记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance')
            conn.commit()
            return True

    def delete_all_employees(self):
        """删除所有员工（慎用）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance')
            cursor.execute('DELETE FROM employees')
            conn.commit()
            return True

    def get_employee_count(self):
        """获取员工总数"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM employees')
            return cursor.fetchone()[0]

    def get_attendance_count(self):
        """获取打卡记录总数"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM attendance')
            return cursor.fetchone()[0]