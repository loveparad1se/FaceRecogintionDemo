import torch
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


class FaceRecognition:
    """
    定义一个 FaceRecognition 类，用于实现人脸识别
    """
    def __init__(self, yolo26_weights_path, username=None, img_path=None, auto_load=False):
        # 加载 YOLOv8 模型，用于检测图像中的人脸
        self.yolo26_model = YOLO(yolo26_weights_path)
        # 加载 FaceNet 模型，用于提取人脸特征
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to('cpu')
        # 初始化一个字典，用于存储数据库的人脸特征
        self.face_features_db = {}
        # 加载测试图像（数据库）并提取人脸特征
        if auto_load and username and img_path:
            self.load_test_images(username, img_path)

    def preprocess_face_img(self, face_img):
        """
        对人脸图像进行预处理
        :param face_img: 需要处理的图像
        :return:返回处理图像后的 PyTorch 张量
        """
        # 将BGR图像转换为RGB图像
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 定义 FaceNet 模型所需的输入图像尺寸
        required_size = (160, 160)
        # 使用OpenCV调整图像尺寸到模型期望的尺寸
        face_img = cv2.resize(face_img, required_size)
        # 将图像数据转换为 PyTorch 张量
        # 对张量的维度进行重排。permute(2, 0, 1)将图像的维度从HWC (高度、宽度、通道) 转换为CHW (通道、高度、宽度)， 在 PyTorch 中，图像张量的标准格式通常要求通道维度在前
        # 所以(2, 0, 1)的作用就是，将原张量索引为2的维度数据放到第一个位置，索引为0的维度数据放到第二个位置，索引为1的维度数据放到第三个位置
        # float()将张量数据类型转换为浮点数
        # /255.0 对张量中的数据进行归一化处理， 图像的像素值范围通常是 0 - 255， 一般会将其归一化到 0 - 1 的范围，有助于模型更好地收敛以及提高模型的泛化能力
        # unsqueeze(0)在张量的第 0 个维度上增加一个维度，因为模型期望输入是4维的 [B, C, H, W]
        # 模型在处理输入数据时，往往期望输入是一个批次（batch）的形式，即使只有一张图像，也需要将其包装成一个批次的形式，也就是在最前面增加一个维度来表示批次大小。
        face_tensor = (torch.tensor(face_img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        # 返回图像归一化后的 PyTorch 张量
        return face_tensor

    def extract_face_feature(self, face_tensor):
        """
        提取人脸特征
        :param face_tensor: 图像预处理后的 PyTorch 张量
        :return: 返回 face_embedding 人脸特征
        """
        # 使用 torch.no_grad() 上下文管理器，表示告诉 PyTorch 不需要计算梯度
        # 这通常用于推理阶段，可以减少内存消耗和提高速度
        with torch.no_grad():
            # 将处理后的图像张量传递给 FaceNet 模型以提取特征
            face_embedding = self.facenet_model(face_tensor)
            # 对提取的人脸特征进行L2归一化
            # 计算特征向量的L2范数
            l2_norm = torch.norm(face_embedding, p=2, dim=1, keepdim=True)
            # 将特征向量除以其L2范数进行归一化
            face_embedding_normalized = face_embedding.div(l2_norm)
        # 将得到的特征张量移动到 CPU 上，并转换为 NumPy 数组
        # 这一步是为了后续处理，如保存特征或进行其他非 PyTorch 操作
        return face_embedding_normalized.cpu().numpy().flatten()

    def load_test_images(self, username, img_path):
        """
        加载测试图像并提取人脸特征
        param test_images_dir: 测试的数据库目录，即包含测试图像的文件夹路径
        return: 无返回值，但会将提取到的人脸特征保存在字典 self.face_features_db 中
        """
        # 将输入的测试图像目录转换为 Path 对象，便于后续的文件路径操作
        frame = cv2.imread(img_path)
        # 遍历 test_images 文件夹下的所有.jpg图像文件
        results = self.yolo26_model.predict(frame, conf=0.7)
        boxes = results[0].boxes
        for box in boxes:
            # 将边界框的坐标（x1, y1, x2, y2）和类别转换为列表形式
            # 注意：xyxy表示边界框的坐标格式，即(x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = box.cls[0].item()
            # 检查检测到的类别是否为 0 (0 是人脸)
            if cls == 0:
                # 根据边界框的坐标从原图像中裁剪出人脸区域
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                # 对裁剪出的人脸图像进行预处理，转换为模型可以接受的输入格式
                face_tensor = self.preprocess_face_img(face_img)
                # 提取人脸特征
                face_feature = self.extract_face_feature(face_tensor)
                # 将提取到的人脸特征和对应的文件名保存在字典self.face_features_db中， {文件名: 人脸特征}
                self.face_features_db[username] = face_feature
        print(f'获取到的字典：{self.face_features_db}')

    def is_same_person(self, face_feature, threshold=0.7):
        face_feature = np.asarray(face_feature).ravel()
        """
        判断人脸特征是否属于数据库中的人
        :param face_feature: 待匹配的人脸特征向量 (numpy array)
        :param threshold: 匹配阈值
        :return: 匹配的人名或 "unknown"
        """
        max_similarity = -1.0
        matched_name = "unknown"

        # 确保输入特征是一维数组
        feat = face_feature.flatten()
    
        for name, db_feature in self.face_features_db.items():
            # 确保数据库特征也是一维数组
            db_feat = db_feature.flatten()
        
            # 计算余弦相似度
            dot_product = np.dot(feat, db_feat)
            norm_product = np.linalg.norm(feat) * np.linalg.norm(db_feat)
            similarity = dot_product / (norm_product + 1e-8)  # 加小值防止除零
        
            print(f'{name} similarity: {similarity:.4f}')
        
            if similarity > max_similarity:
                max_similarity = similarity
                matched_name = name
    
        return matched_name if max_similarity > threshold else "unknown"

    def draw_results(self, frame, boxes):
        """
        在视频帧上绘制检测到的人脸框，并识别和标注人脸。
        :param frame: 视频帧，用于绘制人脸框和人脸名称。
        :param boxes: 检测到的人脸框列表，每个框包含人脸的位置信息。
        :return: 绘制了人脸框和人脸名称的视频帧。
        """
        # 遍历检测到的人脸框列表
        for box in boxes:
            # 解包人脸框的坐标信息，xyxy是一个包含左上角和右下角坐标的列表
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 提取类别标签，这里类别 0 代表人脸
            cls = box.cls[0].item()
            # 如果类别标签为0（即为人脸）
            if cls == 0:
                # 在原始帧上画一个红色的框来标记人脸
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # 从原始帧中裁剪出人脸图像区域
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                # 检查裁剪出的人脸图像是否非空
                if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    # 对裁剪出的人脸图像进行预处理
                    face_tensor = self.preprocess_face_img(face_img)
                    # 从预处理后的人脸图像中提取特征
                    face_feature = self.extract_face_feature(face_tensor)
                    # 使用提取的特征与数据库中的特征进行比较，以确定匹配的姓名
                    matched_name = self.is_same_person(face_feature)
                    # 在人脸框上方绘制匹配的人脸名称，使用绿色字体
                    cv2.putText(frame, matched_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
        # 返回绘制了人脸框和人脸名称的视频帧
        return frame

    def open_camera(self):
        """
        打开摄像头，实时检测和识别视频中的人脸。
        该方法会持续运行，直到用户按下'q'键退出。
        :return: 无返回值
        """
        # 初始化摄像头，参数 0 通常代表默认摄像头
        cap = cv2.VideoCapture(0)
        # 循环读取摄像头帧，直到用户关闭窗口或按下'q'键
        while cap.isOpened():
            # 读取一帧图像，success是一个布尔值，表示读取是否成功
            success, frame = cap.read()
            # 如果成功读取帧
            if success:
                # 使用 YOLOv8 模型对当前帧进行预测，conf是置信度阈值
                results = self.yolo26_model.predict(frame, conf=0.7)
                # 获取检测结果中的边界框信息
                boxes = results[0].boxes
                # 在当前帧上绘制人脸检测结果和匹配的姓名
                annotated_frame = self.draw_results(frame, boxes)
                # 显示带有标注的帧
                cv2.imshow('Face', annotated_frame)
                # 如果按下键盘上的'q'键，则退出循环
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
        # 释放摄像头资源
        cap.release()
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 定义 YOLOv8 模型权重文件的路径
    yolo26_weights_path = './runs/detect/Facenet/exp17/weights/best.pt'
    # 定义测试图像的目录路径
    img_path = ''
    username = ''
    # 创建 FaceRecognition 类的实例，传入 YOLOv8 模型权重路径和测试图像目录
    face_recognition = FaceRecognition(yolo26_weights_path, username,
img_path)
    # 调用 open_camera 方法，打开摄像头进行实时人脸检测和识别
    face_recognition.open_camera()
