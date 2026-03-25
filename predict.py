
# 推理单张图片
from ultralytics import YOLO

# 加载模型
model = YOLO("./runs/detect/Facenet/exp17/weights/best.pt")  # 或自定义模型

# 预测单张图片
results = model("./xgw.jpg")

# 显示结果
results[0].show()                     # 弹窗显示
results[0].save("result.jpg")         # 保存结果图片

# 打印检测信息
for r in results:
    boxes = r.boxes                   # 检测框对象
    if boxes is not None:
        print(f"检测到 {len(boxes)} 个目标")
        for box in boxes:
            cls = int(box.cls[0])                     # 类别ID
            conf = float(box.conf[0])                 # 置信度（转为浮点数）
            xyxy = box.xyxy[0].tolist()            # 坐标（转为列表）
            print(f"类别: {cls}, 置信度: {conf:.2f}, 坐标: {xyxy}")

'''
# 批量推理
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# 对整个文件夹进行预测，自动保存结果
results = model.predict(
    source="path/to/images/",   # 图片文件夹路径
    save=True,                   # 保存结果图片
    save_txt=True,               # 保存标注 TXT 文件
    conf=0.25,                   # 置信度阈值
    iou=0.45,                    # NMS IoU 阈值
    device="cpu",                # 设备
)
'''


# # 视频/摄像头实时推理

# from ultralytics import YOLO

# model = YOLO("./runs/detect/Facenet/exp17/weights/best.pt")

# # # 视频文件预测
# # results = model.predict(
# #     source="video.mp4",
# #     save=True,                   # 保存结果视频
# #     show=True,                   # 实时显示
# # )

# # 摄像头实时预测（0 表示第一个摄像头）
# results = model.predict(
#     source=0,
#     show=True,
#     save=False,
# )