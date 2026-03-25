# from ultralytics import YOLO

# # 加载模型（官方预训练或自定义训练好的）
# model = YOLO("yolov8n.pt")          # 官方模型
# # model = YOLO("path/to/best.pt")   # 自定义模型

# # 执行验证
# metrics = model.val()

# # 查看各项指标
# print(f"mAP50-95: {metrics.box.map}")   # 主要指标
# print(f"mAP50: {metrics.box.map50}")    # IoU=0.5 时的 mAP
# print(f"mAP75: {metrics.box.map75}")    # IoU=0.75 时的 mAP
# print(f"各类别 mAP: {metrics.box.maps}") # 每个类别的 mAP




from ultralytics import YOLO

model = YOLO("./runs/detect/Facenet/exp17/weights/best.pt")

metrics = model.val(
    data="./dataset/mydata.yaml",   # 数据集配置（训练过的模型可省略）
    imgsz=640,              # 验证图像尺寸
    batch=16,               # 批量大小
    conf=0.001,             # 置信度阈值
    iou=0.6,                # NMS IoU 阈值
    device="cpu",           # 设备：cpu / 0 / mps
    save_json=True,         # 保存 COCO JSON 格式结果
    plots=True,             # 保存混淆矩阵等图表
)
# 查看各项指标
print(f"mAP50-95: {metrics.box.map}")   # 主要指标
print(f"mAP50: {metrics.box.map50}")    # IoU=0.5 时的 mAP
print(f"mAP75: {metrics.box.map75}")    # IoU=0.75 时的 mAP
print(f"各类别 mAP: {metrics.box.maps}") # 每个类别的 mAP


