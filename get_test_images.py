import cv2
import os

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 确保保存路径存在，如果不存在则创建
save_path = './test_images'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 初始化图片计数器
count = 0

# 定义全局变量来存储当前帧
current_frame = None


# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global count, current_frame
    if event == cv2.EVENT_RBUTTONDOWN:  # 检测到鼠标右键点击事件
        filename = os.path.join(save_path, f'frame_{count}.jpg')
        cv2.imwrite(filename, current_frame)  # 保存当前帧
        print(f'图片已保存。')
        count += 1


# 设置鼠标回调
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

# 循环读取帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 检查帧是否成功读取
    if not ret:
        print("摄像头无法读取图片帧")
        break

    # 更新当前帧
    current_frame = frame.copy()

    # 显示图片
    cv2.imshow('Frame', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

print("程序已退出。")
