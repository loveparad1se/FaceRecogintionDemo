from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolo26n.pt')


model.train(
    data = './dataset/mydata.yaml',
    epochs=75,
    imgsz=640,
    batch=32,
    device=0,
    project='Facenet',
    name='exp1',
    workers=8,
    )