# FaceRecognition

A simple face recognition check-in system built with **YOLO** (object detection) and **FaceNet** (face embedding).  
This is my first hands-on computer vision project — a learning exercise to understand the full pipeline from data preparation to deployment.

---

## 🧠 Project Overview

The system works in two stages:

1. **Face Detection**  
   YOLO detects faces in an image or video frame and returns bounding boxes.

2. **Face Recognition**  
   The detected face is cropped and passed to FaceNet, which extracts a 512-dimensional feature vector.  
   This vector is compared with stored embeddings to identify the person and record attendance.

---

## 🛠️ Model & Training

| Component | Model | Note |
|-----------|-------|------|
| Face Detection | **YOLOv26n** | Fine-tuned on my custom dataset |
| Face Recognition | **FaceNet** (InceptionResnetV1) | Pre-trained on CASIA-WebFace, used as feature extractor |

The detection model was fine-tuned for **75 epochs** on a small face dataset.  
The final weights are saved as `best.pt`.

> ⚠️ The custom model weights (`best.pt`) are **not included** in this repository due to file size limits.  
> You can use the official `yolo26n.pt` for detection, or train your own.

---

## 📁 Project Structure
FaceRecognition/
├── app.py # Flask web application
├── FaceRecognition.py # Core detection & recognition logic
├── database.py # SQLite database operations
├── templates/
│ └── index.html # Frontend interface
├── requirements.txt # Python dependencies
├── README.md # This file
├── dataset/ # Training dataset (see below)
│ ├── images/
│ └── labels/
└── runs/ # (Optional) trained model weights

## 📂 Dataset

The dataset used for fine-tuning the YOLO face detection model is included in this repository under the `dataset/` folder.

- **Format**: YOLO annotation format
- **Classes**: 1 (face)
- **Split**: Training and validation sets included

If you wish to train your own model, the dataset is ready to use.


## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/loveparad1se/FaceRecognition.git
cd FaceRecognition


Set Up Environment
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows

Install Dependencies
pip install -r requirements.txt

Download or Prepare Models
Option A: Use the official YOLO model (automatically downloaded)
Option B: Train your own model using the provided dataset

To train your own detection model:
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.train(data='dataset/dataset.yaml', epochs=75, imgsz=640)

Run the Application
python app.py
Then open http://127.0.0.1:5000 in your browser

⚠️ Limitations & Future Improvements
This is a learning project, and there are many areas to improve:
❌ No multi-face batch processing — each face is processed individually
❌ Limited to frontal faces (side faces may not be detected reliably)
❌ Simple SQLite backend — not suitable for large-scale deployment
❌ No real-time video stream processing (only static photo capture)
Future ideas:
Add real-time camera feed detection
Improve accuracy with better dataset augmentation
Support multiple simultaneous face recognition in one frame

📚 Acknowledgments
Ultralytics YOLO for the detection framework
FaceNet PyTorch for the recognition model
The open-source community for countless learning resources

📄 License
This project is for personal learning purposes. Feel free to use and modify it for your own practice

👤 Author
loveparad1se — first-time computer vision learner exploring detection and recognition.
Feedback and suggestions are very welcome!