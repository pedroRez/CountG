# backend/contagem_video.py
from ultralytics import YOLO
import cv2

model = YOLO("yolov8-model/best.pt")  # Caminho do modelo treinado

def contar_bois_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        total += len(results[0].boxes)

    cap.release()
    return total