import cv2
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import os
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})



# Caminho para o modelo treinado
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)
# Define a linha virtual
LINE_POSITION = 300  # Ajuste conforme necessário
OFFSET = 10

def run_video_prediction(video_path):
    print(f"[INFO] Iniciando contagem para o vídeo: {video_path}")
    print(f"[INFO] Nome do vídeo: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir o vídeo.")
        return {"error": "Erro ao abrir vídeo"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    counted_ids = set()
    total_count = 0
    class_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Usar rastreamento com IDs persistentes
        results = model.track(frame, persist=True, conf=0.3)[0]

        if results.boxes is not None:
            boxes = results.boxes
            ids = boxes.id  # IDs de rastreamento

            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.int().tolist()
                class_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                track_id = int(ids[i]) if ids is not None else None

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Verifica se cruzou a linha virtual e ainda não foi contado
                if track_id is not None:
                    if (LINE_POSITION - OFFSET) < cy < (LINE_POSITION + OFFSET):
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            total_count += 1
                            class_name = model.names[class_id]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            print(f"[CONTAGEM] Frame {frame_count}: {class_name} cruzou a linha (ID {track_id}).")

    duration_seconds = int(frame_count / fps) if fps > 0 else 0
    video_name = os.path.basename(video_path)

    cap.release()

    print(f"[INFO] Nome do vídeo: {video_name}")
    print(f"[INFO] Duração: {duration_seconds} segundos")
    print(f"[INFO] Contagem finalizada: {total_count} objetos contados.")

    return JSONResponse(content={
        "status": "ok",
        "video": video_name,
        "duration_seconds": duration_seconds,
        "total_frames": frame_count,
        "total_count": total_count,
        "por_classe": class_counts
    })
