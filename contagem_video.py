import cv2
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import os
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

MODEL_PATH = "models/best.pt"
# model = YOLO(MODEL_PATH)
model = YOLO("yolov8n.pt")
LINE_POSITION = 300  # Posição X da linha virtual
OFFSET = 10

def run_video_prediction(video_path):
    print(f"[INFO] Iniciando contagem para o vídeo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Erro ao abrir vídeo"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = f"processed_{os.path.basename(video_path)}"
    output_path = os.path.join("videos", output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    counted_ids = set()
    total_count = 0
    class_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(frame, persist=True, conf=0.3)[0]
        if results.boxes is not None:
            boxes = results.boxes
            ids = boxes.id

            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.int().tolist()
                class_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                track_id = int(ids[i]) if ids is not None else None
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Verifica cruzamento da linha virtual (vertical)
                if track_id is not None:
                    if (LINE_POSITION - OFFSET) < cx < (LINE_POSITION + OFFSET):
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            total_count += 1
                            class_name = model.names[class_id]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            print(f"[CONTAGEM] Frame {frame_count}: {class_name} cruzou a linha (ID {track_id}).")

                # Desenho da bounding box e informações
                label = f"{model.names[class_id]} {conf:.2f} ID:{track_id}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # Desenha linha virtual vertical
        cv2.line(frame, (LINE_POSITION, 0), (LINE_POSITION, height), (0, 0, 255), 2)
        # Mostra a contagem total
        cv2.putText(frame, f"Total: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    duration_seconds = int(frame_count / fps) if fps > 0 else 0
    video_name = os.path.basename(video_path)

    print(f"[INFO] Contagem finalizada: {total_count} objetos contados.")

    return JSONResponse(content={
        "status": "ok",
        "video": video_name,
        "video_processado": output_name,
        "duration_seconds": duration_seconds,
        "total_frames": frame_count,
        "total_count": total_count,
        "por_classe": class_counts
    })
