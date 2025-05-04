import cv2, os, time, asyncio
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import datetime

torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

MODEL_PATH = "models/best.pt"
model = YOLO("yolov8n.pt")  # Substitua por MODEL_PATH se for usar seu modelo treinado
LINE_POSITION = 300
OFFSET = 10
FRAME_SKIP = 1

processing_tasks = {}
progress_status = {}  # nome_do_video: {frame_atual, total_frames_estimado, tempo_restante}


def format_seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


def cancel_task(video_name):
    task = processing_tasks.get(video_name)
    if task and not task.done():
        task.cancel()
        return {"status": "cancelado", "video": video_name}
    return {"status": "não encontrado ou já finalizado", "video": video_name}


async def run_video_prediction(video_path):
    print(f"[INFO] Iniciando contagem para o vídeo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Erro ao abrir vídeo"}

    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_est = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    output_name = f"processed_{os.path.basename(video_path)}"
    output_path = os.path.join("videos", output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    video_name = os.path.basename(video_path)
    frame_count = 0
    counted_ids = set()
    total_count = 0
    class_counts = {}

    try:
        while True:
            await asyncio.sleep(0)  # permite cancelamento
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            processed_frames = frame_count // FRAME_SKIP
            elapsed_time = time.time() - start_time
            average_time_per_frame = elapsed_time / processed_frames if processed_frames else 0
            remaining_frames = (total_frames_est - frame_count) // FRAME_SKIP
            estimated_remaining_time = remaining_frames * average_time_per_frame
            formatted_time = format_seconds_to_hhmmss(estimated_remaining_time)

            progress_status[video_name] = {
                "frame_atual": frame_count,
                "total_frames_estimado": int(total_frames_est),
                "tempo_restante": formatted_time
            }

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

                    if track_id is not None:
                        if (LINE_POSITION - OFFSET) < cx < (LINE_POSITION + OFFSET):
                            if track_id not in counted_ids:
                                counted_ids.add(track_id)
                                total_count += 1
                                class_name = model.names[class_id]
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    label = f"{model.names[class_id]} {conf:.2f} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            cv2.line(frame, (LINE_POSITION, 0), (LINE_POSITION, height), (0, 0, 255), 2)
            cv2.putText(frame, f"Total: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(frame)

    except asyncio.CancelledError:
        print("[INFO] Processamento cancelado pelo usuário.")
        cap.release()
        out.release()
        return JSONResponse(content={"status": "cancelado", "video": video_name})

    cap.release()
    out.release()
    duration_seconds = int(frame_count / fps) if fps > 0 else 0
    progress_status.pop(video_name, None)

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
