import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

LINE_X = 300
FRAME_SKIP = 1
MODEL_PATH = "yolov8n.pt"  # usa modelo base, não o best.pt

def contar_gado_em_video(video_path, video_name, progresso_manager):
    print(f"[INFO] Iniciando contagem para o vídeo: {video_path}")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_count = 0
    por_classe = defaultdict(int)
    track_ids_ja_contados = set()

    # Criar vídeo de saída com as anotações
    output_name = f"processed_{video_name}"
    output_path = f"videos/{output_name}"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_atual = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_atual % FRAME_SKIP == 0:
            # Verifica se foi solicitado cancelamento
            if not progresso_manager.atualizar(video_name, frame_atual, frame_count):
                break

            results = model.track(frame, persist=True, verbose=False)
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for track_id, cls, box in zip(ids, classes, boxes):
                    if track_id in track_ids_ja_contados:
                        continue

                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2  # Para linha vertical

                    # Contagem se cruzar a linha vertical
                    if LINE_X - 10 < x_center < LINE_X + 10:
                        track_ids_ja_contados.add(track_id)
                        total_count += 1
                        nome_classe = model.names[int(cls)]
                        por_classe[nome_classe] += 1

                    # Desenhar caixa e ID
                    label = f"{model.names[int(cls)]} ID:{int(track_id)}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.circle(frame, (int(x_center), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)

            # Desenhar linha vertical
            cv2.line(frame, (LINE_X, 0), (LINE_X, height), (0, 0, 255), 2)
            cv2.putText(frame, f"Total: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            out.write(frame)

        frame_atual += 1

    cap.release()
    out.release()
    print(f"[INFO] Contagem finalizada: {total_count} objetos contados.")

    return {
        "video": video_name,
        "video_processado": output_name,
        "total_frames": frame_count,
        "total_count": total_count,
        "por_classe": dict(por_classe),
    }