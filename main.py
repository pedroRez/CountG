from fastapi import FastAPI, File, UploadFile
import shutil
import os
from extract_frames import extract_frames

app = FastAPI()

def prepare_dataset_folders():
    paths = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
        "videos"
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    prepare_dataset_folders()

    # Salva o vídeo em 'videos/'
    video_path = os.path.join("videos", file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extrai os frames do vídeo para dataset/images/train
    output_folder = "dataset/images/train"
    extract_frames(video_path, output_folder, step=30)

    return {"message": "Vídeo recebido e frames extraídos com sucesso"}
