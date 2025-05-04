from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, asyncio
from contagem_video import run_video_prediction, processing_tasks, cancel_task, progress_status

app = FastAPI()

# Permitir chamadas do React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou defina seu IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "upload_ok", "video": file.filename}

@app.post("/iniciar-processamento/{video_name}")
async def iniciar_processamento(video_name: str):
    video_path = os.path.join(UPLOAD_DIR, video_name)

    if not os.path.exists(video_path):
        return {"error": "Vídeo não encontrado"}

    if video_name in processing_tasks and not processing_tasks[video_name].done():
        return {"status": "já em processamento"}

    task = asyncio.create_task(run_video_prediction(video_path))
    processing_tasks[video_name] = task
    return {"status": "processamento_iniciado", "video": video_name}

@app.get("/status/{video_name}")
def status(video_name: str):
    return progress_status.get(video_name, {"status": "desconhecido"})

@app.post("/cancelar-processamento/{video_name}")
def cancelar(video_name: str):
    return cancel_task(video_name)
