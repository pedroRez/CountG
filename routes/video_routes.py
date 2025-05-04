import os
import shutil
import threading
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from utils.contagem_video import contar_gado_em_video
from utils.gerenciador_progresso import ProgressoManager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import VideoRequest


router = APIRouter()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progresso_manager = ProgressoManager()
processos_em_andamento = {}

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": f"Vídeo {file.filename} enviado com sucesso.",
        "nome_arquivo": file.filename
    }

@router.post("/predict-video/")
async def predict_video(request: VideoRequest):
    video_name = request.nome_arquivo
    video_path = os.path.join(UPLOAD_FOLDER, video_name)

    if not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "Vídeo não encontrado."})

    progresso_manager.iniciar(video_name)

    def processamento():
        try:
            resultado = contar_gado_em_video(video_path, video_name, progresso_manager)
            progresso_manager.finalizar(video_name, resultado)
        except Exception as e:
            progresso_manager.erro(video_name, str(e))

    thread = threading.Thread(target=processamento)
    thread.start()
    processos_em_andamento[video_name] = thread

    return {
    "status": "iniciado",
    "message": "Processamento iniciado.",
    "video_name": video_name
}

@router.get("/progresso/{video_name}")
async def progresso(video_name: str):
    return progresso_manager.status(video_name)

@router.get("/cancelar-processamento/{video_name}")
async def cancelar(video_name: str):
    progresso_manager.cancelar(video_name)
    return {"message": f"Processamento de {video_name} cancelado."}
