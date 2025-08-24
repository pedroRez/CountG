import os
import re
import shutil
import threading
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List

from utils.gerenciador_progresso import ProgressoManager
from utils.contagem_video import contar_gado_em_video
from utils.sftp_handler import upload_file_sftp
from schemas import VideoRequest

router = APIRouter()
DATA_DIR = os.getenv("RENDER_DATA_DIR", "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progresso_manager = ProgressoManager()
processos_em_andamento = {}

@router.post("/upload-video/")
async def upload_video_endpoint(file: UploadFile = File(...)):
    """
    Recebe um vídeo do frontend, salva-o temporariamente no disco do servidor
    com um nome único e retorna esse nome.
    """
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_local_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    print(f"[UPLOAD] Recebendo '{file.filename}', salvando como '{unique_filename}'...")

    try:
        with open(temp_local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[UPLOAD] Vídeo salvo temporariamente em: {temp_local_path}")
    except Exception as e:
        print(f"[UPLOAD ERRO] Falha ao salvar o arquivo temporariamente: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao salvar o arquivo no servidor: {str(e)}")

    # O upload para a HostGator e a limpeza foram movidos para dentro de 'contar_gado_em_video'.
    # Este endpoint agora é muito mais rápido e simples.

    return {
        "message": f"Arquivo '{file.filename}' recebido com sucesso.",
        "nome_arquivo": unique_filename # Retorna o nome único usado no servidor
    }

@router.post("/predict-video/")
async def predict_video_endpoint(request: VideoRequest):
    video_name_on_server = request.nome_arquivo

    # Validação do nome do arquivo para evitar path traversal e caracteres inválidos
    if (
        ".." in video_name_on_server
        or "/" in video_name_on_server
        or "\\" in video_name_on_server
    ):
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido.")

    if not re.fullmatch(r"[\w.-]+", video_name_on_server):
        raise HTTPException(status_code=400, detail="Nome de arquivo contém caracteres inválidos.")

    expected_path = os.path.join(UPLOAD_FOLDER, video_name_on_server)
    abs_path = os.path.abspath(expected_path)
    upload_folder_abs = os.path.abspath(UPLOAD_FOLDER)
    if not abs_path.startswith(upload_folder_abs + os.sep):
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido.")

    if progresso_manager.is_processing(video_name_on_server):
        print(f"[PREDICT AVISO] Vídeo {video_name_on_server} já está sendo processado.")
        return JSONResponse(
            status_code=409,
            content={"status": "em_processamento", "message": "Este vídeo já está sendo processado."}
        )
    
    progresso_manager.iniciar(video_name_on_server)

    # --- FUNÇÃO DA THREAD CORRIGIDA ---
    def processamento_em_thread():
        try:
            print(f"[THREAD] Iniciando a chamada para contar_gado_em_video para: {video_name_on_server}")

            # Chama a função de contagem e armazena o resultado retornado
            resultado = contar_gado_em_video(
                video_path=os.path.join(UPLOAD_FOLDER, video_name_on_server),
                video_name=video_name_on_server,
                progresso_manager=progresso_manager,
                model_choice=request.model_choice,
                orientation=request.orientation,
                target_classes=request.target_classes,
                line_position_ratio=request.line_position_ratio,
            )

            # Se 'resultado' não for None (ou seja, o processamento foi bem-sucedido e não foi cancelado)...
            if resultado is not None:
                # ...chama .finalizar() para atualizar o banco de dados com os resultados.
                print(
                    f"[THREAD] contagem_video retornou um resultado. Finalizando o progresso no banco de dados..."
                )
                progresso_manager.finalizar(video_name_on_server, resultado)
            else:
                # Se resultado for None, o erro ou cancelamento já foi tratado dentro de contar_gado_em_video
                # e o status no banco de dados já foi atualizado para finalizado=True.
                print(
                    f"[THREAD] contagem_video retornou None. O status já deve estar como erro ou cancelado."
                )

        except Exception as e:
            import traceback
            print(
                f"[THREAD ERRO FATAL] Um erro inesperado ocorreu na thread para {video_name_on_server}: {e}"
            )
            traceback.print_exc()
            progresso_manager.erro(video_name_on_server, f"Erro crítico na thread: {str(e)}")
        finally:
            # Remover referência à thread após o término para liberar memória
            processos_em_andamento.pop(video_name_on_server, None)

    thread = threading.Thread(target=processamento_em_thread, daemon=True)
    thread.start()
    processos_em_andamento[video_name_on_server] = thread

    return {
        "status": "iniciado",
        "message": f"Processamento para '{video_name_on_server}' iniciado.",
        "video_name": video_name_on_server
    }

@router.get("/progresso/{video_name}")
async def progresso_endpoint(video_name: str):
    return progresso_manager.status(video_name)

@router.get("/cancelar-processamento/{video_name}")
async def cancelar_endpoint(video_name: str):
    if progresso_manager.cancelar(video_name):
        return {"message": f"Solicitação de cancelamento para {video_name} enviada."}
    return {"message": f"Não foi possível cancelar ou o processo para {video_name} não está ativo."}
