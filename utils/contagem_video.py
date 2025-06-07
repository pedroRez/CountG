import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable

# Importa as funções SFTP do seu handler
from utils.sftp_handler import download_file_sftp, upload_file_sftp, delete_file_sftp

# --- Constantes para Clareza ---
LINE_HORIZONTAL: str = "horizontal"
LINE_VERTICAL: str = "vertical"
LINE_DIAGONAL_FWD: str = "diag_fwd"  # Diagonal como /
LINE_DIAGONAL_BCK: str = "diag_bck"  # Diagonal como \

MOVE_TB: str = "top_bottom";         MOVE_BT: str = "bottom_top"
MOVE_LR: str = "left_right";       MOVE_RL: str = "right_left"
MOVE_TL_BR: str = "topleft_bottomright"; MOVE_BR_TL: str = "bottomright_topleft"
MOVE_TR_BL: str = "topright_bottomleft"; MOVE_BL_TR: str = "bottomleft_topright"


def get_line_and_direction_config(orientation_code: str, width: int, height: int, line_ratio: float = 0.5) -> Tuple[Optional[str], Optional[str], Optional[Tuple], Optional[int], Optional[Tuple]]:
    """Determina tipo de linha, posição e direção com base na orientação."""
    line_type, effective_counting_direction, line_points, line_pos_value, arrow_points = None, None, None, None, None
    orientation_code = str(orientation_code).upper()
    arrow_offset = 60

    if orientation_code == 'S':
        line_type, effective_counting_direction = LINE_HORIZONTAL, MOVE_TB
        line_pos_value = int(height * line_ratio)
        line_points = ((0, line_pos_value), (width, line_pos_value))
        arrow_points = ((width // 2, line_pos_value - arrow_offset), (width // 2, line_pos_value + arrow_offset))
    elif orientation_code == 'N':
        line_type, effective_counting_direction = LINE_HORIZONTAL, MOVE_BT
        line_pos_value = int(height * line_ratio)
        line_points = ((0, line_pos_value), (width, line_pos_value))
        arrow_points = ((width // 2, line_pos_value + arrow_offset), (width // 2, line_pos_value - arrow_offset))
    elif orientation_code == 'E':
        line_type, effective_counting_direction = LINE_VERTICAL, MOVE_LR
        line_pos_value = int(width * line_ratio)
        line_points = ((line_pos_value, 0), (line_pos_value, height))
        arrow_points = ((line_pos_value - arrow_offset, height // 2), (line_pos_value + arrow_offset, height // 2))
    elif orientation_code == 'W':
        line_type, effective_counting_direction = LINE_VERTICAL, MOVE_RL
        line_pos_value = int(width * line_ratio)
        line_points = ((line_pos_value, 0), (line_pos_value, height))
        arrow_points = ((line_pos_value + arrow_offset, height // 2), (line_pos_value - arrow_offset, height // 2))
    elif orientation_code == 'SE': 
        line_type, effective_counting_direction = LINE_DIAGONAL_FWD, MOVE_TL_BR
        line_points, arrow_points = ((0,0), (width,height)), ((width//4, height//4), (3*width//4, 3*height//4))
    elif orientation_code == 'NW': 
        line_type, effective_counting_direction = LINE_DIAGONAL_FWD, MOVE_BR_TL
        line_points, arrow_points = ((0,0), (width,height)), ((3*width//4, 3*height//4), (width//4, height//4))
    elif orientation_code == 'NE': 
        line_type, effective_counting_direction = LINE_DIAGONAL_BCK, MOVE_BL_TR
        line_points, arrow_points = ((0,height),(width,0)), ((width//4, 3*height//4), (3*width//4, height//4))
    elif orientation_code == 'SW': 
        line_type, effective_counting_direction = LINE_DIAGONAL_BCK, MOVE_TR_BL
        line_points, arrow_points = ((0,height),(width,0)), ((3*width//4, height//4), (width//4, 3*height//4))
    else: 
        print(f"[AVISO get_line_config] Orientação '{orientation_code}' desconhecida. Usando padrão Leste (E).")
        line_type, effective_counting_direction = LINE_VERTICAL, MOVE_LR
        line_pos_value = int(width * line_ratio); line_points = ((line_pos_value, 0), (line_pos_value, height))
        arrow_points = ((line_pos_value - arrow_offset, height // 2), (line_pos_value + arrow_offset, height // 2))
    
    print(f"[get_line_config] Para '{orientation_code}': tipo={line_type}, dir={effective_counting_direction}")
    return line_type, effective_counting_direction, line_points, line_pos_value, arrow_points

# !!! ATENÇÃO: ESTA FUNÇÃO PARA DETECÇÃO DIAGONAL É UM ESBOÇO CONCEITUAL. !!!
# A lógica de cruzamento precisa ser refinada com matemática vetorial mais robusta.
def is_crossing_diagonal_line(p_prev_x: int, p_prev_y: int, p_curr_x: int, p_curr_y: int, 
                              line_p1: Tuple[int,int], line_p2: Tuple[int,int], direction: str) -> bool:
    (x1_line, y1_line), (x2_line, y2_line) = line_p1, line_p2
    val_prev = (float(p_prev_y - y1_line) * (x2_line - x1_line)) - (float(p_prev_x - x1_line) * (y2_line - y1_line))
    val_curr = (float(p_curr_y - y1_line) * (x2_line - x1_line)) - (float(p_curr_x - x1_line) * (y2_line - y1_line))
    crossed = (val_prev < 0 and val_curr >= 0) or (val_prev > 0 and val_curr <= 0)
    if not crossed: return False
    if direction == MOVE_TL_BR: return (p_curr_x > p_prev_x and p_curr_y > p_prev_y)
    elif direction == MOVE_BR_TL: return (p_curr_x < p_prev_x and p_curr_y < p_prev_y)
    elif direction == MOVE_BL_TR: return (p_curr_x > p_prev_x and p_curr_y < p_prev_y)
    elif direction == MOVE_TR_BL: return (p_curr_x < p_prev_x and p_curr_y > p_prev_y)
    return False

def contar_gado_em_video(video_path: str,
                         video_name: str, 
                         progresso_manager: Any,
                         model_choice: str = "l", 
                         frame_skip: int = 1,
                         orientation: str = "S", 
                         target_classes: Optional[List[str]] = None,
                         line_position_ratio: float = 0.5,
                         create_annotated_video: bool = False) -> Optional[Dict[str, Any]]:

    sftp_current_action = ""
    def sftp_progress_callback(bytes_transferred: int, total_bytes: int):
        if total_bytes > 0:
            percentage = (bytes_transferred / total_bytes) * 100
            status_message = f"{sftp_current_action}: {percentage:.0f}%"
            progresso_manager.update_status_message(video_name, status_message)
    
    # --- FLUXO DE ARQUIVOS CORRIGIDO ---
    local_video_path = video_path # Caminho onde o vídeo foi salvo temporariamente no servidor
    remote_video_original = f"public_html/kyoday_videos/uploads/{video_name}"

    if not os.path.exists(local_video_path):
        error_msg = f"Arquivo de vídeo local não encontrado em {local_video_path}. O upload inicial falhou."
        if progresso_manager: progresso_manager.erro(video_name, error_msg)
        return None

    # 1. FAZ UPLOAD DO VÍDEO ORIGINAL PARA HOSTGATOR (PARA BACKUP/ARMAZENAMENTO)
    sftp_current_action = "Enviando p/ Servidor"
    if not upload_file_sftp(local_video_path, remote_video_original, progress_callback=sftp_progress_callback):
        error_msg = "Falha ao enviar o vídeo original para a HostGator."
        if progresso_manager: progresso_manager.erro(video_name, error_msg)
        if os.path.exists(local_video_path): os.remove(local_video_path)
        return None

    # --- INÍCIO DO PROCESSAMENTO ---
    model_files: Dict[str, str] = {"n": "yolov8n.pt", "m": "yolov8m.pt", "l": "yolov8l.pt", "p": "best.pt"}
    actual_model_path: str = model_files.get(str(model_choice).lower(), "yolov8l.pt")
    print(f"[CONTAGEM] Modelo: '{actual_model_path}', Orientação: {orientation}, Gerar Vídeo: {create_annotated_video}")

    try: model = YOLO(actual_model_path)
    except Exception as e:
        if progresso_manager: progresso_manager.erro(video_name, f"Falha ao carregar modelo: {e}")
        return None

    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        if progresso_manager: progresso_manager.erro(video_name, "Falha ao abrir o arquivo de vídeo.")
        return None
    
    original_frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps: float = cap.get(cv2.CAP_PROP_FPS)
    _fps: float = fps if fps > 0 else 30.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        if progresso_manager: progresso_manager.erro(video_name, "Dimensões do vídeo inválidas.")
        cap.release(); return None

    line_type, effective_counting_dir, line_points, line_coord_val, arrow_points = \
        get_line_and_direction_config(orientation, width, height, line_position_ratio)
    if line_points is None:
        if progresso_manager: progresso_manager.erro(video_name, f"Configuração de linha inválida para orientação '{orientation}'.")
        cap.release(); return None

    current_total_count: int = 0; current_por_classe: Dict[str, int] = defaultdict(int)
    track_ids_contados: set = set(); track_previous_x: Dict[int, int] = {}; track_previous_y: Dict[int, int] = {}
    
    out: Optional[cv2.VideoWriter] = None; local_output_path = ""; processed_fn = ""
    if create_annotated_video:
        output_dir_local = "videos_processados_temp"; os.makedirs(output_dir_local, exist_ok=True)
        base_name, vid_ext = os.path.splitext(video_name); processed_fn = f"processed_{base_name}{vid_ext}"
        local_output_path = os.path.join(output_dir_local, processed_fn)
        try:
            out = cv2.VideoWriter(local_output_path, cv2.VideoWriter_fourcc(*"mp4v"), _fps, (width, height))
            if not out.isOpened(): raise IOError(f"VideoWriter falhou para {local_output_path}")
        except Exception as e:
            print(f"[ERRO CONTAGEM] VideoWriter: {e}"); cap.release()
            if progresso_manager: progresso_manager.erro(video_name, f"VideoWriter: {e}")
            return None
    
    frame_atual = 0
    while cap.isOpened():
        if progresso_manager and progresso_manager.status(video_name).get("cancelado"):
            print(f"[INFO] Processamento de {video_name} cancelado."); break
        ret, frame = cap.read()
        if not ret: break

        if frame_atual % frame_skip == 0:
            if not progresso_manager.atualizar(video_name, frame_atual, original_frame_count):
                print(f"[INFO] Processamento de {video_name} cancelado via progresso_manager."); break
            
            results = model.track(frame, persist=True, verbose=False, conf=0.3)
            
            annotated_frame = frame.copy() if create_annotated_video else None
            if results[0].boxes is not None and results[0].boxes.id is not None:
                current_tracked_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
                for r_id, cls_id, box_coord in zip(current_tracked_ids, results[0].boxes.cls.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()):
                    track_id = int(r_id); x1, y1, x2, y2 = map(int, box_coord)
                    curr_x, curr_y = (x1+x2)//2, (y1+y2)//2; nome_cls = model.names[int(cls_id)]
                    if track_id not in track_ids_contados:
                        crossed = False; has_prev_pos = track_id in track_previous_x and track_id in track_previous_y
                        if line_type == LINE_HORIZONTAL and track_id in track_previous_y:
                            if (effective_counting_dir==MOVE_TB and track_previous_y[track_id]<line_coord_val and curr_y>=line_coord_val) or \
                               (effective_counting_dir==MOVE_BT and track_previous_y[track_id]>line_coord_val and curr_y<=line_coord_val): crossed = True
                        elif line_type == LINE_VERTICAL and track_id in track_previous_x:
                            if (effective_counting_dir==MOVE_LR and track_previous_x[track_id]<line_coord_val and curr_x>=line_coord_val) or \
                               (effective_counting_dir==MOVE_RL and track_previous_x[track_id]>line_coord_val and curr_x<=line_coord_val): crossed = True
                        elif line_type.startswith("diag") and has_prev_pos:
                            if is_crossing_diagonal_line(track_previous_x[track_id], track_previous_y[track_id], curr_x, curr_y, line_points[0], line_points[1], str(effective_counting_dir)): crossed = True
                        if crossed and (target_classes is None or nome_cls in target_classes):
                            track_ids_contados.add(track_id); current_total_count += 1; current_por_classe[nome_cls] += 1
                    track_previous_x[track_id], track_previous_y[track_id] = curr_x, curr_y
                    if create_annotated_video and annotated_frame is not None:
                        box_color = (0,165,255) if track_id in track_ids_contados else ((200,200,200) if target_classes and nome_cls not in target_classes else (0,255,0))
                        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), box_color, 2)
                        cv2.putText(annotated_frame, f"{nome_cls} ID:{track_id}", (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,box_color,2)
                        cv2.circle(annotated_frame, (curr_x, curr_y), 4, (0,0,255), -1)
                for tid_set in [track_previous_x, track_previous_y]:
                    for tid_to_remove in list(tid_set.keys()):
                        if tid_to_remove not in current_tracked_ids: del tid_set[tid_to_remove]
            
            if create_annotated_video and out is not None and annotated_frame is not None:
                if line_points: cv2.line(annotated_frame, line_points[0], line_points[1], (0,0,255), 3)
                if arrow_points: cv2.arrowedLine(annotated_frame, arrow_points[0], arrow_points[1], (0,255,0), 2, tipLength=0.4)
                info_txt = f"Contagem: {current_total_count}"; cv2.putText(annotated_frame,info_txt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA); cv2.putText(annotated_frame,info_txt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
                out.write(annotated_frame)
        frame_atual += 1
        
    if cap.isOpened(): cap.release()
    if out and out.isOpened(): out.release()

    if progresso_manager and progresso_manager.status(video_name).get("cancelado"): 
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if create_annotated_video and os.path.exists(local_output_path): os.remove(local_output_path)
        return None 

    public_url = "Vídeo processado não foi gerado (opção desabilitada)."
    if create_annotated_video:
        sftp_current_action = "Enviando resultado"
        remote_processed_path = f"public_html/kyoday_videos/processados/{processed_fn}"
        if upload_file_sftp(local_output_path, remote_processed_path, progress_callback=sftp_progress_callback):
            base_url = os.getenv("HG_DOMAIN")
            if not base_url: public_url = "ERRO: HG_DOMAIN não configurado"
            else: public_url = f"{base_url}/kyoday_videos/processados/{processed_fn}"
        else:
            public_url = "ERRO AO FAZER UPLOAD DO VÍDEO PROCESSADO"
            if progresso_manager: progresso_manager.erro(video_name, public_url)
        if os.path.exists(local_output_path): os.remove(local_output_path)

    if os.path.exists(local_video_path): os.remove(local_video_path)
    delete_file_sftp(remote_video_original)

    print(f"[INFO CONTAGEM] Contagem finalizada: {current_total_count} para {video_name}")
    print(f"[INFO CONTAGEM] Status do vídeo processado: {public_url}")
    
    return { "video": video_name, "video_processado": public_url, "total_frames": original_frame_count, "total_count": current_total_count, "por_classe": dict(current_por_classe) }