import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import time # Importado para o progresso_manager no exemplo, mas não usado diretamente aqui

def contar_gado_em_video(video_path,
                         video_name,
                         progresso_manager, # Objeto para gerenciar progresso e cancelamento
                         model_path="yolov8l.pt",
                         frame_skip=1,
                         counting_direction="ltr", # "ltr" (esquerda->direita) ou "rtl" (direita->esquerda)
                         target_classes: list = None,
                         line_position_ratio=0.5): # Posição da linha como % da largura (0.0 a 1.0)
    """
    Realiza a contagem de gado em um vídeo usando YOLO com rastreamento e linha de contagem VERTICAL direcional,
    filtrando por classes alvo. Retorna um dicionário no formato legado para compatibilidade com o frontend.
    """

    print(f"[INFO] Iniciando contagem para o vídeo: {video_path}")
    print(f"[INFO] Usando linha VERTICAL. Direção de contagem: {counting_direction.upper()}") # MENSAGEM DE DEBUG
    if target_classes:
        print(f"[INFO] Contando apenas as classes: {target_classes}")
    else:
        print("[INFO] Nenhuma classe alvo específica, contando todas as classes detectadas que cruzarem.")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o modelo YOLO de {model_path}: {e}")
        # Notificar progresso_manager sobre o erro
        if progresso_manager:
            progresso_manager.erro(os.path.basename(video_path), f"Falha ao carregar modelo: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
        if progresso_manager:
            progresso_manager.erro(os.path.basename(video_path), "Não foi possível abrir o vídeo.")
        return None

    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Evitar divisão por zero se o FPS não puder ser lido
        print("[AVISO] FPS do vídeo é 0. Usando 30 como padrão para VideoWriter.")
        fps = 30 
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        print(f"[ERRO] Dimensões do vídeo inválidas (Width: {width}, Height: {height}) para: {video_path}")
        if progresso_manager:
            progresso_manager.erro(os.path.basename(video_path), "Dimensões do vídeo inválidas.")
        cap.release()
        return None

    # Define a posição da linha de contagem VERTICAL
    LINE_X_POSITION = int(width * line_position_ratio)

    current_total_count = 0
    current_por_classe = defaultdict(int)
    track_ids_contados = set()
    
    track_previous_x_positions = {} 

    output_dir = "videos_processados" # O Render pode não persistir esta pasta entre deploys
    os.makedirs(output_dir, exist_ok=True)
    # Usar um nome de arquivo que inclua o video_name original para evitar sobrescrita
    base_vid_name, vid_ext = os.path.splitext(video_name)
    processed_video_filename = f"processed_{base_vid_name}{vid_ext}"
    output_path = os.path.join(output_dir, processed_video_filename)

    try:
        # Use XVID ou mp4v. H264 pode exigir licenças ou codecs não disponíveis em todos os ambientes.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"VideoWriter não pôde ser aberto para {output_path}. Verifique o codec FourCC e permissões.")
    except Exception as e:
        print(f"[ERRO] Falha ao criar o VideoWriter para {output_path}: {e}")
        if progresso_manager:
            progresso_manager.erro(os.path.basename(video_path), f"Falha no VideoWriter: {e}")
        cap.release()
        return None

    print(f"[INFO] Processando vídeo. Total de frames: {original_frame_count}, FPS: {fps}, Resolução: {width}x{height}")
    print(f"[INFO] Linha de contagem VERTICAL em X = {LINE_X_POSITION}") # MENSAGEM DE DEBUG

    frame_atual = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do stream de vídeo ou erro ao ler frame.")
            break

        if frame_atual % frame_skip == 0:
            if progresso_manager and not progresso_manager.atualizar(os.path.basename(video_name), frame_atual, original_frame_count):
                print(f"[INFO] Processamento do vídeo {os.path.basename(video_name)} cancelado pelo gerenciador.")
                break 

            results = model.track(frame, persist=True, verbose=False, conf=0.3) # Adicionado conf para ajuste
            annotated_frame = frame.copy()
            current_frame_tracked_ids = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                tracked_boxes = results[0].boxes
                ids = tracked_boxes.id.cpu().numpy()
                classes_ids = tracked_boxes.cls.cpu().numpy()
                boxes_coords = tracked_boxes.xyxy.cpu().numpy()

                for track_id_float, cls_id, box in zip(ids, classes_ids, boxes_coords):
                    track_id = int(track_id_float)
                    current_frame_tracked_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box)
                    current_x_center = (x1 + x2) // 2
                    y_center_obj = (y1 + y2) // 2
                    nome_classe_detectada = model.names[int(cls_id)]

                    if track_id not in track_ids_contados:
                        if track_id in track_previous_x_positions:
                            previous_x_center = track_previous_x_positions[track_id]
                            crossed_line_in_direction = False

                            if counting_direction.lower() == "ltr":
                                if previous_x_center < LINE_X_POSITION and current_x_center >= LINE_X_POSITION:
                                    crossed_line_in_direction = True
                            elif counting_direction.lower() == "rtl":
                                if previous_x_center > LINE_X_POSITION and current_x_center <= LINE_X_POSITION:
                                    crossed_line_in_direction = True
                            else: # Fallback ou erro se direção não for ltr/rtl
                                if frame_atual == 0 : # printar só uma vez
                                     print(f"[AVISO] Direção de contagem '{counting_direction}' não reconhecida para linha vertical. Use 'ltr' ou 'rtl'.")
                                # Por padrão, pode não contar ou usar uma direção default
                                pass 
                            
                            if crossed_line_in_direction:
                                if target_classes is None or nome_classe_detectada in target_classes:
                                    track_ids_contados.add(track_id)
                                    current_total_count += 1
                                    current_por_classe[nome_classe_detectada] += 1
                                    # print(f"[CONTAGEM ALVO] ID:{track_id} ({nome_classe_detectada}) cruzou {counting_direction.upper()}. Total: {current_total_count}")
                    
                    track_previous_x_positions[track_id] = current_x_center

                    label = f"{nome_classe_detectada} ID:{track_id}"
                    color = (0, 255, 0)
                    if track_id in track_ids_contados: color = (0,165,255)
                    elif target_classes and nome_classe_detectada not in target_classes: color = (200, 200, 200)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(annotated_frame, (current_x_center, y_center_obj), 4, (0, 0, 255), -1) # Círculo em azul

            for tid_in_memory in list(track_previous_x_positions.keys()):
                if tid_in_memory not in current_frame_tracked_ids:
                    del track_previous_x_positions[tid_in_memory]

            cv2.line(annotated_frame, (LINE_X_POSITION, 0), (LINE_X_POSITION, height), (255, 0, 0), 3) # LINHA AZUL VERTICAL
            
            arrow_start_y = height // 2 # Seta no meio da altura
            arrow_length = 40
            if counting_direction.lower() == "ltr":
                cv2.arrowedLine(annotated_frame, (LINE_X_POSITION - arrow_length -10, arrow_start_y), (LINE_X_POSITION + arrow_length -10, arrow_start_y), (0, 255, 0), 2, tipLength=0.4)
            elif counting_direction.lower() == "rtl":
                cv2.arrowedLine(annotated_frame, (LINE_X_POSITION + arrow_length +10, arrow_start_y), (LINE_X_POSITION - arrow_length +10, arrow_start_y), (0, 255, 0), 2, tipLength=0.4)

            info_text = f"Contagem: {current_total_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            # y_offset = 60 # Removido para simplificar, já que current_por_classe agora só tem 'gado'
            # for nome_cls, num in current_por_classe.items():
            #    cls_text = f"{nome_cls}: {num}" # Se target_classes for None, isso ainda pode ser útil
            #    cv2.putText(annotated_frame, cls_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            #    cv2.putText(annotated_frame, cls_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            #    y_offset += 25
            if current_por_classe: # Mostrar detalhes se houver
                first_class_name = list(current_por_classe.keys())[0]
                first_class_count = current_por_classe[first_class_name]
                details_text = f"{first_class_name}: {first_class_count}" if len(current_por_classe) == 1 else "Ver detalhes no JSON"
                cv2.putText(annotated_frame, details_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, details_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)


            out.write(annotated_frame)

        frame_atual += 1
        if progresso_manager and frame_atual % frame_skip != 0 :
            if not progresso_manager.atualizar(os.path.basename(video_name), frame_atual, original_frame_count, no_processing=True):
                print(f"[INFO] Processamento (frame pulado) {os.path.basename(video_name)} cancelado.")
                break 
    
    if cap.isOpened(): cap.release() # Garante que o cap seja liberado
    if out.isOpened(): out.release() # Garante que o out seja liberado

    if progresso_manager and progresso_manager.cancelados.get(os.path.basename(video_name)):
         print(f"[INFO] Processamento de {os.path.basename(video_name)} foi cancelado. Nenhum resultado retornado.")
         # O progresso_manager já deve ter sido atualizado com erro ou status de cancelado
         return None # Ou um dict indicando cancelamento, se preferir

    print(f"[INFO] Contagem finalizada para {os.path.basename(video_name)}: {current_total_count} objetos alvo contados.")
    print(f"[INFO] Vídeo processado salvo em: {output_path}") # Este caminho é local ao servidor

    return {
        "video": os.path.basename(video_name),
        "video_processado": processed_video_filename, # Apenas o nome, o frontend não acessa o path direto
        "total_frames": original_frame_count,
        "total_count": current_total_count,
        "por_classe": dict(current_por_classe),
    }