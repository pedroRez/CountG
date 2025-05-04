import time

class ProgressoManager:
    def __init__(self):
        self.progresso = {}
        self.cancelados = {}

    def iniciar(self, video_name):
        self.progresso[video_name] = {
            "frame_atual": 0,
            "total_frames_estimado": 1,
            "tempo_inicio": time.time(),
            "tempo_restante": "Calculando...",
            "finalizado": False,
            "resultado": None,
            "erro": None,
        }
        self.cancelados[video_name] = False

    def atualizar(self, video_name, frame_atual, total_estimado):
        if self.cancelados.get(video_name):
            return False

        self.progresso[video_name]["frame_atual"] = frame_atual
        self.progresso[video_name]["total_frames_estimado"] = total_estimado

        elapsed = time.time() - self.progresso[video_name]["tempo_inicio"]
        if frame_atual > 0:
            fps = frame_atual / elapsed
            restante = (total_estimado - frame_atual) / fps if fps > 0 else 0
            self.progresso[video_name]["tempo_restante"] = time.strftime("%H:%M:%S", time.gmtime(restante))
        return True

    def finalizar(self, video_name, resultado):
        self.progresso[video_name].update({
            "finalizado": True,
            "resultado": resultado
        })
        self.cancelados[video_name] = True  # <- nova linha

    def erro(self, video_name, mensagem):
        self.progresso[video_name]["erro"] = mensagem
        self.progresso[video_name]["finalizado"] = True
        self.cancelados[video_name] = True  # <- nova linha

    def status(self, video_name):
        return self.progresso.get(video_name, {"erro": "Vídeo não encontrado."})

    def cancelar(self, video_name):
        self.cancelados[video_name] = True
