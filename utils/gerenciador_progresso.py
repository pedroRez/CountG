import os
import psycopg2
import psycopg2.pool
import time
import json # Para lidar com a coluna JSONB do resultado
from typing import Optional, Dict, Any

# Pega a URL do banco de dados das variáveis de ambiente carregadas pelo load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Pool de Conexões com o Banco de Dados ---
pool = None
try:
    if not DATABASE_URL:
        raise ValueError("A variável de ambiente DATABASE_URL não foi definida.")
    pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn=DATABASE_URL)
    print("[DB] Pool de conexões com PostgreSQL criado com sucesso.")
except Exception as e:
    print(f"[DB ERRO] Falha CRÍTICA ao criar o pool de conexões com PostgreSQL: {e}")
    print("Verifique se o PostgreSQL está rodando e se a DATABASE_URL no seu arquivo .env está correta.")

def create_progress_table_if_not_exists():
    """Garante que a tabela de progresso exista no banco de dados."""
    if not pool: 
        print("[DB AVISO] Pool de conexões não disponível. Tabela não pôde ser verificada/criada.")
        return
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_progress (
                    video_name VARCHAR(255) PRIMARY KEY,
                    frame_atual INTEGER DEFAULT 0,
                    total_frames_estimado INTEGER DEFAULT 1,
                    tempo_inicio DOUBLE PRECISION,
                    tempo_restante VARCHAR(50),
                    finalizado BOOLEAN DEFAULT FALSE,
                    resultado JSONB,
                    erro TEXT,
                    cancelado BOOLEAN DEFAULT FALSE,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            conn.commit()
            print("[DB] Tabela 'video_progress' verificada/criada com sucesso.")
    except Exception as e:
        print(f"[DB ERRO] Falha ao criar/verificar a tabela 'video_progress': {e}")
    finally:
        if conn:
            pool.putconn(conn)

# Chama a função para criar a tabela na inicialização do módulo, uma única vez.
create_progress_table_if_not_exists()

class ProgressoManager:
    """Gerencia o progresso do processamento de vídeo usando um banco de dados PostgreSQL."""

    def _execute_query(self, query: str, params: tuple = (), fetch: Optional[str] = None):
        """Função auxiliar para executar queries no banco de dados usando o pool."""
        if not pool: 
            print("[DB ERRO] Tentativa de executar query sem um pool de conexões válido.")
            return None
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                if fetch == 'one':
                    return cur.fetchone()
                if fetch == 'all':
                    return cur.fetchall()
                conn.commit()
        except Exception as e:
            print(f"[DB ERRO] Falha na query '{query[:60].strip()}...': {e}")
            if conn:
                try: conn.rollback()
                except psycopg2.InterfaceError: conn = None # Conexão provavelmente já fechada/inválida
            return None
        finally:
            if conn:
                pool.putconn(conn)

    def iniciar(self, video_name: str):
        """Inicia ou reseta o progresso para um vídeo no banco de dados."""
        query = """
            INSERT INTO video_progress (video_name, tempo_inicio, tempo_restante, finalizado, cancelado, erro, resultado, frame_atual, total_frames_estimado, last_updated)
            VALUES (%s, %s, %s, %s, %s, NULL, NULL, 0, 1, NOW())
            ON CONFLICT (video_name) DO UPDATE SET
                tempo_inicio = EXCLUDED.tempo_inicio, tempo_restante = EXCLUDED.tempo_restante,
                finalizado = EXCLUDED.finalizado, cancelado = EXCLUDED.cancelado,
                erro = NULL, resultado = NULL, frame_atual = 0, total_frames_estimado = 1,
                last_updated = NOW();
        """
        params = (video_name, time.time(), "Na fila...", False, False)
        self._execute_query(query, params)
        print(f"[DB Progresso] Progresso iniciado/resetado para: {video_name}")

    def is_processing(self, video_name: str) -> bool:
        """Verifica no banco se um vídeo está atualmente em processamento."""
        status = self.status(video_name)
        return bool(status and not status.get("erro") and not status.get("finalizado"))

    def atualizar(self, video_name: str, frame_atual: int, total_estimado: int, no_processing: bool = False) -> bool:
        """Atualiza o progresso do processamento de frames no banco de dados."""
        status = self.status(video_name)
        if not status or status.get("cancelado") or status.get("finalizado"): return False 
        if not no_processing:
            tempo_restante = "Calculando..."
            elapsed = time.time() - status.get("tempo_inicio", time.time())
            if frame_atual > 5 and elapsed > 0.1:
                fps_calc = frame_atual / elapsed
                if fps_calc > 0 and total_estimado > frame_atual:
                    restante_segundos = (total_estimado - frame_atual) / fps_calc
                    tempo_restante = time.strftime("%H:%M:%S", time.gmtime(restante_segundos))
                else: tempo_restante = "Finalizando..."
            query = """
                UPDATE video_progress SET frame_atual = %s, total_frames_estimado = %s, tempo_restante = %s, last_updated = NOW()
                WHERE video_name = %s;
            """
            params = (frame_atual, total_estimado, tempo_restante, video_name)
            self._execute_query(query, params)
        return True

    # --- MÉTODO NOVO QUE ESTAVA FALTANDO ---
    def update_status_message(self, video_name: str, message: str):
        """Atualiza a mensagem de status (usando o campo tempo_restante) para tarefas como SFTP."""
        if self.status(video_name).get("finalizado"): return 
        query = "UPDATE video_progress SET tempo_restante = %s, last_updated = NOW() WHERE video_name = %s;"
        params = (message, video_name)
        self._execute_query(query, params)
        # Removido o print daqui para não poluir o log a cada % de progresso do SFTP

    def finalizar(self, video_name: str, resultado: dict):
        """Marca o processamento como finalizado com sucesso no banco de dados."""
        status = self.status(video_name)
        frame_final = status.get("total_frames_estimado", status.get("frame_atual", 0)) if status else 0
        query = """
            UPDATE video_progress SET finalizado = TRUE, resultado = %s, erro = NULL, tempo_restante = '00:00:00', frame_atual = %s, last_updated = NOW()
            WHERE video_name = %s;
        """
        resultado_json = json.dumps(resultado)
        params = (resultado_json, frame_final, video_name)
        self._execute_query(query, params)
        print(f"[DB Progresso] Finalizado com sucesso para: {video_name}")

    def erro(self, video_name: str, mensagem: str):
        """Marca o processamento como finalizado com erro no banco de dados."""
        query = "UPDATE video_progress SET finalizado = TRUE, erro = %s, tempo_restante = 'Erro', last_updated = NOW() WHERE video_name = %s;"
        if not self.status(video_name).get("erro"): self.iniciar(video_name) # Garante que a linha exista antes de atualizar
        params = (mensagem, video_name)
        self._execute_query(query, params)
        print(f"[DB Progresso] Erro registrado para: {video_name}")

    def status(self, video_name: str) -> Dict[str, Any]:
        """Retorna o status atual de um vídeo do banco de dados."""
        query = "SELECT video_name, frame_atual, total_frames_estimado, tempo_inicio, tempo_restante, finalizado, resultado, erro, cancelado FROM video_progress WHERE video_name = %s;"
        result = self._execute_query(query, (video_name,), fetch='one')
        if result:
            keys = ["video_name", "frame_atual", "total_frames_estimado", "tempo_inicio", "tempo_restante", "finalizado", "resultado", "erro", "cancelado"]
            return dict(zip(keys, result))
        return {"erro": f"Processamento para '{video_name}' não encontrado.", "finalizado": True, "video_name": video_name}

    def cancelar(self, video_name: str) -> bool:
        """Sinaliza no banco de dados que o processamento deve ser cancelado."""
        status = self.status(video_name)
        if status and not status.get("finalizado"):
            query = "UPDATE video_progress SET cancelado = TRUE, finalizado = TRUE, erro = 'Cancelado pelo usuário.', tempo_restante = 'Cancelado', last_updated = NOW() WHERE video_name = %s;"
            self._execute_query(query, (video_name,))
            print(f"[DB Progresso] Cancelamento registrado para: {video_name}")
            return True
        return False