from dotenv import load_dotenv
import os

# --- PASSO CRÍTICO: CARREGAR AS VARIÁVEIS DE AMBIENTE PRIMEIRO! ---
# Esta chamada deve ser uma das primeiras linhas do seu ponto de entrada,
# antes de importar qualquer outro módulo do seu projeto que use essas variáveis.
load_dotenv()

# --- AGORA, IMPORTE O RESTO DA SUA APLICAÇÃO ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import video_routes # Este import agora acontecerá DEPOIS de load_dotenv()

# Cria a instância do FastAPI
app = FastAPI()

# Configuração do CORS (importante para o frontend se comunicar com o backend)
# Permite que seu app React Native (rodando em uma origem diferente)
# se comunique com sua API.
origins = [
    "*" # Para desenvolvimento, '*' é ok. Para produção, seja mais específico.
    # Ex: "http://localhost:8081", "https://seu-pwa.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todos os cabeçalhos
)

# Inclui as rotas do seu arquivo video_routes.py
app.include_router(video_routes.router)

# Rota raiz para teste de saúde (opcional, mas recomendado)
@app.get("/")
def read_root():
    # Verifica se a variável de DB foi carregada, para debug
    db_url_loaded = bool(os.getenv("DATABASE_URL"))
    return {
        "status": "KYO DAY Backend está no ar!",
        "database_url_loaded": db_url_loaded
    }

# Lembre-se que o comando para rodar é:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload