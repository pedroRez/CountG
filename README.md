# CountG

![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/github/actions/workflow/status/USER/CountG/ci.yml?label=build)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

![Demonstração](https://media.giphy.com/media/26BRuo6sLetdllPAQ/giphy.gif)

## Português

### Visão Geral
CountG é um backend em **FastAPI** para contagem e rastreamento de objetos em vídeo utilizando modelos **YOLOv8**.

### Pré-requisitos
- Python 3.10+
- [pip](https://pip.pypa.io/)
- (Opcional) [virtualenv](https://virtualenv.pypa.io/)
- PostgreSQL para persistência de dados

### Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/USER/CountG.git
   cd CountG
   ```
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Execução
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Estrutura do Projeto
```text
.
├── main.py
├── routes/
│   └── video_routes.py
├── models/
├── utils/
├── requirements.txt
├── .env.example
```

### Variáveis de Ambiente
Copie `.env.example` para `.env` e ajuste os valores conforme necessário:
```ini
ROBOFLOW_API_KEY=
DATABASE_URL=
HG_HOST=
HG_USER=
HG_PASS=
HG_PORT=22
HG_DOMAIN=
USE_SFTP=false
CREATE_ANNOTATED_VIDEO=true
OMP_NUM_THREADS=12
```

### Links Relevantes
- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [YOLOv8](https://docs.ultralytics.com/)
- [Shields.io](https://shields.io/)

---

## English

### Overview
CountG is a **FastAPI** backend for object counting and tracking in video using **YOLOv8** models.

### Prerequisites
- Python 3.10+
- [pip](https://pip.pypa.io/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/)
- PostgreSQL for data persistence

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/USER/CountG.git
   cd CountG
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Project Structure
```text
.
├── main.py
├── routes/
│   └── video_routes.py
├── models/
├── utils/
├── requirements.txt
├── .env.example
```

### Environment Variables
Copy `.env.example` to `.env` and adjust as needed:
```ini
ROBOFLOW_API_KEY=
DATABASE_URL=
HG_HOST=
HG_USER=
HG_PASS=
HG_PORT=22
HG_DOMAIN=
USE_SFTP=false
CREATE_ANNOTATED_VIDEO=true
OMP_NUM_THREADS=12
```

### Useful Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [YOLOv8](https://docs.ultralytics.com/)
- [Shields.io](https://shields.io/)

