# Referência da API

Todos os endpoints retornam JSON.

Você pode baixar o esquema OpenAPI completo em formato JSON para usar em
ferramentas como Swagger ou Postman. [Baixe o esquema](../openapi.json) ou
utilize o link no seu explorador de APIs preferido.

## `GET /`
**Resposta**
```json
{
  "status": "KYO DAY Backend is running!",
  "database_url_loaded": true
}
```
```bash
curl http://localhost:8000/
```

## `POST /upload-video/`
Envia um arquivo de vídeo.

**Requisição** (multipart)
- `file`: arquivo de vídeo

**Resposta**
```json
{
  "message": "Arquivo 'video.mp4' recebido com sucesso.",
  "nome_arquivo": "<nome-gerado>.mp4"
}
```
```bash
curl -X POST -F "file=@meu_video.mp4" http://localhost:8000/upload-video/
```

## `POST /predict-video/`
Inicia o processamento de um vídeo previamente enviado.

**Requisição**
```json
{
  "nome_arquivo": "<nome-gerado>.mp4",
  "orientation": "S",
  "model_choice": "l",
  "target_classes": ["cow"],
  "line_position_ratio": 0.5
}
```

**Resposta**
```json
{
  "status": "iniciado",
  "message": "Processamento para '<nome-gerado>.mp4' iniciado.",
  "video_name": "<nome-gerado>.mp4"
}
```
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"nome_arquivo":"<nome-gerado>.mp4","orientation":"S"}' \
  http://localhost:8000/predict-video/
```

## `GET /progresso/{video_name}`
Consulta o progresso do processamento.

**Resposta**
```json
{
  "status": "em_processamento",
  "progresso": 42
}
```
```bash
curl http://localhost:8000/progresso/<nome-gerado>.mp4
```

## `GET /cancelar-processamento/{video_name}`
Cancela o processamento de um vídeo.

**Resposta**
```json
{
  "message": "Solicitação de cancelamento para <nome-gerado>.mp4 enviada."
}
```
```bash
curl http://localhost:8000/cancelar-processamento/<nome-gerado>.mp4
```
