# API Reference

All endpoints return JSON.

You can download the full OpenAPI schema as JSON for use with tools like Swagger
or Postman. [Download the schema](../openapi.json) or paste the link into your
favorite API explorer.

## `GET /`
**Response**
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
Upload a video file.

**Request** (multipart)
- `file`: video file

**Response**
```json
{
  "message": "Arquivo 'video.mp4' recebido com sucesso.",
  "nome_arquivo": "<generated-name>.mp4"
}
```
```bash
curl -X POST -F "file=@my_video.mp4" http://localhost:8000/upload-video/
```

## `POST /predict-video/`
Start processing a previously uploaded video.

**Request**
```json
{
  "nome_arquivo": "<generated-name>.mp4",
  "orientation": "S",
  "model_choice": "l",
  "target_classes": ["cow"],
  "line_position_ratio": 0.5
}
```

**Response**
```json
{
  "status": "iniciado",
  "message": "Processamento para '<generated-name>.mp4' iniciado.",
  "video_name": "<generated-name>.mp4"
}
```
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"nome_arquivo":"<generated-name>.mp4","orientation":"S"}' \
  http://localhost:8000/predict-video/
```

## `GET /progresso/{video_name}`
Check processing progress.

**Response**
```json
{
  "status": "em_processamento",
  "progresso": 42
}
```
```bash
curl http://localhost:8000/progresso/<generated-name>.mp4
```

## `GET /cancelar-processamento/{video_name}`
Cancel processing of a video.

**Response**
```json
{
  "message": "Solicitação de cancelamento para <generated-name>.mp4 enviada."
}
```
```bash
curl http://localhost:8000/cancelar-processamento/<generated-name>.mp4
```
