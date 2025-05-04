from pydantic import BaseModel

class VideoRequest(BaseModel):
    nome_arquivo: str
