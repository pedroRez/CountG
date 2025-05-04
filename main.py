from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import video_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_routes.router)
