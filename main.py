import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import video_routes

# --- CRITICAL STEP: LOAD ENVIRONMENT VARIABLES FIRST! ---
# This call must be one of the first lines of your entry point before importing
# any other module that uses these variables.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastAPI application instance
app = FastAPI()

# CORS configuration (allows frontend to communicate with backend)
# Allows a React Native app (running on a different origin) to talk to the API.
origins = [
    "*",  # For development '*' is fine. For production be more specific.
    # E.g.: "http://localhost:8081", "https://your-pwa.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include routes from video_routes.py
app.include_router(video_routes.router)


# Root endpoint for health check
@app.get("/")
def read_root():
    db_url_loaded = bool(os.getenv("DATABASE_URL"))
    logger.debug("Root endpoint accessed; DATABASE_URL loaded: %s", db_url_loaded)
    return {
        "status": "KYO DAY Backend is running!",
        "database_url_loaded": db_url_loaded,
    }


# Remember to run:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
