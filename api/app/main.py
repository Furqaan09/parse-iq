from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from app.routes.documents import router as documents_router

# Instance of FastAPI
app = FastAPI(title="ParseIQ API", version="0.1.0")

# CORS configuration: Allow browser frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------
# Sample endpoint to verify the API is working
# ----------------------------------------------
@app.get("/status")
def get_status():
    return {"status": "API is running"}


# ------------------------------
# Include the documents router
# ------------------------------
app.include_router(documents_router)


# ---------------------------------
# Static mount for uploaded files
# ---------------------------------
UPLOADS_DIR = Path(__file__).resolve().parents[1] / "storage" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Files will be accessible at /files/<YYYY>/<MM>/<filename>
app.mount("/files", StaticFiles(directory=UPLOADS_DIR), name="files")