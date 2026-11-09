import logging

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.ask import router as ask_router
from app.routes.documents import router as documents_router
from app.settings import STATIC_DIR, STORAGE_DIR

# ------------------------
# Logging configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------
# FastAPI configuration
# -----------------------
app = FastAPI(title="ParseIQ API", version="0.1.0")


# ------------------------------------
# All JSON endpoints live under /api
# ------------------------------------
api = APIRouter(prefix="/api")


# Sample endpoint to verify the API is working
@api.get("/status")
def get_status():
    return {"status": "API is running"}


# ----------------------
# Include all routers
# ----------------------
api.include_router(documents_router)
api.include_router(ask_router)
app.include_router(api)


# -----------------------------------------------
# Upload files at /files/<YYYY>/<MM>/<filename>
# -----------------------------------------------
UPLOADS_DIR = STORAGE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOADS_DIR), name="files")


# --------------------------------------
# Mount the built React app in the end
# --------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
