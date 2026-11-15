import logging

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.ask import router as ask_router
from app.routes.documents import router as documents_router
from app.settings import STATIC_DIR, UPLOADS_DIR, settings

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


# --------------------------------------------------------------------
# Local storage only: serve uploads at /files/<key>.
# With S3, file_url is a presigned link straight to the private bucket
# --------------------------------------------------------------------
if settings.storage_backend == "local":
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=UPLOADS_DIR), name="files")


# --------------------------------------
# Mount the built React app in the end
# --------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
