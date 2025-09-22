from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from app.core.database import get_session
from app.models import Document

# Instance of FastAPI
app = FastAPI(title="ParseIQ API", version="0.1.0")

# CORS configuration: Allow browser frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["hhtp://localhost:5173"], # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample endpoint to verify the API is working
@app.get("/status")
def get_status():
    return {"status": "API is running"}


@app.get("/debug/docs-count")
def docs_count(session: Session = Depends(get_session)):
    count = session.exec(select(Document)).all()
    return {"documents": len(count)}