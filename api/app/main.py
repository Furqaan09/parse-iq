from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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