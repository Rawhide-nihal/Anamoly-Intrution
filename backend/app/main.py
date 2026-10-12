from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router as api_router

app = FastAPI(
    title="CIC-IDS2017 Intrusion Detection Web API",
    description="Backend API for Anomaly-Based Network Intrusion Detection System Using Isolation Forest and DBSCAN",
    version="1.0.0"
)

# Enable CORS for frontend Vite development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {
        "title": "Anomaly-Based Network Intrusion Detection System API",
        "docs_url": "/docs",
        "status": "online"
    }
