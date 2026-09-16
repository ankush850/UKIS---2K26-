"""
Main FastAPI entrypoint for UKIS-2026 Super Resolution Mapping (SRM) System.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path
from backend.api.routes import router as api_router
from backend.aerial.routes import router as aerial_router
from backend.config import BASE_DIR

app = FastAPI(
    title="UKIS-2026 - Netra & Netra Aerial Disaster Assessment",
    description="Sentinel-2 Super Resolution + Drone-Based Disaster Assessment (DMMC, Uttarakhand)",
    version="2.0.0"
)

# Enable CORS for local/remote development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API endpoints
app.include_router(api_router)
app.include_router(aerial_router)

# Mount frontend static directory
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def serve_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "UKIS-2026 SRM Backend Active. Frontend index.html not found."}

@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon():
    favicon_path = frontend_dir / "logo.png"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/png")
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "UKIS-2026 Super Resolution Mapping",
        "usp": "Hallucination-Aware Uncertainty Mapping (MC-Dropout + opensr-test)",
        "resolution": "10m -> 2.5m (4x upsampling)"
    }
