from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

# App routes and configuration
from app.config.database import verify_connection
from app.utils.logger import logger as app_logger
from app.utils.logging_config import setup_logging
from app.routes.auth import auth
from app.routes.stripe import stripe_route
from app.routes.video_export import router as video_export_router
from app.routes.audio_processing import router as audio_processing_router
from app.routes.uploads import router as uploads_router
from app.routes.video_processing import router as video_processing_router
from app.routes.user_jobs import router as user_jobs_router
from app.utils.init_pricing_plans import init_pricing_plans
from app.middleware.auth_middleware import AuthMiddleware

from app.config.settings import settings
from app.utils.r2_storage import R2Storage
# Status management is now handled by individual services (dub_job_service, separation_job_service)
from app.utils.video_downloader import video_download_service
from app.schemas import StatusResponse, ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs, AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, VideoDownloadRequest, VideoDownloadResponse, UploadStatusResponse, VideoDubRequest, VideoDubResponse, VideoDubStatusResponse, VoiceCloneRequest, VoiceCloneResponse

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    logger.info("Starting API initialization...")
    
    # Database connection and pricing plans initialization
    app_logger.info("Starting up — checking MongoDB connection...")
    await verify_connection()
    await init_pricing_plans()
    
    # Create temp directory
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Initialize Fish Speech service
    try:
        from app.services.dub.fish_speech_service import initialize_fish_speech
        if initialize_fish_speech():
            logger.info("✅ Fish Speech service initialized successfully")
        else:
            logger.warning("⚠️ Fish Speech service initialization failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Fish Speech: {e}")
    
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    
    # Start periodic cleanup task (runs every hour)
    async def _cleanup_loop():
        while True:
            video_download_service.cleanup_old_files()
            await asyncio.sleep(3600)
    asyncio.create_task(_cleanup_loop())
    
    yield
    
    # Cleanup on shutdown
    app_logger.info("Shutting down...")
    logger.info("🔄 Shutting down API...")
    # Cleanup Fish Speech service
    try:
        from app.services.dub.fish_speech_service import cleanup_fish_speech
        cleanup_fish_speech()
        logger.info("✅ Fish Speech service cleaned up")
    except Exception as e:
        logger.error(f"❌ Failed to cleanup Fish Speech: {e}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Authentication middleware
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(auth, prefix="/api/auth", tags=["auth"])
app.include_router(stripe_route, prefix="/api/stripe", tags=["stripe"])
app.include_router(video_export_router, prefix="/api", tags=["video-export"])
app.include_router(audio_processing_router, prefix="/api", tags=["audio-processing"])
app.include_router(uploads_router, prefix="", tags=["uploads"])
app.include_router(video_processing_router, prefix="/api", tags=["video-processing"])
app.include_router(user_jobs_router, prefix="/api/jobs", tags=["user-jobs"])

# Global instances
r2_storage = R2Storage()

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status"""
    return StatusResponse(
        status="active",
        message=f"ClearVocals API is running - Voice Cloning API {settings.API_VERSION}"
    )


if __name__ == "__main__":
    import uvicorn
    
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 