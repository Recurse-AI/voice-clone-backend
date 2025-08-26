from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from contextlib import asynccontextmanager

# App routes and configuration
from app.config.database import verify_connection, create_unique_indexes
from app.utils.logging_config import setup_logging
from app.routes.auth import auth
from app.routes.stripe import stripe_route

from app.routes.audio_processing import router as audio_processing_router
from app.routes.uploads import router as uploads_router
from app.routes.video import router as video_processing_router
from app.routes.user_jobs import router as user_jobs_router
from app.utils.init_pricing_plans import init_pricing_plans
from app.middleware.auth_middleware import AuthMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.config.settings import settings
from app.utils.video_downloader import video_download_service
from app.schemas import StatusResponse

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — checking MongoDB connection...")
    await verify_connection()
    await init_pricing_plans()
    await create_unique_indexes()
    
    try:
        from app.utils.init_db_indexes import init_database_indexes
        await init_database_indexes()
    except Exception as e:
        logger.warning(f"Database indexes init failed: {e}")
    
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    try:
        video_download_service.cleanup_old_files()
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup old directories: {cleanup_error}")
    
    # Initialize Fish Speech service
    try:
        from app.services.dub.fish_speech_service import initialize_fish_speech
        if initialize_fish_speech():
            logger.info("✅ Fish Speech service initialized successfully")
        else:
            logger.warning("⚠️ Fish Speech service initialization failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Fish Speech: {e}")
    
    # Initialize WhisperX transcription service
    try:
        logger.info("🚀 Initializing WhisperX transcription service with auto-download...")
        from app.services.dub.whisperx_transcription import initialize_whisperx_transcription
        if initialize_whisperx_transcription():
            logger.info("✅ WhisperX transcription service initialized successfully")
        else:
            logger.warning("⚠️ WhisperX transcription service initialization failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize WhisperX transcription: {e}")
    
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    
    try:
        from app.services.r2_service import get_r2_service, reset_r2_service
        reset_r2_service()
        get_r2_service()
    except Exception as e:
        logger.error(f"Failed to initialize R2 service: {e}")
    
    yield
    
    logger.info("Shutting down...")
    
    try:
        from app.services.dub.fish_speech_service import cleanup_fish_speech
        cleanup_fish_speech()
    except Exception as e:
        logger.error(f"Failed to cleanup Fish Speech: {e}")
    
    try:
        from app.services.dub.whisperx_transcription import cleanup_whisperx_transcription
        cleanup_whisperx_transcription()
    except Exception as e:
        logger.error(f"Failed to cleanup WhisperX transcription: {e}")
    
    try:
        from app.routes.video.dub_routes import get_dub_executor
        from app.routes.audio_processing import get_separation_executor
        
        dub_executor = get_dub_executor()
        separation_executor = get_separation_executor()
        
        dub_executor.shutdown(wait=True)
        separation_executor.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Failed to cleanup ThreadPoolExecutors: {e}")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    max_age=3600
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

app.include_router(auth, prefix="/api/auth", tags=["auth"])
app.include_router(stripe_route, prefix="/api/stripe", tags=["stripe"])

app.include_router(audio_processing_router, prefix="/api", tags=["audio-processing"])
app.include_router(uploads_router, prefix="", tags=["uploads"])
app.include_router(video_processing_router, prefix="/api", tags=["video-processing"])
app.include_router(user_jobs_router, prefix="/api/jobs", tags=["user-jobs"])

@app.get("/", response_model=StatusResponse)
async def root():
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
        workers=1,
        reload=False
    ) 