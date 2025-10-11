from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
import torch
from contextlib import asynccontextmanager

# App routes and configuration
from app.config.database import verify_connection, create_unique_indexes
from app.utils.logging_config import setup_logging
from app.utils.startup_sync import startup_sync
from app.routes.auth import auth
from app.routes.stripe import stripe_route

from app.routes.audio_processing import router as audio_processing_router
from app.routes.uploads import router as uploads_router
from app.routes.resumable_uploads import router as resumable_uploads_router
from app.routes.video import router as video_processing_router
from app.routes.user_jobs import router as user_jobs_router
from app.routes.youtube_transcript import router as youtube_transcript_router
from app.routes.fish_audio import router as fish_audio_router
from app.routes.clip_generation import router as clip_generation_router

from app.middleware.auth_middleware import AuthMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.config.settings import settings
from app.schemas import StatusResponse
from app.utils.cleanup_utils import cleanup_utils
from fastapi.responses import JSONResponse

setup_logging()
logger = logging.getLogger(__name__)

# Environment variables are set in runpod_setup.sh

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏥 Performing startup health checks...")
    
    # Start cookie auto-refresh scheduler
    try:
        from app.services.cookie_scheduler import cookie_scheduler
        import asyncio
        asyncio.create_task(cookie_scheduler.start())
        logger.info("🔄 Cookie auto-refresh scheduler started")
    except Exception as e:
        logger.warning(f"Cookie scheduler not started: {e}")

    mongodb_lock_acquired = await startup_sync.acquire_startup_lock("mongodb_check", timeout=30)

    if mongodb_lock_acquired:
        try:
            logger.info("🔍 Checking MongoDB connection...")
            await verify_connection()
            await startup_sync.mark_task_complete("mongodb_check")
            logger.info("✅ MongoDB health check completed")
        except Exception as e:
            logger.error(f"❌ MongoDB health check failed: {e}")
            raise  # Critical failure - stop startup
        finally:
            await startup_sync.release_startup_lock("mongodb_check")
    else:
        logger.info("⏳ Waiting for MongoDB health check...")
        await startup_sync.wait_for_task_completion("mongodb_check")
    
    # Database initialization - only one worker should do this
    db_lock_acquired = await startup_sync.acquire_startup_lock("database_init", timeout=60)
    
    if db_lock_acquired:
        try:
            logger.info("Initializing database...")
            await create_unique_indexes()
            
            from app.utils.init_db_indexes import init_database_indexes
            await init_database_indexes()
            
            await startup_sync.mark_task_complete("database_init")
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
        finally:
            await startup_sync.release_startup_lock("database_init")
    else:
        logger.info("Waiting for database initialization...")
        await startup_sync.wait_for_task_completion("database_init")
    
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Per-worker cleanup - lightweight, can run on each worker
    try:
        cleanup_utils.cleanup_all_expired()
        logger.info("🧹 Per-worker cleanup completed")
    except Exception as cleanup_error:
        logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")

    # Status reconciler - coordinate startup
    reconciler_lock_acquired = await startup_sync.acquire_startup_lock("status_reconciler_init", timeout=30)

    if reconciler_lock_acquired:
        try:
            from app.utils.status_reconciler import _reconciler
            _reconciler.start()
            await startup_sync.mark_task_complete("status_reconciler_init")
            logger.info("✅ Status reconciler started")
        except Exception as e:
            logger.warning(f"⚠️ Status reconciler warning: {e}")
        finally:
            await startup_sync.release_startup_lock("status_reconciler_init")
    else:
        logger.info("⏳ Waiting for status reconciler...")
        await startup_sync.wait_for_task_completion("status_reconciler_init")

    # Per-worker services - R2 instances created on-demand
    logger.info("✅ R2 service ready (on-demand instances)")

    logger.info("🎯 API server startup completed successfully")
    yield

    logger.info("🔄 API server shutting down...")

    # Cleanup - coordinate status reconciler stop
    cleanup_lock_acquired = await startup_sync.acquire_startup_lock("cleanup", timeout=30)

    if cleanup_lock_acquired:
        try:
            from app.utils.status_reconciler import _reconciler
            _reconciler.stop()
            await startup_sync.mark_task_complete("cleanup")
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
        finally:
            await startup_sync.release_startup_lock("cleanup")
    else:
        logger.info("⏳ Waiting for cleanup...")
        await startup_sync.wait_for_task_completion("cleanup")

    logger.info("✅ API server shutdown complete")

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

allowed_origins = [
    settings.FRONTEND_URL,
    f"https://www.{settings.PUBLIC_HOST}",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5173",
    "https://clearvocals.ai"
]

if settings.DEBUG or settings.NODE_ENV == "development":
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if not settings.DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Upload-Offset"],
)

app.add_middleware(AuthMiddleware)

app.include_router(auth, prefix="/api/auth", tags=["auth"])
app.include_router(stripe_route, tags=["stripe"])

app.include_router(audio_processing_router, prefix="/api", tags=["audio-processing"])
app.include_router(uploads_router, prefix="", tags=["uploads"])
app.include_router(resumable_uploads_router, tags=["resumable-uploads"])
app.include_router(video_processing_router, prefix="/api", tags=["video-processing"])
app.include_router(user_jobs_router, prefix="/api/jobs", tags=["user-jobs"])
app.include_router(youtube_transcript_router, prefix="/api", tags=["youtube-transcript"])
app.include_router(fish_audio_router)
app.include_router(clip_generation_router, prefix="/api/clips", tags=["clip-generation"])

# Mount static files for serving assets like logos
app.mount("/static", StaticFiles(directory="assets"), name="static")


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="active",
        message=f"ClearVocals API is running - Voice Cloning API {settings.API_VERSION}"
    )

@app.get("/health/live")
async def health_live():
    # Process is up and FastAPI is serving
    return {"status": "ok"}

@app.get("/health/ready")
async def health_ready():
    """Enhanced health check with queue monitoring and system stats"""
    from datetime import datetime
    import redis
    import os
    
    result = {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ClearVocals API"
    }
    
    # Check MongoDB
    try:
        await verify_connection()
        result["mongodb"] = "ok"
    except Exception as e:
        error_msg = str(e)[:100]
        result["mongodb"] = f"fail: {error_msg}"
        result["status"] = "degraded"
        logger.error(f"MongoDB health check failed: {error_msg}")
        return JSONResponse(status_code=503, content=result)
    
    # Check Redis/RQ and get queue stats
    redis_status = "unknown"
    try:
        from app.queue.queue_manager import queue_manager
        
        # First check queue manager health
        is_ok = queue_manager.check_health()
        if not is_ok:
            result["redis"] = "fail: queue_manager health check failed"
            result["status"] = "degraded"
            return JSONResponse(status_code=503, content=result)
        
        # Get Redis URL from environment or default
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
        
        # Try to connect to Redis directly
        try:
            r = redis.Redis.from_url(redis_url)
            r.ping()  # Test connection
            redis_status = "ok"
        except redis.ConnectionError:
            # Fallback to localhost if Redis URL fails
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            redis_status = "ok"
        
        # Get queue statistics
        dub_queue = r.llen('rq:queue:dub_queue')
        billing_queue = r.llen('rq:queue:billing_queue')
        separation_queue = r.llen('rq:queue:separation_queue')
        workers = len(r.smembers('rq:workers'))
        failed_jobs = r.llen('rq:queue:failed')
        
        result["redis"] = redis_status
        result["queues"] = {
            "dub_queue": {
                "length": dub_queue, 
                "status": "healthy" if dub_queue < 50 else "overloaded"
            },
            "billing_queue": {
                "length": billing_queue, 
                "status": "healthy" if billing_queue < 20 else "overloaded"
            },
            "separation_queue": {
                "length": separation_queue,
                "status": "healthy" if separation_queue < 30 else "overloaded"
            },
            "failed_queue": {"length": failed_jobs}
        }
        result["workers"] = {"active_count": workers}
        result["metrics"] = {
            "total_queue_load": dub_queue + billing_queue + separation_queue,
            "redis_url": redis_url.split('@')[-1] if '@' in redis_url else redis_url  # Hide auth info
        }
        
    except redis.ConnectionError as e:
        error_msg = f"Redis connection failed: {str(e)[:100]}"
        result["redis"] = f"fail: {error_msg}"
        result["status"] = "degraded"
        logger.error(error_msg)
        return JSONResponse(status_code=503, content=result)
    except Exception as e:
        error_msg = f"Redis health check error: {str(e)[:100]}"
        result["redis"] = f"fail: {error_msg}"
        result["status"] = "degraded"
        logger.error(error_msg)
        return JSONResponse(status_code=503, content=result)
    
    # Add system resources if available
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        result["system"] = {
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(memory.percent, 1),
            "memory_available_gb": round(memory.available / (1024**3), 1),
            "status": "healthy" if cpu < 80 and memory.percent < 85 else "stressed"
        }
    except ImportError:
        result["system"] = {"status": "psutil_not_available"}
    except Exception as e:
        result["system"] = {"status": f"error: {str(e)[:50]}"}
    
    # Final status determination
    if result["status"] == "ok":
        all_healthy = (
            result["mongodb"] == "ok" and
            result["redis"] == "ok" and
            result.get("queues", {}).get("failed_queue", {}).get("length", 0) < 10
        )
        if not all_healthy:
            result["status"] = "degraded"
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        reload=False
    ) 