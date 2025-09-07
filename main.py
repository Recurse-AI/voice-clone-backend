from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import torch
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
    logger.info("Starting up — checking MongoDB connection...")
    await verify_connection()

    # Use startup synchronization to prevent duplicate initialization
    from app.utils.startup_sync import startup_sync
    
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
    
    # Simple cleanup - each worker can do its own cleanup
    try:
        cleanup_utils.cleanup_all_expired()
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup expired resources: {cleanup_error}")
    
    # Start status reconciler
    try:
        from app.utils.status_reconciler import _reconciler
        _reconciler.start()
    except Exception as e:
        logger.error(f"Failed to start status reconciler: {e}")

    # Initialize only lightweight OpenAI service in API server
    try:
        from app.services.openai_service import initialize_openai_service
        initialize_openai_service()
        logger.info("✅ OpenAI service ready")
    except Exception as e:
        logger.warning(f"OpenAI initialization failed: {str(e)[:50]}")

    
    
    # R2 service initialization - only one worker should do this
    r2_lock_acquired = await startup_sync.acquire_startup_lock("r2_init", timeout=30)
    
    if r2_lock_acquired:
        try:
            logger.info("Initializing R2 service...")
            from app.services.r2_service import get_r2_service, reset_r2_service
            reset_r2_service()
            get_r2_service()
            await startup_sync.mark_task_complete("r2_init")
            logger.info("R2 service initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize R2 service: {e}")
        finally:
            await startup_sync.release_startup_lock("r2_init")
    else:
        logger.info("R2 service being initialized by another worker...")
        await startup_sync.wait_for_task_completion("r2_init")
    
    yield
    
    logger.info("🔄 API server shutting down...")
    
    
    try:
        from app.utils.status_reconciler import _reconciler
        _reconciler.stop()
        logger.info("✅ Status reconciler stopped")
    except Exception as e:
        logger.error(f"Failed to cleanup status reconciler: {e}")
    
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

app.include_router(auth, prefix="/api/auth", tags=["auth"])
app.include_router(stripe_route, tags=["stripe"])

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