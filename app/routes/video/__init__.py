"""
Video processing routes module
"""

from fastapi import APIRouter
from .dub_routes import router as dub_router
from .segment_routes import router as segment_router
from .processing_routes import router as processing_router
from .file_routes import router as file_router

# Create main video processing router
router = APIRouter()

# Include all sub-routers
router.include_router(dub_router)
router.include_router(segment_router)
router.include_router(processing_router)
router.include_router(file_router)

# Export the main router
__all__ = ["router"]
