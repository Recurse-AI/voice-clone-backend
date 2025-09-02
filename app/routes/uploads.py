from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import logging
from app.config.constants import (
    CHUNK_SIZE_UPLOAD, MSG_FILE_UPLOADED
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), job_id: str = Form(...)):
    """Simple file upload - accepts job_id from frontend"""
    from app.config.settings import settings

    original_filename = video_file.filename

    try:
        # Create job directory and save file
        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        temp_file_path = os.path.join(job_dir, original_filename)

        # Save file directly without complex validation
        with open(temp_file_path, "wb") as buffer:
            while chunk := await video_file.read(CHUNK_SIZE_UPLOAD):
                buffer.write(chunk)

        file_size = os.path.getsize(temp_file_path)
        file_size_mb = file_size // (1024 * 1024)

        return {
            "success": True,
            "message": MSG_FILE_UPLOADED,
            "job_id": job_id,
            "original_filename": original_filename,
            "file_size_mb": file_size_mb,
            "file_path": temp_file_path
        }

    except Exception as e:
        # Simple cleanup on error
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if 'job_dir' in locals() and os.path.exists(job_dir):
                os.rmdir(job_dir)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
