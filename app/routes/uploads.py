from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from app.schemas import UploadStatusResponse
from app.utils.shared_memory import set_upload_status, update_upload_status, get_upload_status as get_upload_status_data, job_exists
from app.config.constants import (
    ALLOWED_VIDEO_EXTENSIONS, CHUNK_SIZE_UPLOAD, MSG_FILE_UPLOADED,
    MAX_SAFE_PROCESSING_SIZE_MB, ERROR_FILE_TOO_LARGE
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload media file (video/audio) with background processing"""
    from app.services.r2_service import get_r2_service
    from app.config.settings import settings
    
    r2_service = get_r2_service()
    job_id = r2_service.generate_job_id()
    original_filename = video_file.filename
    
    # File size validation for memory safety
    try:
        # Check content-length header for file size
        content_length = None
        if hasattr(video_file, 'size') and video_file.size:
            content_length = video_file.size
        elif hasattr(video_file, 'file') and hasattr(video_file.file, 'seek'):
            # Get file size by seeking to end
            current_pos = video_file.file.tell()
            video_file.file.seek(0, 2)  # Seek to end
            content_length = video_file.file.tell()
            video_file.file.seek(current_pos)  # Reset position
        
        # Validate file size before processing
        if content_length:
            size_mb = content_length / (1024 * 1024)
            if size_mb > MAX_SAFE_PROCESSING_SIZE_MB:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": ERROR_FILE_TOO_LARGE,
                        "max_size_mb": MAX_SAFE_PROCESSING_SIZE_MB,
                        "file_size_mb": round(size_mb, 2)
                    }
                )
    except Exception:
        # If size check fails, continue with processing (fallback behavior)
        pass
    
    try:
        # Change: save as tmp/voice_cloning/dub_{job_id}/{original_filename}
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        temp_file_path = os.path.join(job_dir, original_filename)
        set_upload_status(job_id, {
            "status": "uploading",
            "message": "Saving uploaded file...",
            "original_filename": original_filename,
            "started_at": datetime.now(timezone.utc).isoformat()
        })
        total_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await video_file.read(CHUNK_SIZE_UPLOAD):
                buffer.write(chunk)
                total_size += len(chunk)
        file_size = os.path.getsize(temp_file_path)
        
        # Final file size validation after upload
        size_mb = file_size / (1024 * 1024)
        if size_mb > MAX_SAFE_PROCESSING_SIZE_MB:
            # Clean up uploaded file
            try:
                os.remove(temp_file_path)
                os.rmdir(job_dir)
            except:
                pass
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": ERROR_FILE_TOO_LARGE,
                    "max_size_mb": MAX_SAFE_PROCESSING_SIZE_MB,
                    "file_size_mb": round(size_mb, 2)
                }
            )
        
        update_upload_status(job_id, {
            "message": f"File saved ({file_size // (1024*1024)} MB), starting background processing..."
        })
        background_tasks.add_task(process_file_background_only, job_id, temp_file_path, original_filename, file_size)
        return {
            "success": True,
            "message": MSG_FILE_UPLOADED,
            "job_id": job_id,
            "status_check_url": f"/upload-status/{job_id}",
            "original_filename": original_filename,
            "file_size_mb": file_size // (1024*1024),
            "estimated_time": "2-10 minutes"
        }
    except Exception as e:
        # Cleanup on error
        try:
            job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
            temp_file_path = os.path.join(job_dir, original_filename)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if job_exists(job_id):
                update_upload_status(job_id, {
                    "status": "failed",
                    "message": f"File save failed: {str(e)}"
                })
            from app.services.dub.audio_utils import AudioUtils
            AudioUtils.remove_temp_dir(folder_path=job_dir)
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")

# --- Background process for upload ---
async def process_file_background_only(job_id: str, temp_file_path: str, filename: str, file_size: int):
    from app.services.dub.audio_utils import AudioUtils
    job_dir = os.path.dirname(temp_file_path)
    try:
        update_upload_status(job_id, {
            "message": "Validating uploaded file...",
            "status": "uploading"
        })
        if not os.path.exists(temp_file_path):
            raise Exception("Temporary file not found")
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        update_upload_status(job_id, {"message": "Checking file format...", "status": "uploading"})
        
        # Import audio formats from settings
        from app.config.settings import settings
        allowed_video_extensions = ALLOWED_VIDEO_EXTENSIONS
        allowed_audio_extensions = set(settings.ALLOWED_AUDIO_FORMATS)
        allowed_extensions = allowed_video_extensions.union(allowed_audio_extensions)
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            video_formats = ', '.join(sorted(allowed_video_extensions))
            audio_formats = ', '.join(sorted(allowed_audio_extensions))
            update_upload_status(job_id, {
                "status": "failed", 
                "message": f"Unsupported file format. Allowed video: {video_formats}. Allowed audio: {audio_formats}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            AudioUtils.remove_temp_dir(folder_path=job_dir)
            return
        update_upload_status(job_id, {
            "message": "File uploaded successfully. Ready for processing.",
            "status": "ready",
            "file_url": temp_file_path,
            "ready_for_processing": True
        })
    except Exception as e:
        update_upload_status(job_id, {
            "status": "failed",
            "message": f"Processing failed: {str(e)}"
        })
        AudioUtils.remove_temp_dir(folder_path=job_dir)

@router.get("/upload-status/{job_id}", response_model=UploadStatusResponse)
async def get_upload_status(job_id: str):
    """Get upload status without progress tracking"""
    try:
        if not job_exists(job_id):
            raise HTTPException(status_code=404, detail="Upload ID not found")
        data = get_upload_status_data(job_id)
        # status: uploading, ready, failed
        status = data.get("status", "pending")
        message = data.get("message", "")
        original_filename = data.get("original_filename")
        file_url = data.get("file_url")
        ready_for_processing = data.get("ready_for_processing", False)
        
        return {
            "job_id": job_id,
            "status": status,
            "progress": 0,  # Always 0 during upload phase
            "message": message,
            "original_filename": original_filename,
            "file_url": file_url,
            "ready_for_processing": ready_for_processing
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))