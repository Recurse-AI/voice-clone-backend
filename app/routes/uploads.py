from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from app.schemas import UploadStatusResponse
from app.utils.shared_memory import set_upload_status, update_upload_status, get_upload_status as get_upload_status_data, job_exists
from app.config.constants import ALLOWED_VIDEO_EXTENSIONS, CHUNK_SIZE_UPLOAD, MSG_FILE_UPLOADED

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload media file (video/audio) with background processing"""
    from app.utils.r2_storage import R2Storage
    from app.config.settings import settings
    
    r2_storage = R2Storage()
    job_id = r2_storage.generate_job_id()
    original_filename = video_file.filename
    try:
        # Change: save as tmp/voice_cloning/dub_{job_id}/{original_filename}
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        temp_file_path = os.path.join(job_dir, original_filename)
        set_upload_status(job_id, {
            "status": "uploading",
            "progress": 5,
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
        update_upload_status(job_id, {
            "progress": 15,
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
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        temp_file_path = os.path.join(job_dir, original_filename)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if job_exists(job_id):
            update_upload_status(job_id, {
                "status": "failed",
                "progress": 0,
                "message": f"File save failed: {str(e)}"
            })
        from app.services.dub.audio_utils import AudioUtils
        AudioUtils.remove_temp_dir(folder_path=job_dir)
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")

# --- Background process for upload ---
async def process_file_background_only(job_id: str, temp_file_path: str, filename: str, file_size: int):
    from app.services.dub.audio_utils import AudioUtils
    job_dir = os.path.dirname(temp_file_path)
    try:
        update_upload_status(job_id, {
            "progress": 20, 
            "message": "Validating uploaded file...",
            "status": "uploading"
        })
        if not os.path.exists(temp_file_path):
            raise Exception("Temporary file not found")
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        update_upload_status(job_id, {"progress": 30, "message": "Checking file format...", "status": "uploading"})
        
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
                "progress": 0, 
                "message": f"Unsupported file format. Allowed video: {video_formats}. Allowed audio: {audio_formats}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            AudioUtils.remove_temp_dir(folder_path=job_dir)
            return
        update_upload_status(job_id, {
            "progress": 100,
            "message": "Media file saved locally.",
            "status": "done",
            "file_url": temp_file_path
        })
    except Exception as e:
        update_upload_status(job_id, {
            "status": "failed",
            "progress": 0,
            "message": f"Processing failed: {str(e)}"
        })
        AudioUtils.remove_temp_dir(folder_path=job_dir)

@router.get("/upload-status/{job_id}", response_model=UploadStatusResponse)
async def get_upload_status(job_id: str):
    """Get upload progress status"""
    try:
        if not job_exists(job_id):
            raise HTTPException(status_code=404, detail="Upload ID not found")
        data = get_upload_status_data(job_id)
        # status: pending, uploading, done, failed
        status = data.get("status", "pending")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        original_filename = data.get("original_filename")
        file_url = data.get("file_url")
        return {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "original_filename": original_filename,
            "file_url": file_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))