from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import logging
import uuid
from app.config.constants import CHUNK_SIZE_UPLOAD

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    from app.config.settings import settings
    from app.services.r2_service import R2Service

    original_filename = file.filename
    upload_id = str(uuid.uuid4())

    try:
        temp_dir = os.path.join(settings.TEMP_DIR, "uploads", upload_id)
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, original_filename)

        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE_UPLOAD):
                buffer.write(chunk)

        file_size = os.path.getsize(temp_file_path)
        file_size_mb = file_size / (1024 * 1024)

        r2_service = R2Service()
        sanitized_filename = r2_service._sanitize_filename(original_filename)
        r2_key = f"uploads/{upload_id}/{sanitized_filename}"
        content_type = r2_service._get_content_type(original_filename)
        
        upload_result = r2_service.upload_file(temp_file_path, r2_key, content_type)
        
        if not upload_result.get("success"):
            raise Exception(upload_result.get("error", "R2 upload failed"))

        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except:
            pass

        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_url": upload_result["url"],
            "r2_key": r2_key,
            "original_filename": original_filename,
            "file_size_mb": round(file_size_mb, 2)
        }

    except Exception as e:
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

