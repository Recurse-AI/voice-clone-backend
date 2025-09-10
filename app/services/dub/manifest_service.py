import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING

from app.services.r2_service import R2Service
from app.config.settings import settings

logger = logging.getLogger(__name__)


def ensure_job_dir(job_id: str) -> str:
    """Ensure job directory exists using job_id directly."""
    job_dir = os.path.join(settings.TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def write_json(content: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_url: str) -> Dict[str, Any]:
    import requests
    resp = requests.get(manifest_url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def build_manifest(job_id: str, transcript_id: Optional[str], target_language: str, dubbed_segments: list,
                  vocal_audio_url: Optional[str] = None, instrument_audio_url: Optional[str] = None) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "transcript_id": transcript_id,
        "target_language": target_language,
        "version": 1,
        "vocal_audio_url": vocal_audio_url,
        "instrument_audio_url": instrument_audio_url,
        "segments": [
            {
                "id": seg["id"],
                "segment_index": seg["segment_index"],
                "start": seg["start"],
                "end": seg["end"],
                "duration_ms": seg["duration_ms"],
                "original_text": seg["original_text"],
                "dubbed_text": seg["dubbed_text"],
                "original_audio_file": seg.get("original_audio_file"),
            } for seg in dubbed_segments
        ]
    }


def save_manifest_to_dir(manifest: Dict[str, Any], process_temp_dir: str, job_id: str) -> str:
    manifest_filename = f"manifest_{job_id}.json"
    manifest_path = os.path.join(process_temp_dir, manifest_filename).replace('\\', '/')
    write_json(manifest, manifest_path)
    return manifest_path


def upload_process_dir_to_r2(job_id: str, process_temp_dir: str, r2_service: Optional["R2Service"] = None,
                           exclude_files: List[str] = None) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """
    Upload all files from directory to R2, with option to exclude certain files
    exclude_files: List of filenames to skip uploading (e.g., ['vocal_abc123.wav', 'instrument_abc123.wav'])
    """
    r2 = r2_service or R2Service()
    folder_upload_result = r2.upload_directory(job_id, process_temp_dir, exclude_files=exclude_files or [])
    manifest_url = None
    manifest_key = None
    files_map = {}
    if isinstance(folder_upload_result, dict):
        for fname, res in folder_upload_result.items():
            if res.get("success"):
                files_map[fname] = {"url": res.get("url"), "r2_key": res.get("r2_key")}
        for fname in (f"manifest_{job_id}.json",):
            if fname in files_map:
                manifest_url = files_map[fname]["url"]
                manifest_key = files_map[fname]["r2_key"]
                break
    return folder_upload_result, manifest_url, manifest_key
