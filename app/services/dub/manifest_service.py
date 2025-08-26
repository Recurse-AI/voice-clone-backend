import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING

from app.services.r2_service import get_r2_service
from app.config.settings import settings

if TYPE_CHECKING:
    from app.services.r2_service import R2Service

logger = logging.getLogger(__name__)


def ensure_job_dir(job_id: str) -> str:
    """Ensure job directory exists using consistent dub_{job_id} pattern."""
    job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
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


def build_manifest(job_id: str, transcript_id: Optional[str], target_language: str, dubbed_segments: list) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "transcript_id": transcript_id,
        "target_language": target_language,
        "version": 1,
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
    manifest_filename = f"dubbing_manifest_{job_id}.json"
    manifest_path = os.path.join(process_temp_dir, manifest_filename).replace('\\', '/')
    write_json(manifest, manifest_path)
    return manifest_path


def upload_process_dir_to_r2(job_id: str, process_temp_dir: str, r2_service: Optional["R2Service"] = None) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    r2 = r2_service or get_r2_service()
    folder_upload_result = r2.upload_directory(job_id, process_temp_dir)
    manifest_url = None
    manifest_key = None
    files_map = {}
    if isinstance(folder_upload_result, dict):
        for fname, res in folder_upload_result.items():
            if res.get("success"):
                files_map[fname] = {"url": res.get("url"), "r2_key": res.get("r2_key")}
        for fname in (f"dubbing_manifest_{job_id}.json",):
            if fname in files_map:
                manifest_url = files_map[fname]["url"]
                manifest_key = files_map[fname]["r2_key"]
                break
    return folder_upload_result, manifest_url, manifest_key


def enrich_and_reupload_manifest_with_urls(manifest: Dict[str, Any], manifest_path: str, files_map: Dict[str, Any], manifest_key: Optional[str]) -> Optional[str]:
    try:
        url_map = {k: v["url"] for k, v in files_map.items() if isinstance(v, dict) and v.get("url")}
        for seg in manifest.get("segments", []):
            fname = seg.get("original_audio_file")
            if fname and fname in url_map:
                seg["original_audio_url"] = url_map[fname]
        write_json(manifest, manifest_path)
        if manifest_key:
            r2 = get_r2_service()
            up = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
            return up.get("url") if up.get("success") else None
    except Exception:
        return None
    return None


