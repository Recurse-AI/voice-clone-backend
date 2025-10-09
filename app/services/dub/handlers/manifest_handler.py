import os
import logging
import requests
from typing import Dict, Any, Optional
from app.services.dub.context import DubbingContext
from app.services.dub.manifest_manager import manifest_manager

logger = logging.getLogger(__name__)

class ManifestHandler:
    @staticmethod
    def load_and_restore(context: DubbingContext):
        if not context.manifest:
            return
        
        logger.info(f"Loading manifest for job {context.job_id}")
        manifest_segments = context.manifest.get("segments", [])
        
        if not manifest_segments:
            raise Exception("No segments found in manifest - cannot proceed")
        
        logger.info(f"Manifest contains {len(manifest_segments)} segments")
        
        context.vocal_url = context.manifest.get("vocal_audio_url")
        context.instrument_url = context.manifest.get("instrument_audio_url")
        
        ManifestHandler._download_missing_files(context)
    
    @staticmethod
    def _download_missing_files(context: DubbingContext):
        if not context.manifest:
            return
        
        files_to_check = [
            ("vocal_audio_url", f"vocal_{context.job_id}.wav"),
            ("instrument_audio_url", f"instrument_{context.job_id}.wav")
        ]
        
        for url_key, filename in files_to_check:
            file_path = os.path.join(context.process_temp_dir, filename)
            if not os.path.exists(file_path) and context.manifest.get(url_key):
                try:
                    resp = requests.get(context.manifest[url_key], timeout=60)
                    resp.raise_for_status()
                    with open(file_path, 'wb') as fw:
                        fw.write(resp.content)
                    logger.info(f"Downloaded {filename} from manifest for {context.job_id}")
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
    
    @staticmethod
    def build_and_save(context: DubbingContext) -> Dict[str, Any]:
        from app.services.dub.manifest_service import build_manifest, save_manifest_to_dir
        
        manifest = build_manifest(
            context.job_id,
            context.transcript_id,
            context.target_language,
            context.segments,
            context.vocal_url,
            context.instrument_url,
            context.model_type,
            context.voice_type,
            context.reference_ids,
            context.num_of_speakers,
            context.add_subtitle_to_video
        )
        
        save_manifest_to_dir(manifest, context.process_temp_dir, context.job_id)
        return manifest

