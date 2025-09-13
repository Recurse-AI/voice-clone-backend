import os
from typing import Dict, Any, Optional, Tuple
from app.utils.audio import AudioUtils


class SeparationUtils:
    """Utility functions for audio separation operations"""

    @staticmethod
    def extract_urls_from_clearvocals_response(output: Dict[str, Any]) -> Dict[str, str]:
        urls = {}
        vocal_url = output.get('vocal_audio')
        instrument_url = output.get('instrument_audio')

        if vocal_url:
            urls['vocal_audio'] = vocal_url
        if instrument_url:
            urls['instrument_audio'] = instrument_url

        return urls

    @staticmethod
    def download_separation_files(
        job_id: str,
        job_dir: str,
        runpod_urls: Dict[str, str],
        on_error_callback: Optional[callable] = None
    ) -> Tuple[bool, Dict[str, str]]:
        file_paths = {}
        success = True

        if runpod_urls.get('vocal_audio'):
            vocal_path = os.path.join(job_dir, f"vocal_{job_id}.wav")
            download_result = AudioUtils().download_audio_file(runpod_urls['vocal_audio'], vocal_path)
            if download_result["success"]:
                file_paths['vocal'] = vocal_path
            else:
                if on_error_callback:
                    on_error_callback("Vocal audio download failed.", download_result.get("error"))
                success = False

        if runpod_urls.get('instrument_audio'):
            instrument_path = os.path.join(job_dir, f"instrument_{job_id}.wav")
            download_result = AudioUtils().download_audio_file(runpod_urls['instrument_audio'], instrument_path)
            if download_result["success"]:
                file_paths['instrument'] = instrument_path
            else:
                if on_error_callback:
                    on_error_callback("Instrument audio download failed.", download_result.get("error"))
                success = False

        return success, file_paths

    @staticmethod
    def add_urls_to_folder_upload(
        folder_upload: Dict[str, Any],
        runpod_urls: Dict[str, str],
        job_id: str
    ) -> Dict[str, Any]:
        if not folder_upload:
            folder_upload = {}

        if runpod_urls.get('vocal_audio'):
            vocal_filename = f"vocal_{job_id}.wav"
            folder_upload[vocal_filename] = {
                "url": runpod_urls['vocal_audio'],
                "type": "audio",
                "success": True
            }

        if runpod_urls.get('instrument_audio'):
            instrument_filename = f"instrument_{job_id}.wav"
            folder_upload[instrument_filename] = {
                "url": runpod_urls['instrument_audio'],
                "type": "audio",
                "success": True
            }

        return folder_upload

    @staticmethod
    def store_urls_in_job_details(
        details: Dict[str, Any],
        runpod_urls: Dict[str, str]
    ) -> Dict[str, Any]:
        if runpod_urls:
            details["runpod_urls"] = runpod_urls
        return details

  


# Global instance for easy access
separation_utils = SeparationUtils()
