import logging
import time
import requests
from typing import Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)


class AssemblyAITranscriptionService:
    def __init__(self):
        self.api_key = settings.ASSEMBLYAI_API_KEY
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {"authorization": self.api_key}
        
    def transcribe_audio_file(self, audio_path: str, language_code: str, job_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"AssemblyAI transcription starting for job {job_id}")
            
            audio_url = self._upload_audio(audio_path)
            transcript_id = self._request_transcription(audio_url, language_code)
            transcript_id = self._poll_transcription(transcript_id)
            
            sentences_data = self._get_sentences(transcript_id)
            segments = self._convert_to_whisperx_format(sentences_data)
            
            logger.info(f"AssemblyAI transcription completed for job {job_id}: {len(segments)} segments")
            
            return {
                "success": True,
                "segments": segments,
                "sentences": segments,
                "language": sentences_data.get("language_code", language_code)
            }
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed for job {job_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _upload_audio(self, audio_path: str) -> str:
        logger.info(f"Uploading audio to AssemblyAI: {audio_path}")
        
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/upload",
                headers=self.headers,
                data=f,
                timeout=300
            )
            response.raise_for_status()
        
        audio_url = response.json()["upload_url"]
        logger.info("Audio uploaded successfully")
        return audio_url
    
    def _request_transcription(self, audio_url: str, language_code: str) -> str:
        logger.info("Requesting transcription from AssemblyAI")
        
        payload = {
            "audio_url": audio_url,
            "speech_model": "best",
            "language_detection": True if language_code == "auto_detect" else False,
            "punctuate": True,
            "format_text": True
        }
        
        if language_code != "auto_detect":
            from app.services.language_service import language_service
            assembly_lang = language_service.get_assemblyai_language_code(language_code)
            if assembly_lang:
                payload["language_code"] = assembly_lang
        
        response = requests.post(
            f"{self.base_url}/transcript",
            headers={**self.headers, "content-type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        transcript_id = response.json()["id"]
        logger.info(f"Transcription requested: {transcript_id}")
        return transcript_id
    
    def _poll_transcription(self, transcript_id: str, timeout: int = 600) -> str:
        logger.info(f"Polling transcription status: {transcript_id}")
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Transcription timeout after {timeout}s")
            
            response = requests.get(
                f"{self.base_url}/transcript/{transcript_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            if status == "completed":
                logger.info("Transcription completed")
                return transcript_id
            elif status == "error":
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"AssemblyAI transcription error: {error_msg}")
            
            time.sleep(2)
    
    def _get_sentences(self, transcript_id: str) -> Dict[str, Any]:
        logger.info(f"Fetching sentences from transcript: {transcript_id}")
        
        response = requests.get(
            f"{self.base_url}/transcript/{transcript_id}/sentences",
            headers=self.headers,
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()
    
    def _convert_to_whisperx_format(self, sentences_data: Dict[str, Any]) -> list:
        segments = []
        
        sentences = sentences_data.get("sentences", [])
        if not sentences:
            logger.warning("No sentences in AssemblyAI response")
            return segments
        
        for idx, sentence in enumerate(sentences):
            segments.append({
                "id": f"assembly_{idx:03d}",
                "segment_index": idx,
                "start": sentence.get("start", 0),
                "end": sentence.get("end", 0),
                "duration_ms": sentence.get("end", 0) - sentence.get("start", 0),
                "text": sentence.get("text", "").strip(),
                "confidence": sentence.get("confidence", 0.95)
            })
        
        logger.info(f"Converted {len(segments)} sentences to segments")
        return segments


_assemblyai_service = None

def get_assemblyai_service() -> AssemblyAITranscriptionService:
    global _assemblyai_service
    if _assemblyai_service is None:
        _assemblyai_service = AssemblyAITranscriptionService()
    return _assemblyai_service
