import os
import json
import logging
import time
import threading
import re
import numpy as np
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config.settings import settings
from app.services.language_service import language_service
from .video_processor import VideoProcessor
from .whisperx_transcription import get_whisperx_transcription_service
from .ai_segmentation_service import get_ai_segmentation_service
from .fish_speech_service import get_fish_speech_service
from app.services.r2_service import R2Service   
from .manifest_service import (
    build_manifest,
    save_manifest_to_dir,
    upload_process_dir_to_r2,
)
from app.utils.audio import AudioUtils
from app.services.simple_status_service import status_service, JobStatus

logger = logging.getLogger(__name__)

def _add_language_tag(text: str, language_code: str) -> str:
    if not text or not language_code:
        return text
    return f"{text} [{language_code}]"

class SimpleDubbedAPI:
    PROGRESS_PHASES = {
        "initialization": {"start": 0, "end": 10},
        "separation": {"start": 25, "end": 45},
        "transcription": {"start": 45, "end": 60},
        "dubbing": {"start": 60, "end": 75},
        "review_prep": {"start": 75, "end": 80},
        "reviewing": {"start": 80, "end": 81},
        "voice_cloning": {"start": 81, "end": 91},
        "final_processing": {"start": 91, "end": 96},
        "upload": {"start": 96, "end": 100}
    }
    
    def __init__(self):
        # Multi-worker safe: Service instances use worker queues for model access
        # No direct model loading in this process - delegated to dedicated service workers
        self.transcription_service = get_whisperx_transcription_service()
        self.fish_speech = get_fish_speech_service()
        self._r2_storage = None
        # Cache AI voice sample audio per reference_id for this job lifecycle
        self._ai_voice_reference_cache = {}
    
    @property
    def r2_storage(self):
        if self._r2_storage is None:
            self._r2_storage = R2Service()
        return self._r2_storage
    
    @property  
    def temp_dir(self):
        return settings.TEMP_DIR
    
    def _update_status(self, job_id: str, status: JobStatus, progress: int, details: dict):
        try:
            status_service.update_status(job_id, "dub", status, progress, details)
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
    
    def _charge_review_credits(self, job_id: str):
        """Charge 75% credits when job is ready for review"""
        from app.utils.job_utils import job_utils
# MongoDB client accessed via global sync_client from database config
        
        try:
            # Use global sync client for connection pooling
            from app.config.database import sync_client
            job_doc = sync_client[settings.DB_NAME].dub_jobs.find_one({"job_id": job_id})
            if job_doc and job_doc.get("user_id"):
                job_utils.complete_job_billing_sync(job_id, "dub", job_doc["user_id"], 0.75)
                logger.info(f"✅ Charged 75% credits for job {job_id} ready for review")
        except Exception as e:
            logger.error(f"Failed to charge 75% credits for job {job_id}: {e}")

    def _update_phase_progress(self, job_id: str, phase: str, sub_progress: float, message: str):
        if phase not in self.PROGRESS_PHASES:
            logger.warning(f"Unknown phase '{phase}' for job {job_id}")
            return
        
        sub_progress = min(1.0, max(0.0, sub_progress))
        start, end = self.PROGRESS_PHASES[phase]["start"], self.PROGRESS_PHASES[phase]["end"]
        progress = min(100, max(0, start + int((end - start) * sub_progress)))
        
        # Map phase to appropriate high-level status for UI consistency
        phase_status = {
            "separation": JobStatus.SEPARATING,
            "transcription": JobStatus.TRANSCRIBING,
            "reviewing": JobStatus.REVIEWING,
            "upload": JobStatus.UPLOADING,
        }.get(phase, JobStatus.PROCESSING)

        self._update_status(
            job_id,
            phase_status,
            progress,
            {"message": message, "phase": phase, "sub_progress": sub_progress}
        )
    
    def _should_update_progress(self, completed: int, total: int) -> bool:
        return (completed % max(1, total // 10) == 0 or completed == total or completed <= 3)
    



    def _validate_target_language(self, target_language: str) -> tuple:
        if not language_service.is_dubbing_supported(target_language):
            supported = language_service.get_supported_dubbing_languages()
            error = f"Unsupported target language: {target_language}. Supported: {', '.join(sorted(supported))}"
            return False, error, None
        
        code = language_service.normalize_language_input(target_language)
        logger.info(f"Processing with target language: {target_language} -> {code}")
        return True, None, code
    
    def _setup_voice_config(self, manifest_override: dict, voice_premium_model: bool, voice_type: str, reference_ids: List[str]) -> dict:
        manifest_premium = manifest_override.get("voice_premium_model") if manifest_override else None
        
        if manifest_override:
            voice_type = voice_type or manifest_override.get("voice_type")
            reference_ids = reference_ids or manifest_override.get("reference_ids", [])
        
        final_premium = manifest_premium if manifest_premium is not None else voice_premium_model
        
        self.voice_premium_model = final_premium
        self.voice_type = voice_type
        self.reference_ids = reference_ids or []
        
        return {"premium": final_premium, "voice_type": voice_type, "reference_ids": reference_ids}
    
    def _get_audio_urls(self, manifest_override: dict, separation_urls: dict) -> tuple:
        if manifest_override:
            vocal_url = manifest_override.get("vocal_audio_url")
            instrument_url = manifest_override.get("instrument_audio_url")
        elif separation_urls:
            vocal_url = separation_urls.get("vocal_audio")
            instrument_url = separation_urls.get("instrument_audio")
        else:
            vocal_url = None
            instrument_url = None
        
        if not vocal_url:
            logger.warning("No vocal audio URL available - resume/redub may fail")
        
        return vocal_url, instrument_url
    
    def _handle_review_mode(self, job_id: str, dubbed_segments: list, transcript_id: str, 
                           manifest_override: dict, process_temp_dir: str, target_language: str,
                           vocal_url: str, instrument_url: str, voice_config: dict) -> dict:
        self._update_phase_progress(job_id, "review_prep", 0.5, "Preparing segments for review")
        
        if manifest_override:
            manifest = manifest_override.copy()
            manifest["segments"] = dubbed_segments
            manifest["target_language"] = target_language
            manifest["voice_premium_model"] = voice_config["premium"]
        else:
            manifest = build_manifest(
                job_id, transcript_id, target_language, dubbed_segments,
                vocal_url, instrument_url, voice_config["premium"],
                voice_config["voice_type"], voice_config["reference_ids"]
            )
        
        save_manifest_to_dir(manifest, process_temp_dir, job_id)
        
        exclude_files = []
        if vocal_url:
            exclude_files.append(f"vocal_{job_id}.wav")
        if instrument_url:
            exclude_files.append(f"instrument_{job_id}.wav")
        
        folder_upload, manifest_url, manifest_key = upload_process_dir_to_r2(
            job_id, process_temp_dir, self.r2_storage, exclude_files=exclude_files
        )
        
        if not manifest_url:
            return {"success": False, "error": "Failed to generate manifest URL"}
        
        self._charge_review_credits(job_id)
        
        self._update_status(
            job_id, JobStatus.AWAITING_REVIEW, 80, {
                "message": "Awaiting human review",
                "segments_manifest_url": manifest_url,
                "segments_manifest_key": manifest_key,
                "segments_count": len(dubbed_segments),
                "transcript_id": transcript_id
            }
        )
        
        try:
            from app.utils.cleanup_utils import cleanup_utils
            cleanup_utils.schedule_auto_cleanup(job_id, delay_minutes=60)
        except Exception:
            pass
        
        return {
            "success": True,
            "job_id": job_id,
            "review": {
                "segments_manifest_url": manifest_url,
                "segments_manifest_key": manifest_key,
                "segments_count": len(dubbed_segments),
                "transcript_id": transcript_id
            },
            "folder_upload": folder_upload
        }
    
    def process_dubbed_audio(self, job_id: str, target_language: str,
                           source_video_language: str = None, output_dir: str = None,
                           review_mode: bool = False, manifest_override: Optional[Dict[str, Any]] = None,
                           separation_urls: Optional[Dict[str, str]] = None, video_subtitle: bool = False,
                           voice_premium_model: bool = False, voice_type: Optional[str] = None,
                           reference_ids: Optional[List[str]] = None) -> dict:
        valid, error, target_code = self._validate_target_language(target_language)
        if not valid:
            logger.error(error)
            return {"success": False, "error": error}
        
        try:
            voice_config = self._setup_voice_config(manifest_override, voice_premium_model, voice_type, reference_ids)
            
            self._update_status(job_id, JobStatus.PROCESSING, 60, {"message": "Starting dubbing", "phase": "dubbing"})
            
            ai_segments, transcript_id, transcription_result = self._get_transcription_and_process_with_ai(
                job_id, manifest_override, output_dir, source_video_language,
                target_language, target_code, video_subtitle
            )
            
            if review_mode:
                dubbed_segments = ai_segments
            else:
                self._update_status(job_id, JobStatus.PROCESSING, 65, {"message": "Starting voice cloning", "phase": "voice_cloning"})
                
                source_lang = transcription_result.get("language", "auto_detect") if transcription_result else "auto_detect"
                source_code = language_service.normalize_language_input(source_lang)
                
                dubbed_segments = self._process_voice_cloning_with_ai_segments(
                    job_id, ai_segments, manifest_override, review_mode, output_dir, target_code, source_code
                )
            
            vocal_url, instrument_url = self._get_audio_urls(manifest_override, separation_urls)
            
            if review_mode:
                return self._handle_review_mode(
                    job_id, dubbed_segments, transcript_id, manifest_override, output_dir,
                    target_language, vocal_url, instrument_url, voice_config
                )
            
            return self._generate_final_output(
                job_id, dubbed_segments, output_dir, target_language, transcript_id, vocal_url, instrument_url
            )
            
        except Exception as e:
            logger.error(f"Dubbed processing failed: {str(e)}")
            if locals().get("output_dir"):
                AudioUtils.remove_temp_dir(folder_path=locals().get("output_dir"))
            return {"success": False, "error": str(e)}
    
    def _get_segment_speaker(self, seg_start_ms: int, seg_end_ms: int, speaker_timeline: List[Dict]) -> str:
        seg_start_s = seg_start_ms / 1000.0
        seg_end_s = seg_end_ms / 1000.0
        
        for speaker_seg in speaker_timeline:
            overlap_start = max(seg_start_s, speaker_seg["start"])
            overlap_end = min(seg_end_s, speaker_seg["end"])
            if overlap_start < overlap_end:
                return speaker_seg["speaker"]
        return None
    
    def _assign_reference_id(self, speaker: str) -> str:
        if not speaker or not self.reference_ids:
            return None
        try:
            speaker_index = int(speaker.split("_")[-1])
            if speaker_index < len(self.reference_ids):
                return self.reference_ids[speaker_index]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_ai_voice_reference(self, reference_id: str, source_language_code: str) -> tuple:
        from app.config.settings import settings as _settings
        from app.services.dub.fish_audio_sample_helper import fetch_sample_audio_wav_bytes
        
        cached = self._ai_voice_reference_cache.get(reference_id)
        if cached:
            return cached
        
        audio_bytes, sample_text = fetch_sample_audio_wav_bytes(reference_id, _settings.FISH_AUDIO_API_KEY)
        if audio_bytes:
            self._ai_voice_reference_cache[reference_id] = (audio_bytes, sample_text)
        
        return audio_bytes, sample_text
    
    def _load_reference_audio(self, audio_path: str) -> Optional[bytes]:
        import soundfile as sf
        import io
        
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to load reference audio from {audio_path}: {e}")
            return None
    
    def _generate_with_premium_api(self, tagged_text: str, reference_audio_bytes: bytes, tagged_reference_text: str, 
                                   job_id: str, target_language_code: str, voice_type: str, ai_voice_reference_id: str) -> dict:
        from app.services.dub.fish_audio_api_service import FishAudioAPIService
        
        fish_api = FishAudioAPIService()
        if not fish_api.api_key or not fish_api.api_key.strip():
            return {"success": False}
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🎯 Premium API attempt {attempt}/{max_retries}")
                
                if voice_type == 'ai_voice' and ai_voice_reference_id:
                    result = fish_api.generate_voice_clone(
                        text=tagged_text,
                        reference_audio_bytes=None,
                        reference_text=None,
                        job_id=job_id,
                        target_language_code=target_language_code,
                        reference_id=ai_voice_reference_id
                    )
                else:
                    result = fish_api.generate_voice_clone(
                        text=tagged_text,
                        reference_audio_bytes=reference_audio_bytes,
                        reference_text=tagged_reference_text,
                        job_id=job_id,
                        target_language_code=target_language_code
                    )
                
                if result.get("success"):
                    logger.info(f"✅ Premium API success on attempt {attempt}")
                    return result
                
                logger.warning(f"⚠️ Premium API attempt {attempt} failed: {result.get('error', 'Unknown')}")
                
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    
            except Exception as e:
                logger.error(f"❌ Premium API attempt {attempt} error: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.error(f"❌ Premium API failed after {max_retries} attempts")
        return {"success": False, "error": f"Failed after {max_retries} retries"}
    
    def _generate_with_local_model(self, tagged_text: str, reference_audio_bytes: bytes, 
                                   tagged_reference_text: str, job_id: str, target_language_code: str) -> dict:
        return self.fish_speech.generate_with_reference_audio(
            text=tagged_text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=tagged_reference_text,
            max_new_tokens=1024,
            top_p=0.9,
            repetition_penalty=1.07,
            temperature=0.75,
            job_id=job_id,
            target_language_code=target_language_code
        )
    
    def _save_cloned_audio(self, output_path: str, cloned_path: str, segment_id: str) -> Optional[Dict[str, Any]]:
        import soundfile as sf
        
        if not output_path or not os.path.exists(output_path):
            return None
        
        try:
            audio_data, sample_rate = sf.read(output_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            sf.write(cloned_path, audio_data, sample_rate, subtype='PCM_16')
            os.remove(output_path)
            
            duration_ms = int(len(audio_data) / sample_rate * 1000)
            return {"path": cloned_path, "duration_ms": duration_ms}
        except Exception as e:
            logger.error(f"Failed to process audio for {segment_id}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    
    def _voice_clone_segment(self, dubbed_text: str, reference_audio_path: str, segment_id: str, 
                            original_text: str = "", job_id: str = None, process_temp_dir: str = None, 
                            target_language_code: str = "en", source_language_code: str = "en",
                            segment_reference_id: str = None) -> Optional[Dict[str, Any]]:
        try:
            start_time = time.time()
            
            tagged_text = _add_language_tag(dubbed_text, target_language_code)
            tagged_reference_text = _add_language_tag(original_text or "Reference audio", source_language_code)
            
            logger.info(f"🎯 Voice cloning {segment_id}")
            
            is_premium = getattr(self, 'voice_premium_model', False)
            voice_type = getattr(self, 'voice_type', None)
            ai_voice_id = segment_reference_id if segment_reference_id else (getattr(self, 'reference_ids', [None])[0] if hasattr(self, 'reference_ids') and self.reference_ids else None)
            
            reference_audio_bytes = None
            
            if voice_type == 'ai_voice' and ai_voice_id:
                if is_premium:
                    logger.info(f"🎙️ Using AI voice reference_id directly with premium API: {ai_voice_id}")
                else:
                    logger.info(f"🎙️ Fetching AI voice sample for local model: {ai_voice_id}")
            
            if voice_type == 'ai_voice' and ai_voice_id and not is_premium:
                audio_bytes, sample_text = self._get_ai_voice_reference(ai_voice_id, source_language_code)
                reference_audio_bytes = audio_bytes
                if sample_text:
                    tagged_reference_text = _add_language_tag(sample_text, source_language_code)
            
            if reference_audio_bytes is None:
                reference_audio_bytes = self._load_reference_audio(reference_audio_path)
            
            segment_index = int(segment_id.split('_')[1]) - 1
            cloned_path = os.path.join(process_temp_dir, f"cloned_{job_id}_{segment_index:03d}.wav").replace('\\', '/')
            
            if is_premium:
                result = self._generate_with_premium_api(
                    tagged_text, reference_audio_bytes, tagged_reference_text,
                    job_id, target_language_code, voice_type, ai_voice_id
                )
                
                if not result.get("success"):
                    logger.warning(f"❌ Fish API failed for {segment_id}, using local model")
                    result = self._generate_with_local_model(
                        tagged_text, reference_audio_bytes, tagged_reference_text,
                        job_id, target_language_code
                    )
            else:
                result = self._generate_with_local_model(
                    tagged_text, reference_audio_bytes, tagged_reference_text,
                    job_id, target_language_code
                )
            
            elapsed = time.time() - start_time
            
            if result.get("success"):
                output = self._save_cloned_audio(result.get("output_path"), cloned_path, segment_id)
                if output:
                    logger.info(f"✅ Voice cloning {segment_id} completed in {elapsed:.2f}s")
                    return output
            
            logger.error(f"Voice cloning failed for {segment_id}")
            return None
                
        except Exception as e:
            logger.error(f"Voice cloning error for {segment_id}: {str(e)}")
            return None
    
    def _clone_segment_worker(self, data: dict, actual_index: int, total_segments: int, job_id: str, process_temp_dir: str) -> tuple:
        try:
            start_time = time.time()
            result = self._voice_clone_segment(
                data["dubbed_text"], 
                data["original_audio_path"], 
                data["seg_id"], 
                data["original_text"], 
                job_id=job_id, 
                process_temp_dir=process_temp_dir,
                target_language_code=getattr(self, '_target_language_code', 'en'),
                source_language_code=getattr(self, '_source_language_code', 'en'),
                segment_reference_id=data.get("reference_id")
            )
            elapsed = time.time() - start_time
            logger.info(f"✅ Segment {data['seg_id']} ({actual_index+1}/{total_segments}) completed in {elapsed:.2f}s")
            return (actual_index, result, True)
        except Exception as e:
            logger.error(f"❌ Segment {data.get('seg_id', 'unknown')} failed: {e}")
            return (actual_index, None, False)
    
    def _process_voice_cloning_sequential(self, segments_data: list, job_id: str, process_temp_dir: str) -> list:
        total_segments = len(segments_data)
        batch_size = settings.VOICE_CLONING_BATCH_SIZE
        
        is_premium_api = getattr(self, 'voice_premium_model', False)
        max_workers = settings.VOICE_CLONING_PARALLEL_WORKERS if is_premium_api else 1
        
        mode = "API (parallel)" if is_premium_api else "local s1-mini (sequential)"
        logger.info(f"🚀 Processing {total_segments} segments using {mode} with {max_workers} worker(s)")
        
        results = [None] * total_segments
        
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch_data = segments_data[batch_start:batch_end]
            
            logger.info(f"📦 Batch {batch_start//batch_size + 1}: segments {batch_start+1}-{batch_end}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._clone_segment_worker, data, batch_start + i, total_segments, job_id, process_temp_dir): i
                    for i, data in enumerate(batch_data)
                }
                
                for future in as_completed(futures):
                    actual_index, result, success = future.result()
                    results[actual_index] = result
                    
                    completed = sum(1 for r in results if r is not None)
                    progress = (completed / total_segments) * 0.9 + 0.1
                    
                    try:
                        self._update_phase_progress(job_id, "voice_cloning", progress, f"Voice cloning: {completed}/{total_segments}")
                    except Exception:
                        pass
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"🎯 Completed: {successful}/{total_segments} successful")
        
        try:
            self._update_phase_progress(job_id, "voice_cloning", 1.0, f"Voice cloning complete: {successful}/{total_segments}")
        except Exception:
            pass
        
        return results
    
    
    def _get_transcription_and_process_with_ai(self, job_id: str, manifest_override: Optional[Dict[str, Any]],
                                              process_temp_dir: str, source_video_language: str, 
                                              target_language: str, target_language_code: str, video_subtitle: bool = False) -> tuple:
        """Get transcription and process with AI for optimal segmentation and dubbing"""
        
        transcript_id = None
        transcription_result = None
        
        if manifest_override:
            logger.info("Using manifest override data")
            manifest_segments = manifest_override.get("segments", [])
            if not manifest_segments:
                logger.error(f"No segments found in manifest override. Manifest keys: {list(manifest_override.keys())}")
                logger.error(f"Manifest structure: {json.dumps(manifest_override, indent=2)}")
                raise Exception("No segments found in manifest override - cannot proceed with redub")
            
            logger.info(f"Manifest contains {len(manifest_segments)} segments")
            
            # Manifest segments are already normalized to milliseconds by manifest manager
            logger.info(f"Using {len(manifest_segments)} manifest segments (already in milliseconds format)")
            
            transcription_result = {
                "success": True,
                "segments": manifest_segments,
                "language": manifest_override.get("language", "auto")
            }
            
            transcript_id = manifest_override.get("transcript_id")
            self._download_missing_files(job_id, manifest_override, process_temp_dir)
        else:
            if video_subtitle:
                self._update_phase_progress(job_id, "transcription", 0.0, "Using provided SRT file")
                srt_file_path = os.path.join(process_temp_dir, f"{job_id}.srt")
                if not os.path.exists(srt_file_path):
                    raise Exception(f"SRT file not found at {srt_file_path}")
                
                from app.utils.srt_parser import parse_srt_to_whisperx_format
                transcription_result = parse_srt_to_whisperx_format(srt_file_path)
                
                if not transcription_result["success"]:
                    raise Exception(transcription_result.get("error", "SRT parsing failed"))
                
                transcript_id = f"srt_{int(time.time())}"
                self._update_phase_progress(job_id, "transcription", 1.0, "SRT loaded")
                
            else:
                self._update_phase_progress(job_id, "transcription", 0.0, "Starting audio transcription")
                vocal_file_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")

                if not os.path.exists(vocal_file_path):
                    raise Exception(f"Vocal file not found at {vocal_file_path}")

                transcription_result = self.transcription_service._transcribe_via_service_worker(
                    vocal_file_path, source_video_language, job_id
                )

                if not transcription_result["success"]:
                    AudioUtils.remove_temp_dir(folder_path=process_temp_dir)
                    raise Exception(transcription_result.get("error", "Transcription failed"))

                transcript_id = f"whisperx_{int(time.time())}"
                self._update_phase_progress(job_id, "transcription", 1.0, "Transcription completed")
        
        if not manifest_override:
            try:
                from app.services.dub.speaker_detection_service import speaker_detection_service
                
                vocal_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")
                if os.path.exists(vocal_path):
                    num_speakers = len(self.reference_ids) if self.reference_ids else None
                    logger.info(f"Running speaker detection before AI segmentation (expected speakers: {num_speakers or 'auto'})")
                    speaker_timeline = speaker_detection_service.detect_speakers(vocal_path, job_id, num_speakers)
                    transcription_result["speaker_timeline"] = speaker_timeline
                    
                    for seg in transcription_result.get("segments", []):
                        seg_start_ms = seg.get("start", 0)
                        seg_end_ms = seg.get("end", 0)
                        seg["speaker"] = self._get_segment_speaker(seg_start_ms, seg_end_ms, speaker_timeline)
                    
                    logger.info(f"Tagged {len(transcription_result.get('segments', []))} segments with speakers")
            except Exception as e:
                logger.error(f"Speaker detection failed: {e}, continuing without speaker tags")
        else:
            logger.info("Redub: using manifest speaker info")
        
        self._update_phase_progress(job_id, "dubbing", 0.0, "AI creating segments and dubbing")
        
        # Check if we have manifest with already dubbed texts (resume after review)
        edited_map = self._get_edited_text_map(manifest_override, target_language) if manifest_override else {}
        
        if edited_map:
            # RESUME: Use manifest segments directly, no AI needed
            logger.info(f"RESUME MODE: Using existing segments and translations ({len(edited_map)} segments)")
            ai_segments = self._use_manifest_segments_for_resume(transcription_result, edited_map)
            self._update_phase_progress(job_id, "dubbing", 1.0, "Using reviewed segments and dubbing")
        else:
            # Fresh job or redub - use AI service
            ai_service = get_ai_segmentation_service()
            preserve_segments = bool(manifest_override)  # True for redub, False for fresh
            ai_segments = ai_service.create_optimal_segments_and_dub(
                transcription_result,
                target_language_code,
                preserve_segments=preserve_segments
            )
            
            if preserve_segments:
                logger.info(f"REDUB MODE: Preserved {len(ai_segments)} segments with translation")
                self._update_phase_progress(job_id, "dubbing", 1.0, f"Redub completed {len(ai_segments)} segments")
            else:
                logger.info(f"AI created {len(ai_segments)} optimal segments with S1 dubbing")
                self._update_phase_progress(job_id, "dubbing", 1.0, f"AI completed {len(ai_segments)} segments with S1 dubbing")
        
        for segment in ai_segments:
            speaker = segment.get("speaker")
            segment["reference_id"] = self._assign_reference_id(speaker) if speaker and self.reference_ids else None
        
        logger.info(f"Assigned reference_ids to {len(ai_segments)} segments")
        
        return ai_segments, transcript_id, transcription_result



    def _download_missing_files(self, job_id: str, manifest_override: Dict[str, Any], process_temp_dir: str):
        if not manifest_override:
            return

        # For redub, download parent's files but save with current redub job ID
        files_to_check = [
            ("vocal_audio_url", f"vocal_{job_id}.wav"),
            ("instrument_audio_url", f"instrument_{job_id}.wav")
        ]

        for url_key, filename in files_to_check:
            file_path = os.path.join(process_temp_dir, filename)
            if not os.path.exists(file_path) and manifest_override.get(url_key):
                try:
                    import requests
                    resp = requests.get(manifest_override[url_key], timeout=60)
                    resp.raise_for_status()
                    with open(file_path, 'wb') as fw:
                        fw.write(resp.content)
                    logger.info(f"✅ Downloaded {filename} from parent job for redub {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to download {filename} for job {job_id}: {e}")

    def _get_edited_text_map(self, manifest_override: Optional[Dict[str, Any]], target_language: str) -> dict:
        """Get edited text map for redub/resume scenarios"""
        if not manifest_override:
            return {}
        
        current_target_lang = language_service.normalize_language_input(target_language)
        
       
        manifest_target_lang = manifest_override.get("target_language")
        if manifest_target_lang:
            manifest_target_lang = language_service.normalize_language_input(manifest_target_lang)
            if manifest_target_lang != current_target_lang:
                logger.info(f"Language changed: {manifest_target_lang} → {current_target_lang}, forcing AI translation")
                return {}
        
        # Resume with same language - use existing dubbed_text
        edited_map = {}
        for seg in manifest_override.get("segments", []):
            if seg.get("id") and seg.get("dubbed_text"):
                edited_map[seg["id"]] = seg["dubbed_text"]
        logger.info(f"RESUME: Using {len(edited_map)} existing translations")
        return edited_map
    
    def _use_manifest_segments_for_resume(self, transcription_result: Dict[str, Any], edited_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Use manifest segments directly for resume - no AI processing needed"""
        segments = transcription_result.get("segments", [])
        formatted_segments = []
        
        logger.info(f"RESUME: Processing {len(segments)} manifest segments directly")
        
        for idx, seg in enumerate(segments):
            seg_id = seg.get("id", f"seg_{idx+1:03d}")
            start_ms = int(seg.get("start", 0))
            end_ms = int(seg.get("end", 0))
            duration_ms = seg.get("duration_ms", end_ms - start_ms)
            
            original_text = seg.get("original_text", seg.get("text", "")).strip()
            dubbed_text = seg.get("dubbed_text", "").strip()
            
            # Use edited text from review
            if seg_id in edited_map:
                dubbed_text = edited_map[seg_id]
            elif not dubbed_text:
                dubbed_text = original_text
                
            formatted_segments.append({
                "id": seg_id,
                "segment_index": idx,
                "start": start_ms,
                "end": end_ms,
                "duration_ms": duration_ms,
                "original_text": original_text,
                "dubbed_text": dubbed_text,
                "voice_cloned": seg.get("voice_cloned", False),
                "original_audio_file": seg.get("original_audio_file"),
                "cloned_audio_file": seg.get("cloned_audio_file")
            })
        
        logger.info(f"RESUME: Prepared {len(formatted_segments)} segments for voice cloning")
        return formatted_segments

    def _process_voice_cloning_with_ai_segments(self, job_id: str, ai_segments: List[Dict[str, Any]],
                                                manifest_override: Optional[Dict[str, Any]], 
                                                review_mode: bool, process_temp_dir: str, target_language_code: str = "en", source_language_code: str = "en") -> List[Dict[str, Any]]:
        """Process voice cloning using AI-generated segments"""
        
        if review_mode:
            return ai_segments
        
        self._target_language_code = target_language_code
        self._source_language_code = source_language_code
        self._update_phase_progress(job_id, "voice_cloning", 0.0, "Segmenting audio for voice cloning")
        
        vocal_file_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")
        if not os.path.exists(vocal_file_path):
            raise Exception(f"Vocal file not found for segmentation")
        
        from app.utils.audio import AudioUtils
        audio_utils = AudioUtils()
        
        segments_to_split = []
        for seg in ai_segments:
            if seg.get("original_text", "").strip():
                segments_to_split.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["original_text"]
                })
        
        split_result = audio_utils.split_audio_by_timestamps(vocal_file_path, process_temp_dir, segments_to_split)
        if not split_result["success"]:
            raise Exception(f"Audio segmentation failed: {split_result['error']}")
        
        self._update_phase_progress(job_id, "voice_cloning", 0.1, f"Audio segmented into {len(segments_to_split)} parts")
        
        segments_data = []
        for seg, split_file in zip(ai_segments, split_result.get("split_files", [])):
            if seg.get("original_text", "").strip():
                segments_data.append({
                    "seg_id": seg["id"],
                    "global_idx": seg["segment_index"],
                    "start_ms": seg["start"],
                    "end_ms": seg["end"],
                    "original_text": seg["original_text"],
                    "dubbed_text": seg["dubbed_text"],
                    "original_audio_path": split_file["output_path"]
                })
        
        results = self._process_voice_cloning_sequential(segments_data, job_id, process_temp_dir)
        
        final_segments = []
        for i, data in enumerate(segments_data):
            result = results[i] if i < len(results) else None
            cloned_audio_path = result.get("path") if result else None
            cloned_duration_ms = result.get("duration_ms", data["end_ms"] - data["start_ms"]) if result else data["end_ms"] - data["start_ms"]
            
            segment_json = self._create_segment_data(
                data["seg_id"], data["global_idx"], data["start_ms"], cloned_duration_ms,
                data["original_text"], data["dubbed_text"], data["original_audio_path"],
                cloned_audio_path, job_id
            )
            final_segments.append(segment_json)
        
        return final_segments
    
    def _create_segment_data(self, seg_id: str, segment_index: int, start_ms: int, cloned_duration_ms: int,
                           original_text: str, dubbed_text: str, original_audio_path: str, 
                           cloned_audio_path: str, job_id: str) -> dict:
        start_ms = int(start_ms) if isinstance(start_ms, (float, int)) else start_ms
        end_ms = int(start_ms + cloned_duration_ms)
        duration_ms = int(cloned_duration_ms) if isinstance(cloned_duration_ms, (float, int)) else cloned_duration_ms
        
        return {
            "id": seg_id,
            "segment_index": segment_index + 1,
            "start": start_ms,
            "end": end_ms,
            "duration_ms": duration_ms,
            "original_text": original_text,
            "dubbed_text": dubbed_text,
            "original_audio_file": f"segment_{segment_index:03d}.wav" if original_audio_path else None,
            "cloned_audio_file": f"cloned_{job_id}_{segment_index:03d}.wav" if cloned_audio_path else None,
            "voice_cloned": bool(cloned_audio_path),
            "original_audio_path": original_audio_path,
            "cloned_audio_path": cloned_audio_path,
        }



    def _generate_final_output(self, job_id: str, dubbed_segments: list, process_temp_dir: str,
                              target_language: str, transcript_id: str,
                              vocal_audio_url: str = None, instrument_audio_url: str = None) -> dict:
        """Generate final audio, SRT file, and upload to R2"""
        
        
        self._update_phase_progress(job_id, "final_processing", 0.0, "Reconstructing final audio...")
        logger.info("Reconstructing final audio...")
        
        # Reconstruct final audio
        # Preserve original sample rate if vocal file exists
        orig_vocal_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")
        orig_vocal_path = orig_vocal_path if os.path.exists(orig_vocal_path) else None
        final_audio_path = AudioUtils.reconstruct_final_audio(dubbed_segments, orig_vocal_path, job_id=job_id, process_temp_dir=process_temp_dir)
        
        
        # Generate SRT file and finalize (combining multiple close steps)
        self._update_phase_progress(job_id, "final_processing", 0.5, "Finalizing output files...")
        
        subtitle_path = self._generate_srt_file(job_id, dubbed_segments, process_temp_dir)
        
        
        # Create process summary
        self._create_process_summary(job_id, dubbed_segments, final_audio_path, subtitle_path,
                                   process_temp_dir, target_language, transcript_id)
        
        # Save final manifest (for normal dub / redub lineage)
        try:
            manifest = build_manifest(
                job_id, transcript_id, target_language, dubbed_segments,
                vocal_audio_url, instrument_audio_url,
                getattr(self, 'voice_premium_model', False),
                getattr(self, 'voice_type', None),
                getattr(self, 'reference_id', None)
            )
            save_manifest_to_dir(manifest, process_temp_dir, job_id)
        except Exception as e:
            logger.error(f"Failed to save manifest for job {job_id}: {e}")
            # Don't pass silently - this is critical for manifest availability

        # Create video result if video source is available (before upload)
        video_result = self._create_video_if_available(job_id, process_temp_dir, final_audio_path, 
                                                       vocal_audio_url, instrument_audio_url)

        # Upload to R2 and get results
        return self._upload_and_finalize(job_id, process_temp_dir, final_audio_path, video_result)
    
    def _generate_srt_file(self, job_id: str, dubbed_segments: list, process_temp_dir: str) -> str:
        """Generate SRT subtitle file"""
        logger.info("Generating SRT file...")
        

        processor = VideoProcessor(temp_dir=process_temp_dir)
        subtitle_data = []
        
        for seg in dubbed_segments:
            text = seg["dubbed_text"]
            start = seg["start"] / 1000.0
            end = seg["end"] / 1000.0

            # Use text as single subtitle line (segments are already optimally sized)
            subtitle_data.append({"start": start, "end": end, "text": text})
        
        subtitle_path = os.path.join(process_temp_dir, f"subtitles_{job_id}.srt")
        processor.create_srt_file(subtitle_data, subtitle_path)
        logger.info(f"Subtitle file saved: {subtitle_path}")
        return subtitle_path
    
    def _create_process_summary(self, job_id: str, dubbed_segments: list, final_audio_path: str,
                               subtitle_path: str, process_temp_dir: str, target_language: str,
                               transcript_id: str) -> None:
        """Create and save process summary JSON"""
        
        # Check for instrument and vocal files
        instrument_file = f"instrument_{job_id}.wav"
        vocal_file = f"vocal_{job_id}.wav"
        
        if not os.path.exists(os.path.join(process_temp_dir, instrument_file)):
            instrument_file = None
            
        if not os.path.exists(os.path.join(process_temp_dir, vocal_file)):
            vocal_file = None
        
        process_summary = {
            "success": True,
            "job_id": job_id,
            "segments_count": len(dubbed_segments),
            "target_language": target_language,
            "final_audio_file": os.path.basename(final_audio_path) if final_audio_path else None,
            "subtitle_file": os.path.basename(subtitle_path) if subtitle_path else None,
            "instrument_file": instrument_file,
            "vocal_file": vocal_file,
            "final_video_file": None,  # Video creation disabled
            "processing_timestamp": int(time.time()),
            "segments": dubbed_segments,
            "transcript_id": transcript_id,
        }
        
        summary_filename = f"process_summary_{job_id}.json"
        summary_path = os.path.join(process_temp_dir, summary_filename).replace('\\', '/')
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(process_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Process summary saved: {summary_filename}")
        except Exception as e:
            logger.error(f"Failed to save process summary: {str(e)}")
    
    def _create_video_if_available(self, job_id: str, process_temp_dir: str, final_audio_path: str,
                                   vocal_audio_url: str = None, instrument_audio_url: str = None) -> dict:
        """Create video result if video source is available"""
        try:
            from app.utils.db_sync_operations import get_dub_job_sync
            
            # Get job data to check for video source
            job_data = get_dub_job_sync(job_id)
            if not job_data:
                return {"success": False, "error": "Job not found"}
            
            video_url = job_data.get("video_url")
            local_video_path = job_data.get("local_video_path")
            
            # Skip video creation if neither video URL nor local video path available
            if not video_url and not local_video_path:
                logger.info(f"No video URL or local video path available for job {job_id}, skipping video creation")
                return {"success": True, "skipped": True}
           
            self._update_phase_progress(job_id, "final_processing", 0.7, "Creating video result...")
            logger.info(f"Creating video result for job {job_id}")
            
            # Create video with synchronized video processing logic
            video_result = self._process_video_sync(
                job_id, video_url, local_video_path, final_audio_path, process_temp_dir,
                vocal_audio_url, instrument_audio_url
            )
            
            return video_result
            
        except Exception as e:
            logger.error(f"Failed to create video for job {job_id}: {e}")
            return {"success": False, "error": str(e)}


    def _process_video_sync(self, job_id: str, video_url: str, local_video_path: str, audio_path: str, output_dir: str,
                           vocal_audio_url: str = None, instrument_audio_url: str = None) -> dict:
        """Process video synchronously using existing video processing logic"""
        try:
            from app.config.constants import INSTRUMENT_DEFAULT_VOLUME
            import subprocess
            import os
            import requests
            from pathlib import Path
            
            output_dir_path = Path(output_dir)
            downloaded_video_path = output_dir_path / "source_video.mp4"
            
            # Priority: 1. original.* (from video-dub API download), 2. local_video_path, 3. Download from URL
            if not downloaded_video_path.exists():
                # Check for original.* files from video-dub API initial download
                video_found = False
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']:
                    original_path = output_dir_path / f"original{ext}"
                    if original_path.exists():
                        import shutil
                        shutil.copy2(original_path, downloaded_video_path)
                        logger.info(f"📁 Using original video file: {original_path}")
                        video_found = True
                        break
                
                if not video_found:
                    if local_video_path and os.path.exists(local_video_path):
                        import shutil
                        shutil.copy2(local_video_path, downloaded_video_path)
                        logger.info(f"📁 Copied local video file: {local_video_path}")
                    elif video_url:
                        response = requests.get(video_url)
                        response.raise_for_status()
                        with open(downloaded_video_path, "wb") as f:
                            f.write(response.content)
                        logger.info(f"📥 Downloaded video from URL for job {job_id}")
                    else:
                        return {"success": False, "error": "No valid video source available"}
            else:
                logger.info(f"📁 Video already exists: {downloaded_video_path}")
            
            final_video_path = output_dir_path / f"final_video_{job_id}.mp4"
            
            cmd = ["ffmpeg", "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            
            cmd.extend(["-i", str(downloaded_video_path), "-i", audio_path])
            
            # Add instrument audio if available
            if instrument_audio_url:
                instrument_path = output_dir_path / "instrument_audio.mp3"
                import requests
                response = requests.get(instrument_audio_url)
                response.raise_for_status()
                with open(instrument_path, "wb") as f:
                    f.write(response.content)
                
                cmd.extend(["-i", str(instrument_path)])
                cmd.extend([
                    "-filter_complex", f"[1:a]volume=2.0[dub];[2:a]volume={INSTRUMENT_DEFAULT_VOLUME}[inst];[dub][inst]amix=inputs=2:duration=longest[out]",
                    "-map", "0:v", "-map", "[out]"
                ])
            else:
                cmd.extend(["-map", "0:v", "-map", "1:a", "-filter:a", "volume=2.0"])
            
            cmd.extend([
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(final_video_path)
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {"success": False, "error": "Video processing failed"}
            
            # Upload video to R2
            video_r2_key = self.r2_storage.generate_file_path(job_id, "processed", f"final_video_{job_id}.mp4")
            upload_result = self.r2_storage.upload_file(str(final_video_path), video_r2_key, "video/mp4")
            
            if upload_result["success"]:
                logger.info(f"✅ Video processed and uploaded for job {job_id}")
                return {
                    "success": True,
                    "video_url": upload_result["url"],
                    "video_filename": f"final_video_{job_id}.mp4",
                    "local_path": str(final_video_path)
                }
            else:
                return {"success": False, "error": "Failed to upload video to R2"}
            
        except Exception as e:
            logger.error(f"Video processing failed for job {job_id}: {e}")
            return {"success": False, "error": str(e)}

    def _upload_and_finalize(self, job_id: str, process_temp_dir: str, final_audio_path: str, video_result: dict = None) -> dict:
        """Upload files to R2 and return final results"""
        self._update_phase_progress(job_id, "upload", 0.0, "Uploading and finalizing...")
        
        # Skip all video uploads during folder upload since video is already processed and associated
        exclude_files = []
        import os
        for filename in os.listdir(process_temp_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                exclude_files.append(filename)
        
        if exclude_files:
            logger.info(f"📎 Skipping video files during folder upload: {exclude_files}")
        
        # Upload directory to R2, excluding all video files
        folder_upload_result, manifest_url, manifest_key = upload_process_dir_to_r2(
            job_id, process_temp_dir, self.r2_storage, exclude_files=exclude_files
        )
        
        
        
        logger.info("Dubbed processing completed successfully")
        
        # Prepare result with video data if available
        result_url = None
        video_upload = None
        video_error = None
        
        if video_result:
            if video_result.get("success"):
                result_url = video_result.get("video_url")
                video_upload = {
                    "success": True,
                    "url": video_result.get("video_url"),
                    "filename": video_result.get("video_filename")
                }
                logger.info(f"✅ Video result included for job {job_id}: {result_url}")
            elif not video_result.get("skipped"):
                video_error = video_result.get("error")
                logger.warning(f"⚠️ Video creation failed for job {job_id}: {video_error}")
        
        return {
            "success": True,
            "job_id": job_id,
            "result_url": result_url,  # Video result URL if created
            "result_urls": {},
            "folder_upload": folder_upload_result,
            "manifest_url": manifest_url,
            "manifest_key": manifest_key,
            "video_upload": video_upload,
            "video_error": video_error
        }
    


def get_simple_dubbed_api() -> SimpleDubbedAPI:
    """Create fresh SimpleDubbedAPI instance for multi-worker safety"""
    return SimpleDubbedAPI()
