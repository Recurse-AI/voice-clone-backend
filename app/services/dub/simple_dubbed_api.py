import os
import json
import logging
import time
import threading
import re
import numpy as np
from typing import Optional, Dict, Any

from app.config.settings import settings
from app.services.language_service import language_service
from app.services.openai_service import OpenAIService
from .video_processor import VideoProcessor
from .whisperx_transcription import get_whisperx_transcription_service
from .fish_speech_service import get_fish_speech_service
from app.services.r2_service import R2Service   
from .manifest_service import (
    build_manifest,
    save_manifest_to_dir,
    upload_process_dir_to_r2,
)
from .audio_utils import AudioUtils
from app.services.simple_status_service import status_service, JobStatus

logger = logging.getLogger(__name__)

_simple_dubbed_api_instance = None
_api_lock = threading.Lock()
def smart_chunk(text: str, chunk_size: int = 200, min_size: int = 180) -> list[str]:
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    sentence_break_re = re.compile(r"(?<=[\.\!\?。！？؟؛…])")
    sentences = sentence_break_re.split(text)

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size or len(current) < min_size:
            current += sent
        else:
            chunks.append(current.strip())
            current = sent
    if current.strip():
        chunks.append(current.strip())

    final_chunks: list[str] = []
    for ch in chunks:
        if len(ch) <= chunk_size:
            final_chunks.append(ch)
            continue

        words = ch.split()
        curr_chunk = words[0]
        for word in words[1:]:
            if len(curr_chunk) + 1 + len(word) > chunk_size:
                final_chunks.append(curr_chunk)
                curr_chunk = word
            else:
                curr_chunk += " " + word
        if curr_chunk:
            final_chunks.append(curr_chunk)

    return [c for c in final_chunks if c]



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
        # Only create service instances, don't load models
        # Service workers will handle actual model loading
        self.transcription_service = get_whisperx_transcription_service()
        self.fish_speech = get_fish_speech_service()
        self._r2_storage = None
    
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
    
    def dub_text_batch(self, segments: list, target_language: str = "English", batch_size: int = 10, job_id: str = None) -> list:
        openai_service = OpenAIService()
        return openai_service.translate_dubbing_batch(segments, target_language, batch_size)



    def process_dubbed_audio(self, job_id: str, target_language: str,
                           source_video_language: str = None,
                           output_dir: str = None, review_mode: bool = False,
                           manifest_override: Optional[Dict[str, Any]] = None,
                           separation_urls: Optional[Dict[str, str]] = None,
                           video_subtitle: bool = False) -> dict:
        """
        Complete dubbed audio processing with clean, modular approach.
        Always generates SRT file. No audio mixing or conditional processing.
        """
        # 1. Validate and normalize language
        if not language_service.is_dubbing_supported(target_language):
            supported_langs = language_service.get_supported_dubbing_languages()
            error_msg = f"Unsupported target language: {target_language}. Supported languages: {', '.join(sorted(supported_langs))}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        target_language_code = language_service.normalize_language_input(target_language)
        logger.info(f"Processing with target language: {target_language} -> {target_language_code}")

        try:
            # 2. Use provided output directory (already created by caller)
            process_temp_dir = output_dir

            # Initialize vocal and instrument audio URLs
            vocal_audio_url = None
            instrument_audio_url = None

            # Get transcription data
            raw_sentences, transcript_id = self._get_transcription_data(
                job_id, manifest_override, process_temp_dir, source_video_language, video_subtitle
            )
            
            
            # Transcription complete, bump to 60% before dubbing starts (ensure 46–60% range used)
            self._update_status(
                job_id,
                JobStatus.PROCESSING,
                60,
                {"message": "Transcription complete - starting dubbing", "phase": "dubbing"}
            )

            # Process text dubbing and voice cloning
            dubbed_segments = self._process_dubbing_and_cloning(
                job_id, raw_sentences, target_language, manifest_override, review_mode, process_temp_dir
            )
            
            # Handle URLs based on whether this is redub or original dub (before review mode check)
            if manifest_override:
                # For redub: use existing manifest which already has URLs
                # URLs already preserved from original manifest
                vocal_audio_url = manifest_override.get("vocal_audio_url")
                instrument_audio_url = manifest_override.get("instrument_audio_url")
            else:
                # For original dub: get URLs from separation results
                if separation_urls:
                    vocal_audio_url = separation_urls.get("vocal_audio")
                    instrument_audio_url = separation_urls.get("instrument_audio")

            # Validate that we have vocal URL (critical for resume/redub)
            if not vocal_audio_url:
                logger.error(f"No vocal audio URL available for {job_id} - resume/redub will fail")

            if review_mode:
                self._update_phase_progress(job_id, "review_prep", 0.5, "Preparing segments for human review")

                # Build manifest for review mode
                if manifest_override:
                    manifest = manifest_override.copy()
                    manifest["segments"] = dubbed_segments
                    manifest["target_language"] = target_language  # Update target language for redub
                else:
                    # Build manifest from scratch with separation URLs
                    manifest = build_manifest(job_id, transcript_id, target_language, dubbed_segments,
                                            vocal_audio_url, instrument_audio_url)

                # Save manifest to disk (both redub and original dub cases)
                manifest_path = save_manifest_to_dir(manifest, process_temp_dir, job_id)

                # For review mode, exclude vocal/instrument files since they already have URLs
                exclude_files = []
                if vocal_audio_url:
                    exclude_files.append(f"vocal_{job_id}.wav")
                if instrument_audio_url:
                    exclude_files.append(f"instrument_{job_id}.wav")

                folder_upload_result, manifest_url, manifest_key = upload_process_dir_to_r2(
                    job_id, process_temp_dir, self.r2_storage, exclude_files=exclude_files
                )


                # Ensure manifest_url is available before setting awaiting_review status
                if not manifest_url:
                    logger.error(f"Failed to get manifest URL for review mode job {job_id}")
                    return {"success": False, "error": "Failed to generate manifest URL for review"}
                
                # Charge 75% credits when ready for review
                self._charge_review_credits(job_id)

                # Now set awaiting_review status with manifest details using phase system
                self._update_status(
                    job_id,
                    JobStatus.AWAITING_REVIEW,
                    80,
                    {
                        "message": "Awaiting human review - Please review dubbed text",
                        "segments_manifest_url": manifest_url,
                        "segments_manifest_key": manifest_key,
                        "segments_count": len(dubbed_segments),
                        "transcript_id": transcript_id
                    })

                # Schedule auto cleanup after review window (1h)
                try:
                    from app.utils.cleanup_utils import cleanup_utils
                    cleanup_utils.schedule_auto_cleanup(job_id, delay_minutes=60)
                except Exception:
                    pass
                logger.info(f"Review mode: preserving temp directory for segments access: {process_temp_dir}")
                return {
                    "success": True,
                    "job_id": job_id,
                    "review": {
                        "segments_manifest_url": manifest_url,
                        "segments_manifest_key": manifest_key,
                        "segments_count": len(dubbed_segments),
                        "transcript_id": transcript_id
                    },
                    "folder_upload": folder_upload_result
                }

            
            # Generate final audio and files
            return self._generate_final_output(
                job_id, dubbed_segments, process_temp_dir,
                target_language, transcript_id, vocal_audio_url, instrument_audio_url
            )
        except Exception as e:
            logger.error(f"Dubbed processing failed: {str(e)}")
            # Clean up temp directory on exception
            if locals().get("process_temp_dir"):
                AudioUtils.remove_temp_dir(folder_path=locals().get("process_temp_dir"))
            return {"success": False, "error": str(e)}
    
    def _voice_clone_segment(self, dubbed_text: str, reference_audio_path: str, segment_id: str, original_text: str = "", job_id: str = None, process_temp_dir: str = None) -> Optional[Dict[str, Any]]:
        """Voice clone dubbed text using AI voice cloning service with reference audio and transcript_id in filename"""
        try:
            if not reference_audio_path:
                logger.warning(f"No reference audio path provided for {segment_id}")
                return None
            if not os.path.exists(reference_audio_path):
                logger.warning(f"Reference audio file not found: {reference_audio_path} for {segment_id}")
                return None
            import time
            segment_start_time = time.time()
            logger.info(f"🎯 Voice cloning {segment_id} starting at {time.strftime('%H:%M:%S')} using reference: {reference_audio_path}")
            import soundfile as sf
            import io
            audio_data, sample_rate = sf.read(reference_audio_path)
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Limit reference audio length based on configuration

            max_ref_seconds = settings.MAX_REFERENCE_SECONDS
            total_seconds = len(audio_data) / sample_rate
            if total_seconds > max_ref_seconds:
                # take the beginning slice of length max_ref_seconds for better alignment
                end_sample = int(max_ref_seconds * sample_rate)
                audio_data = audio_data[:end_sample]

            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            reference_audio_bytes = buffer.getvalue()
            if not reference_audio_bytes:
                logger.error(f"Failed to load reference audio: {reference_audio_path}")
                return None
            # Extract segment index from segment_id (seg_001 -> 0)
            segment_index = int(segment_id.split('_')[1]) - 1
            cloned_filename = f"cloned_{job_id}_{segment_index:03d}.wav"
            cloned_path = os.path.join(process_temp_dir, cloned_filename).replace('\\', '/')
            # Split dubbed text into optimized chunks for better GPU utilization  
            text_chunks = smart_chunk(dubbed_text, chunk_size=settings.FISH_SPEECH_CHUNK_SIZE, min_size=150)
            audio_chunks = []
            sample_rate_out = None
            for chunk in text_chunks:
                import time
                chunk_start = time.time()
                
                result = self.fish_speech.generate_with_reference_audio(
                    text=chunk,
                    reference_audio_bytes=reference_audio_bytes,
                    reference_text=original_text or "Reference audio",
                    max_new_tokens=1024,
                    top_p=0.6,
                    repetition_penalty=1.05,
                    temperature=0.6,
                    chunk_length=settings.FISH_SPEECH_CHUNK_SIZE,
                    job_id=job_id
                )

                chunk_time = time.time() - chunk_start
                logger.info(f"Chunk generation took {chunk_time:.2f}s for text: {chunk[:30]}...")

                if result.get("success"):
                    import soundfile as sf
                    import io

                    # Use service worker output directly
                    if "output_path" in result and result["output_path"]:
                        audio, sample_rate = sf.read(result["output_path"])
                        # Clean up temp file immediately
                        os.remove(result["output_path"])

                    if len(audio.shape) > 1:
                        audio = audio[:, 0]
                    audio_chunks.append(audio)
                    sample_rate_out = sample_rate
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
                import soundfile as sf
                buffer = io.BytesIO()
                sf.write(buffer, final_audio, sample_rate_out, format='WAV')
                with open(cloned_path, "wb") as f:
                    f.write(buffer.getvalue())
                duration_ms = int(len(final_audio) / sample_rate_out * 1000)
                total_time = time.time() - segment_start_time
                logger.info(f"✅ Voice cloning {segment_id} completed in {total_time:.2f}s at {time.strftime('%H:%M:%S')}")
                return {"path": cloned_path, "duration_ms": duration_ms}
            return None
                
        except Exception as e:
            logger.error(f"Voice cloning error for {segment_id}: {str(e)}")
            return None
    
    def _process_voice_cloning_sequential(self, segments_data: list, job_id: str, process_temp_dir: str) -> list:
        """Process voice cloning one segment at a time for better GPU stability"""
        import time
        
        results = []
        total_segments = len(segments_data)
        
        logger.info(f"🎯 Processing {total_segments} segments sequentially for stable GPU usage")
        
        for i, data in enumerate(segments_data):
            try:
                segment_start = time.time()
                
                # Add small delay between segments to control GPU utilization
                if i > 0:  # Skip delay for first segment
                    time.sleep(0.2)  # 200ms breathing room for GPU
                
                result = self._voice_clone_segment(
                    data["dubbed_text"], 
                    data["original_audio_path"], 
                    data["seg_id"], 
                    data["original_text"], 
                    job_id=job_id, 
                    process_temp_dir=process_temp_dir
                )
                
                segment_time = time.time() - segment_start
                if result:
                    logger.info(f"✅ Segment {data['seg_id']} ({i+1}/{total_segments}) cloned in {segment_time:.2f}s")
                else:
                    logger.error(f"❌ Segment {data['seg_id']} ({i+1}/{total_segments}) failed after {segment_time:.2f}s")
                
                results.append(result)
                
                # Update progress after each segment completion
                completed_segments = i + 1
                progress_percent = (completed_segments / total_segments) * 0.9  # 90% of voice_cloning phase
                progress_percent += 0.1  # Add initial 10% for segmentation
                
                message = f"Voice cloning: {completed_segments}/{total_segments} segments completed"
                try:
                    self._update_phase_progress(job_id, "voice_cloning", progress_percent, message)
                except Exception as progress_error:
                    logger.warning(f"Failed to update progress: {progress_error}")
                
            except Exception as e:
                logger.error(f"💥 Segment {data.get('seg_id', 'unknown')} failed: {e}")
                results.append(None)
                
                # Update progress even for failed segments
                completed_segments = i + 1
                progress_percent = (completed_segments / total_segments) * 0.9  # 90% of voice_cloning phase
                progress_percent += 0.1  # Add initial 10% for segmentation
                
                message = f"Voice cloning: {completed_segments}/{total_segments} segments processed (some failed)"
                try:
                    self._update_phase_progress(job_id, "voice_cloning", progress_percent, message)
                except Exception as progress_error:
                    logger.warning(f"Failed to update progress: {progress_error}")
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"🎯 Sequential processing completed: {successful}/{total_segments} segments successful")
        
        # Final progress update
        final_message = f"Voice cloning completed: {successful}/{total_segments} segments successful"
        try:
            self._update_phase_progress(job_id, "voice_cloning", 1.0, final_message)
        except Exception as progress_error:
            logger.warning(f"Failed to update final progress: {progress_error}")
        
        return results
    
    
    def _get_transcription_data(self, job_id: str, manifest_override: Optional[Dict[str, Any]],
                               process_temp_dir: str, source_video_language: str, video_subtitle: bool = False) -> tuple:
        """Get transcription data either from manifest override, SRT file, or by transcribing audio"""
        
        transcript_id = None
        raw_sentences = []
        
        if manifest_override:
            # Use existing transcription data - no progress update needed
            logger.info("Using manifest override data - will re-segment vocal audio")
            for idx, seg in enumerate(manifest_override.get("segments", [])):
                raw_sentences.append({
                    "text": seg.get("original_text", ""),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "id": seg.get("id", f"sentence_{idx}")
                })
            transcript_id = manifest_override.get("transcript_id")
            
            self._download_missing_files(job_id, manifest_override, process_temp_dir)
            
        else:
            # Check if we should use SRT file instead of WhisperX
            if video_subtitle:
                # Use SRT file for transcription
                self._update_phase_progress(job_id, "transcription", 0.0, "Using provided SRT file for transcription")
                logger.info("Using SRT file instead of WhisperX transcription")
                
                srt_file_path = os.path.join(process_temp_dir, f"{job_id}.srt")
                if not os.path.exists(srt_file_path):
                    raise Exception(f"SRT file not found at {srt_file_path}. Make sure subtitle file was uploaded.")
                
                # Parse SRT file
                from app.utils.srt_parser import parse_srt_to_whisperx_format
                transcription_result = parse_srt_to_whisperx_format(srt_file_path)
                
                if not transcription_result["success"]:
                    logger.error(f"SRT parsing failed: {transcription_result.get('error')}")
                    raise Exception(transcription_result.get("error", "SRT parsing failed"))
                
                # Convert to raw_sentences format
                raw_sentences = []
                for i, sentence in enumerate(transcription_result["sentences"]):
                    raw_sentences.append({
                        "text": sentence["text"],
                        "start": sentence["start"],
                        "end": sentence["end"],
                        "id": sentence["id"]
                    })
                
                transcript_id = f"srt_{int(time.time())}"
                logger.info(f"Found {len(raw_sentences)} segments from SRT file")
                self._update_phase_progress(job_id, "transcription", 1.0, "SRT transcription completed")
                
            else:
                # Fresh transcription - only transcribe, don't segment yet
                self._update_phase_progress(job_id, "transcription", 0.0, "Starting audio transcription")
                logger.info("Starting fresh transcription (no segmentation)")

                # Get vocal audio path
                vocal_file_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")

                if not os.path.exists(vocal_file_path):
                    raise Exception(f"Vocal file not found at {vocal_file_path}. Make sure separation completed successfully.")

                # Use service worker transcription instead of direct transcription
                # This ensures WhisperX models are loaded in the appropriate worker process
                transcription_result = self.transcription_service._transcribe_via_service_worker(
                    vocal_file_path, source_video_language, job_id
                )

                if not transcription_result["success"]:
                    logger.error(f"Transcription failed: {transcription_result.get('error')}")
                    AudioUtils.remove_temp_dir(folder_path=process_temp_dir)
                    raise Exception(transcription_result.get("error", "Transcription failed"))

                # Convert to raw_sentences format (without output_path since no segmentation yet)
                raw_sentences = []
                for i, sentence in enumerate(transcription_result["sentences"]):
                    raw_sentences.append({
                        "text": sentence["text"],
                        "start": sentence["start"],
                        "end": sentence["end"],
                        "id": sentence["id"]
                    })

                transcript_id = f"whisperx_{int(time.time())}"
                logger.info(f"Found {len(raw_sentences)} raw sentences (segmentation deferred)")
                self._update_phase_progress(job_id, "transcription", 1.0, "Transcription completed")
        
        return raw_sentences, transcript_id

    def _segment_vocal_audio_before_cloning(self, job_id: str, raw_sentences: list, process_temp_dir: str) -> list:
        """Segment vocal audio just before voice cloning and return enhanced segments with output_path"""
        try:
            from .audio_utils import AudioUtils
            audio_utils = AudioUtils()

            # Get vocal audio path
            vocal_file_path = os.path.join(process_temp_dir, f"vocal_{job_id}.wav")

            if not os.path.exists(vocal_file_path):
                raise Exception(f"Vocal file not found at {vocal_file_path}")

            logger.info(f"Starting vocal audio segmentation for voice cloning: {len(raw_sentences)} segments")

            # Prepare segments for splitting
            segments_to_split = []
            for sentence in raw_sentences:
                if sentence.get("text", "").strip():  # Only segment non-empty text
                    segments_to_split.append({
                        "start": sentence["start"],  # Keep in ms (no conversion needed)
                        "end": sentence["end"],     # Keep in ms (no conversion needed)
                        "text": sentence["text"]
                    })

            # Split audio
            split_result = audio_utils.split_audio_by_timestamps(vocal_file_path, process_temp_dir, segments_to_split)
            if not split_result["success"]:
                raise Exception(f"Failed to split audio: {split_result['error']}")

            # Create enhanced segments with output_path
            enhanced_segments = []
            split_files = split_result.get("split_files", [])

            for i, (sentence, split_file) in enumerate(zip(raw_sentences, split_files)):
                if sentence.get("text", "").strip():  # Only include segments with text
                    enhanced_segment = sentence.copy()
                    enhanced_segment.update({
                        "output_path": split_file["output_path"],
                        "duration_ms": split_file["duration_ms"],
                        "segment_index": i
                    })
                    enhanced_segments.append(enhanced_segment)

            logger.info(f"Successfully segmented {len(enhanced_segments)} vocal audio segments")
            return enhanced_segments

        except Exception as e:
            logger.error(f"Vocal audio segmentation failed: {e}")
            raise Exception(f"Audio segmentation failed: {str(e)}")

    def _prepare_dubbing_segments(self, raw_sentences: list) -> tuple:
        segments_to_dub = []
        segment_indices = []
        for i, segment in enumerate(raw_sentences):
            original_text = segment.get("text", "")
            start_ms = segment.get("start", 0)
            end_ms = segment.get("end", 0)
            if original_text.strip():
                segments_to_dub.append({
                    "text": original_text,
                    "duration_ms": end_ms - start_ms,
                    "id": f"seg_{i+1:03d}"
                })
                segment_indices.append(i)
        return segments_to_dub, segment_indices

    def _get_edited_text_map(self, manifest_override: Optional[Dict[str, Any]], target_language: str) -> dict:
        if not manifest_override:
            return {}
        
        is_redub = bool(manifest_override.get("redub_target_language"))
        manifest_target_lang = manifest_override.get("target_language")
        current_target_lang = language_service.normalize_language_input(target_language)
        
        if is_redub and manifest_target_lang:
            manifest_target_lang = language_service.normalize_language_input(manifest_target_lang)
            if manifest_target_lang != current_target_lang:
                return {}
        
        edited_map = {}
        for seg in manifest_override.get("segments", []):
            if seg.get("id") and seg.get("dubbed_text"):
                edited_map[seg["id"]] = seg["dubbed_text"]
        return edited_map

    def _download_missing_files(self, job_id: str, manifest_override: Dict[str, Any], process_temp_dir: str):
        if not manifest_override:
            return

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
                except Exception as e:
                    logger.warning(f"Failed to download {filename} for job {job_id}: {e}")

    def _process_voice_cloning(self, job_id: str, enhanced_sentences: list, dubbed_texts: list, 
                              edited_map: dict, review_mode: bool, process_temp_dir: str) -> list:
        if not review_mode:
            self._update_status(job_id, JobStatus.PROCESSING, 80, {
                "message": "Starting AI voice cloning",
                "phase": "voice_cloning",
                "sub_progress": 0.0
            })

        dubbed_segments = []
        dub_idx = 0
        cloneable_count = sum(1 for s in enhanced_sentences if s.get("text", "").strip()) if not review_mode else 0
        
        # Prepare all segments data
        segments_data = []
        for i, segment in enumerate(enhanced_sentences):
            seg_id = f"seg_{i+1:03d}"
            start_ms = segment.get("start", 0)
            end_ms = segment.get("end", 0)
            original_text = segment.get("text", "")
            original_audio_path = segment.get("output_path", "")
            
            if not original_text.strip():
                continue
                
            dubbed_text = dubbed_texts[dub_idx] if dub_idx < len(dubbed_texts) else ""
            dub_idx += 1
            
            if seg_id in edited_map:
                dubbed_text = edited_map[seg_id]
            
            segments_data.append({
                "seg_id": seg_id, "global_idx": i, "start_ms": start_ms,
                "end_ms": end_ms, "original_text": original_text, "dubbed_text": dubbed_text,
                "original_audio_path": original_audio_path
            })
        
        if not review_mode and segments_data:
            # Process sequentially instead of batch
            results = self._process_voice_cloning_sequential(segments_data, job_id, process_temp_dir)
            
            for i, data in enumerate(segments_data):
                result = results[i] if i < len(results) else None
                cloned_audio_path = result.get("path") if result else None
                cloned_duration_ms = result.get("duration_ms", data["end_ms"] - data["start_ms"]) if result else data["end_ms"] - data["start_ms"]
                
                segment_json = self._create_segment_info(
                    data["seg_id"], data["global_idx"], data["start_ms"], cloned_duration_ms,
                    data["original_text"], data["dubbed_text"], data["original_audio_path"],
                    cloned_audio_path, job_id, process_temp_dir
                )
                dubbed_segments.append(segment_json)
                
                # Update progress
                if cloneable_count > 0:
                    completed_clones = i + 1
                    if self._should_update_progress(completed_clones, cloneable_count):
                        try:
                            sub_progress = completed_clones / cloneable_count
                            self._update_phase_progress(job_id, "voice_cloning", sub_progress,
                                f"Voice cloning: {completed_clones}/{cloneable_count} segments")
                        except Exception:
                            pass
        else:
            # Review mode - no cloning
            for data in segments_data:
                segment_json = self._create_segment_info(
                    data["seg_id"], data["global_idx"], data["start_ms"], data["end_ms"] - data["start_ms"],
                    data["original_text"], data["dubbed_text"], data["original_audio_path"],
                    None, job_id, process_temp_dir
                )
                dubbed_segments.append(segment_json)
        
        return dubbed_segments

    def _process_dubbing_and_cloning(self, job_id: str, raw_sentences: list, target_language: str,
                                     manifest_override: Optional[Dict[str, Any]], review_mode: bool, 
                                     process_temp_dir: str) -> list:
        segments_to_dub, segment_indices = self._prepare_dubbing_segments(raw_sentences)
        target_language_code = language_service.normalize_language_input(target_language)
        
        # Check if we have manifest with already dubbed texts (resume after review)
        edited_map = self._get_edited_text_map(manifest_override, target_language)
        
        if manifest_override and edited_map:
            # Use existing dubbed texts from manifest - no OpenAI call needed
            logger.info(f"✅ RESUME MODE: Using existing dubbed texts from manifest ({len(edited_map)} segments) - skipping OpenAI translation")
            dubbed_texts = []
            for i, segment in enumerate(segments_to_dub):
                seg_id = f"seg_{i+1:03d}"
                dubbed_text = edited_map.get(seg_id, segment.get("text", ""))
                dubbed_texts.append(dubbed_text)
            self._update_phase_progress(job_id, "dubbing", 1.0, "Using reviewed dubbed texts")
        else:
            # Fresh translation needed - call OpenAI
            self._update_phase_progress(job_id, "dubbing", 0.0, "Starting AI text translation with OpenAI")
            dubbed_texts = self.dub_text_batch(segments_to_dub, target_language_code, batch_size=15, job_id=job_id)
            self._update_phase_progress(job_id, "dubbing", 1.0, "Reviewing and editing with AI")
        
        enhanced_sentences = raw_sentences
        if not review_mode:
            self._update_phase_progress(job_id, "voice_cloning", 0.0, "Segmenting vocal audio for voice cloning")
            enhanced_sentences = self._segment_vocal_audio_before_cloning(job_id, raw_sentences, process_temp_dir)
            self._update_phase_progress(job_id, "voice_cloning", 0.1, "Vocal audio segmentation completed")

        return self._process_voice_cloning(job_id, enhanced_sentences, dubbed_texts, edited_map, review_mode, process_temp_dir)
    
    def _create_segment_info(self, seg_id: str, segment_index: int, start_ms: int, cloned_duration_ms: int,
                           original_text: str, dubbed_text: str, original_audio_path: str, 
                           cloned_audio_path: str, job_id: str, process_temp_dir: str) -> dict:
        info_filename = f"segment_{job_id}_{segment_index:03d}_info.json"
        info_path = os.path.join(process_temp_dir, info_filename).replace('\\', '/')
        
        segment_json = {
            "id": seg_id,
            "segment_index": segment_index + 1,
            "start": start_ms,
            "end": start_ms + cloned_duration_ms,
            "duration_ms": cloned_duration_ms,
            "original_text": original_text,
            "dubbed_text": dubbed_text,
            "original_audio_file": f"segment_{segment_index:03d}.wav" if original_audio_path else None,
            "cloned_audio_file": f"cloned_{job_id}_{segment_index:03d}.wav" if cloned_audio_path else None,
            "voice_cloned": bool(cloned_audio_path),
            "original_audio_path": original_audio_path,
            "cloned_audio_path": cloned_audio_path,
        }
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(segment_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved segment info: {info_filename}")
        except Exception as e:
            logger.error(f"Failed to save JSON for {seg_id}: {str(e)}")
        
        return segment_json



    def _generate_final_output(self, job_id: str, dubbed_segments: list, process_temp_dir: str,
                              target_language: str, transcript_id: str,
                              vocal_audio_url: str = None, instrument_audio_url: str = None) -> dict:
        """Generate final audio, SRT file, and upload to R2"""
        
        
        self._update_phase_progress(job_id, "final_processing", 0.0, "Reconstructing final audio...")
        logger.info("Reconstructing final audio...")
        
        # Reconstruct final audio
        final_audio_path = self._reconstruct_final_audio(dubbed_segments, None, job_id=job_id, process_temp_dir=process_temp_dir)
        
        
        # Generate SRT file and finalize (combining multiple close steps)
        self._update_phase_progress(job_id, "final_processing", 0.5, "Finalizing output files...")
        
        subtitle_path = self._generate_srt_file(job_id, dubbed_segments, process_temp_dir)
        
        
        # Create process summary
        self._create_process_summary(job_id, dubbed_segments, final_audio_path, subtitle_path,
                                   process_temp_dir, target_language, transcript_id)
        
        # Save final manifest (for normal dub / redub lineage)
        try:
            manifest = build_manifest(job_id, transcript_id, target_language, dubbed_segments,
                                    vocal_audio_url, instrument_audio_url)
            save_manifest_to_dir(manifest, process_temp_dir, job_id)
        except Exception as e:
            logger.error(f"Failed to save manifest for job {job_id}: {e}")
            # Don't pass silently - this is critical for manifest availability

        # Upload to R2 and get results
        return self._upload_and_finalize(job_id, process_temp_dir, final_audio_path)
    
    def _generate_srt_file(self, job_id: str, dubbed_segments: list, process_temp_dir: str) -> str:
        """Generate SRT subtitle file"""
        logger.info("Generating SRT file...")
        

        processor = VideoProcessor(temp_dir=process_temp_dir)
        subtitle_data = []
        
        for seg in dubbed_segments:
            text = seg["dubbed_text"]
            start = seg["start"] / 1000.0
            end = seg["end"] / 1000.0

            # Use smart_chunk utility to create subtitle lines
            chunks = smart_chunk(text, chunk_size=60, min_size=40)
            total_chars = sum(len(c) for c in chunks) or 1

            char_count_before = 0
            for chunk in chunks:
                chunk_len = len(chunk)
                chunk_start = start + (end - start) * (char_count_before / total_chars)
                char_count_before += chunk_len
                chunk_end = start + (end - start) * (char_count_before / total_chars)
                subtitle_data.append({"start": chunk_start, "end": min(chunk_end, end), "text": chunk})
        
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
    
    def _upload_and_finalize(self, job_id: str, process_temp_dir: str, final_audio_path: str) -> dict:
        """Upload files to R2 and return final results"""
        self._update_phase_progress(job_id, "upload", 0.0, "Uploading and finalizing...")
        
        # Upload entire directory to R2
        folder_upload_result, manifest_url, manifest_key = upload_process_dir_to_r2(
            job_id, process_temp_dir, self.r2_storage
        )
        
        
        
        logger.info("Dubbed processing completed successfully")
        
        
        return {
            "success": True,
            "job_id": job_id,
            "result_url": None,  # No video result
            "result_urls": {},
            "folder_upload": folder_upload_result,
            "manifest_url": manifest_url,
            "manifest_key": manifest_key,
            "video_upload": None,
            "video_error": None
        }
    



    def _reconstruct_final_audio(self, segments: list, original_audio_path: str,
                               job_id: str = None, process_temp_dir: str = None) -> str:
        """Reconstruct final audio with MP3 optimization."""
        if not segments:
            return None

        try:
            import soundfile as sf

            # Get sample rate from original audio
            if original_audio_path and os.path.exists(original_audio_path):
                _, sample_rate = sf.read(original_audio_path, frames=1)
            else:
                sample_rate = 44100

            # Calculate exact duration from segments
            max_end_ms = max(s.get("end", 0) for s in segments)
            duration_samples = int((max_end_ms / 1000.0) * sample_rate)
            final_audio = np.zeros(duration_samples, dtype=np.float32)

            # Place segments
            for segment in segments:
                try:
                    cloned_path = segment.get("cloned_audio_path")
                    if not cloned_path or not os.path.exists(cloned_path):
                        continue

                    cloned_audio, _ = sf.read(cloned_path)
                    if len(cloned_audio.shape) > 1:
                        cloned_audio = np.mean(cloned_audio, axis=1)

                    start_ms = segment.get("start", 0)
                    end_ms = segment.get("end", 0)
                    if start_ms < 0 or end_ms <= start_ms:
                        continue

                    start_sample = int((start_ms / 1000.0) * sample_rate)
                    expected_samples = int(((end_ms - start_ms) / 1000.0) * sample_rate)

                    # Fit audio duration
                    if len(cloned_audio) > expected_samples:
                        cloned_audio = cloned_audio[:expected_samples]
                    elif len(cloned_audio) < expected_samples:
                        padding = expected_samples - len(cloned_audio)
                        cloned_audio = np.pad(cloned_audio, (0, padding), mode="constant")

                    # Normalize volume
                    max_val = np.max(np.abs(cloned_audio))
                    if max_val > 1e-10:
                        cloned_audio = cloned_audio / max_val

                    # Place in final audio
                    end_sample = start_sample + len(cloned_audio)
                    if start_sample >= 0 and end_sample <= len(final_audio):
                        final_audio[start_sample:end_sample] = cloned_audio

                except Exception:
                    continue

            # Save and compress
            final_path = os.path.join(process_temp_dir, f"final_{job_id}.wav")
            sf.write(final_path, final_audio, sample_rate)

            # MP3 compression
            import subprocess
            from app.utils.ffmpeg_helper import get_ffmpeg_path

            ffmpeg_path = get_ffmpeg_path()
            if ffmpeg_path:
                temp_mp3 = final_path.replace('.wav', '_temp.mp3')
                cmd = [ffmpeg_path, '-y', '-i', final_path, '-acodec', 'libmp3lame', '-ab', '320k', temp_mp3]

                if subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0:
                    import shutil
                    shutil.move(temp_mp3, final_path)
                elif os.path.exists(temp_mp3):
                    os.remove(temp_mp3)

            logger.info(f"Final audio: {final_path} ({len(final_audio)/sample_rate:.2f}s)")
            return final_path

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return None


def get_simple_dubbed_api() -> SimpleDubbedAPI:
    """Get or create global SimpleDubbedAPI instance (thread-safe)"""
    global _simple_dubbed_api_instance
    if _simple_dubbed_api_instance is None:
        with _api_lock:
            if _simple_dubbed_api_instance is None:
                _simple_dubbed_api_instance = SimpleDubbedAPI()
    return _simple_dubbed_api_instance
