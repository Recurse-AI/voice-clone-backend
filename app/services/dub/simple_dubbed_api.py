import os
import json
import logging
import time
from app.config.settings import settings
from .assembly_transcription import TranscriptionService
from .fish_speech_service import get_fish_speech_service
from app.services.r2_service import get_r2_service
from .manifest_service import (
    build_manifest,
    save_manifest_to_dir,
    upload_process_dir_to_r2,
    enrich_and_reupload_manifest_with_urls,
)
import numpy as np
from .audio_utils import AudioUtils
from typing import Optional, Dict, Any
from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus, JobType
import threading
import re

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

from app.services.language_service import language_service

class SimpleDubbedAPI:
    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.fish_speech = get_fish_speech_service()
        self._r2_storage = None
    
    @property
    def r2_storage(self):
        if self._r2_storage is None:
            self._r2_storage = get_r2_service()
        return self._r2_storage
    
    @property  
    def temp_dir(self):
        return settings.TEMP_DIR
    
    def _update_status_non_blocking(self, job_id: str, status: ProcessingStatus, progress: int, details: dict, job_type: str = "dub"):
        import asyncio
        manager = get_unified_status_manager()
        job_type_enum = JobType.DUB if job_type == "dub" else JobType.SEPARATION
        
        # Run async update in thread pool for non-blocking execution
        def run_update():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    manager.update_status(job_id, job_type_enum, status, progress, details)
                )
            finally:
                loop.close()
        
        import threading
        threading.Thread(target=run_update, daemon=True).start()
    
    def _check_cancellation(self, job_id: str) -> bool:
        from app.utils.shared_memory import is_job_cancelled
        if is_job_cancelled(job_id):
            self._update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", "error": "Job cancelled by user"
            }, "dub")
            return True
        return False
    
    def dub_text_batch(self, segments: list, target_language: str = "English", batch_size: int = 10, job_id: str = None) -> list:
        all_dubbed = []
        for i in range(0, len(segments), batch_size):
            if job_id and self._check_cancellation(job_id):
                return []
                
            batch = segments[i:i+batch_size]
            prompt_lines = []
            for idx, seg in enumerate(batch):
                prompt_lines.append(f"[{idx+1}] (Target duration: {seg['duration_ms']} ms) {seg['text']}")
            joined_texts = "\n".join(prompt_lines)
            
            system_prompt = (
                f"You are assisting in creating dubbing scripts for the Fish Audio OpenAudio-S1 TTS model.\n"
                f"Translate each input segment into {target_language} (keeping meaning accurate).\n"
                f"Constraints for every translated segment:\n"
                f"1. Try to match the target duration (given in ms) — if you need to lengthen, prefer inserting extra *spaces* between words rather than adding new words.\n"
                f"2. Use the correct alphabet/script for {target_language}; never mix English letters unless the original word is a proper noun or acronym.\n"
                f"3. You MAY optionally use the Fish-Audio emotion/tone markers like (excited), (sad), (whispering) etc. **only** when that better reflects the original intent. Place the marker at the very beginning of the sentence.\n"
                f"4. Do NOT add any explanatory text, numbering, or comments — only the final translated sentences.\n"
                f"5. Return the translated segments in the same order, separated by ||| on a single line."
            )
            
            user_prompt = (
                "Translate each segment below. Return only the translated segments, in order, separated by |||.\n"
                "Segments (with target duration):\n"
                f"{joined_texts}"
            )
            try:
                response = self.transcription_service.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048
                )
                output = response.choices[0].message.content.strip()
                all_dubbed.extend([seg.strip() for seg in output.split("|||")])
            except ConnectionResetError as e:
                logger.warning(f"Connection reset error during OpenAI call, retrying batch {i//batch_size + 1}: {e}")
                # Retry once with a shorter delay
                try:
                    time.sleep(1)
                    response = self.transcription_service.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=2048
                    )
                    output = response.choices[0].message.content.strip()
                    all_dubbed.extend([seg.strip() for seg in output.split("|||")])
                except Exception as retry_error:
                    logger.error(f"Failed to process batch {i//batch_size + 1} after retry: {retry_error}")
                    # Add placeholder translations to maintain segment count
                    all_dubbed.extend([f"[Translation Error for segment {j+1}]" for j in range(len(batch))])
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add placeholder translations to maintain segment count
                all_dubbed.extend([f"[Translation Error for segment {j+1}]" for j in range(len(batch))])
        return all_dubbed

    def _setup_processing_directory(self, job_id: str, output_dir: str = None) -> str:
        """Setup and ensure processing directory exists."""
        if not output_dir:
            output_dir = os.path.join(self.temp_dir, f"dub_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_dubbed_audio(self, audio_url: str, job_id: str, target_language: str, 
                           speakers_count: int = 1, source_video_language: str = None, 
                           output_dir: str = None, review_mode: bool = False, 
                           manifest_override: Optional[Dict[str, Any]] = None) -> dict:
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
            # 2. Setup processing environment
            process_temp_dir = self._setup_processing_directory(job_id, output_dir)
            
            # 🛡️ Check cancellation before transcription
            if self._check_cancellation(job_id):
                return {"success": False, "error": "Job cancelled by user"}
            
            # Get transcription data
            raw_sentences, transcript_id = self._get_transcription_data(
                job_id, audio_url, manifest_override, process_temp_dir, speakers_count, source_video_language
            )
            
            # 🛡️ Check cancellation before dubbing and voice cloning
            if self._check_cancellation(job_id):
                return {"success": False, "error": "Job cancelled by user"}
            
            # Process text dubbing and voice cloning
            dubbed_segments = self._process_dubbing_and_cloning(
                job_id, raw_sentences, target_language, manifest_override, review_mode, process_temp_dir
            )
            
            if review_mode:
                # Preparing review files - update to 80%
                self._update_status_non_blocking(
                    job_id,
                    ProcessingStatus.PROCESSING,
                    80,
                    {"message": "Preparing review files"},
                    "dub",
                )
                logger.info("Preparing review files for human review...")
                # Build and save manifest
                manifest = build_manifest(job_id, transcript_id, target_language, dubbed_segments)
                manifest_path = save_manifest_to_dir(manifest, process_temp_dir, job_id)
                folder_upload_result, manifest_url, manifest_key = upload_process_dir_to_r2(job_id, process_temp_dir, self.r2_storage)

                # best-effort enrich and re-upload
                try:
                    files_map = {}
                    if isinstance(folder_upload_result, dict):
                        for fname, res in folder_upload_result.items():
                            if res.get("success"):
                                files_map[fname] = {"url": res.get("url"), "r2_key": res.get("r2_key")}
                    enriched_url = enrich_and_reupload_manifest_with_urls(manifest, manifest_path, files_map, manifest_key)
                    if enriched_url:
                        manifest_url = enriched_url
                except Exception:
                    pass

                # Ensure manifest_url is available before returning
                if not manifest_url:
                    logger.error(f"Failed to get manifest URL for review mode job {job_id}")
                    return {"success": False, "error": "Failed to generate manifest URL for review"}

                # Don't remove temp directory in review mode - needed for segments access
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

            # 🛡️ Check cancellation before final output generation
            if self._check_cancellation(job_id):
                return {"success": False, "error": "Job cancelled by user"}
            
            # Generate final audio and files
            return self._generate_final_output(
                job_id, dubbed_segments, process_temp_dir, 
                target_language, audio_url, speakers_count, transcript_id
            )
        except Exception as e:
            logger.error(f"Dubbed processing failed: {str(e)}")
            # Clean up temp directory on exception
            if locals().get("process_temp_dir"):
                AudioUtils.remove_temp_dir(folder_path=locals().get("process_temp_dir"))
            return {"success": False, "error": str(e)}
    
    def _voice_clone_segment(self, dubbed_text: str, reference_audio_path: str, segment_id: str, original_text: str = "", speaker_label: str = None, job_id: str = None, process_temp_dir: str = None) -> Optional[Dict[str, Any]]:
        """Voice clone dubbed text using FishSpeechService with reference audio and transcript_id in filename"""
        try:
            if not reference_audio_path:
                logger.warning(f"No reference audio path provided for {segment_id}")
                return None
            if not os.path.exists(reference_audio_path):
                logger.warning(f"Reference audio file not found: {reference_audio_path} for {segment_id}")
                return None
            logger.info(f"Voice cloning {segment_id} using reference: {reference_audio_path}")
            import soundfile as sf
            import io
            audio_data, sample_rate = sf.read(reference_audio_path)
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Limit reference audio length based on configuration
            from app.config.settings import settings
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
            # Split dubbed text into manageable chunks
            text_chunks = smart_chunk(dubbed_text, chunk_size=200, min_size=180)
            audio_chunks = []
            sample_rate_out = None
            seed_val = None
            if speaker_label:
                seed_val = abs(hash(speaker_label)) % (2**32)
            for chunk in text_chunks:
                result = self.fish_speech.generate_with_reference_audio(
                    text=chunk,
                    reference_audio_bytes=reference_audio_bytes,
                    reference_text=original_text or "Reference audio",
                    max_new_tokens=2048,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    temperature=0.7,
                    seed=seed_val,
                    chunk_length=200
                )
                if result.get("success"):
                    import soundfile as sf
                    import io
                    buffer = io.BytesIO(result["audio_data"])
                    audio, sample_rate = sf.read(buffer)
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
                return {"path": cloned_path, "duration_ms": duration_ms}
            return None
                
        except Exception as e:
            logger.error(f"Voice cloning error for {segment_id}: {str(e)}")
            return None
    
    def _get_transcription_data(self, job_id: str, audio_url: str, manifest_override: Optional[Dict[str, Any]], 
                               process_temp_dir: str, speakers_count: int, source_video_language: str) -> tuple:
        """Get transcription data either from manifest override or by transcribing audio"""
        self._update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 45, {"message": "Transcribing audio..."}, "dub")
        logger.info("Starting transcription...")
        
        transcript_id = None
        raw_sentences = []
        
        if manifest_override:
            logger.info("Using manifest override data instead of re-transcribing")
            for idx, seg in enumerate(manifest_override.get("segments", [])):
                raw_sentences.append({
                    "text": seg.get("original_text", ""),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "speaker_label": seg.get("speaker_label"),
                    "output_path": os.path.join(process_temp_dir, seg.get("original_audio_file", f"segment_{idx:03d}.wav")).replace('\\', '/')
                })
            transcript_id = manifest_override.get("transcript_id")
            
            # Download original segment audios to temp dir if URLs available
            self._download_segment_audios(manifest_override, process_temp_dir)
            
        else:
            logger.info("Starting fresh transcription")
            sentences_result = self.transcription_service.get_sentences_and_split_audio(
                audio_url=audio_url,
                output_dir=process_temp_dir,
                speakers_count=speakers_count,
                source_video_language=source_video_language,
                job_id=job_id
            )
            
            if not sentences_result["success"]:
                logger.error(f"Transcription failed: {sentences_result.get('error')}")
                AudioUtils.remove_temp_dir(folder_path=process_temp_dir)
                raise Exception(sentences_result.get("error", "Transcription failed"))
                
            raw_sentences = sentences_result["segments"]
            transcript_id = sentences_result.get("transcript_id")
            logger.info(f"Found {len(raw_sentences)} sentence segments")
        
        return raw_sentences, transcript_id
    
    def _process_dubbing_and_cloning(self, job_id: str, raw_sentences: list, target_language: str,
                                     manifest_override: Optional[Dict[str, Any]], review_mode: bool, 
                                     process_temp_dir: str) -> list:
        """Process text dubbing and voice cloning for all segments"""
        self._update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 55, {"message": "Batch dubbing text..."}, "dub")
        logger.info("Starting batch dubbing...")
        
        # Prepare texts for batch dubbing
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
        
        # Normalize target language for consistent processing
        target_language_code = language_service.normalize_language_input(target_language)
        dubbed_texts = self.dub_text_batch(segments_to_dub, target_language_code, batch_size=15, job_id=job_id)
        
        # OpenAI dubbing completed - update to 75%
        self._update_status_non_blocking(
            job_id,
            ProcessingStatus.PROCESSING,
            75,
            {"message": "Reviewing and editing with AI"},
            "dub",
        )
        logger.info("OpenAI dubbing completed, starting voice processing...")
        
        # Apply edited text overrides if present (only for same target language)
        edited_map = {}
        if manifest_override:
            # Check if this is a redub with language change
            is_redub = bool(manifest_override.get("redub_target_language"))
            manifest_target_lang = manifest_override.get("target_language")
            current_target_lang = language_service.normalize_language_input(target_language)
            
            if is_redub:
                # For redub: compare original manifest language with current target
                if manifest_target_lang:
                    manifest_target_lang = language_service.normalize_language_input(manifest_target_lang)
                    
                    # Only apply overrides if it's same language redub (corrections)
                    if manifest_target_lang == current_target_lang:
                        for seg in manifest_override.get("segments", []):
                            if seg.get("id") and seg.get("dubbed_text"):
                                edited_map[seg["id"]] = seg["dubbed_text"]
            else:
                # Fresh job or continuation: always apply any edited text
                for seg in manifest_override.get("segments", []):
                    if seg.get("id") and seg.get("dubbed_text"):
                        edited_map[seg["id"]] = seg["dubbed_text"]
        
        # Start voice cloning phase at 80% (after review preparation)
        self._update_status_non_blocking(
            job_id,
            ProcessingStatus.PROCESSING,
            80,
            {"message": "Voice cloning and reconstructing final audio"},
            "dub",
        )
        logger.info("Starting voice cloning...")
        
        # Process voice cloning for each segment
        dubbed_segments = []
        dub_idx = 0

        # Calculate dynamic progress range for voice cloning updates (80..89)
        cloneable_count = sum(1 for s in raw_sentences if s.get("text", "").strip()) if not review_mode else 0
        completed_clones = 0
        last_progress_update = 80  # Track last progress to avoid frequent updates

        for i, segment in enumerate(raw_sentences):
            # 🛡️ Check cancellation in voice cloning loop
            if self._check_cancellation(job_id):
                return []  # Return empty list to stop processing
            
            seg_id = f"seg_{i+1:03d}"
            start_ms = segment.get("start", 0)
            end_ms = segment.get("end", 0)
            original_text = segment.get("text", "")
            speaker_label = segment.get("speaker_label")
            original_audio_path = segment.get("output_path", "")
            
            if not original_text.strip():
                continue
                
            dubbed_text = dubbed_texts[dub_idx]
            dub_idx += 1
            
            # Apply edited text override if present
            if seg_id in edited_map:
                dubbed_text = edited_map[seg_id]
            
            # Voice clone if not in review mode
            cloned_audio_path = None
            cloned_duration_ms = end_ms - start_ms
            if not review_mode:
                clone_result = self._voice_clone_segment(
                    dubbed_text, original_audio_path, seg_id, original_text, 
                    speaker_label, job_id=job_id, process_temp_dir=process_temp_dir
                )
                if clone_result:
                    cloned_audio_path = clone_result.get("path")
                    cloned_duration_ms = clone_result.get("duration_ms", end_ms - start_ms)
                
                # Update progress only at significant intervals (every 10% of cloning or every 5 segments)
                try:
                    if cloneable_count > 0:
                        completed_clones += 1
                        # Calculate progress but only update at meaningful intervals
                        progress_value = 80 + min(9, int((completed_clones / max(1, cloneable_count)) * 9))
                        
                        # Only update if progress increased by at least 2% or every 5 segments
                        if (progress_value - last_progress_update >= 2 or 
                            completed_clones % 5 == 0 or 
                            completed_clones == cloneable_count):
                            
                            self._update_status_non_blocking(
                                job_id,
                                ProcessingStatus.PROCESSING,
                                progress_value,
                                {
                                    "message": "Voice cloning and reconstructing final audio",
                                    "cloned_segments": completed_clones,
                                    "total_segments": cloneable_count,
                                },
                                "dub",
                            )
                            last_progress_update = progress_value
                except Exception:
                    # Best-effort progress update; ignore errors
                    pass
            
            # Create segment info and save to JSON
            segment_json = self._create_segment_info(
                seg_id, i, start_ms, cloned_duration_ms, original_text, dubbed_text,
                original_audio_path, cloned_audio_path, speaker_label, job_id, process_temp_dir
            )
            dubbed_segments.append(segment_json)
        
        return dubbed_segments
    
    def _create_segment_info(self, seg_id: str, segment_index: int, start_ms: int, cloned_duration_ms: int,
                           original_text: str, dubbed_text: str, original_audio_path: str, 
                           cloned_audio_path: str, speaker_label: str, job_id: str, process_temp_dir: str) -> dict:
        """Create and save segment information JSON"""
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
            "speaker_label": speaker_label,
        }
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(segment_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved segment info: {info_filename}")
        except Exception as e:
            logger.error(f"Failed to save JSON for {seg_id}: {str(e)}")
        
        return segment_json

    def _download_segment_audios(self, manifest_override: Dict[str, Any], process_temp_dir: str) -> None:
        """Download original segment audios from manifest URLs"""
        for seg in manifest_override.get("segments", []):
            url = seg.get("original_audio_url")
            fname = seg.get("original_audio_file")
            if url and fname:
                try:
                    import requests
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    with open(os.path.join(process_temp_dir, fname), 'wb') as fw:
                        fw.write(resp.content)
                    logger.info(f"Downloaded segment audio: {fname}")
                except Exception as e:
                    logger.warning(f"Failed to download original segment {fname}: {e}")

    def _generate_final_output(self, job_id: str, dubbed_segments: list, process_temp_dir: str, 
                              target_language: str, audio_url: str, speakers_count: int, transcript_id: str) -> dict:
        """Generate final audio, SRT file, and upload to R2"""
        self._update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 90, {"message": "Reconstructing final audio..."}, "dub")
        logger.info("Reconstructing final audio...")
        
        # Reconstruct final audio
        final_audio_path = self._reconstruct_final_audio(dubbed_segments, None, job_id=job_id, process_temp_dir=process_temp_dir)
        
        # Generate SRT file and finalize (combining multiple close steps)
        self._update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 92, {"message": "Finalizing output files..."}, "dub")
        
        subtitle_path = self._generate_srt_file(job_id, dubbed_segments, process_temp_dir)
        
        # Create process summary
        self._create_process_summary(job_id, dubbed_segments, final_audio_path, subtitle_path, 
                                   process_temp_dir, target_language, audio_url, speakers_count, transcript_id)
        
        # Upload to R2 and get results
        return self._upload_and_finalize(job_id, process_temp_dir, final_audio_path)
    
    def _generate_srt_file(self, job_id: str, dubbed_segments: list, process_temp_dir: str) -> str:
        """Generate SRT subtitle file"""
        logger.info("Generating SRT file...")
        
        from .video_processor import VideoProcessor
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
                               audio_url: str, speakers_count: int, transcript_id: str) -> None:
        """Create and save process summary JSON"""
        # Check for instrument and vocal files
        instrument_file = f"instruments_{job_id}.wav"
        vocal_file = f"vocals_{job_id}.wav"
        
        if not os.path.exists(os.path.join(process_temp_dir, instrument_file)):
            instrument_file = None
            
        if not os.path.exists(os.path.join(process_temp_dir, vocal_file)):
            vocal_file = None
        
        process_summary = {
            "success": True,
            "job_id": job_id,
            "segments_count": len(dubbed_segments),
            "audio_url": audio_url,
            "target_language": target_language,
            "speakers_count": speakers_count,
            "final_audio_file": os.path.basename(final_audio_path) if final_audio_path else None,
            "subtitle_file": os.path.basename(subtitle_path) if subtitle_path else None,
            "instrument_file": instrument_file,
            "vocal_file": vocal_file,
            "final_video_file": None,  # Video creation disabled
            "processing_timestamp": int(time.time()),
            "segments": dubbed_segments,
            "transcript_id": transcript_id
        }
        
        summary_filename = f"dubbing_process_summary_{job_id}.json"
        summary_path = os.path.join(process_temp_dir, summary_filename).replace('\\', '/')
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(process_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Process summary saved: {summary_filename}")
        except Exception as e:
            logger.error(f"Failed to save process summary: {str(e)}")
    
    def _upload_and_finalize(self, job_id: str, process_temp_dir: str, final_audio_path: str) -> dict:
        """Upload files to R2 and return final results"""
        self._update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 95, {"message": "Uploading and finalizing..."}, "dub")
        
        # Upload entire directory to R2
        folder_upload_result = self.r2_storage.upload_directory(job_id, process_temp_dir)
        
        # Note: Vocal/instrument files use runpod URLs directly, no R2 upload needed
        
        # All files (including vocals/instruments) are available via folder_upload_result
        
        logger.info("Dubbed processing completed successfully")
        
        # Do not remove temp dir here; caller handles cleanup after final status update
        
        return {
            "success": True,
            "job_id": job_id,
            "result_url": None,  # No video result
            "result_urls": {},
            "folder_upload": folder_upload_result,
            "video_upload": None,
            "video_error": None
        }
    



    def _reconstruct_final_audio(self, segments: list, original_audio_path: str, job_id: str = None, process_temp_dir: str = None) -> str:
        """Reconstruct final audio with exact original duration.
        Missing segments will have silent audio automatically.
        """
        try:
            import soundfile as sf
            if not segments:
                return None
                
            sample_rate = None
            base_dtype = np.float32
            original_duration_samples = 0
            
            # Get original audio properties for exact duration matching
            if original_audio_path and os.path.exists(original_audio_path):
                original_audio, sample_rate = sf.read(original_audio_path)
                if len(original_audio.shape) > 1:
                    original_audio = original_audio[:, 0]
                base_dtype = original_audio.dtype if hasattr(original_audio, 'dtype') else np.float32
                original_duration_samples = len(original_audio)
            else:
                # Fallback: derive sample rate from cloned segments
                for segment in segments:
                    cloned_path = segment.get("cloned_audio_path")
                    if cloned_path and os.path.exists(cloned_path):
                        cloned_audio, sample_rate = sf.read(cloned_path)
                        base_dtype = cloned_audio.dtype if hasattr(cloned_audio, 'dtype') else np.float32
                        break
                if not sample_rate:
                    sample_rate = 44100
                # Calculate duration from segments if no original audio
                segment_end_times = [s.get("end", 0) for s in segments if s.get("end")]
                if segment_end_times:
                    max_end_time_ms = max(segment_end_times)
                    original_duration_samples = int((max_end_time_ms / 1000.0) * sample_rate)
                else:
                    # Fallback to 0 if no valid segments
                    original_duration_samples = 0
            
            # Create final audio array with exact original duration (filled with silence)
            if original_duration_samples <= 0:
                logger.warning("No valid duration found, creating minimal audio")
                original_duration_samples = int(sample_rate)  # 1 second fallback
            final_audio = np.zeros(original_duration_samples, dtype=base_dtype)
            # Place cloned audio segments at their exact positions
            for segment in segments:
                cloned_path = segment.get("cloned_audio_path")
                if not cloned_path or not os.path.exists(cloned_path):
                    continue
                    
                try:
                    cloned_audio, _ = sf.read(cloned_path)
                    if len(cloned_audio.shape) > 1:
                        cloned_audio = np.mean(cloned_audio, axis=1)  # Convert to mono
                    
                    start_ms = segment.get("start", 0)
                    end_ms = segment.get("end", 0)
                    
                    # Validate timing values
                    if start_ms < 0 or end_ms <= start_ms:
                        logger.warning(f"Invalid timing for segment {segment.get('id', 'unknown')}: {start_ms}ms-{end_ms}ms")
                        continue
                        
                    start_sample = int((start_ms / 1000.0) * sample_rate)
                    expected_duration_samples = int(((end_ms - start_ms) / 1000.0) * sample_rate)
                    
                    # Ensure cloned audio fits the expected segment duration
                    if len(cloned_audio) > expected_duration_samples:
                        # Truncate if too long
                        cloned_audio = cloned_audio[:expected_duration_samples]
                    elif len(cloned_audio) < expected_duration_samples:
                        # Pad with silence if too short
                        padding = expected_duration_samples - len(cloned_audio)
                        cloned_audio = np.pad(cloned_audio, (0, padding), mode="constant")
                    
                    # Calculate end position
                    end_sample = start_sample + len(cloned_audio)
                    
                    # Only place audio if it fits within original duration
                    if start_sample >= 0 and end_sample <= original_duration_samples:
                        final_audio[start_sample:end_sample] = cloned_audio
                        logger.info(f"Placed segment {segment.get('id', 'unknown')} at {start_ms}ms-{end_ms}ms")
                    else:
                        logger.warning(f"Segment {segment.get('id', 'unknown')} exceeds original audio duration, skipping")
                        
                except Exception as e:
                    logger.error(f"Failed to process segment {segment.get('id', 'unknown')}: {str(e)}")
                    continue
            # Save final audio with exact original duration
            final_path = os.path.join(process_temp_dir, f"final_dubbed_{job_id}.wav").replace('\\', '/')
            sf.write(final_path, final_audio, sample_rate)
            
            duration_seconds = len(final_audio) / sample_rate
            logger.info(f"Final audio reconstructed: {final_path} (duration: {duration_seconds:.2f}s, {len(final_audio)} samples)")
            return final_path
        except Exception as e:
            logger.error(f"Failed to reconstruct final audio: {str(e)}")
            return None


def get_simple_dubbed_api() -> SimpleDubbedAPI:
    """Get or create global SimpleDubbedAPI instance (thread-safe)"""
    global _simple_dubbed_api_instance
    if _simple_dubbed_api_instance is None:
        with _api_lock:
            if _simple_dubbed_api_instance is None:
                _simple_dubbed_api_instance = SimpleDubbedAPI()
    return _simple_dubbed_api_instance
