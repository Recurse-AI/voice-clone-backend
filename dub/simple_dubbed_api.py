#!/usr/bin/env python3
"""
Simple Dubbed API - Clean Implementation using existing dub folder
Audio URL → Download → Assembly AI → Dub → Voice Clone → Reconstruct → R2 Upload
"""

import os
import json
import logging
import time
from config import settings
from .assembly_transcription import TranscriptionService
from .fish_speech_service import get_fish_speech_service
from r2_storage import R2Storage
import numpy as np
from .audio_utils import AudioUtils
from status_manager import status_manager, ProcessingStatus

logger = logging.getLogger(__name__)

# Supported languages for dubbing and Fish Speech voice synthesis
SUPPORTED_LANGUAGE_NAMES = {
    "english", "chinese", "japanese", "german", "french", "spanish",
    "korean", "arabic", "russian", "dutch", "italian", "polish", "portuguese"
}
SUPPORTED_LANGUAGE_CODES = {
    "en", "zh", "ja", "de", "fr", "es",
    "ko", "ar", "ru", "nl", "it", "pl", "pt"
}

class SimpleDubbedAPI:
    """Simple clean API for dubbed audio processing"""
    
    def __init__(self):
        # Use existing services from dub folder
        self.transcription_service = TranscriptionService()
        self.fish_speech = get_fish_speech_service()  # Use global singleton instance
        self.r2_storage = R2Storage()
        self.temp_dir = settings.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def dub_text_batch(self, segments: list, target_language: str = "English", batch_size: int = 10) -> list:
        """
        Batch dub text using OpenAI with enhanced prompt and batching for large input.
        Each segment: {"text": ..., "duration_ms": ..., "id": ...}
        """
        all_dubbed = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            # Prepare prompt with duration info
            prompt_lines = []
            for idx, seg in enumerate(batch):
                prompt_lines.append(f"[{idx+1}] (Target duration: {seg['duration_ms']} ms) {seg['text']}")
            joined_texts = "\n".join(prompt_lines)
            system_prompt = (
                f"Translate the following texts to {target_language} for voice dubbing.\n"
                f"For each segment, try to match the target duration by adding extra white spaces between words if needed.\n"
                f"The target duration for each segment is given in milliseconds.\n"
                f"Make sure the output uses the correct alphabet/script for {target_language}.\n"
                f"If the target language is not English, do not use English letters.\n"
                f"Break long whitespace naturally, as a human would speak.\n"
                f"The output should sound natural and fluent for native speakers of {target_language}.\n"
                f"If the output needs to be lengthened to match a target duration, it is preferable to add extra white spaces between words within sentences, rather than between sentences.\n"
                f"Do not add extra explanation or comments, only return the translated texts in the same order as input, separated by |||."
            )
            user_prompt = (
                "Translate each segment below. Return only the translated segments, in order, separated by |||.\n"
                "Segments (with target duration):\n"
                f"{joined_texts}"
            )
            response = self.transcription_service.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            output = response.choices[0].message.content.strip()
            all_dubbed.extend([seg.strip() for seg in output.split("|||")])
        return all_dubbed

    def process_dubbed_audio(self, audio_url: str, job_id: str, target_language: str, speakers_count: int = 1, source_video_language: str = None, output_dir: str = None, instrument_path: str = None, video_path: str = None, subtitle: bool = False, instrument: bool = False) -> dict:
        """Complete dubbed audio processing with job_id, expected_speaker, source_video_language, output_dir, instrument mix, subtitle, video merge.

        target_language must be one of the supported languages defined in SUPPORTED_LANGUAGE_NAMES / SUPPORTED_LANGUAGE_CODES.
        """
        # Validate target language early to avoid unnecessary processing
        if target_language.lower() not in SUPPORTED_LANGUAGE_NAMES and target_language.lower() not in SUPPORTED_LANGUAGE_CODES:
            error_msg = f"Unsupported target language: {target_language}"
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": error_msg})
            return {"success": False, "error": error_msg}

        try:
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=45, details={"message": "Transcribing audio..."})
            # Define a dedicated folder for this job inside the TEMP_DIR
            if not output_dir:
                output_dir = os.path.join(self.temp_dir, job_id)
            process_temp_dir = output_dir  # We now keep all generated files inside this folder
            os.makedirs(process_temp_dir, exist_ok=True)
            paragraphs_result = self.transcription_service.get_paragraphs_and_split_audio(
                audio_url=audio_url,
                output_dir=output_dir,
                speakers_count=speakers_count,
                source_video_language=source_video_language
            )
            if not paragraphs_result["success"]:
                status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=50, details={"message": paragraphs_result.get("error", "Transcription failed")})
                # Clean up temp directory on failure
                AudioUtils.remove_temp_dir(folder_path=process_temp_dir)
                return {"success": False, "error": paragraphs_result.get("error", "Transcription failed")}
            raw_paragraphs = paragraphs_result["segments"]
            logger.info(f"Found {len(raw_paragraphs)} segments")
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=55, details={"message": "Batch dubbing..."})
            # Prepare texts for batch dubbing
            segments_to_dub = []
            segment_indices = []
            for i, segment in enumerate(raw_paragraphs):
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
            dubbed_texts = self.dub_text_batch(segments_to_dub, target_language, batch_size=10)
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=65, details={"message": "Voice cloning..."})
            # Assign dubbed texts back to segments and process audio
            dubbed_segments = []
            dub_idx = 0
            for i, segment in enumerate(raw_paragraphs):
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
                cloned_audio_path = self._voice_clone_segment(
                    dubbed_text, original_audio_path, seg_id, original_text, speaker_label, job_id=job_id, process_temp_dir=process_temp_dir
                )
                info_filename = f"segment_{job_id}_{i:03d}_info.json"
                info_path = os.path.join(process_temp_dir, info_filename).replace('\\', '/')
                segment_json = {
                    "id": seg_id,
                    "segment_index": i + 1,
                    "start": start_ms,
                    "end": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "original_text": original_text,
                    "dubbed_text": dubbed_text,
                    "original_audio_file": f"segment_{i:03d}.wav" if original_audio_path else None,
                    "cloned_audio_file": f"cloned_{job_id}_{i:03d}.wav" if cloned_audio_path else None,
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
                dubbed_segments.append(segment_json)
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=75, details={"message": "Reconstructing final audio..."})
            # Step 3: Reconstruct final audio
            final_audio_path = self._reconstruct_final_audio(dubbed_segments, paragraphs_result.get("original_audio"), job_id=job_id, process_temp_dir=process_temp_dir)
            final_mixed_audio_path = final_audio_path
            if instrument and instrument_path and os.path.exists(instrument_path):
                final_mixed_audio_path = os.path.join(process_temp_dir, f"final_mixed_{job_id}.wav")
                AudioUtils.mix_audio_files(final_audio_path, instrument_path, final_mixed_audio_path, ratio1=0.75, ratio2=0.25)
                logger.info(f"Instrument mixed audio saved: {final_mixed_audio_path}")
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=85, details={"message": "Generating subtitles..."})
            # Step 5: Subtitle generation (if subtitle True)
            # --- Always generate SRT file ---
            from dub.video_processor import VideoProcessor
            processor = VideoProcessor(temp_dir=process_temp_dir)
            subtitle_data = []
            for seg in dubbed_segments:
                text = seg["dubbed_text"]
                start = seg["start"] / 1000.0
                end = seg["end"] / 1000.0
                words = text.split()
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i+3])
                    chunk_start = start + (end - start) * (i / max(1, len(words)))
                    chunk_end = start + (end - start) * ((i+3) / max(1, len(words)))
                    subtitle_data.append({"start": chunk_start, "end": min(chunk_end, end), "text": chunk})
            subtitle_path = os.path.join(process_temp_dir, f"subtitles_{job_id}.srt")
            processor.create_srt_file(subtitle_data, subtitle_path)
            logger.info(f"Subtitle file saved: {subtitle_path}")
            status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=90, details={"message": "Creating video..."})
            final_video_path = None
            video_error = None
            if video_path and os.path.exists(video_path):
                from dub.video_processor import VideoProcessor
                processor = VideoProcessor(temp_dir=process_temp_dir)
                if subtitle and subtitle_path and os.path.exists(subtitle_path):
                    result = processor.create_video_with_subtitles(
                        video_path=video_path,
                        audio_path=final_mixed_audio_path,
                        segments_dir=process_temp_dir,
                        audio_id=job_id,
                        instruments_path=instrument_path if instrument else None
                    )
                else:
                    result = processor.create_video_with_audio(
                        video_path=video_path,
                        audio_path=final_mixed_audio_path,
                        audio_id=job_id,
                        instruments_path=instrument_path if instrument else None,
                        segments_dir=process_temp_dir
                    )
                if result.get("success"):
                    final_video_path = result.get("video_path")
                    logger.info(f"Final video created: {final_video_path}")
                else:
                    video_error = result.get('error')
                    logger.error(f"Video creation failed: {video_error}")
            process_summary = {
                "success": True,
                "job_id": job_id,
                "segments_count": len(dubbed_segments),
                "audio_url": audio_url,
                "target_language": target_language,
                "speakers_count": speakers_count,
                "final_audio_file": os.path.basename(final_mixed_audio_path) if final_mixed_audio_path else None,
                "subtitle_file": os.path.basename(subtitle_path) if subtitle_path else None,
                "final_video_file": os.path.basename(final_video_path) if final_video_path else None,
                "processing_timestamp": int(time.time()),
                "segments": dubbed_segments
            }
            summary_filename = f"dubbing_process_summary_{job_id}.json"
            summary_path = os.path.join(process_temp_dir, summary_filename).replace('\\', '/')
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(process_summary, f, ensure_ascii=False, indent=2)
                logger.info(f"Process summary saved: {summary_filename}")
            except Exception as e:
                logger.error(f"Failed to save process summary: {str(e)}")
            # Remove original video before uploading folder to R2
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Original video deleted before R2 upload: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete original video {video_path}: {e}")

            # --- R2 Upload Section ---
            folder_upload_result = self.r2_storage.upload_audio_segments(job_id, process_temp_dir)

            video_upload_res = None
            if final_video_path and os.path.exists(final_video_path):
                r2_video_key = self.r2_storage.generate_file_path(job_id, "", os.path.basename(final_video_path))
                video_upload_res = self.r2_storage.upload_file(final_video_path, r2_video_key, content_type="video/mp4")

            result_urls = {}
            if video_upload_res and video_upload_res.get("success"):
                result_urls["final_video"] = video_upload_res["url"]

            # Update status with detailed information
            status_manager.update_status(
                job_id,
                ProcessingStatus.COMPLETED,
                progress=100,
                details={
                    "message": "Dubbed processing completed",
                    "result_url": result_urls.get("final_video"),
                    "folder_upload": folder_upload_result,
                    "video_upload": video_upload_res,
                    "video_error": video_error
                }
            )

            # Optionally, clean up local temp directory
            try:
                AudioUtils.remove_temp_dir(folder_path=process_temp_dir)
            except Exception:
                pass

            return {
                "success": True,
                "job_id": job_id,
                "result_url": result_urls.get("final_video"),
                "result_urls": result_urls,
                "folder_upload": folder_upload_result,
                "video_upload": video_upload_res,
                "video_error": video_error
            }
        except Exception as e:
            logger.error(f"Dubbed processing failed: {str(e)}")
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": f"Dubbed processing failed: {str(e)}", "error": str(e)})
            # Clean up temp directory on exception
            AudioUtils.remove_temp_dir(folder_path=locals().get("process_temp_dir"))
            return {"success": False, "error": str(e)}
    
    def _voice_clone_segment(self, dubbed_text: str, reference_audio_path: str, segment_id: str, original_text: str = "", speaker_label: str = None, job_id: str = None, process_temp_dir: str = None) -> str:
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
            from config import settings
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
            def smart_chunk(text, chunk_size=200, min_size=180):
                chunks = []
                i = 0
                while i < len(text):
                    end = min(i + chunk_size, len(text))
                    chunk = text[i:end]
                    puncts = [chunk.rfind(p) for p in '.!?']
                    punct = max(puncts)
                    if punct >= min_size - 1:
                        split_at = punct + 1
                    else:
                        space = chunk.rfind(' ')
                        if space >= min_size - 1:
                            split_at = space + 1
                        else:
                            split_at = len(chunk)
                    chunks.append(text[i:i+split_at].strip())
                    i += split_at
                return [c for c in chunks if c]
            text_chunks = smart_chunk(dubbed_text)
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
                    max_new_tokens=1024,
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
                return cloned_path
            return None
                
        except Exception as e:
            logger.error(f"Voice cloning error for {segment_id}: {str(e)}")
            return None
    
    def _reconstruct_final_audio(self, segments: list, original_audio_path: str, job_id: str = None, process_temp_dir: str = None) -> str:
        """Reconstruct final audio with transcript_id in filename"""
        try:
            import soundfile as sf
            if not segments:
                return None
            if not original_audio_path or not os.path.exists(original_audio_path):
                logger.error("Original audio file not found for reconstruction")
                return None
            audio, sample_rate = sf.read(original_audio_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            total_duration = int(max([s["end"] for s in segments]) / 1000.0 * sample_rate)
            total_samples = max(total_duration, len(audio))
            final_audio = np.zeros(total_samples, dtype=audio.dtype)
            for segment in segments:
                cloned_path = segment["cloned_audio_path"]
                if not cloned_path or not os.path.exists(cloned_path):
                    continue
                try:
                    cloned_audio, _ = sf.read(cloned_path)
                    if len(cloned_audio.shape) > 1:
                        cloned_audio = np.mean(cloned_audio, axis=1)
                    start_ms = segment["start"]
                    end_ms = segment["end"]
                    start_sample = int((start_ms / 1000.0) * sample_rate)
                    segment_duration_samples = int(((end_ms - start_ms) / 1000.0) * sample_rate)
                    if len(cloned_audio) > segment_duration_samples:
                        cloned_audio = cloned_audio[:segment_duration_samples]
                    elif len(cloned_audio) < segment_duration_samples:
                        padding = segment_duration_samples - len(cloned_audio)
                        cloned_audio = np.pad(cloned_audio, (0, padding), mode='constant')
                    end_sample = min(start_sample + len(cloned_audio), total_samples)
                    final_audio[start_sample:end_sample] = cloned_audio[:end_sample-start_sample]
                    logger.info(f"Placed {segment['id']} at {start_ms}ms")
                except Exception as e:
                    logger.error(f"Failed to process segment {segment['id']}: {str(e)}")
                    continue
            # Save final audio (with transcript_id)
            final_path = os.path.join(process_temp_dir, f"final_dubbed_{job_id}.wav").replace('\\', '/')
            sf.write(final_path, final_audio, sample_rate)
            logger.info(f"Final audio reconstructed: {final_path}")
            return final_path
        except Exception as e:
            logger.error(f"Failed to reconstruct final audio: {str(e)}")
            return None
