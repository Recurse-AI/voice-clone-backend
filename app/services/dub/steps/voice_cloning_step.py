import os
import time
import logging
import soundfile as sf
from io import BytesIO
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.dub.context import DubbingContext
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.dub.handlers.audio_handler import AudioHandler
from app.config.settings import settings

logger = logging.getLogger(__name__)

def _add_language_tag(text: str, language_code: str) -> str:
    if not text or not language_code:
        return text
    return f"{text} [{language_code}]"

class VoiceCloningStep:
    def __init__(self):
        self._ai_voice_reference_cache = {}
    
    def execute(self, context: DubbingContext):
        ProgressTracker.update_status(
            context.job_id, "processing", 65,
            {"message": "Starting voice cloning", "phase": "voice_cloning"}
        )
        
        ProgressTracker.update_phase_progress(
            context.job_id, "voice_cloning", 0.0, "Segmenting audio for voice cloning"
        )
        
        split_files = AudioHandler.split_audio_segments(context)
        
        ProgressTracker.update_phase_progress(
            context.job_id, "voice_cloning", 0.1,
            f"Audio segmented into {len(split_files)} parts"
        )
        
        segments_data = []
        for seg, split_file in zip(context.segments, split_files):
            if seg.get("original_text", "").strip():
                duration_ms = seg["end"] - seg["start"]
                segments_data.append({
                    "seg_id": seg["id"],
                    "global_idx": seg["segment_index"],
                    "start_ms": seg["start"],
                    "end_ms": seg["end"],
                    "duration_ms": duration_ms,
                    "original_text": seg["original_text"],
                    "dubbed_text": seg["dubbed_text"],
                    "original_audio_path": split_file["output_path"],
                    "speaker": seg.get("speaker"),
                    "reference_id": seg.get("reference_id")
                })
        
        results = self._process_voice_cloning_batch(segments_data, context)
        
        final_segments = []
        for i, data in enumerate(segments_data):
            result = results[i] if i < len(results) else None
            cloned_audio_path = result.get("path") if result else None
            cloned_duration_ms = result.get("duration_ms", data["duration_ms"]) if result else data["duration_ms"]
            
            segment_json = self._create_segment_data(
                data["seg_id"], data["global_idx"], data["start_ms"], cloned_duration_ms,
                data["original_text"], data["dubbed_text"], data["original_audio_path"],
                cloned_audio_path, context.job_id, data.get("speaker"), data.get("reference_id")
            )
            final_segments.append(segment_json)
        
        context.segments = final_segments
    
    def _process_voice_cloning_batch(self, segments_data: list, context: DubbingContext) -> list:
        total_segments = len(segments_data)
        batch_size = settings.VOICE_CLONING_BATCH_SIZE
        max_workers = settings.VOICE_CLONING_PARALLEL_WORKERS if context.model_type in ['best', 'medium'] else 1
        
        mode_map = {
            'best': 'ElevenLabs (parallel)',
            'medium': 'Fish API (parallel)',
            'normal': 'local s1-mini (sequential)'
        }
        mode = mode_map.get(context.model_type, 'local s1-mini (sequential)')
        logger.info(f"Processing {total_segments} segments using {mode} with {max_workers} worker(s)")
        
        results = [None] * total_segments
        
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch_data = segments_data[batch_start:batch_end]
            
            logger.info(f"Batch {batch_start//batch_size + 1}: segments {batch_start+1}-{batch_end}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._clone_segment_worker, data, batch_start + i,
                        total_segments, context
                    ): i
                    for i, data in enumerate(batch_data)
                }
                
                for future in as_completed(futures):
                    actual_index, result, success = future.result()
                    results[actual_index] = result
                    
                    completed = sum(1 for r in results if r is not None)
                    progress = (completed / total_segments) * 0.9 + 0.1
                    
                    try:
                        ProgressTracker.update_phase_progress(
                            context.job_id, "voice_cloning", progress,
                            f"Voice cloning: {completed}/{total_segments}"
                        )
                    except Exception:
                        pass
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Completed: {successful}/{total_segments} successful")
        
        try:
            ProgressTracker.update_phase_progress(
                context.job_id, "voice_cloning", 1.0,
                f"Voice cloning complete: {successful}/{total_segments}"
            )
        except Exception:
            pass
        
        return results
    
    def _clone_segment_worker(self, data: dict, actual_index: int, total_segments: int, context: DubbingContext) -> tuple:
        try:
            start_time = time.time()
            result = self._voice_clone_segment(data, context)
            elapsed = time.time() - start_time
            logger.info(f"Segment {data['seg_id']} ({actual_index+1}/{total_segments}) completed in {elapsed:.2f}s")
            return (actual_index, result, True)
        except Exception as e:
            logger.error(f"Segment {data.get('seg_id', 'unknown')} failed: {e}")
            return (actual_index, None, False)
    
    def _voice_clone_segment(self, data: dict, context: DubbingContext) -> Optional[Dict[str, Any]]:
        try:
            tagged_text = _add_language_tag(data["dubbed_text"], context.target_language_code)
            tagged_reference_text = _add_language_tag(
                data["original_text"] or "Reference audio", context.source_language_code
            )
            
            segment_index = int(data['seg_id'].split('_')[1]) - 1
            cloned_path = os.path.join(
                context.process_temp_dir, f"cloned_{context.job_id}_{segment_index:03d}.wav"
            ).replace('\\', '/')
            
            ai_voice_id = data.get("reference_id")
            reference_audio_bytes = None
            
            if context.voice_type == 'ai_voice' and ai_voice_id and context.model_type == 'normal':
                audio_bytes, sample_text = self._get_ai_voice_reference(
                    ai_voice_id, context.source_language_code
                )
                reference_audio_bytes = audio_bytes
                if sample_text:
                    tagged_reference_text = _add_language_tag(sample_text, context.source_language_code)
            
            if reference_audio_bytes is None:
                reference_audio_bytes = self._load_reference_audio(data["original_audio_path"])
            
            if context.model_type == 'best':
                if not ai_voice_id:
                    return None
                result = self._generate_with_elevenlabs(
                    tagged_text, ai_voice_id, context.job_id,
                    context.target_language_code, segment_index, data["duration_ms"]
                )
            elif context.model_type == 'medium':
                if not ai_voice_id:
                    return None
                result = self._generate_with_premium_api(
                    tagged_text, ai_voice_id, context.job_id, context.target_language_code
                )
                if not result.get("success"):
                    logger.warning(f"Fish API failed, using local model")
                    result = self._generate_with_local_model(
                        tagged_text, reference_audio_bytes, tagged_reference_text,
                        context.job_id, context.target_language_code
                    )
            else:
                result = self._generate_with_local_model(
                    tagged_text, reference_audio_bytes, tagged_reference_text,
                    context.job_id, context.target_language_code
                )
            
            if result.get("success"):
                return self._save_cloned_audio(result.get("output_path"), cloned_path, data['seg_id'])
            
            return None
        except Exception as e:
            logger.error(f"Voice cloning error for {data['seg_id']}: {e}")
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
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            buffer = BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to load reference audio from {audio_path}: {e}")
            return None
    
    def _generate_with_elevenlabs(self, text: str, voice_id: str, job_id: str, target_language_code: str, segment_index: int, target_duration_ms: int) -> dict:
        from app.services.dub.elevenlabs_service import get_elevenlabs_service, BlockedVoiceError
        
        try:
            service = get_elevenlabs_service()
            return service.generate_speech(text, voice_id, target_language_code, job_id, segment_index, target_duration_ms)
        except BlockedVoiceError:
            raise
        except Exception as e:
            logger.error(f"ElevenLabs generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_premium_api(self, tagged_text: str, reference_id: str, job_id: str, target_language_code: str) -> dict:
        from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
        
        if not reference_id:
            return {"success": False, "error": "reference_id is required"}
        
        try:
            service = get_fish_audio_api_service()
            return service.generate_voice_clone(
                text=tagged_text,
                reference_id=reference_id,
                job_id=job_id,
                target_language_code=target_language_code
            )
        except Exception as e:
            logger.error(f"Fish API error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_local_model(self, tagged_text: str, reference_audio_bytes: bytes, tagged_reference_text: str, job_id: str, target_language_code: str) -> dict:
        from app.services.dub.fish_speech_service import get_fish_speech_service
        
        service = get_fish_speech_service()
        return service.generate_with_reference_audio(
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
    
    def _create_segment_data(self, seg_id: str, segment_index: int, start_ms: int, cloned_duration_ms: int,
                           original_text: str, dubbed_text: str, original_audio_path: str, 
                           cloned_audio_path: str, job_id: str, speaker: str = None, reference_id: str = None) -> dict:
        start_ms = int(start_ms)
        end_ms = int(start_ms + cloned_duration_ms)
        duration_ms = int(cloned_duration_ms)
        
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
            "speaker": speaker,
            "reference_id": reference_id
        }

