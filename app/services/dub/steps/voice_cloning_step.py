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

class VoiceCloningStep:
    def __init__(self):
        pass
    
    def execute(self, context: DubbingContext):
        from app.services.simple_status_service import JobStatus
        ProgressTracker.update_status(
            context.job_id, JobStatus.PROCESSING, 65,
            {"message": "Starting voice cloning", "phase": "voice_cloning"}
        )
        
        if not getattr(context, 'audio_already_split', False):
            ProgressTracker.update_phase_progress(
                context.job_id, "voice_cloning", 0.0, "Segmenting audio for voice cloning"
            )
            
            split_files = AudioHandler.split_audio_segments(context)
            
            ProgressTracker.update_phase_progress(
                context.job_id, "voice_cloning", 0.1,
                f"Audio segmented into {len(split_files)} parts"
            )
        else:
            logger.info("Audio already split - skipping segmentation")
            ProgressTracker.update_phase_progress(
                context.job_id, "voice_cloning", 0.1, "Using pre-split audio"
            )
            
            transcription_segments = context.transcription_result.get("segments", [])
            split_files = [
                {"output_path": os.path.join(context.process_temp_dir, seg.get("original_audio_file"))} 
                for seg in transcription_segments
            ]
        
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
            
            segment_json = self._create_segment_data(
                data["seg_id"], data["global_idx"], data["start_ms"], data["end_ms"],
                data["original_text"], data["dubbed_text"], data["original_audio_path"],
                cloned_audio_path, context.job_id, data.get("speaker"), data.get("reference_id")
            )
            final_segments.append(segment_json)
        
        context.segments = final_segments
    
    def _process_voice_cloning_batch(self, segments_data: list, context: DubbingContext) -> list:
        total_segments = len(segments_data)
        max_workers = settings.VOICE_CLONING_PARALLEL_WORKERS if context.model_type in ['best', 'medium'] else 1
        
        mode_map = {
            'best': 'ElevenLabs (parallel)',
            'medium': 'Fish API (parallel)',
            'normal': 'local s1-mini (sequential)'
        }
        mode = mode_map.get(context.model_type, 'local s1-mini (sequential)')
        logger.info(f"Processing {total_segments} segments using {mode} with {max_workers} worker(s)")
        
        results = [None] * total_segments
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._clone_segment_worker, data, i, total_segments, context): i
                for i, data in enumerate(segments_data)
            }
            
            try:
                for future in as_completed(futures):
                    actual_index, result, success = future.result()
                    
                    if not success or result is None:
                        for pending_future in futures:
                            pending_future.cancel()
                        
                        executor.shutdown(wait=False, cancel_futures=True)
                        
                        seg_id = segments_data[actual_index].get('seg_id', 'unknown')
                        error_msg = f"Segment {seg_id} failed - cancelling remaining segments"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    
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
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
        
        successful = sum(1 for r in results if r is not None)
        
        if successful < total_segments:
            failed_count = total_segments - successful
            error_msg = f"Voice cloning failed: {failed_count}/{total_segments} segments failed"
            logger.error(error_msg)
            raise Exception(error_msg)
        
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
            dubbed_text = data["dubbed_text"]
            reference_text = data["original_text"] or "Reference audio"
            segment_index = int(data['seg_id'].split('_')[1]) - 1
            
            provided_reference_id = data.get("reference_id")
            
            if context.model_type == 'best':
                return self._process_elevenlabs_segment(
                    dubbed_text, context, segment_index, data, provided_reference_id
                )
            elif context.model_type == 'medium':
                return self._process_fish_api_segment(
                    dubbed_text, reference_text, context, segment_index, data, provided_reference_id
                )
            else:
                reference_audio_bytes = self._load_reference_audio(data["original_audio_path"])
                if provided_reference_id and context.voice_type == 'ai_voice':
                    audio_bytes, sample_text = self._get_ai_voice_reference(provided_reference_id, context.source_language_code)
                    if audio_bytes:
                        reference_audio_bytes = audio_bytes
                        reference_text = sample_text or reference_text
                
                if not reference_audio_bytes:
                    return None
                
                return self._process_local_segment(
                    dubbed_text, reference_audio_bytes, reference_text, context, segment_index
                )
        except Exception as e:
            logger.error(f"Voice cloning error for {data['seg_id']}: {e}")
            return None
    
    def _get_ai_voice_reference(self, reference_id: str, source_language_code: str) -> tuple:
        from app.config.settings import settings as _settings
        from app.services.dub.fish_audio_sample_helper import fetch_sample_audio_wav_bytes
        
        audio_bytes, sample_text = fetch_sample_audio_wav_bytes(reference_id, _settings.FISH_AUDIO_API_KEY)
        return audio_bytes, sample_text
    
    def _process_elevenlabs_segment(self, text: str, context: DubbingContext, seg_idx: int, data: dict, provided_voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        from app.services.dub.elevenlabs_service import get_elevenlabs_service
        
        service = get_elevenlabs_service()
        voice_id = provided_voice_id
        should_cleanup = False
        
        try:
            if not voice_id:
                audio_bytes = self._load_reference_audio(data["original_audio_path"])
                if not audio_bytes:
                    return None
                
                voice_name = f"{context.job_id}_seg{seg_idx}_{int(time.time())}"
                result = service.create_voice_reference(audio_bytes, voice_name)
                
                if not result.get("success"):
                    return None
                
                voice_id = result["voice_id"]
                should_cleanup = True
            
            gen_result = service.generate_speech(
                text, voice_id, context.target_language_code, 
                context.job_id, seg_idx, data["duration_ms"], speed=1.2
            )
            
            if not gen_result.get("success"):
                return None
            
            cloned_path = os.path.join(
                context.process_temp_dir, f"cloned_{context.job_id}_{seg_idx:03d}.wav"
            )
            return self._save_cloned_audio(gen_result.get("output_path"), cloned_path, data['seg_id'])
            
        finally:
            if should_cleanup and voice_id:
                service.delete_voice(voice_id)
    
    def _process_fish_api_segment(self, text: str, ref_text: str, context: DubbingContext, seg_idx: int, data: dict, provided_reference_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
        
        service = get_fish_audio_api_service()
        reference_id = provided_reference_id
        should_cleanup = False
        
        try:
            if not reference_id:
                audio_bytes = self._load_reference_audio(data["original_audio_path"])
                if not audio_bytes:
                    return self._process_local_segment(text, audio_bytes, ref_text, context, seg_idx)
                
                ref_name = f"{context.job_id}_seg{seg_idx}_{int(time.time())}"
                result = service.create_voice_reference(audio_bytes, ref_name)
                
                if not result.get("success"):
                    audio_bytes = self._load_reference_audio(data["original_audio_path"])
                    return self._process_local_segment(text, audio_bytes, ref_text, context, seg_idx)
                
                reference_id = result["reference_id"]
                should_cleanup = True
            
            gen_result = service.generate_voice_clone(
                text, reference_id, context.job_id, context.target_language_code
            )
            
            if not gen_result.get("success"):
                audio_bytes = self._load_reference_audio(data["original_audio_path"])
                return self._process_local_segment(text, audio_bytes, ref_text, context, seg_idx)
            
            cloned_path = os.path.join(
                context.process_temp_dir, f"cloned_{context.job_id}_{seg_idx:03d}.wav"
            )
            return self._save_cloned_audio(gen_result.get("output_path"), cloned_path, data['seg_id'])
            
        finally:
            if should_cleanup and reference_id:
                service.delete_voice(reference_id)
    
    def _process_local_segment(self, text: str, audio_bytes: bytes, ref_text: str, context: DubbingContext, seg_idx: int) -> Optional[Dict[str, Any]]:
        from app.services.dub.fish_speech_service import get_fish_speech_service
        
        service = get_fish_speech_service()
        result = service.generate_with_reference_audio(
            text=text,
            reference_audio_bytes=audio_bytes,
            reference_text=ref_text,
            max_new_tokens=1024,
            top_p=0.9,
            repetition_penalty=1.07,
            temperature=0.75,
            job_id=context.job_id,
            target_language_code=context.target_language_code
        )
        
        if not result.get("success"):
            return None
        
        cloned_path = os.path.join(
            context.process_temp_dir, f"cloned_{context.job_id}_{seg_idx:03d}.wav"
        )
        return self._save_cloned_audio(result.get("output_path"), cloned_path, f"seg_{seg_idx}")
    
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
    
    def _create_segment_data(self, seg_id: str, segment_index: int, start_ms: int, end_ms: int,
                           original_text: str, dubbed_text: str, original_audio_path: str, 
                           cloned_audio_path: str, job_id: str, speaker: str = None, reference_id: str = None) -> dict:
        start_ms = int(start_ms)
        end_ms = int(end_ms)
        duration_ms = int(end_ms - start_ms)
        
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

