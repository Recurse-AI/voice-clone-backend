import logging
from typing import List, Dict
from app.services.dub.context import DubbingContext

logger = logging.getLogger(__name__)

class ReferenceCreationStep:
    @staticmethod
    def execute(context: DubbingContext):
        if context.reference_ids:
            logger.info(f"Using provided reference_ids: {context.reference_ids}")
            return
        
        if context.model_type not in ["best", "medium"]:
            return
        
        if context.review_mode:
            return
        
        logger.info(f"Creating voice references for {context.model_type} model")
        
        transcription_segments = context.transcription_result.get("segments", [])
        if not transcription_segments:
            logger.warning("No transcription segments available for voice clone creation")
            return
        
        unique_speakers = sorted(set(
            s.get("speaker") for s in transcription_segments if s.get("speaker")
        ))
        
        if not unique_speakers:
            logger.warning("No speakers detected in segments")
            return
        
        logger.info(f"Creating voice clones for {len(unique_speakers)} speakers")
        
        context.reference_ids = ReferenceCreationStep._create_speaker_voice_clones(
            transcription_segments, unique_speakers, context
        )
        
        if context.reference_ids:
            logger.info(f"Created {len(context.reference_ids)} voice clones")
        else:
            model_name = "ElevenLabs" if context.model_type == "best" else "Fish Audio API"
            raise ValueError(f"{model_name} voice clone creation failed. Check API credits.")
    
    @staticmethod
    def _create_speaker_voice_clones(transcription_segments: List[Dict], unique_speakers: List[str], context: DubbingContext) -> List[str]:
        try:
            from app.services.dub.elevenlabs_service import get_elevenlabs_service
            from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
            from app.services.dub.reference_audio_optimizer import get_reference_audio_optimizer
            
            optimizer = get_reference_audio_optimizer()
            reference_ids = []
            
            for speaker in unique_speakers:
                speaker_segments = [
                    s for s in transcription_segments 
                    if s.get("speaker") == speaker and s.get("original_audio_file")
                ]
                
                if not speaker_segments:
                    logger.warning(f"No segments with audio found for {speaker}")
                    continue
                
                if context.model_type == "best":
                    audio_bytes = optimizer.optimize_for_elevenlabs(
                        speaker_segments, speaker, context.process_temp_dir
                    )
                    service = get_elevenlabs_service()
                else:
                    audio_bytes = optimizer.optimize_for_fish(
                        speaker_segments, speaker, context.process_temp_dir
                    )
                    service = get_fish_audio_api_service()
                
                if not audio_bytes:
                    logger.error(f"Failed to optimize audio for {speaker}")
                    return []
                
                voice_name = f"{context.job_id}_{speaker}"
                result = service.create_voice_reference(audio_bytes, voice_name)
                
                if result.get("success"):
                    voice_id = result.get("voice_id") or result.get("reference_id")
                    reference_ids.append(voice_id)
                    context.created_voice_ids.append(voice_id)
                    logger.info(f"Created voice clone for {speaker}: {voice_id}")
                else:
                    logger.error(f"Failed to create voice clone for {speaker}: {result.get('error')}")
                    return []
            
            return reference_ids
            
        except Exception as e:
            logger.error(f"Failed to create speaker voice clones: {e}")
            return []
    
    @staticmethod
    def assign_to_segments(context: DubbingContext):
        for segment in context.segments:
            if not segment.get("reference_id"):
                speaker = segment.get("speaker")
                segment["reference_id"] = ReferenceCreationStep._assign_reference_id(
                    speaker, context.reference_ids
                ) if speaker and context.reference_ids else None
        
        logger.info(f"Assigned reference_ids to {len(context.segments)} segments")
    
    @staticmethod
    def _assign_reference_id(speaker: str, reference_ids: List[str]) -> str:
        if not speaker or not reference_ids:
            return None
        try:
            speaker_index = int(speaker.split("_")[-1])
            if speaker_index < len(reference_ids):
                return reference_ids[speaker_index]
            else:
                return reference_ids[-1]
        except (ValueError, IndexError):
            pass
        return None

