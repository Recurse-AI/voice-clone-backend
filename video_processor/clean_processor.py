"""
Clean Audio Processor - Simplified orchestrator with OpenVoice
Coordinates all voice cloning operations with OpenVoice for superior quality and MIT licensing
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import soundfile as sf

from .voice_cloning import OpenVoiceVoiceCloningService, Args
from .transcription import TranscriptionService
from .segment_manager import SegmentManager
from .audio_reconstructor import AudioReconstructor
from config import settings

logger = logging.getLogger(__name__)

class CleanAudioProcessor:
    """
    Clean, simplified audio processor with OpenVoice
    Focuses on stability and superior quality with MIT licensed voice cloning
    """
    
    def __init__(self):
        self.voice_service = OpenVoiceVoiceCloningService()
        self.transcription_service = TranscriptionService()
        self.segment_manager = SegmentManager(self.transcription_service)
        self.audio_reconstructor = AudioReconstructor(settings.TEMP_DIR)
        
        logger.info("🎙️ CleanAudioProcessor initialized with OpenVoice services")
    
    def load_model(self, model_path: str = None) -> bool:
        """Load the OpenVoice model"""
        return self.voice_service.load_model(model_path)
    
    def is_ready(self) -> bool:
        """Check if processor is ready for voice cloning"""
        return self.voice_service.is_model_loaded()
    
    def process_audio_complete(self, 
                              audio_path: str, 
                              audio_id: str,
                              target_language: str = "English",
                              language_code: Optional[str] = None,
                              speakers_expected: Optional[int] = 1,
                              # OpenVoice parameters
                              max_length: int = None,
                              temperature: float = None,
                              top_p: float = None,
                              repetition_penalty: float = None,
                              speed_factor: float = None,
                              seed: int = None,
                              emotion: str = None,
                              # Legacy compatibility parameters (ignored)
                              cfg_scale: float = None,
                              cfg_filter_top_k: int = None,
                              use_torch_compile: bool = None,
                              silence_padding: float = None,
                              max_tokens: int = None,
                              top_k: int = None) -> Dict[str, Any]:
        """
        Complete audio processing pipeline with OpenVoice
        From transcription to final cloned audio
        """
        
        if not self.is_ready():
            return {"success": False, "error": "OpenVoice model not loaded"}
        
        logger.info(f"🚀 Starting complete audio processing with OpenVoice for {audio_id}")
        pipeline_start = time.time()
        
        try:
            # Step 1: Transcribe audio
            logger.info("📝 Step 1: Transcribing audio...")
            transcript_result = self.transcription_service.transcribe_audio(
                audio_path=audio_path,
                language_code=language_code,
                speakers_expected=speakers_expected,
                audio_id=audio_id
            )
            
            if not transcript_result:
                return {"success": False, "error": "Transcription failed"}
            
            logger.info(f"✅ Transcription completed: {len(transcript_result.get('words', []))} words detected")
            
            # Step 2: Create optimal segments
            logger.info("✂️ Step 2: Creating optimal segments...")
            segments = self.segment_manager.create_optimal_segments(transcript_result)
            
            if not segments:
                return {"success": False, "error": "No segments created"}
            
            logger.info(f"✅ Created {len(segments)} segments")
            
            # Step 3: Save segments and prepare for voice cloning
            logger.info("💾 Step 3: Saving segments...")
            
            # Load original audio for reference
            original_audio, sr = sf.read(audio_path)
            speakers = transcript_result.get('speakers', ['A'])
            detected_language = transcript_result.get('metadata', {}).get('language_code', '')
            
            # Create output directory
            output_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}"
            
            # Save segments with metadata
            self.segment_manager.save_optimal_segments(
                segments=segments,
                audio=original_audio,
                sr=sr,
                output_dir=output_dir,
                speakers=speakers,
                target_language=target_language,
                detected_language=detected_language
            )
            
            # Step 4: Prepare segments for voice cloning
            logger.info("🎯 Step 4: Preparing segments for voice cloning...")
            
            cloning_segments = self._prepare_segments_for_cloning(
                segments, 
                output_dir / "segments",
                audio_id
            )
            
            # Step 5: Validate and create OpenVoice parameters
            logger.info("⚙️ Step 5: Setting up OpenVoice parameters...")
            
            voice_args = self.voice_service.validate_parameters(
                max_length=max_length or max_tokens,  # Map legacy parameter
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                emotion=emotion or settings.OPENVOICE_DEFAULT_EMOTION
            )
            
            logger.info(f"📊 Using OpenVoice parameters: temp={voice_args.temperature}, seed={voice_args.seed}, emotion={voice_args.emotion}")
            
            # Step 6: Perform voice cloning with OpenVoice
            logger.info("🎙️ Step 6: Performing OpenVoice voice cloning...")
            
            cloning_result = self.voice_service.process_segments_batch(
                segments=cloning_segments,
                audio_id=audio_id,
                custom_args=voice_args
            )
            
            if not cloning_result.get("success", False):
                return {"success": False, "error": "OpenVoice voice cloning failed", "details": cloning_result}
            
            logger.info(f"✅ OpenVoice cloning completed: {cloning_result['successful_segments']}/{cloning_result['total_segments']} segments")
            
            # Step 7: Reconstruct final audio
            logger.info("🔧 Step 7: Reconstructing final audio...")
            
            reconstruction_result = self.audio_reconstructor.reconstruct_final_audio(
                segments_dir=str(output_dir),
                audio_id=audio_id,
                include_instruments=False  # Simplified - no instrument mixing for stability
            )
            
            if not reconstruction_result.get("success", False):
                return {"success": False, "error": "Audio reconstruction failed", "details": reconstruction_result}
            
            pipeline_time = time.time() - pipeline_start
            
            logger.info(f"🎉 Complete OpenVoice processing finished in {pipeline_time:.2f}s")
            
            # Return comprehensive results
            return {
                "success": True,
                "audio_id": audio_id,
                "processing_time": pipeline_time,
                "model_used": "OpenVoice",
                "transcription": {
                    "total_words": len(transcript_result.get('words', [])),
                    "speakers_detected": len(speakers),
                    "language_detected": detected_language,
                    "duration": transcript_result.get('duration', 0)
                },
                "segmentation": {
                    "total_segments": len(segments),
                    "segments_created": len(cloning_segments)
                },
                "voice_cloning": {
                    "model": "OpenVoice",
                    "total_segments": cloning_result['total_segments'],
                    "successful_segments": cloning_result['successful_segments'],
                    "failed_segments": cloning_result['failed_segments'],
                    "parameters_used": {
                        "temperature": voice_args.temperature,
                        "seed": voice_args.seed,
                        "emotion": voice_args.emotion,
                        "repetition_penalty": voice_args.repetition_penalty
                    }
                },
                "output": {
                    "final_audio_path": reconstruction_result.get("final_audio_path"),
                    "segments_directory": str(output_dir),
                    "cloned_segments_directory": cloning_result.get("output_directory")
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Complete OpenVoice processing failed: {str(e)}")
            return {"success": False, "error": f"OpenVoice processing pipeline failed: {str(e)}"}
    
    def process_voice_cloning_only(self, 
                                  segments_dir: str, 
                                  audio_id: str,
                                  **voice_params) -> Dict[str, Any]:
        """
        Process only voice cloning for existing segments using OpenVoice
        Useful when segments are already prepared
        """
        
        if not self.is_ready():
            return {"success": False, "error": "OpenVoice model not loaded"}
        
        logger.info(f"🎙️ Starting OpenVoice voice cloning only for {audio_id}")
        
        try:
            # Load existing segments
            segments_path = Path(segments_dir)
            if not segments_path.exists():
                return {"success": False, "error": f"Segments directory not found: {segments_dir}"}
            
            # Prepare segments for cloning
            cloning_segments = self._prepare_segments_for_cloning(
                None,  # We'll load from files
                segments_path,
                audio_id
            )
            
            if not cloning_segments:
                return {"success": False, "error": "No segments found for cloning"}
            
            # Validate parameters for OpenVoice
            voice_args = self.voice_service.validate_parameters(**voice_params)
            
            # Perform OpenVoice voice cloning
            cloning_result = self.voice_service.process_segments_batch(
                segments=cloning_segments,
                audio_id=audio_id,
                custom_args=voice_args
            )
            
            return cloning_result
            
        except Exception as e:
            logger.error(f"❌ OpenVoice voice cloning only failed: {str(e)}")
            return {"success": False, "error": f"OpenVoice voice cloning failed: {str(e)}"}
    
    def _prepare_segments_for_cloning(self, 
                                     segments: Optional[List[Dict]], 
                                     segments_dir: Path,
                                     audio_id: str) -> List[Dict]:
        """
        Prepare segments for OpenVoice voice cloning by loading metadata
        """
        
        if segments:
            # Use provided segments (from fresh processing)
            return self._convert_segments_to_cloning_format(segments, segments_dir)
        else:
            # Load segments from metadata files (for voice cloning only)
            return self._load_segments_from_metadata(segments_dir)
    
    def _convert_segments_to_cloning_format(self, 
                                          segments: List[Dict], 
                                          segments_dir: Path) -> List[Dict]:
        """
        Convert segment manager format to OpenVoice voice cloning format
        """
        cloning_segments = []
        
        # Segments are saved in a "segments" subdirectory by segment_manager
        actual_segments_dir = segments_dir / "segments"
        
        for i, segment in enumerate(segments):
            # Find corresponding audio file in the correct subdirectory
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = actual_segments_dir / audio_filename
            
            # Find corresponding metadata in the correct subdirectory
            metadata_filename = f"segment_{i+1:03d}_metadata.json"
            metadata_path = actual_segments_dir / metadata_filename
            
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                cloning_segment = {
                    'segment_index': metadata.get('segment_index', i + 1),
                    'audio_path': str(audio_path) if audio_path.exists() else None,
                    'original_text': metadata.get('original_text', ''),
                    'english_text': metadata.get('english_text', ''),
                    'start': metadata.get('start', 0.0),
                    'end': metadata.get('end', 0.0),
                    'duration': metadata.get('duration', 0.0),
                    'speaker': metadata.get('speaker', 'A'),
                    'confidence': metadata.get('confidence', 0.0)
                }
                
                cloning_segments.append(cloning_segment)
                logger.info(f"Converted segment {i+1}: '{cloning_segment['english_text'][:50]}...'")
        
        return cloning_segments
    
    def _load_segments_from_metadata(self, segments_dir: Path) -> List[Dict]:
        """
        Load segments from existing metadata files for OpenVoice processing
        """
        cloning_segments = []
        
        # Segments are saved in a "segments" subdirectory by segment_manager
        actual_segments_dir = segments_dir / "segments"
        
        if not actual_segments_dir.exists():
            logger.warning(f"Segments subdirectory not found: {actual_segments_dir}")
            return cloning_segments
        
        # Find all metadata files in the correct subdirectory
        metadata_files = sorted(list(actual_segments_dir.glob("*_metadata.json")))
        
        logger.info(f"Found {len(metadata_files)} metadata files for OpenVoice processing in {actual_segments_dir}")
        
        for metadata_file in metadata_files:
            try:
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                cloning_segment = {
                    'segment_index': metadata.get('segment_index', 1),
                    'audio_path': metadata.get('audio_path'),
                    'original_text': metadata.get('original_text', ''),
                    'english_text': metadata.get('english_text', ''),
                    'start': metadata.get('start', 0.0),
                    'end': metadata.get('end', 0.0),
                    'duration': metadata.get('duration', 0.0),
                    'speaker': metadata.get('speaker', 'A'),
                    'confidence': metadata.get('confidence', 0.0)
                }
                
                cloning_segments.append(cloning_segment)
                logger.info(f"Loaded segment {cloning_segment['segment_index']}: '{cloning_segment['english_text'][:50]}...'")
                
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(cloning_segments)} segments for OpenVoice voice cloning")
        return cloning_segments
    
    def get_processing_stats(self, audio_id: str) -> Dict[str, Any]:
        """
        Get processing statistics for an audio ID
        """
        try:
            output_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}"
            
            if not output_dir.exists():
                return {"success": False, "error": "Processing directory not found"}
            
            # Count segments
            segments_dir = output_dir / "segments"
            metadata_files = list(segments_dir.glob("*_metadata.json")) if segments_dir.exists() else []
            
            # Count cloned segments
            cloned_dir = output_dir / "cloned_segments"
            cloned_files = list(cloned_dir.glob("cloned_segment_*.wav")) if cloned_dir.exists() else []
            
            # Check final audio
            final_audio_path = output_dir / f"final_output_{audio_id}.wav"
            
            return {
                "success": True,
                "audio_id": audio_id,
                "model_used": "OpenVoice",
                "total_segments": len(metadata_files),
                "cloned_segments": len(cloned_files),
                "final_audio_exists": final_audio_path.exists(),
                "final_audio_path": str(final_audio_path) if final_audio_path.exists() else None,
                "processing_complete": len(metadata_files) > 0 and len(cloned_files) == len(metadata_files) and final_audio_path.exists()
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def cleanup_processing_files(self, audio_id: str) -> bool:
        """
        Clean up processing files for an audio ID
        """
        try:
            import shutil
            output_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}"
            
            if output_dir.exists():
                shutil.rmtree(output_dir)
                logger.info(f"🧹 Cleaned up OpenVoice processing files for {audio_id}")
                return True
            
            return True  # Already clean
            
        except Exception as e:
            logger.error(f"Error cleaning up files for {audio_id}: {str(e)}")
            return False
    
    def clear_model(self):
        """Clear OpenVoice model from memory"""
        self.voice_service.clear_model()
        logger.info("🧹 OpenVoice model cleared from memory")