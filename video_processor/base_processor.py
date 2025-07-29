"""
Simplified Base Audio Processor - Core functionality only
Voice cloning now handled by CleanAudioProcessor
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from .transcription import TranscriptionService
from .segment_manager import SegmentManager
from .audio_utils import AudioUtils
from .file_manager import FileManager
from .audio_reconstructor import AudioReconstructor
from .video_processor import VideoProcessor
from .runpod_service import RunPodService
from config import settings


class AudioProcessor:
    """Simplified audio processor - core functionality only (no voice cloning)"""
    
    def __init__(self, temp_dir: str = "./tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core services (no voice cloning service)
        self.transcription_service = TranscriptionService()
        self.segment_manager = SegmentManager(self.transcription_service)
        self.audio_utils = AudioUtils()
        self.file_manager = FileManager()
        self.audio_reconstructor = AudioReconstructor(str(self.temp_dir))
        self.video_processor = VideoProcessor(str(self.temp_dir))
        self.runpod_service = RunPodService()
        
        logger.info("✅ Simplified AudioProcessor initialized (voice cloning handled separately)")
    
    def process_video_with_separation(self, video_path: str, audio_id: str, 
                                    target_language: str = "English",
                                    language_code: Optional[str] = None,
                                    speakers_expected: Optional[int] = 1,
                                    include_instruments: bool = True) -> Dict[str, Any]:
        """Process video with proper vocal/instrument separation using RunPod"""
        try:
            logger.info(f"🎬 Processing video with separation for {audio_id}")
            
            # Extract audio from video using audio_utils
            audio_output_path = self.temp_dir / f"extracted_audio_{audio_id}.wav"
            extraction_result = self.audio_utils.extract_audio_from_video(video_path, str(audio_output_path))
            
            if not extraction_result.get("success", False):
                return {"success": False, "error": f"Failed to extract audio from video: {extraction_result.get('error', 'Unknown error')}"}
            
            audio_path = str(audio_output_path)
            
            # Perform vocal/instrument separation if requested
            vocal_path = audio_path  # Default to original audio
            instruments_path = None
            
            if include_instruments:
                try:
                    logger.info(f"🎵 Starting vocal/instrument separation for {audio_id}")
                    
                    # Upload audio to R2 for RunPod processing
                    from r2_storage import R2Storage
                    r2_storage = R2Storage()
                    
                    # Upload extracted audio to get URL for RunPod
                    upload_result = r2_storage.upload_file(
                        local_path=audio_path,
                        r2_key=f"processing/{audio_id}/extracted_audio.wav"
                    )
                    
                    if upload_result.get("success"):
                        audio_url = upload_result.get("url")
                        
                        # Submit to RunPod for separation
                        separation_result = self.runpod_service.process_audio_separation(audio_url)
                        
                        if separation_result.get("success"):
                            # Wait for completion
                            completion_result = self.runpod_service.wait_for_completion(separation_result["id"])
                            
                            if completion_result.get("status") == "COMPLETED":  # Changed from get("success") to status check
                                # Download separated files
                                output = completion_result.get("output", {})  # Get output object
                                vocal_url = output.get("vocal_audio")  # Get from output object
                                instruments_url = output.get("instrument_audio")  # Get from output object
                                
                                if vocal_url:
                                    vocal_path = str(self.temp_dir / f"vocal_separated_{audio_id}.wav")
                                    self._download_file(vocal_url, vocal_path)
                                    logger.info(f"✅ Downloaded separated vocal: {vocal_path}")
                                
                                if instruments_url:
                                    instruments_path = str(self.temp_dir / f"instruments_separated_{audio_id}.wav")
                                    self._download_file(instruments_url, instruments_path)
                                    logger.info(f"✅ Downloaded separated instruments: {instruments_path}")
                            else:
                                logger.warning("RunPod separation completed but failed to get results, using original audio")
                        else:
                            logger.warning("RunPod separation failed, using original audio")
                    else:
                        logger.warning("Failed to upload audio for separation, using original audio")
                        
                except Exception as e:
                    logger.warning(f"Separation failed: {str(e)}, using original audio")
            
            # Process audio segments using vocal track (or original if separation failed)
            segments_result = self.process_audio_segments(
                audio_path=vocal_path,
                audio_id=audio_id,
                target_language=target_language,
                language_code=language_code,
                speakers_expected=speakers_expected
            )
            
            if not segments_result.get("success", False):
                return {"success": False, "error": "Failed to process audio segments"}
            
            return {
                "success": True,
                "segments_dir": segments_result.get("segments_dir"),
                "audio_path": vocal_path,  # Return vocal path for voice cloning
                "instruments_path": instruments_path,  # Return instruments path for mixing
                "total_segments": segments_result.get("total_segments", 0),
                "separation_performed": include_instruments and instruments_path is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Video processing with separation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_audio_segments(self, audio_path: str, audio_id: str, 
                             target_language: str = "English",
                             language_code: Optional[str] = None,
                             speakers_expected: Optional[int] = 1,
                             original_audio_details: Optional[Dict] = None) -> Dict[str, Any]:
        """Process audio into segments (no voice cloning)"""
        try:
            logger.info(f"🎵 Processing audio segments for {audio_id}")
            
            # Get audio duration for metadata
            audio_data, sr = sf.read(audio_path)
            audio_duration = len(audio_data) / sr
            
            # Transcribe audio
            logger.info("📝 Transcribing audio...")
            transcript_result = self.transcription_service.transcribe_audio(
                audio_path=audio_path,
                language_code=language_code,
                speakers_expected=speakers_expected,
                audio_id=audio_id,
                original_duration=audio_duration
            )
            
            if not transcript_result:
                return {"success": False, "error": "Transcription failed"}
            
            # Create segments
            logger.info("✂️ Creating optimal segments...")
            segments = self.segment_manager.create_optimal_segments(transcript_result)
            
            if not segments:
                return {"success": False, "error": "No segments created"}
            
            # Save segments
            logger.info("💾 Saving segments...")
            output_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}"
            
            speakers = transcript_result.get('speakers', ['A'])
            detected_language = transcript_result.get('metadata', {}).get('language_code', '')
            
            self.segment_manager.save_optimal_segments(
                segments=segments,
                audio=audio_data,
                sr=sr,
                output_dir=output_dir,
                speakers=speakers,
                target_language=target_language,
                detected_language=detected_language,
                original_audio_details=original_audio_details
            )
            
            logger.info(f"✅ Created {len(segments)} segments for processing")
            
            return {
                "success": True,
                "segments_dir": str(output_dir),
                "total_segments": len(segments),
                "speakers": speakers,
                "detected_language": detected_language
            }
            
        except Exception as e:
            logger.error(f"❌ Audio segments processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio from cloned segments"""
        try:
            return self.audio_reconstructor.reconstruct_final_audio(
                segments_dir=segments_dir,
                audio_id=audio_id,
                include_instruments=include_instruments,
                instruments_path=instruments_path
            )
        except Exception as e:
            logger.error(f"❌ Audio reconstruction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with subtitles"""
        try:
            return self.video_processor.create_video_with_subtitles(
                video_path=video_path,
                audio_path=audio_path,
                segments_dir=segments_dir,
                audio_id=audio_id,
                instruments_path=instruments_path
            )
        except Exception as e:
            logger.error(f"❌ Video creation with subtitles failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                               audio_id: str, instruments_path: Optional[str] = None,
                               segments_dir: Optional[str] = None) -> Dict[str, Any]:
        """Create video with audio"""
        try:
            return self.video_processor.create_video_with_audio(
                video_path=video_path,
                audio_path=audio_path,
                audio_id=audio_id,
                instruments_path=instruments_path,
                segments_dir=segments_dir
            )
        except Exception as e:
            logger.error(f"❌ Video creation with audio failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            segments_path = Path(segments_dir)
            if not segments_path.exists():
                return {"success": False, "error": "Segments directory not found"}
            
            # Count segments
            segments_subdir = segments_path / "segments"
            if segments_subdir.exists():
                metadata_files = list(segments_subdir.glob("*_metadata.json"))
                total_segments = len(metadata_files)
            else:
                total_segments = 0
            
            # Count cloned segments
            cloned_dir = segments_path / "cloned_segments"
            if cloned_dir.exists():
                cloned_files = list(cloned_dir.glob("cloned_segment_*.wav"))
                cloned_segments = len(cloned_files)
            else:
                cloned_segments = 0
            
            completion_percentage = (cloned_segments / total_segments * 100) if total_segments > 0 else 0
            
            return {
                "success": True,
                "total_segments": total_segments,
                "cloned_segments": cloned_segments,
                "completion_percentage": completion_percentage,
                "processing_complete": cloned_segments == total_segments and total_segments > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting processing stats: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            import shutil
            temp_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 Cleaned up temp files for {audio_id}")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
    
    def _download_file(self, url: str, local_path: str) -> bool:
        """Download file from URL to local path"""
        try:
            import requests
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            return False
