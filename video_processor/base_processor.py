"""
Base Audio Processor Module - Simplified for Voice Cloning
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import soundfile as sf

from config import settings
from .audio_reconstructor import AudioReconstructor
from .audio_utils import AudioUtils
from .file_manager import FileManager
from .runpod_service import RunPodService
from .segment_manager import SegmentManager
from .transcription import TranscriptionService
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Simplified audio processor for voice cloning"""
    
    def __init__(self, temp_dir: str = "./tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.transcription_service = TranscriptionService()
        self.segment_manager = SegmentManager()
        self.audio_utils = AudioUtils()
        self.file_manager = FileManager()
        self.audio_reconstructor = AudioReconstructor(str(self.temp_dir))
        self.video_processor = VideoProcessor(str(self.temp_dir))
        self.runpod_service = RunPodService()
    
    def process_audio_segments(self, audio_path: str, audio_id: str, 
                             target_language: str = "English",
                             language_code: Optional[str] = None,
                             speakers_expected: Optional[int] = 1,
                             original_audio_details: Optional[Dict] = None) -> Dict[str, Any]:
        """Process audio segments for voice cloning"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Calculate original audio duration
            original_duration = len(audio) / sr
            logger.info(f"Original audio duration: {original_duration:.2f} seconds")
            
            # Transcribe audio
            transcript_data = self.transcription_service.transcribe_audio(
                audio_path, language_code, speakers_expected, audio_id, original_duration
            )
            
            # Update progress for transcription completion
            try:
                from status_manager import status_manager, ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=50, 
                    details={"message": "Transcription completed, preparing segments"}
                )
            except:
                pass
            
            # Create optimal segments
            segments = self.segment_manager.create_optimal_segments(
                transcript_data, target_language, audio_id
            )
            
            if not segments:
                raise ValueError("No valid segments created from transcript")
            
            logger.info(f"Created {len(segments)} segments for processing")
            
            # Create output directory
            output_dir = self.temp_dir / f"segments_{audio_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            self.file_manager.create_directory_structure(output_dir, transcript_data['speakers'])
            
            # Save segments
            detected_language = transcript_data.get('metadata', {}).get('language_code', 'en')
            self.segment_manager.save_optimal_segments(
                segments, audio, sr, output_dir, 
                transcript_data['speakers'], target_language, detected_language,
                original_audio_details
            )
            
            # Voice cloning with Fish Speech
            cloning_result = self._perform_voice_cloning(
                output_dir, audio_path, audio_id, segments
            )
            
            # Audio reconstruction - create complete dubbed track
            if cloning_result.get("success"):
                reconstruction_result = self._perform_audio_reconstruction(
                    output_dir, audio_id, False, None  # No instruments for vocal-only processing
                )
            else:
                reconstruction_result = {"success": False, "error": "Voice cloning failed"}
            
            return {
                "success": True,
                "segments_dir": str(output_dir),
                "audio_id": audio_id,
                "speakers": transcript_data.get('speakers', []),
                "total_segments": len(segments),
                "total_duration": transcript_data.get('duration', original_duration),
                "language_code": language_code,
                "detected_speakers": len(transcript_data.get('speakers', [])),
                "speakers_expected": speakers_expected,
                "voice_cloning": cloning_result,
                "audio_reconstruction": reconstruction_result
            }
            
        except Exception as e:
            return {"success": False, "error": f"Audio processing failed: {str(e)}"}
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio from cloned segments"""
        result = self.audio_reconstructor.reconstruct_final_audio(
            segments_dir, audio_id, include_instruments, instruments_path
        )
        
        # Add final_audio_path key for compatibility
        if result.get("success") and "output_path" in result:
            result["final_audio_path"] = result["output_path"]
        
        return result
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with subtitles"""
        return self.video_processor.create_video_with_subtitles(
            video_path, audio_path, segments_dir, audio_id, instruments_path
        )
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                               audio_id: str, instruments_path: Optional[str] = None,
                               segments_dir: Optional[str] = None) -> Dict[str, Any]:
        """Create video with new audio only"""
        return self.video_processor.create_video_with_audio(
            video_path, audio_path, audio_id, instruments_path, segments_dir
        )
    
    def process_video_with_separation(self, video_path: str, audio_id: str, 
                                    target_language: str = "English",
                                    language_code: Optional[str] = None,
                                    speakers_expected: Optional[int] = 1,
                                    include_instruments: bool = False) -> Dict[str, Any]:
        """Process video with RunPod vocal/instrument separation"""
        try:
            # Extract audio from video
            audio_temp_path = self.temp_dir / f"{audio_id}_extracted_audio.wav"
            extract_result = self.audio_utils.extract_audio_from_video(video_path, str(audio_temp_path))
            
            if not extract_result["success"]:
                return {"success": False, "error": f"Audio extraction failed: {extract_result['error']}"}
            
            from r2_storage import R2Storage
            r2_storage = R2Storage()
            
            upload_result = r2_storage.upload_file(
                str(audio_temp_path),
                f"temp/{audio_id}_audio.wav",
                "audio/wav"
            )
            
            if not upload_result["success"]:
                return {"success": False, "error": f"Failed to upload audio: {upload_result.get('error', 'Unknown error')}"}
            
            audio_url = upload_result["url"]
            
            # Update progress for audio separation completion
            try:
                from status_manager import status_manager, ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=40, 
                    details={"message": "Audio separation completed, preparing transcription"}
                )
            except:
                pass

            # Process with RunPod Queue Service
            from runpod_queue_service import runpod_queue_service
            
            completion_result = runpod_queue_service.process_audio_separation_sync(
                audio_url, 
                caller_info="video_processing"
            )
            
            if completion_result.get("status") != "COMPLETED":
                error_msg = completion_result.get("error", "Unknown error")
                return {"success": False, "error": f"RunPod job failed: {error_msg}"}
            
            # Update progress for transcription completion
            try:
                from status_manager import status_manager, ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=50, 
                    details={"message": "Transcription completed, preparing segments"}
                )
            except:
                pass
            
            # Validate output
            if not completion_result.get("output") or not completion_result["output"].get("vocal_audio"):
                return {"success": False, "error": "No output URLs provided"}
            
            # Download separated audio
            vocal_path = self.temp_dir / f"{audio_id}_vocal.wav"
            instrument_path = self.temp_dir / f"{audio_id}_instruments.wav"
            
            vocal_download = self.audio_utils.download_audio_file(
                completion_result["output"]["vocal_audio"], str(vocal_path)
            )
            
            instrument_download = self.audio_utils.download_audio_file(
                completion_result["output"]["instrument_audio"], str(instrument_path)
            )
            
            if not vocal_download["success"] or not instrument_download["success"]:
                return {"success": False, "error": "Failed to download separated audio"}
            
            # Store separated audio locally for backup and future use
            self._store_separated_audio_locally(str(vocal_path), str(instrument_path), audio_id)
            
            # Calculate original audio details
            import soundfile as sf
            vocal_audio, vocal_sr = sf.read(str(vocal_path))
            original_audio_details = {
                "duration": len(vocal_audio) / vocal_sr,
                "sample_rate": vocal_sr,
                "channels": len(vocal_audio.shape) if len(vocal_audio.shape) > 1 else 1,
                "processing_type": "video_with_separation"
            }
            
            # Process vocal audio
            segment_result = self.process_audio_segments(
                str(vocal_path), 
                audio_id, 
                target_language,
                language_code=language_code,
                speakers_expected=speakers_expected,
                original_audio_details=original_audio_details
            )
            
            if not segment_result.get("success", True):
                return {"success": False, "error": f"Audio segmentation failed: {segment_result.get('error', 'Unknown error')}"}
            
            # Get cloning and reconstruction results
            voice_cloning = segment_result.get("voice_cloning", {})
            audio_reconstruction = segment_result.get("audio_reconstruction", {})
            
            # If voice cloning succeeded, perform final reconstruction with instruments if requested
            if voice_cloning.get("success") and audio_reconstruction.get("success"):
                final_reconstruction = self._perform_audio_reconstruction(
                    Path(segment_result["segments_dir"]), audio_id, include_instruments, 
                    str(instrument_path) if include_instruments else None
                )
            else:
                final_reconstruction = audio_reconstruction
            
            return {
                "success": True,
                "segments_dir": segment_result["segments_dir"],
                "vocal_path": str(vocal_path),
                "instruments_path": str(instrument_path),
                "audio_id": audio_id,
                "speakers": segment_result["speakers"],
                "total_segments": segment_result["total_segments"],
                "total_duration": segment_result["total_duration"],
                "detected_speakers": segment_result.get("detected_speakers", len(segment_result.get("speakers", []))),
                "voice_cloning": voice_cloning,
                "audio_reconstruction": audio_reconstruction,
                "final_dubbed_audio": final_reconstruction
            }
            
        except Exception as e:
            return {"success": False, "error": f"Video processing failed: {str(e)}"}
    
    def cleanup_temp_files(self, audio_id: str, keep_final_output: bool = True) -> Dict[str, Any]:
        """Clean up temporary files"""
        return self.file_manager.cleanup_temp_files(audio_id, keep_final_output)
    
    def _store_separated_audio_locally(self, vocal_path: str, instrument_path: str, audio_id: str):
        """Store separated vocal and instrument audio locally"""
        try:
            from utils import local_storage
            
            # Store vocal audio
            with open(vocal_path, 'rb') as f:
                vocal_content = f.read()
            
            vocal_result = local_storage.store_audio(
                audio_id, 
                vocal_content, 
                f"{audio_id}_vocal_separated.wav"
            )
            
            # Store instrument audio
            with open(instrument_path, 'rb') as f:
                instrument_content = f.read()
            
            instrument_result = local_storage.store_audio(
                audio_id, 
                instrument_content, 
                f"{audio_id}_instruments_separated.wav"
            )
            
            if vocal_result.get("success") and instrument_result.get("success"):
                logger.info(f"Separated audio stored locally for {audio_id}")
                logger.info(f"Vocal: {vocal_result['local_path']}")
                logger.info(f"Instruments: {instrument_result['local_path']}")
            else:
                logger.warning(f"Failed to store separated audio locally: vocal={vocal_result.get('success')}, instruments={instrument_result.get('success')}")
                
        except Exception as e:
            logger.warning(f"Failed to store separated audio locally: {str(e)}")
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics from segments directory"""
        try:
            from pathlib import Path
            import json
            
            segments_path = Path(segments_dir)
            
            # Read segmentation summary
            summary_file = segments_path / "segmentation_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    
                return {
                    "total_segments": summary_data.get("total_segments", 0),
                    "total_duration": summary_data.get("total_duration", 0),
                    "speakers": summary_data.get("speakers", []),
                    "segmentation_method": summary_data.get("segmentation_method", "unknown"),
                    "segment_files_count": len(summary_data.get("segment_files", []))
                }
            else:
                # Fallback: count files manually
                segments_folder = segments_path / "segments"
                if segments_folder.exists():
                    audio_files = list(segments_folder.glob("*.wav"))
                    return {
                        "total_segments": len(audio_files),
                        "total_duration": 0,
                        "speakers": [],
                        "segmentation_method": "file_count",
                        "segment_files_count": len(audio_files)
                    }
                
                return {
                    "total_segments": 0,
                    "total_duration": 0,
                    "speakers": [],
                    "segmentation_method": "none",
                    "segment_files_count": 0
                }
                
        except Exception as e:
            logger.warning(f"Failed to get processing stats: {str(e)}")
            return {
                "total_segments": 0,
                "total_duration": 0,
                "speakers": [],
                "segmentation_method": "error",
                "segment_files_count": 0
            }
    
    def _perform_voice_cloning(self, segments_dir: Path, reference_audio_path: str, 
                              audio_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform voice cloning for all segments using Fish Speech"""
        try:
            logger.info(f"Starting voice cloning for {len(segments)} segments")
            
            # Get Fish Speech service (prefer app state, fallback to global)
            from .voice_cloning import get_fish_speech_service
            fish_service = get_fish_speech_service()
            
            # Create cloned directory
            cloned_dir = segments_dir / "cloned"
            cloned_dir.mkdir(exist_ok=True)
            
            # Update progress
            try:
                from status_manager import status_manager, ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=70, 
                    details={"message": "Starting voice cloning with Fish Speech..."}
                )
            except:
                pass
            
            cloning_stats = {
                "total_segments": len(segments),
                "successful_clones": 0,
                "failed_clones": 0,
                "skipped_segments": 0,
                "cloning_errors": [],
                "model_used": "fish_speech_openaudio_s1"
            }
            
            for i, segment in enumerate(segments):
                try:
                    # Update progress for each segment
                    segment_progress = 70 + int((i / len(segments)) * 15)  # 70% to 85%
                    try:
                        status_manager.update_status(
                            audio_id, 
                            ProcessingStatus.PROCESSING, 
                            progress=segment_progress,
                            details={"message": f"Cloning voice for segment {i+1}/{len(segments)}"}
                        )
                    except:
                        pass
                    
                    # Skip silence segments for voice cloning (keep for reconstruction)
                    segment_type = segment.get("type", "speech")
                    if segment_type == "silence":
                        cloning_stats["skipped_segments"] += 1
                        logger.info(f"Skipping silence segment {segment.get('segment_index', i+1)} for voice cloning")
                        continue
                    
                    # Get segment metadata file
                    segment_index = segment.get("segment_index", i + 1)
                    metadata_file = segments_dir / "segments" / f"segment_{segment_index:03d}_metadata.json"
                    
                    if not metadata_file.exists():
                        logger.warning(f"Metadata file not found for segment {segment_index}")
                        cloning_stats["failed_clones"] += 1
                        continue
                    
                    # Load metadata
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        segment_metadata = json.load(f)
                    
                    # Perform voice cloning
                    cloning_result = fish_service.clone_voice_for_segment(
                        segment_metadata, reference_audio_path, audio_id
                    )
                    
                    # Define expected path for both success and failure cases
                    expected_cloned_path = cloned_dir / f"cloned_segment_{segment_index:03d}.wav"
                    
                    if cloning_result.get("success"):
                        # Move cloned audio to expected location
                        cloned_audio_path = cloning_result["cloned_audio_path"]
                        
                        if Path(cloned_audio_path) != expected_cloned_path:
                            shutil.move(str(cloned_audio_path), str(expected_cloned_path))
                        
                        cloning_stats["successful_clones"] += 1
                        logger.info(f"Successfully cloned voice for segment {segment_index}")
                        
                    else:
                        error_msg = cloning_result.get("error", "Unknown error")
                        cloning_stats["failed_clones"] += 1
                        cloning_stats["cloning_errors"].append({
                            "segment_index": segment_index,
                            "error": error_msg
                        })
                        logger.error(f"Voice cloning failed for segment {segment_index}: {error_msg}")
                        
                        # Create fallback silent audio for failed segments
                        self._create_fallback_audio(segment_metadata, expected_cloned_path)
                        
                except Exception as e:
                    logger.error(f"Error processing segment {i+1}: {str(e)}")
                    cloning_stats["failed_clones"] += 1
                    cloning_stats["cloning_errors"].append({
                        "segment_index": segment.get("segment_index", i+1),
                        "error": str(e)
                    })
            
            # Final progress update
            try:
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=85,
                    details={"message": f"Voice cloning completed: {cloning_stats['successful_clones']}/{cloning_stats['total_segments']} successful"}
                )
            except:
                pass
            
            logger.info(f"Voice cloning completed: {cloning_stats['successful_clones']}/{cloning_stats['total_segments']} successful")
            
            return {
                "success": True,
                "stats": cloning_stats
            }
            
        except Exception as e:
            logger.error(f"Voice cloning process failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stats": {
                    "total_segments": len(segments) if segments else 0,
                    "successful_clones": 0,
                    "failed_clones": len(segments) if segments else 0,
                    "model_used": "fish_speech_openaudio_s1"
                }
            }
    
    def _create_fallback_audio(self, segment_metadata: Dict[str, Any], output_path: Path):
        """Create silent fallback audio for failed voice cloning"""
        try:
            duration = segment_metadata.get("duration", 1.0)
            sample_rate = segment_metadata.get("sample_rate", 44100)
            
            # Create silent audio
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            sf.write(str(output_path), silence, sample_rate)
            
            logger.info(f"Created fallback silent audio for segment {segment_metadata.get('segment_index', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to create fallback audio: {str(e)}")
    
    def _perform_audio_reconstruction(self, segments_dir: Path, audio_id: str, 
                                    include_instruments: bool = False, 
                                    instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform complete audio reconstruction from cloned segments"""
        try:
            logger.info(f"🎵 Starting audio reconstruction for {audio_id}")
            
            # Update progress
            try:
                from status_manager import status_manager, ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=85, 
                    details={"message": "Reconstructing dubbed audio from cloned segments..."}
                )
            except:
                pass
            
            # Perform reconstruction
            reconstruction_result = self.audio_reconstructor.reconstruct_final_audio(
                str(segments_dir), audio_id, include_instruments, instruments_path
            )
            
            if reconstruction_result.get("success"):
                # Update progress to completion
                try:
                    stats = reconstruction_result.get("reconstruction_stats", {})
                    cloned_segments = stats.get("cloned_segments", 0)
                    total_segments = stats.get("speech_segments", 0)
                    
                    status_manager.update_status(
                        audio_id, 
                        ProcessingStatus.PROCESSING, 
                        progress=95,
                        details={
                            "message": f"Audio reconstruction completed: {cloned_segments}/{total_segments} segments dubbed",
                            "reconstruction_stats": stats
                        }
                    )
                except:
                    pass
                
                logger.info(f"✅ Audio reconstruction completed successfully")
                
                return {
                    "success": True,
                    "dubbed_audio_path": reconstruction_result["output_path"],
                    "final_audio_path": reconstruction_result["output_path"],  # Add missing key
                    "duration": reconstruction_result["duration"],
                    "stats": reconstruction_result.get("reconstruction_stats", {})
                }
            
            else:
                error_msg = reconstruction_result.get("error", "Unknown reconstruction error")
                logger.error(f"❌ Audio reconstruction failed: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Audio reconstruction process failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
