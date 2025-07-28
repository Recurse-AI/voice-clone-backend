"""
Base Audio Processor Module - Simplified for Voice Cloning
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)

from .transcription import TranscriptionService
from .segment_manager import SegmentManager
from .audio_utils import AudioUtils
from .file_manager import FileManager
from .audio_reconstructor import AudioReconstructor
from .video_processor import VideoProcessor
from .voice_cloning import VoiceCloningService
from .runpod_service import RunPodService
from config import settings


class AudioProcessor:
    """Simplified audio processor for voice cloning"""
    
    def __init__(self, temp_dir: str = "./tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.transcription_service = TranscriptionService()
        self.segment_manager = SegmentManager(self.transcription_service)
        self.audio_utils = AudioUtils()
        self.file_manager = FileManager()
        self.voice_cloning_service = VoiceCloningService(self.segment_manager)
        self.audio_reconstructor = AudioReconstructor(str(self.temp_dir))
        self.video_processor = VideoProcessor(str(self.temp_dir))
        self.runpod_service = RunPodService()
    
    def load_dia_model(self, repo_id: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model for voice cloning"""
        return self.voice_cloning_service.load_dia_model(repo_id)
    
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
            
            # Create segments
            segments = self.segment_manager.create_optimal_segments(transcript_data)
            
            if not segments:
                return {"success": False, "error": "No viable segments created"}
            
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
            
            return {
                "success": True,
                "segments_dir": str(output_dir),
                "audio_id": audio_id,
                "speakers": transcript_data['speakers'],
                "total_segments": len(segments),
                "total_duration": transcript_data['duration'],
                "language_code": language_code,
                "detected_speakers": len(transcript_data['speakers']),
                "speakers_expected": speakers_expected
            }
            
        except Exception as e:
            return {"success": False, "error": f"Audio processing failed: {str(e)}"}
    
    def clone_voice_segments(self, segments_dir: str, audio_id: str, 
                           # Enhanced parameter controls from Colab
                           max_tokens: int = None,
                           cfg_scale: float = None, 
                           temperature: float = None,
                           top_p: float = None,
                           cfg_filter_top_k: int = None,
                           speed_factor: float = None,
                           seed: Optional[int] = None, 
                           use_torch_compile: bool = None) -> Dict[str, Any]:
        """Enhanced voice cloning with advanced parameter controls and unified segment processing"""
        logger.info(f"Starting enhanced unified voice cloning for audio_id: {audio_id}")
        logger.info(f"Parameters: max_tokens={max_tokens}, cfg_scale={cfg_scale}, temp={temperature}, top_p={top_p}, speed={speed_factor}")
        
        if not self.voice_cloning_service.is_model_loaded():
            logger.error("Dia model not loaded")
            return {"success": False, "error": "Dia model not loaded"}
        
        logger.info("Dia model is loaded successfully")
        
        try:
            segments_path = Path(segments_dir)
            if not segments_path.exists():
                logger.error(f"Segments directory not found: {segments_dir}")
                return {"success": False, "error": "Segments directory not found"}
            
            # Get all JSON metadata files from segments subdirectory
            segments_subdir = segments_path / "segments"
            if not segments_subdir.exists():
                logger.error(f"Segments subdirectory not found: {segments_subdir}")
                return {"success": False, "error": "Segments subdirectory not found"}
            
            metadata_files = sorted(list(segments_subdir.glob("*_metadata.json")))
            logger.info(f"Found {len(metadata_files)} metadata files in {segments_subdir}")
            
            if not metadata_files:
                logger.error("No segment metadata files found")
                return {"success": False, "error": "No segment metadata files found"}
            
            # Collect all segments
            all_segments = []
            base_seed = seed if seed is not None else random.randint(1, 1000000)
            
            for json_file in metadata_files:
                try:
                    logger.info(f"Loading metadata from: {json_file}")
                    
                    with open(json_file, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    # Create segment data for unified processing
                    segment_data = {
                        'segment_index': metadata.get('segment_index', 1),
                        'audio_path': metadata.get('audio_path'),
                        'original_text': metadata.get('original_text', ''),
                        'english_text': metadata.get('english_text', ''),
                        'start': metadata.get('start', 0.0),
                        'end': metadata.get('end', 0.0),
                        'duration': metadata.get('duration', 0.0),
                        'speaker': metadata.get('speaker', 'A'),
                        'confidence': metadata.get('confidence', 0.0),
                        'word_count': metadata.get('word_count', 0)
                    }
                    
                    # Validate required data
                    if not segment_data.get('audio_path') or not os.path.exists(segment_data['audio_path']):
                        logger.warning(f"Audio file missing for segment: {json_file}")
                        # Still add the segment but mark it for silence generation
                        segment_data['audio_path'] = None
                    
                    # Ensure speaker is set
                    if not segment_data.get('speaker'):
                        segment_data['speaker'] = 'A'  # Default speaker
                    
                    all_segments.append(segment_data)
                    logger.info(f"Successfully added segment from {json_file.name} for speaker {segment_data.get('speaker', 'A')}")
                    
                except Exception as e:
                    logger.error(f"Error loading metadata from {json_file}: {e}")
                    continue
            
            if not all_segments:
                logger.error("No valid segments found for processing")
                return {"success": False, "error": "No valid segments found"}
            
            logger.info(f"Collected {len(all_segments)} total segments for enhanced unified processing")
            
            # Process all segments together with enhanced parameters
            logger.info("Calling enhanced unified voice cloning service")
            cloning_result = self.voice_cloning_service.clone_voice_segments(
                segments=all_segments, 
                max_tokens=max_tokens,
                cfg_scale=cfg_scale, 
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                speed_factor=speed_factor,
                seed=base_seed, 
                use_torch_compile=use_torch_compile,
                audio_id=audio_id
            )
            
            if not cloning_result.get('success', False):
                logger.error(f"Unified cloning failed: {cloning_result.get('error', 'Unknown error')}")
                return cloning_result
            
            logger.info(f"Unified cloning successful: {cloning_result.get('successful_clones', 0)} segments cloned")
            
            return {
                "success": True,
                "total_segments": len(all_segments),
                "successful_clones": cloning_result.get('successful_clones', 0),
                "cloned_segments": cloning_result.get('cloned_segments', []),
                "seed_used": base_seed,
                "speaker_seeds": cloning_result.get('speaker_seeds', {}),
                "cloning_duration": cloning_result.get('cloning_duration', 0)
            }
            
        except Exception as e:
            logger.error(f"Unified voice cloning failed: {str(e)}")
            return {"success": False, "error": f"Unified voice cloning failed: {str(e)}"}
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio from cloned segments"""
        return self.audio_reconstructor.reconstruct_final_audio(
            segments_dir, audio_id, include_instruments, instruments_path
        )
    
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
                                    speakers_expected: Optional[int] = 1) -> Dict[str, Any]:
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
            
            # Process with RunPod Queue Service
            from runpod_queue_service import runpod_queue_service
            
            completion_result = runpod_queue_service.process_audio_separation_sync(
                audio_url, 
                caller_info="video_processing"
            )
            
            if completion_result.get("status") != "COMPLETED":
                error_msg = completion_result.get("error", "Unknown error")
                return {"success": False, "error": f"RunPod job failed: {error_msg}"}
            
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
            
            return {
                "success": True,
                "segments_dir": segment_result["segments_dir"],
                "vocal_path": str(vocal_path),
                "instruments_path": str(instrument_path),
                "audio_id": audio_id,
                "speakers": segment_result["speakers"],
                "total_segments": segment_result["total_segments"],
                "total_duration": segment_result["total_duration"],
                "detected_speakers": segment_result.get("detected_speakers", len(segment_result.get("speakers", [])))
            }
            
        except Exception as e:
            return {"success": False, "error": f"Video processing failed: {str(e)}"}
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            if hasattr(self.transcription_service, 'translation_cache'):
                self.transcription_service.translation_cache.clear()
            
            self.voice_cloning_service.clear_cache()
            self.file_manager.cleanup_temp_files(audio_id)
            self.video_processor.cleanup_temp_files(audio_id)
            self.audio_reconstructor.cleanup_temp_files(audio_id)
            
            temp_files = [
                f"{audio_id}_extracted_audio.wav",
                f"{audio_id}_vocal.wav",
                f"{audio_id}_instruments.wav"
            ]
            
            for temp_file in temp_files:
                temp_path = self.temp_dir / temp_file
                if temp_path.exists():
                    temp_path.unlink()
            
            segments_dir = self.temp_dir / f"segments_{audio_id}"
            if segments_dir.exists():
                import shutil
                shutil.rmtree(segments_dir)
                
        except Exception:
            pass
    
    def validate_and_repair_segments(self, segments_dir: str) -> Dict[str, Any]:
        """Validate and repair existing segment metadata files"""
        return self.file_manager.validate_and_repair_metadata(segments_dir)
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get clean processing statistics without unnecessary speaker breakdown"""
        try:
            segments_path = Path(segments_dir)
            
            # Check unified structure
            segments_folder = segments_path / "segments"
            cloned_folder = segments_path / "cloned"
            metadata_folder = segments_path / "metadata"
            
            stats = {
                "total_segments": 0,
                "total_cloned": 0,
                "speakers": [],
                "completion_rate": 0,
                "has_processing_summary": False,
                "has_cloning_summary": False,
                "has_reconstruction_summary": False,
                "transcription_source": "AssemblyAI"
            }
            
            if not segments_folder.exists():
                logger.warning(f"Segments folder not found: {segments_folder}")
                return stats
            
            # Collect basic statistics
            metadata_files = list(segments_folder.glob("*_metadata.json"))
            cloned_files = list(cloned_folder.glob("*.wav")) if cloned_folder.exists() else []
            
            stats["total_segments"] = len(metadata_files)
            stats["total_cloned"] = len(cloned_files)
            
            # Get unique speakers (simple list)
            speakers_found = set()
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    speaker = metadata.get('speaker', 'A')
                    speakers_found.add(speaker)
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {metadata_file.name}: {e}")
                    continue
            
            stats["speakers"] = sorted(list(speakers_found))
            
            # Calculate completion rate
            if stats["total_segments"] > 0:
                stats["completion_rate"] = round((stats["total_cloned"] / stats["total_segments"]) * 100, 1)
            
            # Check for summary files
            if metadata_folder.exists():
                stats["has_processing_summary"] = (metadata_folder / "processing_metadata.json").exists()
                stats["has_cloning_summary"] = (metadata_folder / "cloning_summary.json").exists()
                stats["has_reconstruction_summary"] = (metadata_folder / "reconstruction_summary.json").exists()
            
            logger.info(f"Processing stats: {stats['total_segments']} segments, {stats['total_cloned']} cloned, {stats['completion_rate']}% complete")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                "total_segments": 0,
                "total_cloned": 0,
                "speakers": [],
                "completion_rate": 0,
                "has_processing_summary": False,
                "has_cloning_summary": False,
                "has_reconstruction_summary": False,
                "transcription_source": "AssemblyAI"
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        stats = {
            "translation_cache_size": len(getattr(self.transcription_service, 'translation_cache', {}))
        }
        return stats
