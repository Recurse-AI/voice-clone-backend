"""
Base Audio Processor Module

Main orchestrator class that ties all audio processing components together.
"""

import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import soundfile as sf
import numpy as np
import librosa
from datetime import datetime

from .transcription import TranscriptionService
from .voice_cloning import VoiceCloningService
from .audio_utils import AudioUtils
from .segment_manager import SegmentManager
from .audio_reconstructor import AudioReconstructor
from .file_manager import FileManager
from .video_processor import VideoProcessor
from .runpod_service import RunPodService


class AudioProcessor:
    """Main audio processor that orchestrates all processing steps"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.transcription_service = TranscriptionService()
        self.voice_cloning_service = VoiceCloningService()
        self.audio_utils = AudioUtils()
        self.segment_manager = SegmentManager(self.transcription_service)
        self.audio_reconstructor = AudioReconstructor(str(self.temp_dir))
        self.file_manager = FileManager(str(self.temp_dir))
        self.video_processor = VideoProcessor(str(self.temp_dir))
        self.runpod_service = RunPodService()
    
    def load_dia_model(self, repo_id: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model for voice cloning"""
        return self.voice_cloning_service.load_dia_model(repo_id)
    
    def process_audio_segments(self, audio_path: str, audio_id: str, 
                             target_language: str = "English",
                             language_code: str = "en",
                             speakers_expected: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audio file into segments for voice cloning
        
        Args:
            audio_path: Path to audio file
            audio_id: Unique identifier for this processing session
            target_language: Target language for translation (e.g., "English", "Spanish")
            language_code: Language code for AssemblyAI (e.g., "en", "es", "fr")
            speakers_expected: Expected number of speakers (1-10)
        """
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Transcribe audio with universal model
            transcript_data = self.transcription_service.transcribe_audio(
                audio_path, 
                language_code=language_code,
                speakers_expected=speakers_expected
            )
            
            # Detect silent parts
            silent_parts = self.audio_utils.detect_silent_parts(audio, sr)
            
            # Create segments
            segments = self.segment_manager.create_speaker_segments(transcript_data)
            
            if not segments:
                return {"success": False, "error": "No valid segments created"}
            
            # Set up output directory
            output_dir = self.temp_dir / f"processed_audio_segments" / audio_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            self.file_manager.create_directory_structure(output_dir, transcript_data['speakers'])
            
            # Save segments
            self.segment_manager.save_segments(
                segments, audio, sr, output_dir, 
                transcript_data['speakers'], target_language
            )
            
            # Save silent parts
            self.file_manager.save_silent_parts(silent_parts, audio, sr, output_dir)
            
            # Select and save reference segments
            reference_segments = self.segment_manager.select_reference_segments(
                segments, transcript_data['speakers']
            )
            self.file_manager.save_reference_segments(
                reference_segments, audio, sr, output_dir, transcript_data['speakers']
            )
            
            # Save multi-speaker reference if applicable
            self.file_manager.save_multi_speaker_reference(
                audio, sr, output_dir, transcript_data['speakers']
            )
            
            # Save metadata
            metadata = {
                **transcript_data.get('metadata', {}),
                "target_language": target_language,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            self.file_manager.save_metadata(
                transcript_data, segments, silent_parts, 
                output_dir, audio_id, audio_path, metadata
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
            return {"success": False, "error": str(e)}
    
    def clone_voice_segments(self, segments_dir: str, audio_id: str, 
                           temperature: float = 1.3, cfg_scale: float = 3.0, 
                           top_p: float = 0.95, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments using Dia model"""
        if not self.voice_cloning_service.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            segments_path = Path(segments_dir)
            
            # Load segment data
            segment_data_list = []
            for speaker_dir in segments_path.iterdir():
                if speaker_dir.is_dir() and speaker_dir.name.startswith("speaker_"):
                    segments_subdir = speaker_dir / "segments"
                    reference_subdir = speaker_dir / "reference"
                    
                    # Get reference audio
                    reference_audio_path = None
                    reference_text = None
                    
                    for ref_file in reference_subdir.glob("*_REFERENCE.wav"):
                        reference_audio_path = str(ref_file)
                        # Get corresponding reference text
                        ref_json = ref_file.with_suffix('').with_suffix('_metadata.json')
                        if ref_json.exists():
                            with open(ref_json, 'r', encoding='utf-8') as f:
                                import json
                                ref_data = json.load(f)
                                reference_text = ref_data.get('text', '')
                        break
                    
                    # Process segments
                    for segment_file in segments_subdir.glob("*.json"):
                        with open(segment_file, 'r', encoding='utf-8') as f:
                            import json
                            segment_data = json.load(f)
                        
                        segment_data['reference_audio_path'] = reference_audio_path
                        segment_data['reference_text'] = reference_text
                        segment_data['segments_dir'] = str(segments_subdir)
                        segment_data['segment_file'] = str(segment_file)
                        segment_data_list.append(segment_data)
            
            # Clone segments
            cloning_result = self.voice_cloning_service.clone_voice_segments(
                segment_data_list, temperature, cfg_scale, top_p, seed
            )
            
            if not cloning_result['success']:
                return cloning_result
            
            # Save cloned audio files
            for cloned_segment in cloning_result['cloned_segments']:
                if cloned_segment['success'] and cloned_segment['cloned_audio'] is not None:
                    segment_data = cloned_segment['original_data']
                    segments_dir_path = Path(segment_data['segments_dir'])
                    
                    # Adjust audio length to match original
                    original_audio_path = Path(segment_data['segment_file']).with_suffix('.wav')
                    if original_audio_path.exists():
                        original_audio, _ = sf.read(original_audio_path)
                        original_duration = len(original_audio) / 44100
                        
                        cloned_audio = self.audio_utils.adjust_audio_length(
                            cloned_segment['cloned_audio'], original_duration, 44100
                        )
                        
                        # Save cloned audio
                        cloned_filename = f"cloned_{original_audio_path.stem}.wav"
                        cloned_path = segments_dir_path / cloned_filename
                        sf.write(cloned_path, cloned_audio, 44100)
            
            return {
                "success": True,
                "cloned_segments_count": len([s for s in cloning_result['cloned_segments'] if s['success']]),
                "total_segments": len(segment_data_list),
                "audio_id": audio_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio from cloned segments"""
        return self.audio_reconstructor.reconstruct_final_audio(
            segments_dir, audio_id, include_instruments, instruments_path
        )
    
    def generate_subtitles(self, segments_dir: str, audio_id: str) -> Dict[str, Any]:
        """Generate subtitle file"""
        return self.audio_reconstructor.generate_subtitles(segments_dir, audio_id)
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with high-quality subtitles"""
        return self.video_processor.create_video_with_subtitles(
            video_path, audio_path, segments_dir, audio_id, instruments_path
        )
    
    def process_video_with_separation(self, video_path: str, audio_id: str, 
                                    target_language: str = "English",
                                    language_code: str = "en",
                                    speakers_expected: Optional[int] = None) -> Dict[str, Any]:
        """Process video with RunPod vocal/instrument separation"""
        try:
            # Extract audio from video
            audio_temp_path = self.temp_dir / f"{audio_id}_extracted_audio.wav"
            extract_result = self.audio_utils.extract_audio_from_video(video_path, str(audio_temp_path))
            
            if not extract_result["success"]:
                return {"success": False, "error": f"Audio extraction failed: {extract_result['error']}"}
            
            # Upload audio to temporary storage for RunPod
            r2_storage = None
            try:
                from r2_storage import R2Storage
                r2_storage = R2Storage()
                
                # Upload extracted audio
                upload_result = r2_storage.upload_file(
                    str(audio_temp_path),
                    f"temp/{audio_id}_audio.wav",
                    "audio/wav"
                )
                
                if not upload_result["success"]:
                    return {"success": False, "error": "Failed to upload audio for processing"}
                
                audio_url = upload_result["url"]
                
            except Exception as e:
                return {"success": False, "error": f"Upload failed: {str(e)}"}
            
            # Process with RunPod
            separation_result = self.runpod_service.process_audio_separation(audio_url)
            
            if not separation_result["success"]:
                return {"success": False, "error": f"RunPod processing failed: {separation_result['error']}"}
            
            # Wait for completion
            completion_result = self.runpod_service.wait_for_completion(separation_result["job_id"])
            
            if not completion_result["success"]:
                return {"success": False, "error": f"RunPod job failed: {completion_result['error']}"}
            
            # Download separated audio files
            vocal_path = self.temp_dir / f"{audio_id}_vocal.wav"
            instrument_path = self.temp_dir / f"{audio_id}_instruments.wav"
            
            vocal_download = self.audio_utils.download_audio_file(
                completion_result["vocal_audio"], str(vocal_path)
            )
            
            instrument_download = self.audio_utils.download_audio_file(
                completion_result["instrument_audio"], str(instrument_path)
            )
            
            if not vocal_download["success"] or not instrument_download["success"]:
                return {"success": False, "error": "Failed to download separated audio"}
            
            # Process vocal audio through normal pipeline
            segment_result = self.process_audio_segments(
                str(vocal_path), 
                audio_id, 
                target_language,
                language_code=language_code,
                speakers_expected=speakers_expected
            )
            
            if not segment_result["success"]:
                return {"success": False, "error": segment_result["error"]}
            
            return {
                "success": True,
                "segments_dir": segment_result["segments_dir"],
                "vocal_path": str(vocal_path),
                "instruments_path": str(instrument_path),
                "audio_id": audio_id,
                "speakers": segment_result["speakers"],
                "total_segments": segment_result["total_segments"],
                "total_duration": segment_result["total_duration"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        self.file_manager.cleanup_temp_files(audio_id)
        self.video_processor.cleanup_temp_files(audio_id)
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.file_manager.get_processing_stats(segments_dir)
