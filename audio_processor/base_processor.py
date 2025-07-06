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

from .transcription import TranscriptionService
from .voice_cloning import VoiceCloningService
from .audio_utils import AudioUtils
from .segment_manager import SegmentManager
from .audio_reconstructor import AudioReconstructor
from .file_manager import FileManager


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
    
    def load_dia_model(self, repo_id: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model for voice cloning"""
        return self.voice_cloning_service.load_dia_model(repo_id)
    
    def process_audio_segments(self, audio_path: str, audio_id: str, 
                             target_language: str = "English") -> Dict[str, Any]:
        """Process audio file into segments for voice cloning"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Transcribe audio
            transcript_data = self.transcription_service.transcribe_audio(audio_path)
            
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
            self.file_manager.save_metadata(
                transcript_data, segments, silent_parts, 
                output_dir, audio_id, audio_path
            )
            
            return {
                "success": True,
                "segments_dir": str(output_dir),
                "audio_id": audio_id,
                "speakers": transcript_data['speakers'],
                "total_segments": len(segments),
                "total_duration": transcript_data['duration']
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
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        self.file_manager.cleanup_temp_files(audio_id)
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.file_manager.get_processing_stats(segments_dir)
