"""
Base Audio Processor Module - Optimized for Dia Voice Cloning

Main orchestrator for voice cloning pipeline with speaker-wise batch processing.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .transcription import TranscriptionService
from .segment_manager import SegmentManager
from .audio_utils import AudioUtils
from .file_manager import FileManager
from .audio_reconstructor import AudioReconstructor
from .video_processor import VideoProcessor
from .voice_cloning import VoiceCloningService
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
                             language_code: Optional[str] = None,
                             speakers_expected: Optional[int] = 1) -> Dict[str, Any]:
        """Process audio segments optimized for Dia voice cloning"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Transcribe audio with detailed error handling
            try:
                transcript_data = self.transcription_service.transcribe_audio(
                    audio_path, language_code, speakers_expected
                )
            except Exception as e:
                return {"success": False, "error": f"Transcription failed: {str(e)}"}
            
            # Create optimal segments for Dia model with detailed error handling
            try:
                segments = self.segment_manager.create_optimal_segments(transcript_data)
            except Exception as e:
                return {"success": False, "error": f"Segment creation failed: {str(e)}"}
            
            if not segments:
                return {"success": False, "error": "No viable segments created"}
            
            # Create output directory
            output_dir = self.temp_dir / f"segments_{audio_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            try:
                self.file_manager.create_directory_structure(output_dir, transcript_data['speakers'])
            except Exception as e:
                return {"success": False, "error": f"Directory creation failed: {str(e)}"}
            
            # Save optimal segments
            try:
                self.segment_manager.save_optimal_segments(
                    segments, audio, sr, output_dir, 
                    transcript_data['speakers'], target_language
                )
            except Exception as e:
                return {"success": False, "error": f"Segment saving failed: {str(e)}"}
            
            # Select optimal references
            try:
                reference_segments = self.segment_manager.select_optimal_references(
                    segments, transcript_data['speakers']
                )
                self.file_manager.save_reference_segments(
                    reference_segments, audio, sr, output_dir, transcript_data['speakers']
                )
            except Exception as e:
                return {"success": False, "error": f"Reference segment processing failed: {str(e)}"}
            
            # Save metadata
            try:
                metadata = {
                    **transcript_data.get('metadata', {}),
                    "target_language": target_language,
                    "processing_timestamp": datetime.now().isoformat(),
                    "optimization": "dia_optimal_segments"
                }
                
                self.file_manager.save_metadata(
                    transcript_data, segments, [], 
                    output_dir, audio_id, audio_path, metadata
                )
            except Exception as e:
                return {"success": False, "error": f"Metadata saving failed: {str(e)}"}
            
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
    
    def clone_voice_segments_speaker_wise(self, segments_dir: str, audio_id: str, 
                                        temperature: float = 1.3, cfg_scale: float = 3.0, 
                                        top_p: float = 0.95, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments speaker by speaker for optimal batch processing"""
        if not self.voice_cloning_service.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            segments_path = Path(segments_dir)
            all_cloned_segments = {}
            total_successful_clones = 0
            
            # Process each speaker separately
            for speaker_dir in segments_path.iterdir():
                if not (speaker_dir.is_dir() and speaker_dir.name.startswith("speaker_")):
                    continue
                
                speaker = speaker_dir.name.replace("speaker_", "")
                segments_subdir = speaker_dir / "segments"
                reference_subdir = speaker_dir / "reference"
                
                # Load reference for this speaker
                reference_audio_path = None
                reference_text = None
                
                for ref_file in reference_subdir.glob("*_REFERENCE.wav"):
                    reference_audio_path = str(ref_file)
                    ref_json_name = ref_file.stem + "_metadata.json"
                    ref_json = ref_file.parent / ref_json_name
                    if ref_json.exists():
                        with open(ref_json, 'r', encoding='utf-8') as f:
                            import json
                            ref_data = json.load(f)
                            reference_text = ref_data.get('text', '')
                    break
                
                if not reference_audio_path:
                    continue
                
                # Load all segments for this speaker
                speaker_segments = []
                for segment_file in segments_subdir.glob("*.json"):
                    with open(segment_file, 'r', encoding='utf-8') as f:
                        import json
                        segment_data = json.load(f)
                    
                    segment_data['reference_audio_path'] = reference_audio_path
                    segment_data['reference_text'] = reference_text
                    segment_data['segments_dir'] = str(segments_subdir)
                    segment_data['segment_file'] = str(segment_file)
                    speaker_segments.append(segment_data)
                
                if not speaker_segments:
                    continue
                
                # Clone all segments for this speaker at once
                cloning_result = self.voice_cloning_service.clone_voice_segments(
                    speaker_segments, temperature, cfg_scale, top_p, seed
                )
                
                if not cloning_result['success']:
                    continue
                
                # Save cloned audio with exact length preservation
                speaker_successful = 0
                for cloned_segment in cloning_result['cloned_segments']:
                    if cloned_segment['success'] and cloned_segment['cloned_audio'] is not None:
                        segment_data = cloned_segment['original_data']
                        segments_dir_path = Path(segment_data['segments_dir'])
                        
                        # Get original audio for exact length matching
                        original_audio_path = Path(segment_data['segment_file']).with_suffix('.wav')
                        if original_audio_path.exists():
                            original_audio, _ = sf.read(original_audio_path)
                            original_duration = len(original_audio) / 44100
                            
                            # Ensure cloned audio matches exact original length
                            cloned_audio = self._preserve_exact_length(
                                cloned_segment['cloned_audio'], original_duration, 44100
                            )
                            
                            # Save cloned audio
                            cloned_filename = f"cloned_{original_audio_path.stem}.wav"
                            cloned_path = segments_dir_path / cloned_filename
                            sf.write(cloned_path, cloned_audio, 44100)
                            speaker_successful += 1
                
                all_cloned_segments[speaker] = {
                    'total_segments': len(speaker_segments),
                    'successful_clones': speaker_successful,
                    'reference_used': reference_audio_path
                }
                total_successful_clones += speaker_successful
            
            return {
                "success": True,
                "cloned_segments_count": total_successful_clones,
                "cloned_by_speaker": all_cloned_segments,
                "audio_id": audio_id,
                "processing_method": "speaker_wise_batch"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _preserve_exact_length(self, cloned_audio: np.ndarray, target_duration: float, sample_rate: int) -> np.ndarray:
        """Preserve exact length of original audio for dubbing"""
        target_samples = int(target_duration * sample_rate)
        current_samples = len(cloned_audio)
        
        if current_samples == target_samples:
            return cloned_audio
        elif current_samples > target_samples:
            # Trim excess
            return cloned_audio[:target_samples]
        else:
            # Pad with silence
            padding = np.zeros(target_samples - current_samples)
            return np.concatenate([cloned_audio, padding])
    
    def clone_voice_segments(self, segments_dir: str, audio_id: str, 
                           temperature: float = 1.3, cfg_scale: float = 3.0, 
                           top_p: float = 0.95, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments using speaker-wise batch processing"""
        return self.clone_voice_segments_speaker_wise(
            segments_dir, audio_id, temperature, cfg_scale, top_p, seed
        )
    
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
        """Create video with high-quality subtitles"""
        return self.video_processor.create_video_with_subtitles(
            video_path, audio_path, segments_dir, audio_id, instruments_path
        )
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                               audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with new audio only (no subtitles)"""
        return self.video_processor.create_video_with_audio(
            video_path, audio_path, audio_id, instruments_path
        )
    
    def process_video_with_separation(self, video_path: str, audio_id: str, 
                                    target_language: str = "English",
                                    language_code: Optional[str] = None,
                                    speakers_expected: Optional[int] = 1) -> Dict[str, Any]:
        """Process video with RunPod vocal/instrument separation"""
        try:
            # Validate RunPod service is properly configured
            if not hasattr(self, 'runpod_service') or not self.runpod_service:
                return {"success": False, "error": "RunPod service not initialized"}
            
            # Check if RunPod credentials are configured
            if not self.runpod_service.api_key or not self.runpod_service.base_url:
                return {"success": False, "error": "RunPod credentials not configured"}
            
            # Extract audio from video
            audio_temp_path = self.temp_dir / f"{audio_id}_extracted_audio.wav"
            extract_result = self.audio_utils.extract_audio_from_video(video_path, str(audio_temp_path))
            
            if not extract_result["success"]:
                return {"success": False, "error": f"Audio extraction failed: {extract_result['error']}"}
            
            from r2_storage import R2Storage
            r2_storage = R2Storage()
            
            # Validate R2 storage is configured
            if not r2_storage.bucket_name:
                return {"success": False, "error": "R2 storage not configured"}
            
            if not r2_storage.client:
                return {"success": False, "error": "R2 client not initialized"}
            
            upload_result = r2_storage.upload_file(
                str(audio_temp_path),
                f"temp/{audio_id}_audio.wav",
                "audio/wav"
            )
            
            if not upload_result["success"]:
                return {"success": False, "error": f"Failed to upload audio for processing: {upload_result.get('error', 'Unknown error')}"}
            
            audio_url = upload_result["url"]
            
            # Process with RunPod
            try:
                separation_result = self.runpod_service.process_audio_separation(audio_url)
                
                if not separation_result or not separation_result.get("id"):
                    return {"success": False, "error": "RunPod service returned invalid response"}
                
                # Wait for completion
                completion_result = self.runpod_service.wait_for_completion(separation_result["id"])
                
                if completion_result.get("status") == "FAILED":
                    return {"success": False, "error": f"RunPod job failed: {completion_result.get('error', 'Unknown error')}"}
                
                if completion_result.get("status") != "COMPLETED":
                    return {"success": False, "error": f"RunPod job did not complete successfully: {completion_result.get('status', 'Unknown status')}"}
                
                # Validate output URLs
                if not completion_result.get("output") or not completion_result["output"].get("vocal_audio") or not completion_result["output"].get("instrument_audio"):
                    return {"success": False, "error": "RunPod job completed but no output URLs provided"}
                
            except Exception as e:
                return {"success": False, "error": f"RunPod processing failed: {str(e)}"}
            
            # Download separated audio files
            vocal_path = self.temp_dir / f"{audio_id}_vocal.wav"
            instrument_path = self.temp_dir / f"{audio_id}_instruments.wav"
            
            vocal_download = self.audio_utils.download_audio_file(
                completion_result["output"]["vocal_audio"], str(vocal_path)
            )
            
            instrument_download = self.audio_utils.download_audio_file(
                completion_result["output"]["instrument_audio"], str(instrument_path)
            )
            
            if not vocal_download["success"] or not instrument_download["success"]:
                return {"success": False, "error": f"Failed to download separated audio: vocal={vocal_download.get('error', 'Unknown')}, instrument={instrument_download.get('error', 'Unknown')}"}
            
            # Validate downloaded files
            if not vocal_path.exists() or vocal_path.stat().st_size == 0:
                return {"success": False, "error": "Downloaded vocal file is empty or missing"}
            
            if not instrument_path.exists() or instrument_path.stat().st_size == 0:
                return {"success": False, "error": "Downloaded instrument file is empty or missing"}
            
            # Process vocal audio through normal pipeline
            segment_result = self.process_audio_segments(
                str(vocal_path), 
                audio_id, 
                target_language,
                language_code=language_code,
                speakers_expected=speakers_expected
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
            return {"success": False, "error": f"Unexpected error in video processing: {str(e)}"}
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            # Clean up file manager temp files
            self.file_manager.cleanup_temp_files(audio_id)
            
            # Clean up video processor temp files
            self.video_processor.cleanup_temp_files(audio_id)
            
            # Clean up audio reconstructor temp files
            self.audio_reconstructor.cleanup_temp_files(audio_id)
            
            # Clean up our own temp files
            temp_files_to_clean = [
                f"{audio_id}_extracted_audio.wav",
                f"{audio_id}_vocal.wav",
                f"{audio_id}_instruments.wav"
            ]
            
            for temp_file in temp_files_to_clean:
                temp_path = self.temp_dir / temp_file
                if temp_path.exists():
                    temp_path.unlink()
            
            # Clean up segments directory
            segments_dir = self.temp_dir / f"segments_{audio_id}"
            if segments_dir.exists():
                import shutil
                shutil.rmtree(segments_dir)
                
        except Exception:
            pass
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.file_manager.get_processing_stats(segments_dir)
