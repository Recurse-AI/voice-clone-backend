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
        self.voice_cloning_service = VoiceCloningService()
        self.audio_reconstructor = AudioReconstructor(str(self.temp_dir))
        self.video_processor = VideoProcessor(str(self.temp_dir))
        self.runpod_service = RunPodService()
    
    def load_dia_model(self, repo_id: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model for voice cloning"""
        return self.voice_cloning_service.load_dia_model(repo_id)
    
    def process_audio_segments(self, audio_path: str, audio_id: str, 
                             target_language: str = "English",
                             language_code: Optional[str] = None,
                             speakers_expected: Optional[int] = 1) -> Dict[str, Any]:
        """Process audio segments for voice cloning"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Transcribe audio
            transcript_data = self.transcription_service.transcribe_audio(
                audio_path, language_code, speakers_expected, audio_id
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
                transcript_data['speakers'], target_language, detected_language
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
                           temperature: float = 1.3, cfg_scale: float = 3.0, 
                           top_p: float = 0.95, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments speaker by speaker - simplified"""
        print(f"Starting clone_voice_segments for audio_id: {audio_id}")
        
        if not self.voice_cloning_service.is_model_loaded():
            print("ERROR: Dia model not loaded")
            return {"success": False, "error": "Dia model not loaded"}
        
        print("Dia model is loaded successfully")
        
        try:
            segments_path = Path(segments_dir)
            print(f"Segments directory: {segments_path}")
            
            total_successful_clones = 0
            seeds_used = {}
            speaker_results = {}
            
            base_seed = seed or settings.DEFAULT_SEED
            print(f"Base seed: {base_seed}")
            
            # Process each speaker directory
            speaker_dirs = list(segments_path.iterdir())
            print(f"Found {len(speaker_dirs)} directories in segments_path")
            
            for speaker_dir in speaker_dirs:
                if not speaker_dir.is_dir() or not speaker_dir.name.startswith("speaker_"):
                    print(f"Skipping non-speaker directory: {speaker_dir.name}")
                    continue
                
                speaker = speaker_dir.name.replace("speaker_", "")
                print(f"Processing speaker: {speaker}")
                
                segments_subdir = speaker_dir / "segments"
                
                if not segments_subdir.exists():
                    print(f"Segments subdirectory not found: {segments_subdir}")
                    continue
                
                # Calculate speaker seed
                speaker_index = ord(speaker) - ord('A')
                speaker_seed = base_seed + (speaker_index * settings.SPEAKER_SEED_OFFSET)
                print(f"Speaker {speaker} seed: {speaker_seed}")
                
                # Collect all segments for this speaker
                speaker_segments = []
                json_files = sorted(list(segments_subdir.glob("*_metadata.json")))  # Sort files for consistent ordering
                print(f"Found {len(json_files)} metadata files for speaker {speaker}")
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            import json
                            segment_data = json.load(f)
                        
                        if not segment_data.get('english_text', '').strip():
                            print(f"Skipping segment with no english_text: {json_file}")
                            continue
                        
                        segment_data['segments_dir'] = str(segments_subdir)
                        segment_data['segment_file'] = str(json_file)
                        speaker_segments.append(segment_data)
                        print(f"Added segment from {json_file.name} to speaker {speaker}")
                        
                    except Exception as e:
                        print(f"Error loading metadata from {json_file}: {e}")
                        continue
                
                if not speaker_segments:
                    print(f"No valid segments found for speaker {speaker}")
                    continue
                
                print(f"Collected {len(speaker_segments)} segments for speaker {speaker}")
                
                # Clone segments for this speaker
                print(f"Calling voice cloning service for speaker {speaker}")
                cloning_result = self.voice_cloning_service.clone_voice_segments(
                    speaker_segments, temperature, cfg_scale, top_p, speaker_seed
                )
                
                print(f"Cloning result success: {cloning_result.get('success', False)}")
                if not cloning_result.get('success', False):
                    print(f"Cloning failed for speaker {speaker}: {cloning_result.get('error', 'Unknown error')}")
                    continue
                
                seeds_used[speaker] = speaker_seed
                
                # Process cloned segments
                cloned_segments = cloning_result.get('cloned_segments', [])
                print(f"Received {len(cloned_segments)} cloned segments for speaker {speaker}")
                
                speaker_successful = 0
                for idx, cloned_segment in enumerate(cloned_segments):
                    if cloned_segment.get('success', False) and cloned_segment.get('cloned_audio') is not None:
                        try:
                            segment_data = cloned_segment['original_data']
                            segments_dir_path = Path(segment_data['segments_dir'])
                            
                            segment_index = segment_data.get('segment_index', idx + 1)
                            cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
                            cloned_path = segments_dir_path / cloned_filename
                            
                            # Save cloned audio
                            sf.write(cloned_path, cloned_segment['cloned_audio'], 44100)
                            print(f"Saved cloned audio: {cloned_path}")
                            
                            # Verify file was created
                            if not cloned_path.exists():
                                print(f"ERROR: Failed to create cloned audio file: {cloned_path}")
                                continue
                            
                            # Update metadata
                            metadata_file = Path(segment_data['segment_file'])
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            metadata.update({
                                'cloned_audio_file': cloned_filename,
                                'cloned_audio_path': str(cloned_path),
                                'cloned_audio_exists': True,
                                'cloning_successful': True,
                                'cloning_seed': speaker_seed,
                                'cloning_parameters': {
                                    'temperature': temperature,
                                    'cfg_scale': cfg_scale,
                                    'top_p': top_p
                                },
                                'processing_status': 'cloning_completed'
                            })
                            
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, ensure_ascii=False, indent=2)
                            
                            print(f"Updated metadata for segment {segment_index}: {metadata_file}")
                            speaker_successful += 1
                            
                        except Exception as e:
                            print(f"Error saving cloned segment {idx}: {e}")
                            continue
                    else:
                        print(f"Cloned segment {idx} failed or has no audio data")
                
                speaker_results[speaker] = {
                    'speaker': speaker,
                    'total_segments': len(speaker_segments),
                    'successful_clones': speaker_successful,
                    'seed_used': speaker_seed,
                    'cloning_parameters': {
                        'temperature': temperature,
                        'cfg_scale': cfg_scale,
                        'top_p': top_p
                    }
                }
                total_successful_clones += speaker_successful
                
                print(f"Speaker {speaker} completed: {speaker_successful}/{len(speaker_segments)} successful")
                
                self.voice_cloning_service.clear_cache()
            
            print(f"Total successful clones: {total_successful_clones}")
            
            # Save cloning summary
            cloning_summary = {
                'audio_id': audio_id,
                'total_successful_clones': total_successful_clones,
                'cloned_by_speaker': speaker_results,
                'seeds_used': seeds_used,
                'cloning_timestamp': str(datetime.now())
            }
            
            summary_path = segments_path / "metadata" / "cloning_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(cloning_summary, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "cloned_segments": total_successful_clones,
                "cloned_segments_count": total_successful_clones,
                "cloned_by_speaker": speaker_results,
                "seeds_used": seeds_used,
                "audio_id": audio_id
            }
            
        except Exception as e:
            print(f"Exception in clone_voice_segments: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.voice_cloning_service.clear_cache()
    
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
                               audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with new audio only"""
        return self.video_processor.create_video_with_audio(
            video_path, audio_path, audio_id, instruments_path
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
            
            # Process with RunPod
            separation_result = self.runpod_service.process_audio_separation(audio_url)
            
            if not separation_result or not separation_result.get("id"):
                return {"success": False, "error": "RunPod service returned invalid response"}
            
            # Wait for completion
            completion_result = self.runpod_service.wait_for_completion(separation_result["id"])
            
            if completion_result.get("status") != "COMPLETED":
                return {"success": False, "error": f"RunPod job failed: {completion_result.get('status', 'Unknown status')}"}
            
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
            
            # Process vocal audio
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
        """Get processing statistics"""
        return self.file_manager.get_processing_stats(segments_dir)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        stats = {
            "translation_cache_size": len(getattr(self.transcription_service, 'translation_cache', {}))
        }
        return stats
