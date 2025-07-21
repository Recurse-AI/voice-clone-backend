"""
File Manager Module - Simplified with Metadata Validation
"""

import json
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileManager:
    """Simplified file manager with metadata validation"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_directory_structure(self, base_dir: Path, speakers: List[str]):
        """Create directory structure for segments"""
        (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}"
            (speaker_dir / "segments").mkdir(parents=True, exist_ok=True)
    
    def validate_and_repair_metadata(self, segments_dir: str) -> Dict[str, Any]:
        """Validate and repair metadata files in segments directory"""
        segments_path = Path(segments_dir)
        validation_results = {
            "total_files_checked": 0,
            "files_repaired": 0,
            "files_with_errors": 0,
            "missing_english_text": 0,
            "missing_essential_fields": 0,
            "speakers_processed": [],
            "validation_timestamp": str(datetime.now())
        }
        
        try:
            for speaker_dir in segments_path.glob("speaker_*"):
                if not speaker_dir.is_dir():
                    continue
                
                speaker = speaker_dir.name.replace("speaker_", "")
                validation_results["speakers_processed"].append(speaker)
                
                segments_subdir = speaker_dir / "segments"
                if not segments_subdir.exists():
                    continue
                
                for json_file in segments_subdir.glob("*_metadata.json"):
                    validation_results["total_files_checked"] += 1
                    
                    try:
                        # Load existing metadata
                        with open(json_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Check if repair is needed
                        needs_repair = False
                        
                        # Validate essential fields
                        if not metadata.get('segment_index'):
                            segment_num = self._extract_segment_number(json_file.name)
                            metadata['segment_index'] = segment_num
                            needs_repair = True
                            validation_results["missing_essential_fields"] += 1
                        
                        if not metadata.get('speaker'):
                            metadata['speaker'] = speaker
                            needs_repair = True
                            validation_results["missing_essential_fields"] += 1
                        
                        if not metadata.get('speaker_index'):
                            metadata['speaker_index'] = ord(speaker) - ord('A') + 1
                            needs_repair = True
                            validation_results["missing_essential_fields"] += 1
                        
                        # Check English text
                        if not metadata.get('english_text', '').strip():
                            # Try to create English text from original text
                            original_text = metadata.get('original_text', metadata.get('text', ''))
                            if original_text:
                                speaker_num = metadata.get('speaker_index', ord(speaker) - ord('A') + 1)
                                english_text = self._create_fallback_english_text(original_text, speaker_num)
                                metadata['english_text'] = english_text
                                needs_repair = True
                                validation_results["missing_english_text"] += 1
                        
                        # Add missing paths
                        if not metadata.get('audio_path'):
                            audio_file = metadata.get('audio_file', f"segment_{metadata['segment_index']:03d}.wav")
                            metadata['audio_path'] = str(segments_subdir / audio_file)
                            needs_repair = True
                        
                        if not metadata.get('cloned_audio_file'):
                            metadata['cloned_audio_file'] = f"cloned_segment_{metadata['segment_index']:03d}.wav"
                            needs_repair = True
                        
                        if not metadata.get('cloned_audio_path'):
                            metadata['cloned_audio_path'] = str(segments_subdir / metadata['cloned_audio_file'])
                            needs_repair = True
                        
                        # Check if cloned audio actually exists
                        cloned_path = Path(metadata['cloned_audio_path'])
                        metadata['cloned_audio_exists'] = cloned_path.exists()
                        
                        # Add processing status
                        if not metadata.get('processing_status'):
                            if metadata.get('cloned_audio_exists'):
                                metadata['processing_status'] = 'cloning_completed'
                            else:
                                metadata['processing_status'] = 'ready_for_cloning'
                            needs_repair = True
                        
                        # Add metadata completeness flag
                        metadata['metadata_complete'] = True
                        metadata['last_validated'] = str(datetime.now())
                        
                        # Save repaired metadata
                        if needs_repair:
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, ensure_ascii=False, indent=2)
                            validation_results["files_repaired"] += 1
                        
                    except Exception as e:
                        validation_results["files_with_errors"] += 1
                        logger.error(f"Error processing {json_file}: {str(e)}")
                        continue
            
            # Save validation summary
            summary_path = segments_path / "metadata" / "validation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            
            return validation_results
            
        except Exception as e:
            validation_results["validation_error"] = str(e)
            return validation_results
    
    def _extract_segment_number(self, filename: str) -> int:
        """Extract segment number from filename"""
        import re
        match = re.search(r'segment_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 1
    
    def _create_fallback_english_text(self, text: str, speaker_num: int) -> str:
        """Create fallback English text with proper formatting"""
        if not text.strip():
            return f"[S{speaker_num}] Audio segment content"
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= 10:
                lines.append(f"[S{speaker_num}] {' '.join(current_line)}")
                current_line = []
        
        if current_line:
            lines.append(f"[S{speaker_num}] {' '.join(current_line)}")
        
        return '\n'.join(lines) if lines else f"[S{speaker_num}] {text}"
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            for item in self.temp_dir.iterdir():
                if audio_id in item.name:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception:
            pass
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        try:
            segments_path = Path(segments_dir)
            
            # Count segments by speaker
            segments_by_speaker = {}
            cloned_by_speaker = {}
            total_segments = 0
            total_cloned = 0
            speakers = []
            
            for speaker_dir in segments_path.glob("speaker_*"):
                if speaker_dir.is_dir():
                    speaker = speaker_dir.name.replace("speaker_", "")
                    speakers.append(speaker)
                    
                    segments_subdir = speaker_dir / "segments"
                    if segments_subdir.exists():
                        segment_count = len(list(segments_subdir.glob("*_metadata.json")))
                        cloned_count = len(list(segments_subdir.glob("cloned_segment_*.wav")))
                        
                        segments_by_speaker[speaker] = segment_count
                        cloned_by_speaker[speaker] = cloned_count
                        total_segments += segment_count
                        total_cloned += cloned_count
            
            # Check for summary files
            metadata_dir = segments_path / "metadata"
            has_processing_summary = (metadata_dir / "processing_metadata.json").exists()
            has_cloning_summary = (metadata_dir / "cloning_summary.json").exists()
            has_reconstruction_summary = (metadata_dir / "reconstruction_summary.json").exists()
            
            return {
                "total_segments": total_segments,
                "total_cloned": total_cloned,
                "speakers": speakers,
                "segments_by_speaker": segments_by_speaker,
                "cloned_by_speaker": cloned_by_speaker,
                "completion_rate": (total_cloned / total_segments * 100) if total_segments > 0 else 0,
                "has_processing_summary": has_processing_summary,
                "has_cloning_summary": has_cloning_summary,
                "has_reconstruction_summary": has_reconstruction_summary,
                "transcription_source": "AssemblyAI"
            }
            
        except Exception as e:
            return {"error": str(e)}
