"""
File Manager Module - Simplified with Metadata Validation
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FileManager:
    """Simplified file manager with metadata validation"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_directory_structure(self, base_dir: Path, speakers: List[str]):
        """Create simplified directory structure - single segments folder for all speakers"""
        (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (base_dir / "segments").mkdir(parents=True, exist_ok=True)
        (base_dir / "cloned").mkdir(parents=True, exist_ok=True)
    
    def validate_and_repair_metadata(self, segments_dir: str) -> Dict[str, Any]:
        """Validate and repair metadata files in unified segments directory"""
        segments_path = Path(segments_dir)
        validation_results = {
            "total_files_checked": 0,
            "files_repaired": 0,
            "files_with_errors": 0,
            "missing_english_text": 0,
            "missing_essential_fields": 0,
            "segments_processed": 0,
            "validation_timestamp": str(datetime.now())
        }
        
        try:
            # Check if segments folder exists
            segments_folder = segments_path / "segments"
            if not segments_folder.exists():
                logger.error(f"Segments folder not found: {segments_folder}")
                validation_results["error"] = "Segments folder not found"
                return validation_results
            
            # Process all metadata files in segments folder
            json_files = list(segments_folder.glob("*_metadata.json"))
            logger.info(f"Found {len(json_files)} metadata files to validate")
            
            for json_file in json_files:
                validation_results["total_files_checked"] += 1
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Track if file needs repair
                    needs_repair = False
                    
                    # Check essential fields
                    essential_fields = ['segment_index', 'audio_file', 'audio_path', 'english_text', 'speaker']
                    missing_fields = [field for field in essential_fields if field not in metadata or not metadata[field]]
                    
                    if missing_fields:
                        validation_results["missing_essential_fields"] += 1
                        logger.warning(f"Missing essential fields in {json_file.name}: {missing_fields}")
                        needs_repair = True
                    
                    # Check english_text specifically
                    if not metadata.get('english_text', '').strip():
                        validation_results["missing_english_text"] += 1
                        logger.warning(f"Missing english_text in {json_file.name}")
                        needs_repair = True
                    
                    # Auto-repair missing fields if possible
                    if needs_repair:
                        # Try to repair basic fields
                        if 'segment_index' not in metadata:
                            # Extract segment number from filename
                            try:
                                import re
                                match = re.search(r'segment_(\d+)_metadata\.json', json_file.name)
                                if match:
                                    metadata['segment_index'] = int(match.group(1))
                            except:
                                metadata['segment_index'] = validation_results["total_files_checked"]
                        
                        if 'speaker' not in metadata or not metadata['speaker']:
                            metadata['speaker'] = 'A'  # Default speaker
                        
                        if 'audio_file' not in metadata:
                            base_name = json_file.stem.replace('_metadata', '')
                            metadata['audio_file'] = f"{base_name}.wav"
                        
                        if 'audio_path' not in metadata:
                            metadata['audio_path'] = str(segments_folder / metadata['audio_file'])
                        
                        # Update cloned paths to new structure
                        cloned_folder = segments_path / "cloned"
                        if 'cloned_audio_file' in metadata:
                            metadata['cloned_audio_path'] = str(cloned_folder / metadata['cloned_audio_file'])
                        
                        # Save repaired metadata
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                        
                        validation_results["files_repaired"] += 1
                        logger.info(f"Repaired metadata file: {json_file.name}")
                    
                    validation_results["segments_processed"] += 1
                    
                except Exception as e:
                    validation_results["files_with_errors"] += 1
                    logger.error(f"Error processing {json_file.name}: {e}")
                    continue
            
            logger.info(f"Validation completed: {validation_results['segments_processed']} segments processed")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)
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
