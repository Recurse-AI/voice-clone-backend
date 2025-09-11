import json
import os
import logging
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ManifestManager:
    
    def __init__(self):
        self.temp_dir = Path("./tmp")
    
    def _normalize_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        normalized = segment.copy()
        
        # Better logic: check if values are floats (likely seconds) vs ints (likely milliseconds)
        if 'start' in normalized and isinstance(normalized['start'], (float, int)):
            if isinstance(normalized['start'], float) and normalized['start'] < 1000:
                # Float under 1000 = seconds, convert to milliseconds
                normalized['start'] = int(normalized['start'] * 1000)
            else:
                # Int or large value = already milliseconds
                normalized['start'] = int(normalized['start'])
        
        if 'end' in normalized and isinstance(normalized['end'], (float, int)):
            if isinstance(normalized['end'], float) and normalized['end'] < 1000:
                # Float under 1000 = seconds, convert to milliseconds
                normalized['end'] = int(normalized['end'] * 1000)
            else:
                # Int or large value = already milliseconds
                normalized['end'] = int(normalized['end'])
        
        if 'duration_ms' in normalized and isinstance(normalized['duration_ms'], (float, int)):
            if isinstance(normalized['duration_ms'], float) and normalized['duration_ms'] < 1000:
                # Float under 1000 = seconds, convert to milliseconds
                normalized['duration_ms'] = int(normalized['duration_ms'] * 1000)
            else:
                # Int or large value = already milliseconds
                normalized['duration_ms'] = int(normalized['duration_ms'])
        
        if 'segment_index' in normalized and isinstance(normalized['segment_index'], (float, int)):
            normalized['segment_index'] = int(normalized['segment_index'])
        
        return normalized
        
    def create_manifest(self, job_id: str, transcript_id: Optional[str], target_language: str, 
                       segments: list, vocal_audio_url: Optional[str] = None, 
                       instrument_audio_url: Optional[str] = None, voice_premium_model: bool = False) -> Dict[str, Any]:
        normalized_segments = [self._normalize_segment(seg) for seg in segments]
        return {
            "job_id": job_id,
            "transcript_id": transcript_id,
            "target_language": target_language,
            "version": 1,
            "vocal_audio_url": vocal_audio_url,
            "instrument_audio_url": instrument_audio_url,
            "voice_premium_model": voice_premium_model,
            "segments": normalized_segments
        }
    
    def load_manifest(self, manifest_url_or_path: str) -> Dict[str, Any]:
        if manifest_url_or_path.startswith(('http://', 'https://')):
            return self._load_from_url(manifest_url_or_path)
        else:
            return self._load_from_file(manifest_url_or_path)
    
    def _load_from_url(self, url: str) -> Dict[str, Any]:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_manifest(self, manifest: Dict[str, Any], job_id: str, 
                     temp_dir: Optional[str] = None) -> str:
        if temp_dir is None:
            temp_dir = self.temp_dir / job_id
        
        os.makedirs(temp_dir, exist_ok=True)
        manifest_path = os.path.join(temp_dir, f"manifest_{job_id}.json")
        
        normalized_manifest = self._normalize_manifest(manifest)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_manifest, f, ensure_ascii=False, indent=2)
        
        return manifest_path
    
    def _normalize_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        normalized = manifest.copy()
        if 'segments' in normalized:
            normalized['segments'] = [self._normalize_segment(seg) for seg in normalized['segments']]
        return normalized
    
    def add_segment(self, manifest: Dict[str, Any], segment: Dict[str, Any]) -> Dict[str, Any]:
        updated_manifest = manifest.copy()
        if 'segments' not in updated_manifest:
            updated_manifest['segments'] = []
        
        normalized_segment = self._normalize_segment(segment)
        updated_manifest['segments'].append(normalized_segment)
        return updated_manifest
    
    def update_segment(self, manifest: Dict[str, Any], segment_id: str, 
                      updates: Dict[str, Any]) -> Dict[str, Any]:
        updated_manifest = manifest.copy()
        segments = updated_manifest.get('segments', [])
        
        for i, seg in enumerate(segments):
            if seg.get('id') == segment_id:
                updated_seg = seg.copy()
                updated_seg.update(updates)
                segments[i] = self._normalize_segment(updated_seg)
                break
        
        return updated_manifest
    
    def get_segment(self, manifest: Dict[str, Any], segment_id: str) -> Optional[Dict[str, Any]]:
        segments = manifest.get('segments', [])
        for seg in segments:
            if seg.get('id') == segment_id:
                return seg
        return None
    
    def delete_segment(self, manifest: Dict[str, Any], segment_id: str) -> Dict[str, Any]:
        updated_manifest = manifest.copy()
        segments = updated_manifest.get('segments', [])
        updated_manifest['segments'] = [seg for seg in segments if seg.get('id') != segment_id]
        return updated_manifest
    
    def get_segments_for_subtitles(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        segments = manifest.get('segments', [])
        subtitle_data = []
        
        for seg in segments:
            start_sec = seg.get('start', 0) / 1000.0
            end_sec = seg.get('end', 0) / 1000.0
            text = seg.get('dubbed_text') or seg.get('original_text', '')
            
            if text:
                subtitle_data.append({
                    "start": start_sec,
                    "end": end_sec, 
                    "text": text
                })
        
        return subtitle_data
    
    def increment_version(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        updated_manifest = manifest.copy()
        current_version = updated_manifest.get('version', 1)
        updated_manifest['version'] = current_version + 1
        return updated_manifest

manifest_manager = ManifestManager()
