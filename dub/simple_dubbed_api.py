#!/usr/bin/env python3
"""
Simple Dubbed API - Clean Implementation using existing dub folder
Audio URL → Download → Assembly AI → Dub → Voice Clone → Reconstruct → R2 Upload
"""

import os
import json
import logging
import time
from config import settings
from .assembly_transcription import TranscriptionService
from .fish_speech_service import get_fish_speech_service
from r2_storage import R2Storage
import numpy as np

logger = logging.getLogger(__name__)

class SimpleDubbedAPI:
    """Simple clean API for dubbed audio processing"""
    
    def __init__(self):
        # Use existing services from dub folder
        self.transcription_service = TranscriptionService()
        self.fish_speech = get_fish_speech_service()  # Use global singleton instance
        self.r2_storage = R2Storage()
        self.temp_dir = settings.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def dub_text(self, text: str, target_language: str = "English") -> str:
        """Dub text using OpenAI from existing service"""
        response = self.transcription_service.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Translate to {target_language} naturally for voice dubbing."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=1000  # Increased max tokens
        )
        return response.choices[0].message.content.strip()
    
    def process_dubbed_audio(self, audio_url: str, speakers_count: int = 2, target_language: str = "English") -> dict:
        """Complete dubbed audio processing"""
        try:
            logger.info(f"Starting dubbed processing for {speakers_count} speakers")
            
            # Step 1: Use existing transcription service 
            paragraphs_result = self.transcription_service.get_paragraphs_and_split_audio(
                audio_url, None, speakers_count
            )
            
            if not paragraphs_result["success"]:
                raise Exception(f"Transcription failed: {paragraphs_result.get('error', 'Unknown error')}")
            
            raw_paragraphs = paragraphs_result["segments"]
            transcript_id = paragraphs_result.get("transcript_id", "unknown")
            
            logger.info(f"Transcription completed: {transcript_id}")
            logger.info(f"Found {len(raw_paragraphs)} segments")
            
            # Step 2: Process each segment  
            dubbed_segments = []
            
            for i, segment in enumerate(raw_paragraphs):
                seg_id = f"seg_{i+1:03d}"
                
                # Extract segment info
                start_ms = segment.get("start", 0)
                end_ms = segment.get("end", 0)
                original_text = segment.get("text", "")
                speaker = segment.get("speaker", "Unknown")
                original_audio_path = segment.get("output_path", "")
                
                logger.info(f"Processing segment {seg_id}: speaker={speaker}, audio_path={original_audio_path}")
                
                # Skip if no text
                if not original_text.strip():
                    continue
                
                # Dub text
                dubbed_text = self.dub_text(original_text, target_language)
                
                # Voice clone dubbed text with reference
                cloned_audio_path = self._voice_clone_segment(dubbed_text, original_audio_path, seg_id, original_text)
                
                # Create clean segment info
                segment_json = {
                    "id": seg_id,
                    "segment_index": i + 1,
                    "start": start_ms,
                    "end": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "original_text": original_text,
                    "dubbed_text": dubbed_text,
                    "speaker": speaker,
                    "original_audio_file": f"segment_{i:03d}.wav" if original_audio_path else None,
                    "cloned_audio_file": f"cloned_{i:03d}.wav" if cloned_audio_path else None,
                    "voice_cloned": bool(cloned_audio_path),
                    "original_audio_path": original_audio_path,
                    "cloned_audio_path": cloned_audio_path
                }
                
                # Save JSON file with clean name
                json_filename = f"segment_{i:03d}_info.json"
                json_path = os.path.join(self.temp_dir, json_filename).replace('\\', '/')
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(segment_json, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved segment info: {json_filename}")
                except Exception as e:
                    logger.error(f"Failed to save JSON for {seg_id}: {str(e)}")
                
                dubbed_segments.append(segment_json)
            
            # Step 3: Reconstruct final audio
            final_audio_path = self._reconstruct_final_audio(dubbed_segments, paragraphs_result.get("original_audio"))
            
            # Step 4: Upload to R2
            final_audio_url = None
            if final_audio_path:
                r2_result = self.r2_storage.upload_file(
                    final_audio_path,
                    f"dubbed_final_{int(time.time())}.wav"
                )
                if r2_result.get("success"):
                    final_audio_url = r2_result.get("public_url")
            
            # Step 5: Save complete process summary
            process_summary = {
                "success": True,
                "transcript_id": transcript_id,
                "segments_count": len(dubbed_segments),
                "audio_url": audio_url,
                "target_language": target_language,
                "speakers_count": speakers_count,
                "final_audio_url": final_audio_url,
                "final_audio_file": os.path.basename(final_audio_path) if final_audio_path else None,
                "processing_timestamp": int(time.time()),
                "segments": dubbed_segments
            }
            
            # Save process summary JSON
            summary_filename = f"dubbing_process_summary_{int(time.time())}.json"
            summary_path = os.path.join(self.temp_dir, summary_filename).replace('\\', '/')
            
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(process_summary, f, ensure_ascii=False, indent=2)
                logger.info(f"Process summary saved: {summary_filename}")
            except Exception as e:
                logger.error(f"Failed to save process summary: {str(e)}")
            
            result = process_summary
            
            # Note: Keep all files for inspection - no cleanup
            logger.info(f"Process completed. Files saved in: {self.temp_dir}")
            logger.info(f"Check: segment_xxx_info.json files and cloned_xxx.wav files")
            logger.info(f"Summary: {summary_filename}")
            
            logger.info("Dubbed processing completed")
            return result
            
        except Exception as e:
            logger.error(f"Dubbed processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    

    
    def _voice_clone_segment(self, dubbed_text: str, reference_audio_path: str, segment_id: str, original_text: str = "") -> str:
        """Voice clone dubbed text using FishSpeechService with reference audio"""
        try:
            if not reference_audio_path:
                logger.warning(f"No reference audio path provided for {segment_id}")
                return None
                
            if not os.path.exists(reference_audio_path):
                logger.warning(f"Reference audio file not found: {reference_audio_path} for {segment_id}")
                return None
            
            logger.info(f"Voice cloning {segment_id} using reference: {reference_audio_path}")
            
            # Reference audio ৭ সেকেন্ডের বেশি হলে শুধু প্রথম ৭ সেকেন্ড পাঠাও
            import soundfile as sf
            import io
            max_ref_sec = 7
            audio_data, sample_rate = sf.read(reference_audio_path)
            max_samples = int(max_ref_sec * sample_rate)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # শুধু প্রথম চ্যানেল
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            reference_audio_bytes = buffer.getvalue()
            
            if not reference_audio_bytes:
                logger.error(f"Failed to load reference audio: {reference_audio_path}")
                return None
            
            # Voice clone using FishSpeechService with clean naming
            # Extract segment index from segment_id (seg_001 -> 0)
            segment_index = int(segment_id.split('_')[1]) - 1
            cloned_filename = f"cloned_{segment_index:03d}.wav"
            cloned_path = os.path.join(self.temp_dir, cloned_filename).replace('\\', '/')
            
            # Intelligent chunking: sentence বা word boundary-তে chunk শেষ করো
            def smart_chunk(text, chunk_size=200, min_size=180):
                chunks = []
                i = 0
                while i < len(text):
                    end = min(i + chunk_size, len(text))
                    chunk = text[i:end]
                    # প্রথমে sentence boundary খুঁজো
                    puncts = [chunk.rfind(p) for p in '.!?']
                    punct = max(puncts)
                    if punct >= min_size - 1:
                        split_at = punct + 1
                    else:
                        # না পেলে word boundary (space)
                        space = chunk.rfind(' ')
                        if space >= min_size - 1:
                            split_at = space + 1
                        else:
                            # fallback: ২০০ char
                            split_at = len(chunk)
                    chunks.append(text[i:i+split_at].strip())
                    i += split_at
                return [c for c in chunks if c]
            text_chunks = smart_chunk(dubbed_text)
            audio_chunks = []
            sample_rate_out = None
            for chunk in text_chunks:
                result = self.fish_speech.generate_with_reference_audio(
                    text=chunk,
                    reference_audio_bytes=reference_audio_bytes,
                    reference_text=original_text or "Reference audio",
                    max_new_tokens=1024,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    temperature=0.8,
                    chunk_length=200
                )
                print(result)
                if result.get("success"):
                    import soundfile as sf
                    import io
                    buffer = io.BytesIO(result["audio_data"])
                    audio, sample_rate = sf.read(buffer)
                    if len(audio.shape) > 1:
                        audio = audio[:, 0]
                    audio_chunks.append(audio)
                    sample_rate_out = sample_rate
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
                import soundfile as sf
                buffer = io.BytesIO()
                sf.write(buffer, final_audio, sample_rate_out, format='WAV')
                with open(cloned_path, "wb") as f:
                    f.write(buffer.getvalue())
                return cloned_path
            else:
                logger.error(f"Voice cloning failed for {segment_id}: No audio chunks generated.")
                return None
                
        except Exception as e:
            logger.error(f"Voice cloning error for {segment_id}: {str(e)}")
            return None
    
    def _reconstruct_final_audio(self, segments: list, original_audio_path: str) -> str:
        """Reconstruct final audio maintaining exact timing"""
        try:
            import soundfile as sf
            import numpy as np
            
            # Get original audio duration
            original_audio, sample_rate = sf.read(original_audio_path)
            total_duration = len(original_audio) / sample_rate
            total_samples = len(original_audio)
            
            # Create final audio array
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            logger.info(f"Reconstructing audio: {total_duration:.2f}s, {len(segments)} segments")
            
            for segment in segments:
                if not segment.get("voice_cloned") or not segment.get("cloned_audio"):
                    continue
                
                # Load cloned audio
                cloned_path = os.path.join(self.temp_dir, segment["cloned_audio"])
                if not os.path.exists(cloned_path):
                    continue
                
                try:
                    cloned_audio, _ = sf.read(cloned_path)
                    if len(cloned_audio.shape) > 1:
                        cloned_audio = np.mean(cloned_audio, axis=1)  # Convert to mono
                    
                    # Calculate timing
                    start_ms = segment["start"]
                    end_ms = segment["end"]
                    start_sample = int((start_ms / 1000.0) * sample_rate)
                    segment_duration_samples = int(((end_ms - start_ms) / 1000.0) * sample_rate)
                    
                    # Adjust cloned audio length to match segment duration
                    if len(cloned_audio) > segment_duration_samples:
                        cloned_audio = cloned_audio[:segment_duration_samples]
                    elif len(cloned_audio) < segment_duration_samples:
                        # Pad with silence
                        padding = segment_duration_samples - len(cloned_audio)
                        cloned_audio = np.pad(cloned_audio, (0, padding), mode='constant')
                    
                    # Place in final audio
                    end_sample = min(start_sample + len(cloned_audio), total_samples)
                    final_audio[start_sample:end_sample] = cloned_audio[:end_sample-start_sample]
                    
                    logger.info(f"Placed {segment['id']} at {start_ms}ms")
                    
                except Exception as e:
                    logger.error(f"Failed to process segment {segment['id']}: {str(e)}")
                    continue
            
            # Save final audio
            final_path = os.path.join(self.temp_dir, f"final_dubbed_{int(time.time())}.wav").replace('\\', '/')
            sf.write(final_path, final_audio, sample_rate)
            
            logger.info(f"Final audio reconstructed: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to reconstruct final audio: {str(e)}")
            return None
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {str(e)}")

# API endpoint function
def create_dubbed_audio_simple(audio_url: str, speakers_count: int = 2, target_language: str = "English") -> dict:
    """Main API function for simple dubbed audio creation"""
    processor = SimpleDubbedAPI()
    return processor.process_dubbed_audio(audio_url, speakers_count, target_language)