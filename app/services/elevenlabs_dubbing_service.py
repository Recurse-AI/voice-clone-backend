"""
ElevenLabs Dubbing Service
Handles end-to-end dubbing using ElevenLabs API
"""
import logging
import os
import subprocess
import asyncio
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ElevenLabsDubbingService:
    """Service for processing dubbing via ElevenLabs API"""
    
    def __init__(self):
        from app.services.r2_service import R2Service
        self.r2_service = R2Service()
    
    def process_dubbing(
        self,
        job_id: str,
        target_language: str,
        source_video_language: str,
        job_dir: str,
        audio_path: str,
        num_of_speakers: int,
        separation_urls: dict
    ) -> Dict:
        """
        Main orchestrator for ElevenLabs dubbing
        
        Returns:
            dict: {"success": bool, "result_urls": dict, "dubbing_id": str}
        """
        try:
            logger.info(f"Starting ElevenLabs dubbing for job {job_id}")
            
            original_video_url, duration = self._get_job_info(job_id)
            has_original_video = bool(original_video_url)
            
            logger.info(f"Job {job_id}: has_video={has_original_video}, duration={duration}s")
            
            self._update_status(job_id, 15, "Preparing video for ElevenLabs", "preparation")
            
            blank_video_path = self._create_blank_video(audio_path, job_dir, job_id, duration)
            if not blank_video_path:
                return {"success": False, "error": "Failed to create blank video"}
            
            self._update_status(job_id, 35, "Submitting to ElevenLabs API", "elevenlabs_submit")
            
            dubbing_id = asyncio.run(self._submit_to_elevenlabs(
                blank_video_path, target_language, source_video_language, num_of_speakers, job_id
            ))
            
            if not dubbing_id:
                return {"success": False, "error": "Failed to submit to ElevenLabs"}
            
            logger.info(f"ElevenLabs dubbing submitted: {dubbing_id}")
            
            poll_success = asyncio.run(self._poll_status(job_id, dubbing_id, target_language))
            
            if not poll_success:
                return {"success": False, "error": "Dubbing failed or timeout"}
            
            dubbed_video_path = asyncio.run(self._download_result(
                dubbing_id, target_language, job_dir, job_id
            ))
            
            if not dubbed_video_path:
                return {"success": False, "error": "Failed to download result"}
            
            self._update_status(job_id, 88, "Extracting dubbed audio", "audio_extraction")
            
            dubbed_audio_path = self._extract_audio(dubbed_video_path, job_dir, job_id)
            if not dubbed_audio_path:
                return {"success": False, "error": "Failed to extract audio"}
            
            self._update_status(job_id, 90, "Mixing audio", "mixing")
            
            final_audio_path = self._mix_with_instrumental(
                dubbed_audio_path, separation_urls, job_dir, job_id
            )
            
            final_video_path = None
            if has_original_video:
                self._update_status(job_id, 93, "Merging with original video", "video_merge")
                logger.info(f"🎬 Original video detected, merging audio with video...")
                final_video_path = self._merge_with_original_video(
                    original_video_url, final_audio_path, job_dir, job_id
                )
                if final_video_path:
                    logger.info(f"✅ Video merge successful: {final_video_path}")
                else:
                    logger.warning(f"⚠️ Video merge failed, will return audio only")
            else:
                logger.info(f"📢 No original video, returning audio only")
            
            self._update_status(job_id, 96, "Uploading results", "upload_results")
            
            result_urls = self._upload_results(final_audio_path, final_video_path, job_id)
            
            # Upload separated vocal and instrumental tracks to R2 for permanent storage
            separation_r2_urls = self._upload_separated_tracks(separation_urls, job_dir, job_id)
            
            logger.info(f"📤 Upload complete - Audio: {result_urls.get('audio_url')}, Video: {result_urls.get('video_url')}")
            logger.info(f"🎵 Separation tracks uploaded - Vocal: {separation_r2_urls.get('vocal_url')}, Instrumental: {separation_r2_urls.get('instrumental_url')}")
            logger.info(f"✅ ElevenLabs dubbing completed for job {job_id}")
            
            return {
                "success": True,
                "result_urls": result_urls,
                "separation_r2_urls": separation_r2_urls,
                "dubbing_id": dubbing_id
            }
            
        except Exception as e:
            logger.error(f"ElevenLabs dubbing failed for {job_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _get_job_info(self, job_id: str) -> Tuple[Optional[str], float]:
        """Get job info from database"""
        from app.utils.db_sync_operations import get_dub_job_sync
        
        job_data = get_dub_job_sync(job_id)
        if not job_data:
            logger.error(f"Job {job_id} not found in database")
            return None, 0
        
        video_url = job_data.get("video_url")
        duration = job_data.get("duration", 0)
        
        return video_url, duration
    
    def _update_status(self, job_id: str, progress: int, message: str, phase: str):
        """Update job status"""
        from app.services.simple_status_service import status_service, JobStatus
        
        status_service.update_status(
            job_id, "dub", JobStatus.PROCESSING, progress,
            {"message": message, "phase": phase}
        )
    
    def _create_blank_video(
        self, audio_path: str, job_dir: str, job_id: str, duration: float
    ) -> Optional[str]:
        """Create a blank video (black screen) with audio"""
        try:
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            output_path = os.path.join(job_dir, f"{job_id}_blank.mp4")
            ffmpeg = get_ffmpeg_path()
            
            cmd = [
                ffmpeg, "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}",
                "-i", audio_path,
                "-shortest",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-b:a", "192k",
                output_path
            ]
            
            logger.info(f"Creating blank video: {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return None
            
            if not os.path.exists(output_path):
                logger.error("Blank video file not created")
                return None
            
            logger.info(f"✅ Blank video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create blank video: {e}")
            return None
    
    async def _submit_to_elevenlabs(
        self,
        video_path: str,
        target_language: str,
        source_language: str,
        num_speakers: int,
        job_id: str
    ) -> Optional[str]:
        """Submit dubbing request to ElevenLabs API using REST API"""
        try:
            import httpx
            from app.config.settings import settings
            from app.services.language_service import language_service
            
            if not settings.ELEVENLABS_API_KEY:
                logger.error("ELEVENLABS_API_KEY not configured")
                return None
            
            target_lang_code = language_service.normalize_language_input(target_language)
            
            # Prepare form data
            files = {
                "file": (os.path.basename(video_path), open(video_path, "rb"), "video/mp4")
            }
            
            data = {
                "target_lang": target_lang_code,
                "mode": "automatic",
                "num_speakers": num_speakers,
                "watermark": "true"
            }
            
            # Only add source_lang if explicitly provided
            if source_language and source_language.lower() not in ["auto", "auto_detect", "none", "auto detect"]:
                source_lang_code = language_service.normalize_language_input(source_language)
                data["source_lang"] = source_lang_code
                logger.info(f"Submitting to ElevenLabs: target={target_lang_code}, source={source_lang_code}")
            else:
                logger.info(f"Submitting to ElevenLabs: target={target_lang_code}, source=auto-detect")
            
            logger.info(f"API params: {data}")
            
            # Submit dubbing request via REST API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.elevenlabs.io/v1/dubbing",
                    headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
                    files=files,
                    data=data
                )
            
            if response.status_code != 200:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            dubbing_id = result.get("dubbing_id")
            logger.info(f"✅ ElevenLabs dubbing created: {dubbing_id}")
            
            return dubbing_id
                
        except Exception as e:
            logger.error(f"Failed to submit to ElevenLabs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            # Close the file if it was opened
            if 'files' in locals():
                for file_tuple in files.values():
                    if hasattr(file_tuple[1], 'close'):
                        file_tuple[1].close()
    
    async def _poll_status(self, job_id: str, dubbing_id: str, target_language: str) -> bool:
        """Poll ElevenLabs API until dubbing is complete using REST API"""
        try:
            import httpx
            from app.config.settings import settings
            
            max_attempts = 60
            attempt = 0
            
            logger.info(f"Polling ElevenLabs status for {dubbing_id}")
            
            while attempt < max_attempts:
                await asyncio.sleep(10)
                attempt += 1
                
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(
                            f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}",
                            headers={"xi-api-key": settings.ELEVENLABS_API_KEY}
                        )
                    
                    if response.status_code != 200:
                        logger.warning(f"Status check failed: {response.status_code}")
                        continue
                    
                    metadata = response.json()
                    status = metadata.get("status")
                    logger.info(f"ElevenLabs status: {status} (attempt {attempt}/{max_attempts})")
                    
                    progress = min(40 + int((attempt / max_attempts) * 45), 85)
                    self._update_status(
                        job_id, progress, f"Dubbing in progress: {status}", "elevenlabs_dubbing"
                    )
                    
                    if status == "dubbed":
                        logger.info(f"✅ ElevenLabs dubbing completed")
                        return True
                    elif status == "dubbing_failed":
                        error = metadata.get("error", "Unknown error")
                        logger.error(f"ElevenLabs dubbing failed: {error}")
                        return False
                    
                except Exception as e:
                    logger.warning(f"Status check error: {e}")
                    continue
            
            logger.error(f"ElevenLabs timeout after {max_attempts} attempts")
            return False
                
        except Exception as e:
            logger.error(f"Failed to poll ElevenLabs status: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _download_result(
        self, dubbing_id: str, target_language: str, job_dir: str, job_id: str
    ) -> Optional[str]:
        """Download dubbed video from ElevenLabs using REST API"""
        try:
            import httpx
            from app.config.settings import settings
            from app.services.language_service import language_service
            
            lang_code = language_service.normalize_language_input(target_language)
            output_path = os.path.join(job_dir, f"{job_id}_dubbed_elevenlabs.mp4")
            
            logger.info(f"Downloading dubbed file from ElevenLabs for language: {lang_code}")
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(
                    f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{lang_code}",
                    headers={"xi-api-key": settings.ELEVENLABS_API_KEY}
                )
            
            if response.status_code != 200:
                logger.error(f"Download failed: {response.status_code} - {response.text}")
                return None
            
            # Write the downloaded content to file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded dubbed video: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Failed to download ElevenLabs result: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_audio(self, video_path: str, job_dir: str, job_id: str) -> Optional[str]:
        """Extract audio from dubbed video"""
        try:
            from app.utils.audio.audio_utils import AudioUtils
            
            output_path = os.path.join(job_dir, f"{job_id}_dubbed_audio.wav")
            audio_utils = AudioUtils()
            
            result = audio_utils.extract_audio_from_video(video_path, output_path)
            
            if not result.get("success"):
                logger.error(f"Audio extraction failed: {result.get('error')}")
                return None
            
            logger.info(f"✅ Extracted dubbed audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None
    
    def _mix_with_instrumental(
        self, dubbed_audio_path: str, separation_urls: dict, job_dir: str, job_id: str
    ) -> str:
        """Mix dubbed audio with instrumental track"""
        try:
            import requests
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            instrument_url = separation_urls.get("instrument_audio")
            
            if not instrument_url:
                logger.info("No instrumental track, using dubbed audio as-is")
                return dubbed_audio_path
            
            logger.info("Mixing dubbed audio with instrumental")
            
            instrument_path = os.path.join(job_dir, f"{job_id}_instrument.wav")
            response = requests.get(instrument_url, timeout=120)
            response.raise_for_status()
            
            with open(instrument_path, "wb") as f:
                f.write(response.content)
            
            output_path = os.path.join(job_dir, f"{job_id}_final_audio.wav")
            ffmpeg = get_ffmpeg_path()
            
            cmd = [
                ffmpeg, "-y",
                "-i", dubbed_audio_path,
                "-i", instrument_path,
                "-filter_complex",
                "[0:a]volume=2.0[dub];[1:a]volume=0.95[inst];[dub][inst]amix=inputs=2:duration=longest[out]",
                "-map", "[out]",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Audio mixing failed: {result.stderr}")
                return dubbed_audio_path
            
            logger.info(f"✅ Mixed audio created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to mix audio: {e}")
            return dubbed_audio_path
    
    def _merge_with_original_video(
        self, video_url: str, audio_path: str, job_dir: str, job_id: str
    ) -> Optional[str]:
        """Merge dubbed audio with original video"""
        try:
            import requests
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            from app.config.settings import settings
            
            logger.info("Merging audio with original video")
            
            original_video_path = os.path.join(job_dir, f"{job_id}_original_video.mp4")
            response = requests.get(video_url, timeout=300)
            response.raise_for_status()
            
            with open(original_video_path, "wb") as f:
                f.write(response.content)
            
            output_path = os.path.join(job_dir, f"{job_id}_final_video.mp4")
            ffmpeg = get_ffmpeg_path()
            
            cmd = [ffmpeg, "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            
            cmd.extend([
                "-i", original_video_path,
                "-i", audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                output_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Video merge failed: {result.stderr}")
                return None
            
            logger.info(f"✅ Final video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge video: {e}")
            return None
    
    def _upload_results(self, audio_path: str, video_path: Optional[str], job_id: str) -> Dict:
        """Upload final audio, video, and separated tracks to R2"""
        try:
            result_urls = {}
            
            # Upload final dubbed audio
            audio_key = self.r2_service.generate_file_path(job_id, "final", "audio.wav")
            audio_upload = self.r2_service.upload_file(audio_path, audio_key)
            
            if audio_upload.get("success"):
                result_urls["audio_url"] = audio_upload["url"]
                logger.info(f"✅ Audio uploaded: {result_urls['audio_url']}")
            
            # Upload final video if exists
            if video_path and os.path.exists(video_path):
                video_key = self.r2_service.generate_file_path(job_id, "final", "video.mp4")
                video_upload = self.r2_service.upload_file(video_path, video_key)
                
                if video_upload.get("success"):
                    result_urls["video_url"] = video_upload["url"]
                    logger.info(f"✅ Video uploaded: {result_urls['video_url']}")
            
            return result_urls
            
        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            return {"audio_url": None, "video_url": None}
    
    def _upload_separated_tracks(self, separation_urls: dict, job_dir: str, job_id: str) -> Dict:
        """Download and upload separated vocal and instrumental tracks to R2 for permanent storage"""
        try:
            import requests
            
            separation_r2_urls = {}
            
            vocal_url = separation_urls.get("vocal_audio")
            instrument_url = separation_urls.get("instrument_audio")
            
            # Download and upload vocal track
            if vocal_url:
                try:
                    logger.info(f"Downloading vocal track from RunPod...")
                    vocal_path = os.path.join(job_dir, f"vocal_{job_id}.wav")
                    response = requests.get(vocal_url, timeout=120)
                    response.raise_for_status()
                    
                    with open(vocal_path, "wb") as f:
                        f.write(response.content)
                    
                    # Upload to R2
                    vocal_key = self.r2_service.generate_file_path(job_id, "separation", "vocal.wav")
                    vocal_upload = self.r2_service.upload_file(vocal_path, vocal_key)
                    
                    if vocal_upload.get("success"):
                        separation_r2_urls["vocal_url"] = vocal_upload["url"]
                        logger.info(f"✅ Vocal track uploaded to R2: {separation_r2_urls['vocal_url']}")
                    
                    # Clean up local file
                    if os.path.exists(vocal_path):
                        os.remove(vocal_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to upload vocal track: {e}")
            
            # Download and upload instrumental track
            if instrument_url:
                try:
                    logger.info(f"Downloading instrumental track from RunPod...")
                    instrument_path = os.path.join(job_dir, f"instrument_{job_id}.wav")
                    response = requests.get(instrument_url, timeout=120)
                    response.raise_for_status()
                    
                    with open(instrument_path, "wb") as f:
                        f.write(response.content)
                    
                    # Upload to R2
                    instrument_key = self.r2_service.generate_file_path(job_id, "separation", "instrumental.wav")
                    instrument_upload = self.r2_service.upload_file(instrument_path, instrument_key)
                    
                    if instrument_upload.get("success"):
                        separation_r2_urls["instrumental_url"] = instrument_upload["url"]
                        logger.info(f"✅ Instrumental track uploaded to R2: {separation_r2_urls['instrumental_url']}")
                    
                    # Clean up local file
                    if os.path.exists(instrument_path):
                        os.remove(instrument_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to upload instrumental track: {e}")
            
            return separation_r2_urls
            
        except Exception as e:
            logger.error(f"Failed to upload separated tracks: {e}")
            return {}


def get_elevenlabs_dubbing_service() -> ElevenLabsDubbingService:
    """Get singleton instance of ElevenLabs dubbing service"""
    return ElevenLabsDubbingService()

