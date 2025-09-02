from fastapi import APIRouter, UploadFile, File, Form
import logging
import os
from app.schemas import (
    TimelineAudioSegment,
    VideoProcessingOptions,
    VideoProcessingResponse,
)
from typing import Optional, List

import uuid
from pathlib import Path

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/process-video-complete", response_model=VideoProcessingResponse)
async def process_video_complete(
    video_file: Optional[UploadFile] = File(None),
    dubbed_audio_url: Optional[str] = Form(None),
    instrument_audio_url: Optional[str] = Form(None),
    timeline_audio: Optional[str] = Form(None),
    subtitle_url: Optional[str] = Form(None),
    options: str = Form("{}")
):
    """
    Complete video processing API - handles video, audio, subtitles, and timeline reconstruction.
    
    ALL INPUTS ARE OPTIONAL - But at least ONE must be provided:
    - video_file: Video file upload
    - dubbed_audio_url: Dubbed audio URL  
    - instrument_audio_url: Background music URL
    - timeline_audio: JSON array of audio segments
    - subtitle_url: SRT file URL
    - options: Processing options (resolution, format, etc.)
    
    Response:
    - download_url: Final processed file
    - output_type: "video" or "audio"
    """
    import subprocess
    import requests
    import tempfile
    import json
    from typing import List
    
    job_id = str(uuid.uuid4())
    logger.info(f"🎬 Starting video processing job {job_id}")
    
    try:
        # 1. Parse options
        options_dict = json.loads(options) if options != "{}" else {}
        processing_options = VideoProcessingOptions(**options_dict)
        
        # 2. Parse timeline audio if provided
        timeline_segments: List[TimelineAudioSegment] = []
        if timeline_audio:
            try:
                timeline_data = json.loads(timeline_audio)
                timeline_segments = [TimelineAudioSegment(**seg) for seg in timeline_data]
                logger.info(f"📋 Parsed {len(timeline_segments)} timeline segments")
            except Exception as e:
                return VideoProcessingResponse(
                    success=False,
                    message="Invalid timeline audio format",
                    job_id=job_id,
                    error=f"Timeline parsing failed: {str(e)}",
                    error_code="INVALID_TIMELINE"
                )
        
        # 3. Validate: At least one input required
        has_input = any([
            video_file,
            dubbed_audio_url,
            instrument_audio_url, 
            len(timeline_segments) > 0
        ])
        
        if not has_input:
            return VideoProcessingResponse(
                success=False,
                message="No input provided",
                job_id=job_id,
                error="At least one input required: video_file, dubbed_audio_url, instrument_audio_url, or timeline_audio",
                error_code="NO_INPUT"
            )
        
        # 4. Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"📁 Created temp directory: {temp_path}")
            
            # === STEP 1: Download all remote files ===
            video_path = None
            dubbed_audio_path = None
            instrument_audio_path = None
            subtitle_path = None
            
            # Save video file if provided
            if video_file:
                video_filename = video_file.filename or f"video_{job_id}.mp4"
                video_ext = Path(video_filename).suffix or ".mp4"
                video_path = temp_path / f"input_video{video_ext}"
                
                content = await video_file.read()
                with open(video_path, "wb") as f:
                    f.write(content)
                logger.info(f"💾 Video saved: {len(content)/1024/1024:.1f} MB")
            
            # Download dubbed audio if provided
            if dubbed_audio_url:
                dubbed_audio_path = temp_path / "dubbed_audio.mp3"
                await _download_file(dubbed_audio_url, dubbed_audio_path, "Dubbed audio")
            
            # Download instrument audio if provided
            if instrument_audio_url:
                instrument_audio_path = temp_path / "instrument_audio.mp3"
                await _download_file(instrument_audio_url, instrument_audio_path, "Instrument audio")
            
            # Download subtitle file if provided
            if subtitle_url:
                subtitle_path = temp_path / "subtitles.srt"
                await _download_file(subtitle_url, subtitle_path, "Subtitle file")
            
            # === STEP 2: Process Audio ===
            final_audio_path = None
            
            # Option A: Timeline audio reconstruction (acts as dubbed_audio)
            if len(timeline_segments) > 0:
                logger.info(f"🎵 Reconstructing timeline audio from {len(timeline_segments)} segments")
                reconstructed_audio = await _reconstruct_timeline_audio(timeline_segments, temp_path, job_id)
                if not reconstructed_audio:
                    return VideoProcessingResponse(
                        success=False,
                        message="Timeline audio reconstruction failed",
                        job_id=job_id,
                        error="Failed to reconstruct timeline audio",
                        error_code="TIMELINE_FAILED"
                    )
                
                # Treat reconstructed audio as dubbed_audio and mix with instrument if present
                logger.info("🎵 Treating timeline audio as dubbed audio")
                final_audio_path = await _mix_audio_files(
                    reconstructed_audio, instrument_audio_path, 
                    temp_path, processing_options.instrument_volume, job_id
                )
                if not final_audio_path:
                    return VideoProcessingResponse(
                        success=False,
                        message="Audio mixing with timeline failed",
                        job_id=job_id,
                        error="Failed to mix timeline audio with instrument",
                        error_code="TIMELINE_MIX_FAILED"
                    )
            
            # Option B: Regular audio mixing (dubbed + instrument)
            elif dubbed_audio_path or instrument_audio_path:
                logger.info("🎵 Processing regular audio mixing")
                final_audio_path = await _mix_audio_files(
                    dubbed_audio_path, instrument_audio_path, 
                    temp_path, processing_options.instrument_volume, job_id
                )
                if not final_audio_path:
                    return VideoProcessingResponse(
                        success=False,
                        message="Audio mixing failed",
                        job_id=job_id,
                        error="Failed to mix audio files",
                        error_code="AUDIO_MIX_FAILED"
                    )
            
            # === STEP 3: Video Processing ===
            output_path = None
            output_type = "audio" if processing_options.audio_only else "video"
            
            if processing_options.audio_only:
                # Audio-only output with proper format
                if final_audio_path:
                    audio_format = processing_options.audio_format or "mp3"
                    output_path = temp_path / f"final_audio_{job_id}.{audio_format}"
                    
                    # Convert to requested audio format if needed
                    if audio_format != "mp3":
                        import subprocess
                        convert_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(final_audio_path),
                            "-c:a", _get_audio_codec(audio_format),
                            str(output_path)
                        ]
                        result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode != 0:
                            logger.warning(f"Audio format conversion failed, using original: {result.stderr}")
                            import shutil
                            shutil.copy2(final_audio_path, output_path)
                        else:
                            logger.info(f"🎵 Audio converted to {audio_format}")
                    else:
                        import shutil
                        shutil.copy2(final_audio_path, output_path)
                    logger.info("🎵 Audio-only output prepared")
                else:
                    return VideoProcessingResponse(
                        success=False,
                        message="No audio to output",
                        job_id=job_id,
                        error="Audio-only requested but no audio available",
                        error_code="NO_AUDIO"
                    )
            
            elif video_path:
                # Video processing with audio replacement
                logger.info("🎬 Processing video with audio replacement")
                output_path = await _process_video_with_audio(
                    video_path, final_audio_path, subtitle_path,
                    temp_path, processing_options, job_id
                )
                if not output_path:
                    return VideoProcessingResponse(
                        success=False,
                        message="Video processing failed",
                        job_id=job_id,
                        error="Failed to process video",
                        error_code="VIDEO_PROCESSING_FAILED"
                    )
            
            else:
                return VideoProcessingResponse(
                    success=False,
                    message="No video provided for video output",
                    job_id=job_id,
                    error="Video file required when audio_only=false",
                    error_code="NO_VIDEO"
                )
            
            # === STEP 4: Upload result ===
            logger.info("☁️ Uploading to R2...")
            from app.services.r2_service import get_r2_service
            r2_service = get_r2_service()
            
            output_filename = output_path.name
            r2_key = f"processed/{job_id}/{output_filename}"
            
            upload_result = r2_service.upload_file(str(output_path), r2_key)
            
            if not upload_result.get("success"):
                return VideoProcessingResponse(
                    success=False,
                    message="Upload failed",
                    job_id=job_id,
                    error=upload_result.get("error"),
                    error_code="UPLOAD_FAILED"
                )
            
            # === STEP 5: Success response ===
            file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
            duration_seconds = await _get_media_duration(output_path)
            
            logger.info(f"✅ Processing completed: {file_size_mb} MB, {duration_seconds}s")
            
            return VideoProcessingResponse(
                success=True,
                message="Video processing completed successfully",
                job_id=job_id,
                download_url=upload_result["url"],
                output_filename=output_filename,
                output_type=output_type,
                file_size_mb=file_size_mb,
                duration_seconds=duration_seconds,
                applied_options=processing_options
            )
            
    except subprocess.TimeoutExpired:
        return VideoProcessingResponse(
            success=False,
            message="Processing timeout",
            job_id=job_id,
            error="Processing exceeded 15-minute limit",
            error_code="TIMEOUT"
        )
        
    except Exception as e:
        logger.error(f"💥 Processing job {job_id} failed: {str(e)}")
        return VideoProcessingResponse(
            success=False,
            message="Processing failed",
            job_id=job_id,
            error=str(e),
            error_code="UNEXPECTED_ERROR"
        )


# Helper functions for process_video_complete

async def _download_file(url: str, file_path: Path, description: str) -> None:
    """Download file from URL to local path"""
    try:
        import requests
        logger.info(f"📥 Downloading {description} from: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"✅ {description} downloaded: {len(response.content)} bytes")
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {str(e)}")
        raise


async def _reconstruct_timeline_audio(
    segments: List[TimelineAudioSegment], 
    temp_path: Path, 
    job_id: str
) -> Optional[Path]:
    """Reconstruct audio from timeline segments with proper time positioning"""
    try:
        import subprocess
        
        # Download all segment audio files
        segment_files = []
        for i, segment in enumerate(segments):
            segment_path = temp_path / f"segment_{i}.mp3"
            await _download_file(segment.audio_url, segment_path, f"Timeline segment {i}")
            segment_files.append((segment_path, segment.start, segment.end))
        
        # Calculate total duration
        total_duration = max(seg.end for seg in segments)
        logger.info(f"🎵 Total timeline duration: {total_duration}ms")
        
        # Build complex FFmpeg command to handle all segments at once
        output_path = temp_path / f"timeline_audio_{job_id}.mp3"
        
        # Create the command with base silent audio and all segment inputs
        cmd = ["ffmpeg", "-y"]
        
        # Add silent base audio as first input
        cmd.extend(["-f", "lavfi", "-i", f"anullsrc=duration={total_duration/1000}:sample_rate=44100"])
        
        # Add all segment files as inputs (no offset here)
        input_count = 1  # Start from 1 since 0 is the silent base
        delay_filters = []
        mix_inputs = ["[0:a]"]  # Start with silent base
        
        for segment_path, start_ms, end_ms in segment_files:
            cmd.extend(["-i", str(segment_path)])
            
            # Create adelay filter for this segment with volume 1.0
            delay_ms = start_ms
            delay_filter = f"[{input_count}:a]volume=1.0,adelay={delay_ms}|{delay_ms}[delayed{input_count}]"
            delay_filters.append(delay_filter)
            mix_inputs.append(f"[delayed{input_count}]")
            input_count += 1
        
        # Build filter complex for delaying and mixing all inputs
        if len(delay_filters) > 0:
            # Combine delay filters with mixing
            all_filters = ";".join(delay_filters)
            mix_filter = f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[out]"
            filter_complex = f"{all_filters};{mix_filter}"
            
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])
        else:
            # No segments, just use the base audio
            cmd.extend(["-map", "0:a"])
        
        cmd.extend(["-c:a", "mp3", str(output_path)])
        
        logger.info(f"🎵 Reconstructing timeline with adelay filters: {len(segment_files)} segments")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"❌ Timeline reconstruction failed: {result.stderr}")
            return None
        
        logger.info(f"✅ Timeline audio reconstructed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Timeline reconstruction failed: {str(e)}")
        return None


async def _mix_audio_files(
    dubbed_path: Optional[Path], 
    instrument_path: Optional[Path],
    temp_path: Path, 
    instrument_volume: float,
    job_id: str
) -> Optional[Path]:
    """Mix dubbed audio with instrument audio"""
    try:
        import subprocess
        
        output_path = temp_path / f"mixed_audio_{job_id}.mp3"
        
        if dubbed_path and instrument_path:
            # Mix both files
            logger.info(f"🎵 Mixing dubbed + instrument (volume: {instrument_volume})")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(dubbed_path),
                "-i", str(instrument_path),
                "-filter_complex", f"[0:a]volume=1.0[dub];[1:a]volume={instrument_volume}[inst];[dub][inst]amix=inputs=2:duration=longest[out]",
                "-map", "[out]",
                str(output_path)
            ]
        elif dubbed_path:
            # Only dubbed audio
            logger.info("🎵 Using dubbed audio only")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(dubbed_path),
                "-filter:a", "volume=1.0",
                "-c:a", "mp3",
                str(output_path)
            ]
        elif instrument_path:
            # Only instrument audio
            logger.info(f"🎵 Using instrument audio only (volume: {instrument_volume})")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(instrument_path),
                "-filter:a", f"volume={instrument_volume}",
                "-c:a", "mp3",
                str(output_path)
            ]
        else:
            return None
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"❌ Audio mixing failed: {result.stderr}")
            return None
        
        logger.info(f"✅ Audio mixing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Audio mixing error: {str(e)}")
        return None


async def _process_video_with_audio(
    video_path: Path,
    audio_path: Optional[Path],
    subtitle_path: Optional[Path],
    temp_path: Path,
    options: VideoProcessingOptions,
    job_id: str
) -> Optional[Path]:
    """Process video with audio replacement and optional subtitles"""
    try:
        import subprocess
        
        video_ext = video_path.suffix
        output_path = temp_path / f"final_video_{job_id}{video_ext}"
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        
        # Add audio input if provided
        if audio_path:
            cmd.extend(["-i", str(audio_path)])
        
        # Video filters
        filters = []
        
        # Add subtitle filter if provided
        if subtitle_path and options.include_subtitles:
            subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
            filters.append(f"subtitles='{subtitle_str}'")
        
        # Resolution scaling
        if options.resolution == "720p":
            filters.append("scale=1280:720")
        elif options.resolution == "4k":
            filters.append("scale=3840:2160")
        
        # Apply video filters
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Audio handling
        if audio_path:
            cmd.extend(["-c:a", "aac", "-map", "0:v", "-map", "1:a"])
        else:
            cmd.extend(["-c:a", "copy"])
        
        # Video codec and quality
        if options.quality == "high":
            cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "slow"])
        else:
            cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "fast"])
        
        # Output format
        if options.format == "webm":
            cmd.extend(["-f", "webm", "-c:v", "libvpx-vp9"])
        elif options.format == "mov":
            cmd.extend(["-f", "mov"])
        
        cmd.append(str(output_path))
        
        logger.info(f"🎬 Processing video: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            logger.error(f"❌ Video processing failed: {result.stderr}")
            return None
        
        logger.info(f"✅ Video processing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Video processing error: {str(e)}")
        return None


async def _get_media_duration(file_path: Path) -> Optional[float]:
    """Get media file duration in seconds"""
    try:
        import subprocess
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _get_audio_codec(audio_format: str) -> str:
    """Get appropriate audio codec for format"""
    codec_map = {
        "mp3": "mp3",
        "wav": "pcm_s16le",
        "aac": "aac"
    }
    return codec_map.get(audio_format.lower(), "mp3")
