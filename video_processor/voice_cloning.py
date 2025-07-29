"""
OpenVoice Voice Cloning Service - MIT License Implementation
Dedicated voice cloning solution with accurate tone color cloning and style control
OpenVoice by MyShell AI and MIT - Perfect for GPU-based voice cloning
"""

import time
import logging
import warnings
import gc
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import soundfile as sf
import torch
import random

from config import settings
from .audio_utils import AudioUtils

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"🎲 Set global seed to {seed}")

def add_silence(audio: np.ndarray, duration_sec: float = 1.0, sample_rate: int = 24000) -> np.ndarray:
    """Add silence to audio array"""
    silence_samples = int(duration_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([audio, silence])

def apply_speed_adjustment(audio: np.ndarray, speed: float) -> np.ndarray:
    """Apply speed adjustment to audio using resampling"""
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)
    except ImportError:
        logger.warning("librosa not available, skipping speed adjustment")
        return audio

class Args:
    """OpenVoice generation arguments"""
    def __init__(self, **kwargs):
        # OpenVoice specific parameters
        self.max_length = kwargs.get('max_length', settings.OPENVOICE_MAX_LENGTH)
        self.temperature = kwargs.get('temperature', settings.OPENVOICE_TEMPERATURE)
        self.top_p = kwargs.get('top_p', settings.OPENVOICE_TOP_P)
        self.repetition_penalty = kwargs.get('repetition_penalty', settings.OPENVOICE_REPETITION_PENALTY)
        self.seed = kwargs.get('seed', settings.OPENVOICE_DEFAULT_SEED)
        self.emotion = kwargs.get('emotion', settings.OPENVOICE_DEFAULT_EMOTION)
        self.chunk_length = kwargs.get('chunk_length', settings.OPENVOICE_CHUNK_LENGTH)
        self.max_retries = kwargs.get('max_retries', settings.OPENVOICE_MAX_RETRIES)
        self.use_autocast = kwargs.get('use_autocast', settings.OPENVOICE_USE_AUTOCAST)
        self.compile_model = kwargs.get('compile_model', settings.OPENVOICE_COMPILE)
        
        # Style control parameters
        self.enable_emotion = kwargs.get('enable_emotion', settings.OPENVOICE_ENABLE_EMOTION_CONTROL)
        self.enable_accent = kwargs.get('enable_accent', settings.OPENVOICE_ENABLE_ACCENT_CONTROL)
        self.enable_rhythm = kwargs.get('enable_rhythm', settings.OPENVOICE_ENABLE_RHYTHM_CONTROL)

class OpenVoiceVoiceCloningService:
    """
    OpenVoice Voice Cloning Service - MIT Licensed
    Dedicated voice cloning with accurate tone color cloning and flexible style control
    Optimized for GPU usage with zero-shot cross-lingual capabilities
    """
    
    def __init__(self):
        self.openvoice_model = None
        self.tone_color_converter = None
        self.device = self._detect_device()
        self.sample_rate = settings.OPENVOICE_SAMPLE_RATE
        
        # Default parameters from settings
        self.default_args = Args(
            max_length=settings.OPENVOICE_MAX_LENGTH,
            temperature=settings.OPENVOICE_TEMPERATURE,
            top_p=settings.OPENVOICE_TOP_P,
            repetition_penalty=settings.OPENVOICE_REPETITION_PENALTY,
            seed=settings.OPENVOICE_DEFAULT_SEED,
            emotion=settings.OPENVOICE_DEFAULT_EMOTION,
            chunk_length=settings.OPENVOICE_CHUNK_LENGTH,
            max_retries=settings.OPENVOICE_MAX_RETRIES,
            use_autocast=settings.OPENVOICE_USE_AUTOCAST,
            compile_model=settings.OPENVOICE_COMPILE
        )
        
        logger.info(f"🎙️ Initialized OpenVoice Service (MIT License) on device: {self.device}")
        
        # Automatically attempt to load model during initialization
        try:
            if self.load_model():
                logger.info("✅ OpenVoice model loaded successfully during initialization")
            else:
                logger.warning("⚠️ OpenVoice model failed to load during initialization - will retry on first use")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load OpenVoice model during init: {str(e)} - will retry on first use")
    
    def _detect_device(self):
        """Detect the best available device for inference"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_path: str = None) -> bool:
        """Load OpenVoice model - Commercial Grade Setup"""
        try:
            logger.info(f"🎙️ Loading OpenVoice model...")
            start_time = time.time()
            
            # Import OpenVoice components
            try:
                from openvoice import se_extractor
                from openvoice.api import BaseSpeakerTTS, ToneColorConverter
                logger.info("✅ OpenVoice modules imported successfully")
                
            except ImportError as e:
                logger.error(f"❌ OpenVoice not installed: {str(e)}")
                return False
            
            # Model paths - using official structure
            model_dir = "/workspace/voice-clone-backend/models/openvoice"
            base_config = f"{model_dir}/checkpoints/base_speakers/EN/config.json"
            base_ckpt = f"{model_dir}/checkpoints/base_speakers/EN/checkpoint.pth"
            converter_config = f"{model_dir}/checkpoints/converter/config.json"
            converter_ckpt = f"{model_dir}/checkpoints/converter/checkpoint.pth"
            
            # Check if models exist and are valid (not corrupted)
            if not self._models_exist_and_valid([base_config, base_ckpt, converter_config, converter_ckpt]):
                logger.info("📥 Downloading OpenVoice models (first time only)...")
                self._download_models(model_dir)
            else:
                logger.info("✅ OpenVoice models already exist, skipping download")
            
            # Load Base Speaker TTS Model
            try:
                self.openvoice_model = BaseSpeakerTTS(base_config, device=str(self.device))
                self.openvoice_model.load_ckpt(base_ckpt)
                logger.info("✅ OpenVoice Base TTS model loaded")
                
            except Exception as e:
                logger.error(f"❌ Failed to load base model: {e}")
                return False
            
            # Load Tone Color Converter
            try:
                self.tone_color_converter = ToneColorConverter(converter_config, device=str(self.device))
                self.tone_color_converter.load_ckpt(converter_ckpt)
                logger.info("✅ OpenVoice Tone Color Converter loaded")
                
            except Exception as e:
                logger.error(f"❌ Failed to load converter: {e}")
                return False
            
            # Enable torch.compile if requested
            if settings.OPENVOICE_COMPILE:
                try:
                    self.openvoice_model = torch.compile(self.openvoice_model)
                    self.tone_color_converter = torch.compile(self.tone_color_converter)
                    logger.info("✅ Models compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Failed to compile: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"✅ OpenVoice loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load OpenVoice: {str(e)}")
            return False
    
    def _models_exist_and_valid(self, file_paths: List[str]) -> bool:
        """Check if all model files exist and have reasonable sizes"""
        try:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.info(f"Missing file: {file_path}")
                    return False
                
                # Check file size (avoid corrupted downloads)
                file_size = os.path.getsize(file_path)
                if file_path.endswith('.pth') and file_size < 1024 * 1024:  # .pth files should be > 1MB
                    logger.info(f"File too small (possibly corrupted): {file_path}")
                    return False
                elif file_path.endswith('.json') and file_size < 100:  # .json files should be > 100 bytes
                    logger.info(f"Config file too small: {file_path}")
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Error checking model files: {e}")
            return False
    
    def _download_models(self, model_dir: str):
        """Download OpenVoice models from official sources"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Download using wget/curl for reliability
            import subprocess
            
            logger.info("📥 Downloading official OpenVoice checkpoints...")
            
            # Create checkpoints structure
            os.makedirs(f"{model_dir}/checkpoints/base_speakers/EN", exist_ok=True)
            os.makedirs(f"{model_dir}/checkpoints/converter", exist_ok=True)
            
            # Download base model files
            base_urls = {
                "config.json": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/base_speakers/EN/config.json",
                "checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/base_speakers/EN/checkpoint.pth"
            }
            
            # Download converter files
            converter_urls = {
                "config.json": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter/config.json", 
                "checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter/checkpoint.pth"
            }
            
            # Download base model
            for filename, url in base_urls.items():
                output_path = f"{model_dir}/checkpoints/base_speakers/EN/{filename}"
                logger.info(f"📥 Downloading {filename}...")
                subprocess.run([
                    "wget", "-O", output_path, url, 
                    "--progress=bar", "--no-check-certificate"
                ], check=True)
            
            # Download converter model
            for filename, url in converter_urls.items():
                output_path = f"{model_dir}/checkpoints/converter/{filename}"
                logger.info(f"📥 Downloading {filename}...")
                subprocess.run([
                    "wget", "-O", output_path, url,
                    "--progress=bar", "--no-check-certificate"
                ], check=True)
            
            logger.info("✅ All OpenVoice models downloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to download models: {e}")
            raise
        
    def is_model_loaded(self) -> bool:
        """Check if OpenVoice models are loaded"""
        return self.openvoice_model is not None and self.tone_color_converter is not None
    
    def _split_text_into_chunks(self, text: str, max_length: int = None) -> List[str]:
        """Split long text into multiple chunks for voice cloning instead of truncating"""
        if max_length is None:
            max_length = settings.OPENVOICE_MAX_TEXT_LENGTH
            
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while len(remaining_text) > max_length:
            # Find the best place to split within max_length
            chunk = remaining_text[:max_length]
            
            # Find the last space to avoid cutting words
            last_space = chunk.rfind(' ')
            last_period = chunk.rfind('.')
            last_comma = chunk.rfind(',')
            last_semicolon = chunk.rfind(';')
            
            # Use the latest punctuation or space as cutoff point
            cutoff_points = [last_space, last_period, last_comma, last_semicolon]
            best_cutoff = max([p for p in cutoff_points if p > max_length * 0.7])  # At least 70% of max length
            
            if best_cutoff > 0:
                chunk_text = chunk[:best_cutoff].strip()
                remaining_text = remaining_text[best_cutoff:].strip()
            else:
                # If no good cutoff point, split at max length
                chunk_text = chunk.strip()
                remaining_text = remaining_text[max_length:].strip()
            
            if chunk_text:
                chunks.append(chunk_text)
        
        # Add the remaining text if any
        if remaining_text.strip():
            chunks.append(remaining_text.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks (original: {len(text)} chars, max per chunk: {max_length})")
        return chunks

    def _concatenate_audio_chunks(self, audio_chunks: List[np.ndarray], gap_duration: float = 0.1) -> np.ndarray:
        """Concatenate multiple audio chunks with small gaps between them"""
        if not audio_chunks:
            return np.array([])
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # Create small gap between chunks
        gap_samples = int(gap_duration * self.sample_rate)
        gap = np.zeros(gap_samples, dtype=audio_chunks[0].dtype)
        
        # Concatenate all chunks with gaps
        result = audio_chunks[0]
        for i in range(1, len(audio_chunks)):
            result = np.concatenate([result, gap, audio_chunks[i]])
        
        logger.info(f"Concatenated {len(audio_chunks)} audio chunks with {gap_duration}s gaps")
        return result

    def generate_with_retry(self, chunk_text: str, audio_prompt: Optional[str] = None, 
                           custom_args: Optional[Args] = None) -> Optional[np.ndarray]:
        """
        Generate voice cloned audio using OpenVoice with retry logic and text chunking for long texts
        """
        # Split text into chunks if it's too long
        text_chunks = self._split_text_into_chunks(chunk_text)
        if not text_chunks:
            logger.warning("Empty or invalid text provided for voice cloning")
            return None
        
        # Ensure model is loaded before processing
        if not self.is_model_loaded():
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                logger.error("Failed to load OpenVoice model")
                return None
        
        args = custom_args or self.default_args
        
        # Process each chunk separately
        audio_chunks = []
        for i, text_chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: '{text_chunk[:50]}...' ({len(text_chunk)} chars)")
            
            chunk_audio = self._generate_single_chunk(text_chunk, audio_prompt, args)
            if chunk_audio is not None:
                audio_chunks.append(chunk_audio)
            else:
                logger.error(f"Failed to generate audio for chunk {i+1}")
                return None
        
        # Concatenate all audio chunks
        if audio_chunks:
            final_audio = self._concatenate_audio_chunks(audio_chunks, gap_duration=0.1)
            logger.info(f"✅ Generated complete audio from {len(text_chunks)} chunks: '{chunk_text[:50]}...' (total: {len(chunk_text)} chars)")
            return final_audio
        else:
            logger.error("No audio chunks were generated successfully")
            return None
    
    def _generate_single_chunk(self, chunk_text: str, audio_prompt: Optional[str] = None, 
                              args: Optional[Args] = None) -> Optional[np.ndarray]:
        """Generate audio for a single text chunk with retry logic"""
        retries = 0
        
        while retries <= args.max_retries:
            try:
                # Set seed for reproducibility
                set_seed(args.seed + retries)
                
                # Process text with style controls if enabled
                processed_text = self._process_text_with_style(chunk_text, args)
                
                # Generate audio using OpenVoice
                with torch.inference_mode():
                    if args.use_autocast and self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            audio = self._generate_openvoice_audio(
                                text=processed_text,
                                reference_audio=audio_prompt,
                                args=args
                            )
                    else:
                        audio = self._generate_openvoice_audio(
                            text=processed_text,
                            reference_audio=audio_prompt,
                            args=args
                        )
                
                if audio is not None:
                    # Cleanup memory
                    self._cleanup_memory()
                    
                    logger.debug(f"✅ Generated audio chunk: '{chunk_text[:30]}...' (attempt {retries + 1})")
                    return audio
                else:
                    logger.warning(f"⚠️ Chunk generation attempt {retries + 1} returned None")
                
            except Exception as e:
                logger.warning(f"⚠️ Chunk generation attempt {retries + 1} failed: {str(e)}")
            
            retries += 1
            
            if retries <= args.max_retries:
                logger.debug(f"🔄 Retrying chunk generation (attempt {retries + 1}/{args.max_retries + 1})")
                time.sleep(1)  # Short delay before retry
        
        logger.error(f"❌ Failed to generate audio chunk after {args.max_retries + 1} attempts")
        return None
    
    def _generate_openvoice_audio(self, text: str, reference_audio: Optional[str], args: Args) -> Optional[np.ndarray]:
        """Generate audio using OpenVoice with voice cloning"""
        try:
            # Import se_extractor from OpenVoice package
            from openvoice import se_extractor
            
            # Step 1: Generate base speech using Base TTS
            logger.debug("🎯 Step 1: Generating base speech...")
            
            # Use proper OpenVoice styles (from Colab example)
            style_options = ['default', 'friendly', 'cheerful', 'sad']
            output_dir = tempfile.mkdtemp()
            src_path = os.path.join(output_dir, 'tmp.wav')
            
            # Try different styles until one works
            generation_success = False
            for style in style_options:
                try:
                    logger.debug(f"🎯 Trying style: {style}")
                    # Use proper OpenVoice TTS call (based on Colab example)
                    self.openvoice_model.tts(
                        text,
                        src_path,
                        speaker=style,
                        language='English'
                    )
                    if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                        logger.debug(f"✅ Successfully used style: {style}")
                        generation_success = True
                        break
                except Exception as e:
                    logger.debug(f"❌ Style '{style}' failed: {str(e)}")
                    # Clean up failed attempt
                    if os.path.exists(src_path):
                        os.remove(src_path)
                    continue
            
            if not generation_success:
                logger.error("❌ All style options failed")
                return None
            
            # Step 2: Clone tone color if reference audio provided
            if reference_audio and os.path.exists(reference_audio):
                logger.debug("🎨 Step 2: Cloning tone color...")
                
                # Extract speaker embedding from reference
                target_se, audio_name = se_extractor.get_se(
                    reference_audio, 
                    self.tone_color_converter, 
                    target_dir=output_dir, 
                    vad=True
                )
                
                # Apply tone color conversion
                save_path = os.path.join(output_dir, 'output.wav')
                
                # Encode source audio
                source_se = se_extractor.get_se(
                    src_path, 
                    self.tone_color_converter, 
                    target_dir=output_dir
                )[0]
                
                # Convert tone color
                self.tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=save_path,
                    message="Converting..."
                )
                
                final_audio_path = save_path
            else:
                # Use base audio without tone color conversion
                final_audio_path = src_path
            
            # Load and return audio
            if os.path.exists(final_audio_path):
                audio, sr = sf.read(final_audio_path)
                
                # Resample if needed
                if sr != self.sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                
                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Normalize audio
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.9
                
                # Remove tail silence after voice cloning
                audio = AudioUtils.remove_tail_silence(audio, self.sample_rate, threshold=0.01, min_silence_duration=0.3)
                
                # Cleanup temp files
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
                
                logger.info(f"✅ Generated audio: {len(audio)} samples at {self.sample_rate}Hz")
                return audio
            else:
                logger.error("Final audio file not found")
                return None
                
        except Exception as e:
            logger.error(f"OpenVoice generation error: {str(e)}")
            return None
    
    def _process_text_with_style(self, text: str, args: Args) -> str:
        """Process text with OpenVoice style controls"""
        processed_text = text.strip()
        
        # OpenVoice uses natural text without special markers
        # Style control is handled through model parameters
        if args.enable_emotion and args.emotion != "neutral":
            # Add emotional context naturally
            emotion_context = {
                "happy": "Speaking with joy and enthusiasm: ",
                "sad": "Speaking with sadness: ",
                "excited": "Speaking with great excitement: ",
                "calm": "Speaking calmly and peacefully: ",
                "confident": "Speaking with confidence: ",
                "nervous": "Speaking with some nervousness: "
            }
            
            context = emotion_context.get(args.emotion.lower(), "")
            if context:
                processed_text = context + processed_text
        
        return processed_text
    
    def process_segments_batch(self, segments: List[Dict], audio_id: str, 
                              custom_args: Optional[Args] = None) -> Dict[str, Any]:
        """
        Process multiple segments with OpenVoice voice cloning
        """
        # Ensure model is loaded before processing
        if not self.is_model_loaded():
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return {"success": False, "error": "Failed to load OpenVoice model"}
        
        logger.info(f"🎙️ Starting OpenVoice batch processing for {len(segments)} segments")
        batch_start = time.time()
        
        args = custom_args or self.default_args
        successful_segments = 0
        failed_segments = 0
        processing_details = []
        
        # Create output directory
        output_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}" / "cloned_segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            try:
                segment_start = time.time()
                
                # Extract segment information
                segment_index = segment.get('segment_index', i + 1)
                text = segment.get('english_text', segment.get('original_text', ''))
                reference_audio = segment.get('audio_path')
                duration = segment.get('duration', 0.0)
                speaker = segment.get('speaker', 'A')
                
                # Split text into chunks if needed
                text_chunks = self._split_text_into_chunks(text)
                if not text_chunks:
                    logger.warning(f"⚠️ Segment {segment_index}: Empty or invalid text after chunking, skipping")
                    failed_segments += 1
                    continue
                
                total_text_length = sum(len(chunk) for chunk in text_chunks)
                logger.info(f"🎯 Processing segment {segment_index}: {len(text_chunks)} chunks, total {total_text_length} chars with OpenVoice")
                
                # Generate speaker-specific seed for consistency
                speaker_seed = args.seed + hash(speaker) % 1000
                segment_args = Args(**args.__dict__)
                segment_args.seed = speaker_seed
                
                # Generate cloned audio using chunking
                cloned_audio = self.generate_with_retry(
                    chunk_text=text,  # The method will handle chunking internally
                    audio_prompt=reference_audio,
                    custom_args=segment_args
                )
                
                if cloned_audio is not None:
                    # Remove tail silence after voice cloning  
                    cloned_audio = AudioUtils.remove_tail_silence(cloned_audio, self.sample_rate, threshold=0.01, min_silence_duration=0.3)
                    
                    # Adjust audio duration to match original
                    if duration > 0:
                        target_samples = int(duration * self.sample_rate)
                        cloned_audio = self._adjust_audio_duration(cloned_audio, target_samples)
                    
                    # Save cloned segment
                    output_filename = f"cloned_segment_{segment_index:03d}.wav"
                    output_path = output_dir / output_filename
                    
                    sf.write(output_path, cloned_audio, self.sample_rate)
                    
                    # Update segment with cloned audio path
                    segment['cloned_audio_path'] = str(output_path)
                    segment['cloned_duration'] = len(cloned_audio) / self.sample_rate
                    
                    successful_segments += 1
                    
                    segment_time = time.time() - segment_start
                    logger.info(f"✅ Segment {segment_index} completed in {segment_time:.2f}s")
                    
                    processing_details.append({
                        "segment_index": segment_index,
                        "text": text[:100],
                        "original_text_length": len(text),
                        "chunks_count": len(text_chunks),
                        "total_chunks_length": total_text_length,
                        "speaker": speaker,
                        "duration": duration,
                        "processing_time": segment_time,
                        "status": "success"
                    })
                    
                else:
                    logger.error(f"❌ Segment {segment_index}: OpenVoice generation failed")
                    failed_segments += 1
                    
                    processing_details.append({
                        "segment_index": segment_index,
                        "text": text[:100],
                        "original_text_length": len(text),
                        "chunks_count": len(text_chunks),
                        "total_chunks_length": total_text_length,
                        "speaker": speaker,
                        "status": "failed",
                        "error": "Generation failed"
                    })
                
                # Periodic memory cleanup
                if i % settings.OPENVOICE_MEMORY_CLEANUP_FREQUENCY == 0:
                    self._cleanup_memory()
                
            except Exception as e:
                logger.error(f"❌ Error processing segment {i + 1}: {str(e)}")
                failed_segments += 1
                
                # Get text and speaker for error reporting
                text = segment.get('english_text', segment.get('original_text', ''))
                speaker = segment.get('speaker', 'A')
                text_chunks = self._split_text_into_chunks(text)
                total_text_length = sum(len(chunk) for chunk in text_chunks)
                
                processing_details.append({
                    "segment_index": segment.get('segment_index', i + 1),
                    "text": text[:100],
                    "original_text_length": len(text),
                    "chunks_count": len(text_chunks),
                    "total_chunks_length": total_text_length,
                    "speaker": speaker,
                    "status": "failed",
                    "error": str(e)
                })
        
        batch_time = time.time() - batch_start
        
        logger.info(f"🎉 OpenVoice batch processing completed in {batch_time:.2f}s")
        logger.info(f"📊 Results: {successful_segments} successful, {failed_segments} failed")
        
        return {
            "success": successful_segments > 0,
            "total_segments": len(segments),
            "successful_segments": successful_segments,
            "failed_segments": failed_segments,
            "processing_time": batch_time,
            "output_directory": str(output_dir),
            "processing_details": processing_details
        }
    
    def _adjust_audio_duration(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Adjust audio duration to match target length - improved for silence-trimmed audio"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples < target_samples:
            # Pad with silence at the end (after silence removal, we may need to add some back)
            padding = np.zeros(target_samples - current_samples, dtype=audio.dtype)
            logger.debug(f"Padding audio with {len(padding)} samples ({len(padding)/self.sample_rate:.2f}s)")
            return np.concatenate([audio, padding])
        else:
            # Audio is longer than target - trim from the end (preserve the content)
            logger.debug(f"Trimming audio from {current_samples} to {target_samples} samples")
            return audio[:target_samples]
    
    def _cleanup_memory(self):
        """Clean up GPU/CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def clear_model(self):
        """Clear OpenVoice models from memory"""
        if self.openvoice_model is not None:
            del self.openvoice_model
            self.openvoice_model = None
        if self.tone_color_converter is not None:
            del self.tone_color_converter
            self.tone_color_converter = None
        self._cleanup_memory()
        logger.info("🧹 OpenVoice models cleared from memory")
    
    def test_text_chunking(self, test_text: str) -> Dict[str, Any]:
        """Test method to demonstrate text chunking functionality"""
        logger.info(f"Testing text chunking for: '{test_text[:100]}...'")
        
        chunks = self._split_text_into_chunks(test_text)
        
        result = {
            "original_text": test_text,
            "original_length": len(test_text),
            "chunks_count": len(chunks),
            "chunks": [{"index": i+1, "text": chunk, "length": len(chunk)} for i, chunk in enumerate(chunks)],
            "total_chunks_length": sum(len(chunk) for chunk in chunks),
            "max_chunk_length": max(len(chunk) for chunk in chunks) if chunks else 0,
            "min_chunk_length": min(len(chunk) for chunk in chunks) if chunks else 0
        }
        
        logger.info(f"Chunking result: {len(chunks)} chunks, total length preserved: {result['total_chunks_length']}")
        return result

    def validate_parameters(self, **kwargs) -> Args:
        """Validate and create OpenVoice generation arguments"""
        # Use defaults from settings, override with provided kwargs
        validated_args = Args()
        
        for key, value in kwargs.items():
            if value is not None and hasattr(validated_args, key):
                setattr(validated_args, key, value)
        
        logger.info(f"📝 Validated OpenVoice parameters: temp={validated_args.temperature}, seed={validated_args.seed}")
        return validated_args

# Backward compatibility aliases
FishSpeechVoiceCloningService = OpenVoiceVoiceCloningService
CleanVoiceCloningService = OpenVoiceVoiceCloningService 