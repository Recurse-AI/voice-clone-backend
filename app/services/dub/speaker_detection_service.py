import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from typing import Dict, Any, List
from app.config.settings import settings

logger = logging.getLogger(__name__)

class SpeakerDetectionService:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.hf_token = settings.HF_TOKEN
        
    def preload_model(self):
        if self.pipeline is not None:
            return
        logger.info("🚀 Preloading speaker detection model...")
        self._initialize_pipeline()
        logger.info("✅ Speaker detection model preloaded successfully!")
        
    def _initialize_pipeline(self):
        if self.pipeline is not None:
            return
        
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for speaker detection")
            
        self.device = torch.device("cuda")
        logger.info(f"Initializing speaker detection on GPU: {torch.cuda.get_device_name(0)}")
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
        self.pipeline.to(self.device)
        
    def detect_speakers(self, audio_path: str, job_id: str = None) -> List[Dict[str, Any]]:
        try:
            self._initialize_pipeline()
            
            logger.info(f"Loading audio: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            logger.info("Running GPU speaker diarization")
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            
            diarization = self.pipeline(audio_dict)
            
            results = []
            if hasattr(diarization, 'itertracks'):
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    results.append({
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "duration": float(turn.end - turn.start)
                    })
            
            num_speakers = len(set([r["speaker"] for r in results]))
            logger.info(f"Detected {num_speakers} speakers, {len(results)} segments")
            
            return results
            
        except Exception as e:
            logger.error(f"Speaker detection failed: {e}")
            raise

speaker_detection_service = SpeakerDetectionService()