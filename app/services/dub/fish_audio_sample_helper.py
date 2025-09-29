import io
import logging
from typing import Optional, Tuple

import soundfile as sf


def fetch_sample_audio_wav_bytes(reference_id: str, api_key: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetch first sample audio for a Fish Audio model and return as WAV bytes.

    Returns (wav_bytes, sample_text). If fetch or decode fails, returns (None, None).
    """
    try:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(f"https://api.fish.audio/model/{reference_id}", headers=headers)
            if resp.status_code != 200:
                logging.warning(f"Fish model lookup failed {resp.status_code}")
                return None, None
            data = resp.json()
            samples = data.get("samples") or []
            sample = samples[0] if samples else None
            sample_url = sample.get("audio") if isinstance(sample, dict) else None
            sample_text = sample.get("text") if isinstance(sample, dict) else None
            if not sample_url:
                return None, None

            audio_resp = client.get(sample_url)
            if audio_resp.status_code != 200 or not audio_resp.content:
                return None, None

            try:
                audio_arr, sr = sf.read(io.BytesIO(audio_resp.content))
                if hasattr(audio_arr, "shape") and len(audio_arr.shape) > 1:
                    audio_arr = audio_arr[:, 0]
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio_arr, sr, format="WAV")
                return wav_buf.getvalue(), sample_text
            except Exception:
                return None, None
    except Exception:
        return None, None


