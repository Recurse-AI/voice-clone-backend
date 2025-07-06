# Voice Cloning API - Production Backend

A production-ready voice cloning API built with FastAPI, integrating Dia voice cloning model with intelligent audio processing and cloud storage.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Deployment](#deployment)
- [Examples](#examples)

## Features

- **Voice Cloning**: Advanced voice cloning using Dia 1.6B model
- **Audio Processing**: Intelligent speaker-aware audio segmentation  
- **AssemblyAI Transcription**: Professional-grade transcription with speaker diarization
- **Cloud Storage**: R2 bucket integration for scalable storage
- **Subtitle Generation**: Automatic subtitle generation in SRT format
- **Instrument Mixing**: Optional background instrument mixing
- **Standalone Deployment**: No external dependencies - completely self-contained
- **RESTful API**: Clean, documented API endpoints

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, no authentication is required. Configure authentication as needed for production use.

---

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get API status and information.

#### Response

```json
{
  "status": "active",
  "message": "Voice Cloning API is running",
  "details": {
    "version": "1.0.0",
    "features": {
      "voice_cloning": true,
      "subtitle_generation": true,
      "instrument_mixing": true,
      "r2_storage": true
    }
  }
}
```

### 2. Health Check

**GET** `/health`

Check API health and component status.

#### Response

```json
{
  "status": "healthy",
  "message": "API is healthy and ready to process requests",
  "details": {
    "dia_model_loaded": true,
    "r2_configured": true,
    "temp_dir": "/tmp/voice_cloning"
  }
}
```

### 3. Process Audio (Main Endpoint)

**POST** `/process-audio`

Process audio file with voice cloning and return final output with R2 storage details.

#### Request

**Content-Type**: `multipart/form-data`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_file` | file | ✅ | - | Main audio file to process |
| `instruments_file` | file | ❌ | - | Optional instruments file to mix |
| `seed` | integer | ❌ | random | Seed for reproducible results |
| `include_instruments` | boolean | ❌ | false | Whether to include instruments in final audio |
| `generate_subtitles` | boolean | ❌ | true | Whether to generate subtitles |
| `temperature` | float | ❌ | 1.3 | Voice cloning temperature (0.1-2.0) |
| `cfg_scale` | float | ❌ | 3.0 | CFG scale for voice cloning (1.0-5.0) |
| `top_p` | float | ❌ | 0.95 | Top-p for voice cloning (0.1-1.0) |
| `target_language` | string | ❌ | "English" | Target language for translation |

#### File Constraints

- **Supported formats**: `.wav`, `.mp3`, `.flac`, `.m4a`
- **Maximum file size**: 100MB
- **Recommended duration**: 1-300 seconds for optimal processing

#### Response

```json
{
  "success": true,
  "audio_id": "audio_20241201_123456_abcd1234",
  "message": "Audio processing completed successfully",
  "processing_details": {
    "segments_processed": 15,
    "total_duration": 120.5,
    "speakers_detected": ["speaker_A", "speaker_B"],
    "processing_stats": {
      "total_segments": 15,
      "speakers": ["A", "B"],
      "segments_by_speaker": {"A": 8, "B": 7},
      "silent_parts_count": 3,
      "transcription_source": "AssemblyAI",
      "dia_guidelines_followed": true
    }
  },
  "r2_storage": {
    "bucket": "your-bucket-name",
    "base_path": "voice-cloning/2024/12/01/audio_20241201_123456_abcd1234",
    "segments_upload": {
      "success": true,
      "files_uploaded": 25,
      "total_size_mb": 45.2
    },
    "final_audio_upload": {
      "success": true,
      "url": "https://your-bucket.r2.dev/voice-cloning/.../final_output.wav",
      "size_mb": 12.5
    },
    "subtitles_upload": {
      "success": true,
      "url": "https://your-bucket.r2.dev/voice-cloning/.../subtitles.srt",
      "size_kb": 2.1
    },
    "access_url": "https://your-bucket.r2.cloudflarestorage.com/..."
  },
  "final_audio_url": "https://your-bucket.r2.dev/.../final_output.wav",
  "subtitles_url": "https://your-bucket.r2.dev/.../subtitles.srt",
  "original_audio_details": {
    "filename": "input.wav",
    "duration": 120.5,
    "sample_rate": 44100,
    "channels": 1,
    "size_mb": 11.8
  },
  "seed_used": 42
}
```

### 4. Processing Status

**GET** `/status/{audio_id}`

Get processing status for a specific audio ID.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_id` | string | Unique audio processing ID |

#### Response

**Processing Complete:**
```json
{
  "status": "completed",
  "message": "Processing completed successfully",
  "audio_id": "audio_20241201_123456_abcd1234",
  "details": {
    "final_audio_available": true,
    "r2_storage_info": {
      "bucket": "your-bucket-name",
      "base_path": "voice-cloning/2024/12/01/audio_20241201_123456_abcd1234",
      "access_url": "https://your-bucket.r2.dev/..."
    }
  }
}
```

**Processing In Progress:**
```json
{
  "status": "processing",
  "message": "Processing in progress",
  "audio_id": "audio_20241201_123456_abcd1234"
}
```

**Error:**
```json
{
  "status": "error",
  "message": "Error description here",
  "audio_id": "audio_20241201_123456_abcd1234"
}
```

### 5. Download Files (Development)

**GET** `/download/{audio_id}/{file_type}`

Download processed files directly (for development/testing when R2 is not configured).

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_id` | string | Unique audio processing ID |
| `file_type` | string | File type to download (`final` or `subtitles`) |

#### Response

Returns the requested file as a download.

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid audio format. Supported formats: ['.wav', '.mp3', '.flac', '.m4a']"
}
```

### 404 Not Found
```json
{
  "detail": "File not found"
}
```

### 413 Payload Too Large
```json
{
  "detail": "File too large. Maximum size: 100.0MB"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Processing failed: Detailed error message"
}
```

---

## Processing Pipeline

1. **Audio Upload**: Validate format and size
2. **AssemblyAI Transcription**: Professional transcription with speaker diarization
3. **Intelligent Segmentation**: Speaker-aware audio segmentation
4. **Voice Cloning**: Generate cloned voice using Dia model with optimized parameters
5. **Audio Reconstruction**: Rebuild audio with precise timing
6. **Instrument Mixing**: Optional background instrument mixing
7. **Subtitle Generation**: Generate synchronized subtitles in SRT format
8. **Cloud Storage**: Upload all results to R2 bucket with organized structure
9. **Response**: Return comprehensive response with URLs and metadata

---

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# R2 Cloudflare Storage Configuration
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
R2_REGION=auto

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# AssemblyAI Configuration
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

# GPU Configuration
CUDA_AVAILABLE=true
```

### Advanced Configuration

Edit `config.py` for advanced settings:

```python
# File Upload Configuration
MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a"]

# Voice Cloning Defaults
DEFAULT_TEMPERATURE: float = 1.3
DEFAULT_CFG_SCALE: float = 3.0
DEFAULT_TOP_P: float = 0.95

# Server Configuration
HOST: str = "0.0.0.0"
PORT: int = 8000
```

---

## R2 Storage Structure

Files are organized in R2 bucket with the following structure:

```
voice-cloning/
├── 2024/
│   ├── 12/
│   │   ├── 01/
│   │   │   ├── audio_20241201_123456_abcd1234/
│   │   │   │   ├── segments/
│   │   │   │   │   ├── speaker_A/
│   │   │   │   │   │   ├── segments/
│   │   │   │   │   │   └── reference/
│   │   │   │   │   └── speaker_B/
│   │   │   │   │       ├── segments/
│   │   │   │   │       └── reference/
│   │   │   │   ├── metadata/
│   │   │   │   ├── silent_parts/
│   │   │   │   ├── final/
│   │   │   │   │   └── final_output.wav
│   │   │   │   ├── subtitles/
│   │   │   │   │   └── subtitles.srt
│   │   │   │   └── summary/
│   │   │   │       └── processing_summary.json
```

---

## Examples

### Example 1: Basic Voice Cloning

```bash
curl -X POST "http://localhost:8000/process-audio" \
  -F "audio_file=@input.wav" \
  -F "generate_subtitles=true" \
  -F "temperature=1.5" \
  -F "seed=42"
```

### Example 2: Voice Cloning with Instruments

```bash
curl -X POST "http://localhost:8000/process-audio" \
  -F "audio_file=@vocals.wav" \
  -F "instruments_file=@music.wav" \
  -F "include_instruments=true" \
  -F "temperature=1.3" \
  -F "cfg_scale=3.0" \
  -F "top_p=0.95"
```

### Example 3: Check Processing Status

```bash
curl -X GET "http://localhost:8000/status/audio_20241201_123456_abcd1234"
```

### Example 4: Python SDK Usage

```python
import requests

# Upload and process audio
files = {'audio_file': open('input.wav', 'rb')}
data = {
    'seed': 42,
    'temperature': 1.5,
    'generate_subtitles': True
}

response = requests.post(
    'http://localhost:8000/process-audio',
    files=files,
    data=data
)

result = response.json()
print(f"Audio ID: {result['audio_id']}")
print(f"Final Audio URL: {result['final_audio_url']}")
print(f"Subtitles URL: {result['subtitles_url']}")
```

---

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t voice-cloning-api .

# Run container
docker run -p 8000:8000 \
  -e R2_ACCESS_KEY_ID=your_key \
  -e R2_SECRET_ACCESS_KEY=your_secret \
  -e R2_BUCKET_NAME=your_bucket \
  -e R2_ENDPOINT_URL=your_endpoint \
  -e OPENAI_API_KEY=your_openai_key \
  -e ASSEMBLYAI_API_KEY=your_assemblyai_key \
  voice-cloning-api
```

### RunPod Deployment

1. Push Docker image to registry
2. Create RunPod template with GPU support
3. Set environment variables in RunPod dashboard
4. Deploy with recommended instance: RTX 4090 or better

### Production Considerations

- Configure proper authentication and rate limiting
- Set up monitoring and logging
- Use HTTPS in production
- Configure proper CORS settings
- Set up backup and disaster recovery for R2 storage

---

## Support

For issues and questions:
1. Check the logs for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure R2 bucket permissions are configured properly
4. Test with smaller audio files first

---

## License

Production Voice Cloning API - Internal Use 