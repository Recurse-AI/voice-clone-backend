"""
Quick test script for new voice cloning implementation
"""

import requests
import time
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_video_processing():
    """Test video processing with new implementation"""
    
    # Test video URL (replace with your test video)
    video_url = "https://pub-74887e7c8b114b879fd5038c6e8af446.r2.dev/hindi_podcast.mp4"
    
    # Start processing
    print("Starting video processing...")
    response = requests.post(
        f"{BASE_URL}/process-video",
        data={
            "video_url": video_url,
            "include_instruments": True,
            "generate_subtitles": True,
            "temperature": 1.3,
            "cfg_scale": 3.0,
            "top_p": 0.95,
            "target_language": "English",
            "language_code": None,  # Auto-detect
            "speakers_expected": 2
        }
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    audio_id = result["audio_id"]
    print(f"Processing started with audio_id: {audio_id}")
    
    # Check status
    while True:
        status_response = requests.get(f"{BASE_URL}/status/{audio_id}")
        status = status_response.json()
        
        print(f"\nStatus: {status['status']}")
        print(f"Progress: {status.get('progress', 0)}%")
        print(f"Message: {status['message']}")
        
        if status['status'] in ['completed', 'failed']:
            if status['status'] == 'completed':
                print("\nProcessing completed successfully!")
                print(json.dumps(status.get('details', {}), indent=2))
            else:
                print(f"\nProcessing failed: {status.get('details', {}).get('error', 'Unknown error')}")
            break
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    test_video_processing() 