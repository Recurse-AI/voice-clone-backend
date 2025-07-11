#!/usr/bin/env python3
"""
Quick test script to check process-video API and status
"""
import requests
import time
import json

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_VIDEO_URL = "https://pub-74887e7c8b114b879fd5038c6e8af446.r2.dev/hindi_podcast.mp4"

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            health_data = response.json()
            print(f"   - Dia Model Loaded: {health_data.get('details', {}).get('dia_model_loaded', 'Unknown')}")
            print(f"   - R2 Configured: {health_data.get('details', {}).get('r2_configured', 'Unknown')}")
            print(f"   - Active Processing: {health_data.get('details', {}).get('active_processing_count', 0)}")
            return True
        else:
            print(f"❌ API Health Check: FAILED - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Check: FAILED - {str(e)}")
        return False

def test_process_video():
    """Test process-video endpoint"""
    print("\n🔄 Testing process-video endpoint...")
    
    try:
        # Start processing
        data = {
            "video_url": TEST_VIDEO_URL,
            "target_language": "English",
            "include_instruments": True,
            "generate_subtitles": True,
            "speakers_expected": 2,
            "seed": 12345
        }
        
        response = requests.post(f"{API_BASE_URL}/process-video", data=data)
        
        if response.status_code == 200:
            result = response.json()
            audio_id = result.get("audio_id")
            print(f"✅ Video Processing Started: {audio_id}")
            print(f"   - Status Check URL: {result.get('status_check_url')}")
            print(f"   - Logs URL: {result.get('logs_url')}")
            return audio_id
        else:
            print(f"❌ Video Processing Failed: Status {response.status_code}")
            print(f"   - Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Video Processing Error: {str(e)}")
        return None

def test_status_check(audio_id):
    """Test status check endpoint"""
    print(f"\n🔄 Testing status check for {audio_id}...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/status/{audio_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status")
            message = result.get("message")
            details = result.get("details", {})
            
            print(f"✅ Status Check: {status}")
            print(f"   - Message: {message}")
            
            if details:
                print(f"   - Details: {json.dumps(details, indent=2)}")
            
            return status
        else:
            print(f"❌ Status Check Failed: Status {response.status_code}")
            print(f"   - Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Status Check Error: {str(e)}")
        return None

def test_logs(audio_id):
    """Test logs endpoint"""
    print(f"\n🔄 Testing logs for {audio_id}...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/logs/{audio_id}")
        
        if response.status_code == 200:
            print("✅ Logs Retrieved Successfully")
            print("   - First 500 characters of log:")
            print(f"   {response.text[:500]}")
            return True
        else:
            print(f"❌ Logs Retrieval Failed: Status {response.status_code}")
            print(f"   - Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Logs Retrieval Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting API Test Suite...")
    
    # Test API health
    if not test_api_health():
        print("❌ API is not healthy. Exiting...")
        return
    
    # Test process-video
    audio_id = test_process_video()
    if not audio_id:
        print("❌ Could not start video processing. Exiting...")
        return
    
    # Test status check
    status = test_status_check(audio_id)
    if not status:
        print("❌ Could not get status. Exiting...")
        return
    
    # Test logs
    test_logs(audio_id)
    
    # Monitor processing for a few iterations
    print(f"\n🔄 Monitoring processing for {audio_id}...")
    for i in range(5):
        time.sleep(10)
        current_status = test_status_check(audio_id)
        
        if current_status in ["completed", "failed"]:
            print(f"✅ Processing finished with status: {current_status}")
            break
        elif current_status == "processing":
            print(f"⏳ Still processing... (Check {i+1}/5)")
        else:
            print(f"🔄 Status: {current_status}")
    
    print("\n📊 Test Summary:")
    print(f"   - Audio ID: {audio_id}")
    print(f"   - Final Status: {current_status}")
    print(f"   - Test Video URL: {TEST_VIDEO_URL}")

if __name__ == "__main__":
    main() 