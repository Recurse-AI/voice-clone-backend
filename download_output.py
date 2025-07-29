#!/usr/bin/env python3
"""
Download Latest OpenVoice Output Audio

This script finds and downloads the latest generated English audio file
from the OpenVoice processing for the user to listen to.
"""

import os
import shutil
from pathlib import Path
import glob

def find_latest_output_audio():
    """Find the most recent OpenVoice output audio file"""
    
    # Look for audio files in the temp directory
    base_dirs = [
        "./tmp/voice_cloning/",
        "/workspace/voice-clone-backend/tmp/voice_cloning/"
    ]
    
    latest_audio = None
    latest_time = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Look for final output audio files
        pattern = f"{base_dir}**/final_output_*.wav"
        audio_files = glob.glob(pattern, recursive=True)
        
        for audio_file in audio_files:
            file_time = os.path.getmtime(audio_file)
            if file_time > latest_time:
                latest_time = file_time
                latest_audio = audio_file
    
    return latest_audio

def download_audio():
    """Download the latest audio file to current directory"""
    
    print("🎙️ Looking for latest OpenVoice output...")
    
    latest_audio = find_latest_output_audio()
    
    if not latest_audio:
        print("❌ No OpenVoice output audio found!")
        print("💡 Make sure to run the voice cloning process first.")
        return False
    
    # Copy to current directory with a simple name
    output_name = "openvoice_cloned_english.wav"
    
    try:
        shutil.copy2(latest_audio, output_name)
        print(f"✅ Downloaded: {output_name}")
        print(f"📁 Source: {latest_audio}")
        
        # Get file size
        size_mb = os.path.getsize(output_name) / (1024 * 1024)
        print(f"📊 Size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to copy audio: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎵 OpenVoice Audio Downloader")
    print("=" * 40)
    
    success = download_audio()
    
    if success:
        print("\n🎉 Success! You can now listen to the cloned English audio.")
        print("🎧 File: openvoice_cloned_english.wav")
    else:
        print("\n❌ Failed to download audio file.")
        print("🔄 Try running the voice cloning process again.")