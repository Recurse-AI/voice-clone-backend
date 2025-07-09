#!/usr/bin/env python3
"""
Quick API Test Script
Tests basic endpoints without requiring video files
"""

import requests
import json

BASE_URL = "https://yb2q88dplgj2tv-8000.proxy.runpod.net"

def quick_test():
    """Quick test of basic endpoints"""
    print("🚀 Quick API Test")
    print("=" * 30)
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Working: {data['message']}")
            print(f"   📋 Version: {data['details']['version']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Working: {data['message']}")
            print(f"   🧠 DIA Model: {data['details']['dia_model_loaded']}")
            print(f"   ☁️ R2 Storage: {data['details']['r2_configured']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test API docs
    print("\n3. Testing API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("   ✅ OpenAPI docs accessible")
        else:
            print(f"   ❌ Docs failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Docs error: {e}")
    
    print("\n" + "=" * 30)
    print("🎯 Quick test completed!")

if __name__ == "__main__":
    quick_test() 