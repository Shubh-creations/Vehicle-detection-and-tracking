#!/usr/bin/env python3
"""
Simple launcher script for the Advanced Vehicle Detection System
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import cv2
        import numpy
        import imutils
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False

def check_files():
    """Check if required files exist."""
    required_files = [
        "advanced_vehicle_detector.py",
        "cars.xml",
        "vehicle_detection_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Main launcher function."""
    print("🚗 Advanced Vehicle Detection System Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check files
    if not check_files():
        return 1
    
    print("\n🎯 Starting vehicle detection system...")
    print("💡 Press 'q' to quit, 's' for screenshot, 'r' to reset stats")
    print("⚡ Use --gpu flag for GPU acceleration")
    print("📹 Use --save-video to record output")
    print("-" * 50)
    
    try:
        # Run the main detection system
        subprocess.run([sys.executable, "advanced_vehicle_detector.py"] + sys.argv[1:])
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error running detection system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
