#!/usr/bin/env python3
"""
Simple test script to verify the system structure without requiring OpenCV
"""

import sys
import os
import json

def test_file_structure():
    """Test if all required files are present."""
    required_files = [
        "advanced_vehicle_detector.py",
        "cars.xml",
        "vehicle_detection_config.json",
        "requirements.txt",
        "README.md",
        "run_detection.py",
        "performance_test.py",
        "setup.py"
    ]
    
    print("🔍 Testing file structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required files present")
        return True

def test_configuration():
    """Test configuration file."""
    print("\n🔧 Testing configuration...")
    
    try:
        with open('vehicle_detection_config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = [
            'cascade_file', 'camera_index', 'resize_width',
            'scale_factor', 'min_neighbors', 'enable_gpu',
            'enable_tracking', 'enable_analytics'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ {key}: Missing")
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n❌ Missing configuration keys: {', '.join(missing_keys)}")
            return False
        else:
            print("\n✅ Configuration file is valid")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_imports():
    """Test Python imports (without actually importing)."""
    print("\n📦 Testing import structure...")
    
    try:
        # Read the main file and check for imports
        with open('advanced_vehicle_detector.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            'import cv2',
            'import numpy',
            'import imutils',
            'import time',
            'import json',
            'from collections import defaultdict, deque',
            'from dataclasses import dataclass'
        ]
        
        missing_imports = []
        for import_line in required_imports:
            if import_line in content:
                print(f"✅ {import_line}")
            else:
                print(f"❌ {import_line}")
                missing_imports.append(import_line)
        
        if missing_imports:
            print(f"\n❌ Missing imports: {', '.join(missing_imports)}")
            return False
        else:
            print("\n✅ All required imports found")
            return True
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_requirements():
    """Test requirements file."""
    print("\n📋 Testing requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'opencv-python',
            'numpy',
            'imutils'
        ]
        
        missing_packages = []
        for package in required_packages:
            if package in requirements:
                print(f"✅ {package}")
            else:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            return False
        else:
            print("\n✅ Requirements file is complete")
            return True
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚗 Advanced Vehicle Detection System - Structure Test")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_configuration,
        test_imports,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System structure is correct.")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the system: python run_detection.py")
        print("3. Or use the batch file: run_vehicle_detection.bat")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
