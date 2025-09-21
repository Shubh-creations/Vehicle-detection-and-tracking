#!/usr/bin/env python3
"""
Setup script for the Advanced Vehicle Detection System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_opencv():
    """Check OpenCV installation and CUDA support."""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Check CUDA support
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            print(f"✅ CUDA support detected: {cuda_devices} devices")
        else:
            print("ℹ️ CUDA support not available (CPU processing will be used)")
        
        return True
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def check_camera():
    """Check camera availability."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected")
            cap.release()
            return True
        else:
            print("⚠️ No camera detected on index 0")
            return False
    except Exception as e:
        print(f"⚠️ Camera check failed: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)."""
    if platform.system() == "Windows":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Vehicle Detection System.lnk")
            target = os.path.join(os.getcwd(), "run_detection.py")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target}"'
            shortcut.WorkingDirectory = os.getcwd()
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("✅ Desktop shortcut created")
        except ImportError:
            print("ℹ️ Desktop shortcut creation skipped (requires pywin32)")
        except Exception as e:
            print(f"⚠️ Could not create desktop shortcut: {e}")

def run_system_test():
    """Run a quick system test."""
    print("\n🧪 Running system test...")
    try:
        from advanced_vehicle_detector import AdvancedVehicleDetector
        
        # Test initialization
        detector = AdvancedVehicleDetector()
        print("✅ System initialization successful")
        
        # Test configuration
        print(f"📊 Configuration loaded: {len(detector.config)} settings")
        
        return True
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚗 Advanced Vehicle Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Check OpenCV
    if not check_opencv():
        return 1
    
    # Check camera
    check_camera()
    
    # Run system test
    if not run_system_test():
        return 1
    
    # Create desktop shortcut
    create_desktop_shortcut()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Quick Start:")
    print("   python run_detection.py")
    print("   python advanced_vehicle_detector.py --gpu")
    print("\n📚 For more options, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
