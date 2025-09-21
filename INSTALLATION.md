# 🛠️ Installation Guide

This guide will help you install and set up the Advanced Vehicle Detection System on different platforms.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Camera**: USB webcam or built-in camera
- **GPU** (Optional): NVIDIA GPU with CUDA support for acceleration

### Hardware Recommendations

| Hardware | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| CPU | Dual-core 2.0GHz | Quad-core 2.5GHz | 8-core 3.0GHz+ |
| RAM | 4GB | 8GB | 16GB+ |
| GPU | Integrated | GTX 1060 | RTX 3070+ |
| Storage | 2GB SSD | 10GB SSD | 50GB NVMe SSD |

## 🚀 Quick Installation

### Windows

1. **Download Python**
   ```bash
   # Download from https://python.org
   # Make sure to check "Add Python to PATH" during installation
   ```

2. **Open Command Prompt**
   ```cmd
   # Navigate to project directory
   cd "C:\Users\shubh\Downloads\vehicle detection and tracking"
   ```

3. **Run Setup**
   ```cmd
   python setup.py
   ```

4. **Start Detection**
   ```cmd
   # Double-click run_vehicle_detection.bat
   # OR
   python run_detection.py
   ```

### macOS

1. **Install Python**
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.9
   
   # OR download from https://python.org
   ```

2. **Open Terminal**
   ```bash
   cd "/path/to/vehicle detection and tracking"
   ```

3. **Run Setup**
   ```bash
   python3 setup.py
   ```

4. **Start Detection**
   ```bash
   python3 run_detection.py
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Navigate to Project**
   ```bash
   cd "/path/to/vehicle detection and tracking"
   ```

3. **Create Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Run Setup**
   ```bash
   python setup.py
   ```

5. **Start Detection**
   ```bash
   python run_detection.py
   ```

## 🔧 Manual Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify OpenCV Installation

```python
import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

### Step 3: Test Camera Access

```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera working")
    cap.release()
else:
    print("❌ Camera not accessible")
```

## 🎮 GPU Acceleration Setup (Optional)

### NVIDIA CUDA Installation

1. **Check GPU Compatibility**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow platform-specific installation instructions

3. **Install OpenCV with CUDA Support**
   ```bash
   # Uninstall existing OpenCV
   pip uninstall opencv-python opencv-contrib-python
   
   # Install OpenCV with CUDA support
   pip install opencv-python-gpu
   ```

4. **Verify CUDA Support**
   ```python
   import cv2
   print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
   ```

### Troubleshooting GPU Issues

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvcc --version

# Check OpenCV CUDA support
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

**Problem**: OpenCV compilation errors
```bash
# Use pre-compiled binaries
pip install opencv-python-gpu==4.5.5.64
```

## 📱 Edge Device Setup

### Raspberry Pi 4

1. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv
   ```

2. **Optimize Configuration**
   ```bash
   # Edit vehicle_detection_config.json
   # Use "raspberry_pi" preset settings
   ```

3. **Performance Tips**
   ```bash
   # Increase GPU memory split
   sudo raspi-config
   # Advanced Options > Memory Split > 128
   
   # Enable camera
   sudo raspi-config
   # Interface Options > Camera > Enable
   ```

### NVIDIA Jetson Nano

1. **Flash JetPack**
   - Download from [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)
   - Follow official installation guide

2. **Install OpenCV with CUDA**
   ```bash
   sudo apt install python3-opencv
   ```

3. **Optimize Performance**
   ```bash
   # Enable maximum performance mode
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

## 🔍 Verification and Testing

### Run System Test

```bash
python performance_test.py --duration 30
```

### Test Different Configurations

```bash
python performance_test.py --compare
```

### Check System Status

```bash
python -c "
from advanced_vehicle_detector import AdvancedVehicleDetector
detector = AdvancedVehicleDetector()
print('✅ System ready')
print(f'GPU enabled: {detector.gpu_enabled}')
print(f'Config loaded: {len(detector.config)} settings')
"
```

## 🚨 Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **Camera not found**
   ```bash
   # Check camera index
   python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
   ```

3. **Permission denied errors**
   ```bash
   # Linux/macOS: Add user to video group
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

4. **Low FPS performance**
   ```bash
   # Check system resources
   htop  # Linux/macOS
   taskmgr  # Windows
   ```

5. **Memory issues**
   ```bash
   # Reduce image resolution in config
   # Set resize_width to 320 or lower
   ```

### Performance Optimization

1. **CPU Optimization**
   ```bash
   # Set process priority (Windows)
   tasklist /fi "imagename eq python.exe"
   wmic process where name="python.exe" CALL setpriority "high priority"
   ```

2. **Memory Optimization**
   ```bash
   # Close unnecessary applications
   # Increase virtual memory (Windows)
   # Add swap space (Linux)
   ```

## 📞 Support

### Getting Help

1. **Check Documentation**
   - Read README.md
   - Review configuration options
   - Check troubleshooting section

2. **Run Diagnostics**
   ```bash
   python setup.py  # System check
   python performance_test.py  # Performance test
   ```

3. **Create Issue Report**
   - Include system specifications
   - Provide error messages
   - Attach configuration files

### System Information Collection

```bash
python -c "
import platform
import sys
import cv2
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
"
```

---

**Need help? Check the troubleshooting section or create an issue on GitHub.**
