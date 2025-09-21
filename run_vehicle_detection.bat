@echo off
echo ========================================
echo Advanced Vehicle Detection System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "advanced_vehicle_detector.py" (
    echo ERROR: advanced_vehicle_detector.py not found
    pause
    exit /b 1
)

if not exist "cars.xml" (
    echo ERROR: cars.xml not found
    pause
    exit /b 1
)

echo Starting Vehicle Detection System...
echo.
echo Controls:
echo   Q - Quit
echo   S - Save screenshot
echo   R - Reset statistics
echo.
echo Press any key to start...
pause >nul

REM Run the detection system
python advanced_vehicle_detector.py

echo.
echo Vehicle Detection System stopped.
pause
