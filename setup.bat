@echo off
echo ======================================================================
echo  Real-Time Crowd Density Analysis & Stampede Risk Detection System
echo ======================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.9 - 3.11.
    pause
    exit /b 1
)

:: Create Virtual Environment
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

:: Activate Virtual Environment & Install Dependencies
echo [2/4] Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

:: Initialize Database & Create Directories
echo [3/4] Initializing Database & Folder Structure...
python -c "from app import initialize_database; initialize_database()"

:: Download/Pre-cache YOLOv8 Model
echo [4/4] Loading and caching YOLOv8 Person Detector...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo.
echo ======================================================================
echo  Setup Completed Successfully!
echo  Default Demo Account:
echo    Username: demo  (or demo@example.com)
echo    Password: Demo@123
echo.
echo  To run the system, execute: run.bat
echo ======================================================================
pause
