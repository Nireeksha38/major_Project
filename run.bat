@echo off
echo ======================================================================
echo  Starting Real-Time Crowd Density Analysis & Stampede Risk System
echo ======================================================================
echo.

if not exist "venv" (
    echo [WARNING] Virtual environment not found. Running setup.bat first...
    call setup.bat
)

call venv\Scripts\activate.bat

echo Starting Flask Surveillance Server on http://127.0.0.1:5000 ...
echo Demo Account: Username: demo | Password: Demo@123
echo.
python app.py
pause
