@echo off
echo Installing dependencies...

REM Navigate to project root
cd /d "%~dp0..\.."

if exist "venv" (
    echo Using existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    REM Check Global Python version before creating venv
    python -c "import sys; v=sys.version_info; exit(1) if v.major==3 and v.minor>=13 else exit(0)"
    if errorlevel 1 (
        echo.
        echo [ERROR] Global Python 3.13+ detected! PyTorch does not support Python 3.13/3.14 yet.
        echo Please directly run 'scripts\windows\reinstall_venv.bat' to force Python 3.10 usage.
        echo.
        pause
        exit /b 1
    )

    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Verify active python version using explicit path
venv\Scripts\python -c "import sys; v=sys.version_info; exit(1) if v.major==3 and v.minor>=13 else exit(0)"
if errorlevel 1 (
    echo.
    echo [ERROR] Virtual Environment Python is 3.13+!
    echo Please delete 'venv' folder and run 'scripts\windows\reinstall_venv.bat'.
    echo.
    venv\Scripts\python --version
    pause
    exit /b 1
)

echo Installing requirements...
venv\Scripts\python -m pip install -r requirements.txt

echo.
echo Downloading V3 Models (SD 1.5, ControlNet, IP-Adapter)...
python scripts/download_models_v3.py

echo.
echo Installation complete! Run 'scripts\windows\run.bat' to start the server.
pause
