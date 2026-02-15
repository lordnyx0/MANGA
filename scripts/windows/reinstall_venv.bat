@echo off
echo ===================================================
echo Reinstalling Virtual Environment with Python 3.10
echo ===================================================

REM Navigate to project root
cd /d "%~dp0..\.."

echo Killing running python processes to release file locks...
taskkill /F /IM python.exe >nul 2>&1

if exist "venv" (
    echo Deleting existing venv...
    rmdir /s /q venv
)

echo Creating new venv with Python 3.10...
py -3.10 -m venv venv

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create venv with Python 3.10.
    echo Please ensure Python 3.10 is installed and 'py' launcher is available.
    pause
    exit /b 1
)

echo.
echo Virtual environment created successfully with Python 3.10.
echo Now installing dependencies...
echo.

call scripts\windows\install.bat
