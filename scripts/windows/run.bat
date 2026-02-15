@echo off
echo Starting MangaAutoColor Pro API...

REM Navigate to project root
cd /d "%~dp0..\.."

call venv\Scripts\activate.bat
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
