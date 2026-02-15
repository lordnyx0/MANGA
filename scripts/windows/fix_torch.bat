@echo off
echo Fixing PyTorch installation...

REM Navigate to project root
cd /d "%~dp0..\.."

if not exist "venv" (
    echo Virtual environment not found! Run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Uninstalling broken packages...
pip uninstall torch torchvision torchaudio -y

echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Verification:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

pause
