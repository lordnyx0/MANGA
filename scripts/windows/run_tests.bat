@echo off
echo Running MangaAutoColor Tests...

set PYTHON_EXEC=python
if exist "venv\Scripts\python.exe" (
    echo [INFO] Using venv python...
    set PYTHON_EXEC=venv\Scripts\python.exe
)

REM Set PYTHONPATH to ensure imports work correctly
set PYTHONPATH=%CD%

REM Run Unit Tests
echo [1/3] Running Unit Tests...
%PYTHON_EXEC% -m pytest tests/unit -v -o addopts=
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Unit Tests Failed!
    exit /b 1
)

REM Run Integration Tests
echo [2/3] Running Integration Tests...
%PYTHON_EXEC% -m pytest tests/integration -v -o addopts=
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Integration Tests Failed!
    exit /b 1
)

REM Run E2E Tests
echo [3/3] Running E2E Tests...
%PYTHON_EXEC% -m pytest tests/e2e -v -o addopts=
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] E2E Tests Failed!
    exit /b 1
)

echo [SUCCESS] All tests passed!
exit /b 0
