@echo off
title GlacierVision - AI Cryosphere Monitoring Platform
color 0b

echo =====================================================================
echo                GLACIERVISION AI RESEARCH PLATFORM
echo =====================================================================
echo [INFO] Initializing GlacierVision application...

:: Dynamically set directory to the folder containing this batch script
cd /d "%~dp0"
echo [INFO] Project Directory: %CD%
echo.

:: Detect Python launcher or executable
set "PY_CMD="
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    goto :FOUND_PY
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
    goto :FOUND_PY
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto :FOUND_PY
)

echo [ERROR] Python was not found in your system PATH.
echo Please ensure Python 3.10+ is installed and added to PATH.
echo.
pause
exit /b 1

:FOUND_PY
echo [INFO] Using Python: %PY_CMD%
echo [INFO] Starting GlacierVision via run.py...
echo.

%PY_CMD% run.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
