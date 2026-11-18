@echo off
setlocal enabledelayedexpansion
title GlacierVision - AI Cryosphere Monitoring Platform
color 0b

echo =====================================================================
echo                GLACIERVISION AI RESEARCH PLATFORM
echo =====================================================================
echo [INFO] Initializing one-click launcher...

:: Change to directory where this script lives (portable across folders)
cd /d "%~dp0"
echo [INFO] Working Directory: %CD%

:: 1. Check for local virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment (venv)...
    call "venv\Scripts\activate.bat"
    set "PY_CMD=python"
    goto :START_RUN
)

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
    set "PY_CMD=python"
    goto :START_RUN
)

:: 2. Check for py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
    goto :START_RUN
)

:: 3. Check for standard python executable
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto :START_RUN
)

echo [ERROR] No Python installation found in system PATH.
echo Please install Python 3.10+ from https://www.python.org/
echo.
pause
exit /b 1

:START_RUN
echo [INFO] Using Python: %PY_CMD%
echo.

:: Run the cross-platform run.py script
%PY_CMD% run.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] GlacierVision terminated with exit code %ERRORLEVEL%.
    pause
)
