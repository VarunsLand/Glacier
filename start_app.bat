@echo off
title GlacierVision - AI Cryosphere Monitoring Platform
color 0b

echo =====================================================================
echo                GLACIERVISION AI RESEARCH PLATFORM
echo =====================================================================
echo [INFO] Initializing application...

REM Change directory to where this batch file is located
cd /d "%~dp0"
echo [INFO] Working Directory: %CD%

REM Check for virtual environment
set "PY_CMD="

if exist "venv\Scripts\python.exe" (
    echo [INFO] Found local virtual environment: venv
    set "PY_CMD=venv\Scripts\python.exe"
    goto :START_RUN
)

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found local virtual environment: .venv
    set "PY_CMD=.venv\Scripts\python.exe"
    goto :START_RUN
)

REM Check for py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
    goto :START_RUN
)

REM Check for standard python executable
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto :START_RUN
)

echo.
echo [ERROR] No Python installation found in system PATH.
echo Please install Python 3.10+ from https://www.python.org/
echo.
pause
exit /b 1

:START_RUN
echo [INFO] Using Python command: %PY_CMD%
echo.

REM Launch the cross-platform run.py
%PY_CMD% run.py

echo.
echo =====================================================================
echo GlacierVision has finished.
echo =====================================================================
pause
