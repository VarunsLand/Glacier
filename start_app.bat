@echo off
title GlacierVision - AI Cryosphere Monitoring Platform
color 0b

echo =====================================================================
echo                GLACIERVISION AI RESEARCH PLATFORM
echo =====================================================================
echo [INFO] Initializing GlacierVision application...
echo [INFO] Project Directory: C:\Users\Varun's LOQ\Desktop\GlacierResearch
echo.

cd /d "C:\Users\Varun's LOQ\Desktop\GlacierResearch"

:: Detect Python launcher or executable
set "PY_CMD="
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
echo [INFO] Using Python launcher: %PY_CMD%
echo [INFO] Starting Streamlit server on http://localhost:8501 ...
echo.

:: Launch default browser after a 2-second delay in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8501"

:: Start Streamlit server
%PY_CMD% -m streamlit run app.py --server.port 8501 --server.headless false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
