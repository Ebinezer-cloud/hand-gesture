@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Jarvis Gesture Interface
echo.
echo  ===== J.A.R.V.I.S. GESTURE INTERFACE =====
echo.

rem ---------- 1. find a suitable Python (3.10 - 3.13) ----------
set "PYCMD="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined PYCMD (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD goto :nopython
echo [1/5] Python found: %PYCMD%

rem ---------- 2. virtual environment ----------
if exist ".venv\Scripts\python.exe" goto :venvok
echo [2/5] Creating virtual environment...
%PYCMD% -m venv .venv
if errorlevel 1 goto :venvfail
:venvok
set "VPY=%~dp0.venv\Scripts\python.exe"
echo [2/5] Virtual environment ready

rem ---------- 3. packages ----------
if exist ".venv\installed.ok" goto :pkgok
echo [3/5] Installing packages - this takes a few minutes the first time...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pipfail
"%VPY%" -c "import cv2, mediapipe, numpy, PIL" >nul 2>&1
if errorlevel 1 goto :pipfail
echo ok> ".venv\installed.ok"
:pkgok
echo [3/5] Packages ready

rem ---------- 4. hologram images ----------
if exist "assets\heart.png" goto :assetsok
echo [4/5] Generating hologram images...
"%VPY%" tools\generate_assets.py
if errorlevel 1 goto :assetfail
:assetsok
echo [4/5] Images ready

rem ---------- 5. hand-tracking model (checked on every start, downloaded once) ----------
:modelcheck
echo [5/5] Checking hand-tracking model - downloads about 8 MB the first time...
"%VPY%" tools\fetch_model.py
if errorlevel 1 goto :modelfail
echo [5/5] Model ready
echo.
echo  Starting... show your hand to the camera.  Press Q in the window to quit.
echo.

"%VPY%" main.py %*
if errorlevel 1 goto :runfail
goto :end

:nopython
echo.
echo  ERROR: Python 3.10 - 3.13 was not found.
echo  Install Python 3.12 from https://www.python.org/downloads/
echo  and TICK "Add python.exe to PATH" in the installer, then run this file again.
goto :end

:venvfail
echo.
echo  ERROR: could not create the virtual environment.
echo  Delete the ".venv" folder if it exists and run this file again.
goto :end

:pipfail
echo.
echo  ERROR: package installation failed. Check your internet connection and run again.
echo  If it keeps failing, install Python 3.12 and delete the ".venv" folder.
if exist ".venv\installed.ok" del ".venv\installed.ok"
goto :end

:assetfail
echo.
echo  ERROR: could not generate the images in the assets folder.
goto :end

:modelfail
echo.
echo  ------------------------------------------------------------
echo  The model could not be downloaded automatically.
echo  A browser tab and the "models" folder will open now.
echo.
echo   1. The browser downloads  hand_landmarker.task  by itself.
echo   2. Wait until it finishes, then come back to this window
echo      and press any key. The file is picked up from your
echo      Downloads folder automatically.
echo  ------------------------------------------------------------
if not exist "models" mkdir "models"
start "" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
start "" explorer "%~dp0models"
pause
goto :modelcheck

:runfail
echo.
echo  Jarvis stopped with an error - see the messages above.
echo  Tip: run   run.bat --check   to test the camera and model.

:end
echo.
pause
endlocal
