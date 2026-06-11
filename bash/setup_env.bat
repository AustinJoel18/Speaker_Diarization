@echo off
REM =============================================================================
REM setup_env.bat -- Complete NeMo Environment Setup for Windows
REM =============================================================================

REM Force the script to run from the root project folder
cd /d "%~dp0.."

echo ============================================================
echo   NeMo Speaker Diarization -- Windows Environment Setup
echo ============================================================
echo.

set ENV_NAME=nemo2
set PYTHON_VERSION=3.10

REM --- Step 1: Install Miniconda ---
echo [Step 1] Setting up isolated Miniconda environment...
if not exist ".\miniconda" (
    echo    Downloading Miniconda...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile 'miniconda_installer.exe'"
    echo    Installing Miniconda silently, this may take a minute...
    start /wait "" miniconda_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%cd%\miniconda
    del miniconda_installer.exe
) else (
    echo    Miniconda already installed.
)

if not exist ".\miniconda\Scripts\activate.bat" (
    echo.
    echo [ERROR] Miniconda installation failed! Please check your permissions or antivirus.
    pause
    exit /b 1
)

REM --- Step 2: Create Environment ---
echo.
echo [Step 2] Creating and activating Conda environment...
call .\miniconda\Scripts\activate.bat

REM Automatically accept Anaconda's new Terms of Service to prevent freezing
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>nul

call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to create Conda environment! Please check the errors above.
    pause
    exit /b 1
)
call conda activate %ENV_NAME%

REM --- Step 3: Install PyTorch ---
echo.
echo [Step 3] Installing PyTorch...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo    NVIDIA GPU detected -- installing CUDA PyTorch
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo    No NVIDIA GPU detected -- installing CPU-only PyTorch
    pip install torch torchaudio
)

REM --- Step 4: Install NeMo ---
echo.
echo [Step 4] Installing NVIDIA NeMo toolkit...
REM Hack to bypass the C++ compiler requirement on Windows for youtokentome
echo    Bypassing C++ compiler requirement for youtokentome...
mkdir fake_yokentome
mkdir fake_yokentome\youtokentome
echo from setuptools import setup > fake_yokentome\setup.py
echo setup(name='youtokentome', version='1.0.6', packages=['youtokentome']) >> fake_yokentome\setup.py
echo import sys > fake_yokentome\youtokentome\__init__.py
echo from unittest.mock import MagicMock >> fake_yokentome\youtokentome\__init__.py
echo sys.modules[__name__] = MagicMock() >> fake_yokentome\youtokentome\__init__.py
pip install .\fake_yokentome >nul 2>nul
rmdir /S /Q fake_yokentome

pip install nemo_toolkit[asr]==1.21.0
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to install NeMo toolkit! Please check the errors above.
    pause
    exit /b 1
)

REM --- Step 5: Install dependencies ---
echo.
echo [Step 5] Installing Python dependencies from requirements.txt...
pip install -r requirements.txt

REM --- Step 6: Install ffmpeg ---
echo.
echo [Step 6] Installing ffmpeg...
if not exist ".\miniconda\Scripts\ffmpeg.exe" (
    echo    Downloading static ffmpeg binary to avoid Conda DLL issues...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'ffmpeg.zip'"
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_extracted' -Force"
    copy /Y ffmpeg_extracted\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe .\miniconda\Scripts\ffmpeg.exe
    rmdir /S /Q ffmpeg_extracted
    del ffmpeg.zip
) else (
    echo    ffmpeg already installed.
)

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   You can now run the diarization!
echo   Simply double-click run_diarization.bat in the main folder.
echo.
pause