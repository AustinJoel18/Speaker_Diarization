@echo off
echo ============================================================
echo   NeMo Speaker Diarization
echo ============================================================
echo.

if not exist ".\miniconda\Scripts\activate.bat" (
    echo [ERROR] Environment not found! Please double-click bash\setup_env.bat first to install the AI.
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating environment...
call .\miniconda\Scripts\activate.bat nemo

echo.
echo [2/3] Patching NeMo to ensure compatibility...
python scripts\patch_nemo.py

echo.
echo [3/3] Running diarization...
echo.
echo Please look in your 'data' folder and type the name of the audio file you want to process.
echo (For example: toy-example.wav)
echo.
set /p audio_file="File name: "

echo.
echo ============================================================
echo   Processing data\%audio_file%...
echo   (This might take a few minutes depending on the file length)
echo ============================================================
python scripts\run_nemo_cascaded_diarization.py --audio data\%audio_file%

echo.
echo ============================================================
echo   Diarization complete! 
echo   Please check the 'outputs\nemo\pred_rttms\' folder for your results.
echo ============================================================
pause
