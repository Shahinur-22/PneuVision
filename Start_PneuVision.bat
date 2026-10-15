@echo off
title PneuVision - Pneumonia Detection System

cd /d "%~dp0"

echo ==========================================
echo          Starting PneuVision
echo ==========================================
echo.
echo Please wait while the model is loading...
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found.
    echo.
    echo Expected location:
    echo venv\Scripts\python.exe
    echo.
    pause
    exit /b
)

if not exist "models\densenet169_pneumonia.pth" (
    echo ERROR: DenseNet169 model file not found.
    echo.
    echo Expected location:
    echo models\densenet169_pneumonia.pth
    echo.
    pause
    exit /b
)

"venv\Scripts\python.exe" "app\app.py"

echo.
echo PneuVision has stopped.
pause