@echo off
REM Quick Installation Script for Chest X-Ray Classification
REM Run this to install all dependencies with GPU support

echo ============================================================
echo  Chest X-Ray Classification - Quick Install
echo ============================================================
echo.

REM Check if nvidia-smi exists
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: nvidia-smi not found!
    echo You might not have an NVIDIA GPU or drivers installed.
    echo.
    set /p continue="Continue with CPU-only installation? (y/n): "
    if /i not "%continue%"=="y" exit /b
    set INSTALL_TYPE=cpu
) else (
    echo Detected NVIDIA GPU
    nvidia-smi | findstr "CUDA Version"
    echo.
    echo Select installation type:
    echo   1. CUDA 12.1 ^(Recommended for RTX 30/40 series^)
    echo   2. CUDA 11.8 ^(For older GPUs^)
    echo   3. CPU Only ^(No GPU^)
    echo.
    set /p choice="Enter choice (1/2/3): "
    
    if "%choice%"=="1" set INSTALL_TYPE=cu121
    if "%choice%"=="2" set INSTALL_TYPE=cu118
    if "%choice%"=="3" set INSTALL_TYPE=cpu
)

echo.
echo ============================================================
echo  Step 1: Installing PyTorch
echo ============================================================

if "%INSTALL_TYPE%"=="cu121" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%INSTALL_TYPE%"=="cu118" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing PyTorch ^(CPU only^)...
    pip install torch torchvision
)

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 2: Installing Core Dependencies
echo ============================================================

pip install numpy pillow opencv-python scikit-learn matplotlib seaborn

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Core dependencies installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 3: Installing ML Libraries
echo ============================================================

pip install timm efficientnet-pytorch torchsummary

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ML libraries installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 4: Installing API Dependencies
echo ============================================================

pip install fastapi uvicorn python-multipart tqdm

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: API dependencies installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 5: Verifying Installation
echo ============================================================

python check_gpu.py

echo.
echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Place your dataset in: D:\project_root\chest_xray\
echo   2. Train models: python scripts\train_all.py
echo   3. Run API: python -m uvicorn api.app:app --reload
echo   4. Open frontend: start frontend\index.html
echo.
pause
