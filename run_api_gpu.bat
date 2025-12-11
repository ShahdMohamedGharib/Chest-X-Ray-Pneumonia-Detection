@echo off
REM Activate the environment with GPU support
call "C:\Users\PC\Desktop\python vs\env1\Scripts\activate.bat"

REM Change to project directory
cd /d "C:\Users\PC\Desktop\python vs\Project on drive-20251210T195341Z-1-001\Project on drive"

REM Verify PyTorch CUDA
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

REM Run API
echo.
echo Starting API server with GPU support...
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

pause

