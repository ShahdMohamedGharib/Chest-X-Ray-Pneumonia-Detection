# Activate the environment with GPU support
& "C:\Users\PC\Desktop\python vs\env1\Scripts\Activate.ps1"

# Change to project directory
Set-Location "C:\Users\PC\Desktop\python vs\Project on drive-20251210T195341Z-1-001\Project on drive"

# Verify PyTorch CUDA
Write-Host "Checking PyTorch CUDA support..." -ForegroundColor Cyan
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

Write-Host "`nStarting API server with GPU support..." -ForegroundColor Green
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

