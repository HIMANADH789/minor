$ErrorActionPreference = "Stop"

Write-Host "=== 1. Checking Environment ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe check_env.py

Write-Host "`n=== 2. Downloading Dataset ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe data\download_ecg5000.py

Write-Host "`n=== 3. Loading and Splitting Dataset ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe loaders\ecg5000_loader.py

Write-Host "`n=== 4. Running MiniRocket ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe models\rocket_model.py

Write-Host "`n=== 5. Running InceptionTime ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe experiments\train_inceptiontime.py

Write-Host "`n=== 6. Generating Benchmark Report ===" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe results\benchmark_report.py

Write-Host "`n=== Pipeline Complete ===" -ForegroundColor Green
