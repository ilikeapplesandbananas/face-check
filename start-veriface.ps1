Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "         LAUNCHING VERIFACE AI           " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$backendDir = Join-Path $PSScriptRoot "backend"
$frontendDir = Join-Path $PSScriptRoot "frontend"
$pythonPath = Join-Path $backendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "[!] Virtual environment not found at $pythonPath" -ForegroundColor Red
    exit 1
}

Write-Host "[+] Starting FastAPI AI Backend on http://localhost:8000..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath $pythonPath -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory $backendDir -PassThru

Start-Sleep -Seconds 3

Write-Host "[+] Starting Next.js Web Frontend on http://localhost:3000..." -ForegroundColor Green
$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory $frontendDir -PassThru

Write-Host ""
Write-Host "VeriFace is online!" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C or close the terminal windows to stop."
