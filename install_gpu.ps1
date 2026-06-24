# Sets up GPU training for TF 2.10 on Windows (no admin needed).
# Installs Miniforge silently, then creates a local CUDA 11.2 + cuDNN 8.1 env at .\.cuda
# whose DLLs main.py auto-loads (see src/gpu.py). Run:
#   powershell -ExecutionPolicy Bypass -File .\install_gpu.ps1
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CondaRoot   = "$env:USERPROFILE\miniforge3"
$Conda       = "$CondaRoot\Scripts\conda.exe"
$CudaEnv     = Join-Path $ProjectRoot ".cuda"
$Installer   = "$env:TEMP\Miniforge3-Windows-x86_64.exe"
$Url         = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"

if (-not (Test-Path $Conda)) {
    Write-Output "[1/3] Downloading Miniforge..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $Installer
    Write-Output "[2/3] Installing Miniforge (silent) to $CondaRoot ..."
    Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/S","/D=$CondaRoot" -Wait
    $deadline = (Get-Date).AddMinutes(5)
    while (-not (Test-Path $Conda) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
} else {
    Write-Output "[1-2/3] Miniforge already installed at $CondaRoot"
}
if (-not (Test-Path $Conda)) { throw "conda.exe not found after install: $Conda" }

Write-Output "[3/3] Creating CUDA env (cudatoolkit=11.2, cudnn=8.1) at $CudaEnv ..."
& $Conda create -y -p $CudaEnv -c conda-forge --override-channels cudatoolkit=11.2 cudnn=8.1.0
& $Conda clean -y --all | Out-Null

$Bin = Join-Path $CudaEnv "Library\bin"
Write-Output "CUDA DLLs found in ${Bin}:"
Get-ChildItem $Bin -Filter "cud*.dll" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
if (Test-Path (Join-Path $Bin "cudnn64_8.dll")) {
    Write-Output "=== CUDA SETUP OK ==="
} else {
    Write-Output "=== CUDA SETUP INCOMPLETE (see log above) ==="
}
