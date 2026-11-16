# Creates the dedicated Python 3.9 virtual environment and installs the project dependencies INSIDE it only.
# Usage (PowerShell, from the DSIN-modern folder):  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
$ErrorActionPreference = "Stop"
$Python39 = "C:\Users\SURESH\AppData\Local\Programs\Python\Python39\python.exe"
Set-Location $PSScriptRoot

if (-not (Test-Path $Python39)) { throw "Python 3.9 not found at $Python39 - edit the path in this script." }
Write-Host "Using $(& $Python39 --version)"

$free = [math]::Round((Get-PSDrive ((Get-Location).Drive.Name)).Free / 1GB, 1)
$ram  = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "Free disk: $free GB   Total RAM: $ram GB"
if ($free -lt 4) { throw "Need at least ~4 GB free disk space for TensorFlow + data." }

if (-not (Test-Path ".\venv\Scripts\python.exe")) { & $Python39 -m venv venv }
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-modern.txt
# DeepCTR 0.9.4 declares no TensorFlow requirement; --no-deps keeps pip from touching the pinned TF stack.
.\venv\Scripts\python.exe -m pip install deepctr==0.9.4 --no-deps
.\venv\Scripts\python.exe verify_environment.py
