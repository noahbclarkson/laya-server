<#
.SYNOPSIS
    Sets up laya-server natively on Windows: a Python 3.13 venv, PyTorch, Laya,
    and the server's dependencies. Safe to re-run.

.DESCRIPTION
    With no switches, installs AMD's ROCm build of PyTorch for a Radeon GPU.
    AMD tests ROCm on Windows against AMD Software: Adrenalin Edition 26.8.1.
    Ends by running check_gpu.py, which shows the device the server will use.

.PARAMETER GfxTarget
    The Radeon's architecture, from AMD's ROCm compatibility matrix:
    gfx1102 for an RX 7600, gfx1100 for an RX 7900, gfx1201 for an RX 9070.

.PARAMETER Cpu
    Install PyTorch's CPU build instead, for a machine without a supported GPU.

.EXAMPLE
    .\setup.ps1                     # RX 7600
    .\setup.ps1 -GfxTarget gfx1100  # RX 7900 XTX
    .\setup.ps1 -Cpu
#>
param(
    [string]$GfxTarget = 'gfx1102',
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Checked {
    & $args[0] $args[1..($args.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $($args -join ' ')" }
}

if (-not (Test-Path .venv)) { Invoke-Checked py -3.13 -m venv .venv }
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

Invoke-Checked $python -m pip install --upgrade pip

# PyTorch first, from the hardware-specific index, so Laya's `torch`
# requirement is already met and pip does not replace it.
if ($Cpu) {
    Invoke-Checked $python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0'
} else {
    Invoke-Checked $python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
        "torch[device-$GfxTarget]==2.13.0+rocm10.0.0"
}

Invoke-Checked $python -m pip install -r requirements.txt

Invoke-Checked $python check_gpu.py
