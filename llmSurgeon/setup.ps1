<#
.SYNOPSIS
  One-command setup for llmSurgeon on a fresh Windows machine.

.DESCRIPTION
  Creates the venv, installs torch (CUDA build if an NVIDIA GPU is present, CPU otherwise),
  installs dependencies, downloads the model, and runs the test suite.

  The steering vectors are already in the repo, so nothing has to be re-derived.

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -WithLargeModel      # also fetch Qwen2.5-3B (~6 GB, more accurate, barely steerable)
  .\setup.ps1 -SkipModel           # environment only
#>
param(
  [switch]$WithLargeModel,
  [switch]$SkipModel,
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root "software\.venv"
$py   = Join-Path $venv "Scripts\python.exe"
$pip  = Join-Path $venv "Scripts\pip.exe"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Good($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- python
Step "Python"
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pyCmd) { throw "python is not on PATH. Install Python 3.10+ and retry." }
$ver = (& python --version 2>&1)
Good "$ver at $($pyCmd.Source)"

# ---------------------------------------------------------------- disk
Step "Disk space"
$drive = Get-PSDrive -Name ((Get-Item $root).PSDrive.Name)
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Good "$freeGB GB free"
$needGB = 6
if ($WithLargeModel) { $needGB = 12 }
if ($freeGB -lt $needGB) {
  Warn "Less than $needGB GB free. torch is ~3 GB, the 1.5B model ~3 GB, the 3B ~6 GB."
}

# ---------------------------------------------------------------- venv
Step "Virtual environment"
if (Test-Path $py) {
  Good "already exists at software\.venv"
} else {
  & python -m venv $venv
  if (-not (Test-Path $py)) { throw "venv creation failed" }
  Good "created software\.venv"
}
& $py -m pip install --upgrade pip --quiet

# ---------------------------------------------------------------- gpu / torch
Step "PyTorch"
$hasNvidia = $false
try {
  $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
  if ($LASTEXITCODE -eq 0 -and $smi) { $hasNvidia = $true; Good "GPU: $smi" }
} catch { }

$torchInstalled = $false
try {
  $probe = & $py -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>$null
  if ($LASTEXITCODE -eq 0) { $torchInstalled = $true; Good "already installed: $probe" }
} catch { }

if (-not $torchInstalled) {
  if ($hasNvidia) {
    # IMPORTANT: on Windows, plain `pip install torch` gives the CPU-ONLY wheel.
    # cuda.is_available() would be False and everything would run ~20x slower.
    Warn "Installing the CUDA build (~3 GB). Plain pip would give a CPU-only wheel."
    & $pip install torch --index-url https://download.pytorch.org/whl/cu130
  } else {
    Warn "No NVIDIA GPU found; installing the CPU build. Use DTYPE=fp32 on CPU -- bf16 is 3.6x slower there."
    & $pip install torch
  }
}

# ---------------------------------------------------------------- deps
Step "Dependencies"
& $pip install -r (Join-Path $root "software\requirements.txt") --quiet
Good "transformers, accelerate, pyserial, treys"

$check = & $py -c "import torch,transformers,serial; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| transformers',transformers.__version__,'| pyserial',serial.__version__)"
Good $check
if ($hasNvidia -and $check -notmatch "cuda True") {
  Warn "GPU present but torch cannot see CUDA. Reinstall with:"
  Warn "  software\.venv\Scripts\pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130"
}

# ---------------------------------------------------------------- models
if (-not $SkipModel) {
  Step "Model download"
  # HF_HUB_DISABLE_XET: the xet backend throttles itself to one connection and crawls at
  # tens of KB/s. With it off, the same download runs at full speed.
  $env:HF_HUB_DISABLE_XET = "1"
  Warn "If this looks stuck, it isn't -- the 1.5B is ~3 GB."
  & $py -c @"
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
                      allow_patterns=['*.safetensors', '*.json', '*.txt'])
print('ok:', p)
"@
  Good "Qwen2.5-1.5B-Instruct ready (this is the default -- steering works on it)"

  if ($WithLargeModel) {
    & $py -c @"
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-3B-Instruct',
                      allow_patterns=['*.safetensors', '*.json', '*.txt'])
print('ok:', p)
"@
    Good "Qwen2.5-3B-Instruct ready (72% accurate but barely steerable -- see CONTINUE-HERE.md)"
  }
}

# ---------------------------------------------------------------- vectors
Step "Steering vectors"
$vecs = Get-ChildItem (Join-Path $root "software") -Filter "risk_vec_*.pt" -ErrorAction SilentlyContinue
if ($vecs) {
  foreach ($v in $vecs) { Good "$($v.Name) ($([math]::Round($v.Length/1KB)) KB)" }
  Good "Already built -- do NOT re-run build_risk_vec.py unless you change MODEL, describe(), or the layer."
} else {
  Warn "No vectors found. Rebuild with:  software\.venv\Scripts\python software\build_risk_vec.py"
}

# ---------------------------------------------------------------- tests
if (-not $SkipTests) {
  Step "Tests"
  Push-Location (Join-Path $root "software")
  try { & $py tests.py 2>&1 | Select-Object -Last 4 } finally { Pop-Location }
}

# ---------------------------------------------------------------- done
Step "Next"
Write-Host @"
  Dashboard (recommended):
    software\.venv\Scripts\python software\ui.py          then open http://127.0.0.1:8000
    software\.venv\Scripts\python software\ui.py --serial  with the Arduino attached

  Terminal demo:
    software\.venv\Scripts\python software\debate.py

  Hardware: open arduino\panel\panel.ino in the Arduino IDE, board = Arduino Mega 2560.
  Check the encoder first -- it has never been confirmed working. See CONTINUE-HERE.md.
"@
