[CmdletBinding()]
param(
    [string]$VenvPath = ".venv-yolo11",
    [string]$PythonExecutable = "python",
    [switch]$TrustOfficialHosts,
    [string]$ProxyUrl
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvFullPath = Join-Path $repoRoot $VenvPath
$python = Join-Path $venvFullPath "Scripts\python.exe"
$ultralyticsConfig = Join-Path $repoRoot ".ultralytics"
$certificateCandidates = @(
    (Join-Path $env:USERPROFILE "miniconda3\lib\site-packages\certifi\cacert.pem"),
    (Join-Path $env:USERPROFILE "anaconda3\Lib\site-packages\certifi\cacert.pem")
)
$certificate = $certificateCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($ProxyUrl) {
    $proxyUri = [Uri]$ProxyUrl
    if ($proxyUri.Scheme -notin @("http", "https")) {
        throw "ProxyUrl must use an http or https URL, for example http://127.0.0.1:7897."
    }
    $env:HTTP_PROXY = $proxyUri.AbsoluteUri.TrimEnd('/')
    $env:HTTPS_PROXY = $proxyUri.AbsoluteUri.TrimEnd('/')
    $env:NO_PROXY = "127.0.0.1,localhost"
    Write-Host "Using proxy for this installation only: $($proxyUri.AbsoluteUri.TrimEnd('/'))"
}

New-Item -ItemType Directory -Force -Path $ultralyticsConfig | Out-Null
$env:YOLO_CONFIG_DIR = $ultralyticsConfig

# Fail before creating a virtual environment if PATH resolves to an incompatible
# interpreter (for example, an older Conda Python with a broken Windows cert store).
& $PythonExecutable -c "import ssl, sys; assert sys.version_info >= (3, 9), sys.version; ssl.create_default_context(); print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "The selected interpreter cannot create a usable TLS context or is older than Python 3.9. Pass -PythonExecutable with a supported interpreter path."
}

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment at $venvFullPath using $PythonExecutable"
    & $PythonExecutable -m venv $venvFullPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment. Pass -PythonExecutable with a Python 3.9+ interpreter path."
    }
}

& $python -c "import ssl, sys; assert sys.version_info >= (3, 9), sys.version; ssl.create_default_context(); print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "The existing virtual environment is not usable. Remove $venvFullPath and recreate it with a Python 3.9+ interpreter that can load the Windows certificate store."
}

if ($certificate) {
    $env:SSL_CERT_FILE = $certificate
    $env:REQUESTS_CA_BUNDLE = $certificate
    Write-Host "Using CA bundle: $certificate"
}

# --isolated prevents user-level mirrors and options from changing this pinned baseline.
$pipPrefix = @("--isolated")
if ($TrustOfficialHosts) {
    Write-Warning "TLS certificate verification is bypassed only for the official PyPI and PyTorch package hosts."
    $pipPrefix += @(
        "--trusted-host", "pypi.org",
        "--trusted-host", "files.pythonhosted.org",
        "--trusted-host", "download.pytorch.org"
    )
}

& $python -m pip @pipPrefix install --upgrade pip --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# CUDA 12.8 is compatible with the current driver and supports the RTX 50-series GPU.
& $python -m pip @pipPrefix install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA 12.8 installation failed." }

& $python -m pip @pipPrefix install ultralytics==8.3.5 opencv-python==4.10.0.84 --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw "Ultralytics baseline dependencies installation failed." }

& $python -c "import torch, ultralytics; assert torch.cuda.is_available(), 'CUDA is not available'; print(f'ultralytics={ultralytics.__version__}'); print(f'torch={torch.__version__}'); print(f'gpu={torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }

Write-Host "Baseline environment is ready. Activate it with: .\$VenvPath\Scripts\Activate.ps1"
