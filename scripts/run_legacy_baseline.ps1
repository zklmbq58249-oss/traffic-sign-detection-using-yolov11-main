[CmdletBinding()]
param(
    [string]$DataYaml,
    [string]$VenvPath = ".venv-yolo11",
    [string]$ModelPath = "model/traffic_sign_detector.pt",
    [string]$SourcePath = "data/input",
    [int]$ImageSize = 416,
    [int]$Batch = 16,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "$VenvPath\Scripts\python.exe"
$yolo = Join-Path $repoRoot "$VenvPath\Scripts\yolo.exe"
$model = Join-Path $repoRoot $ModelPath
$source = Join-Path $repoRoot $SourcePath
$project = Join-Path $repoRoot "runs\baseline_yolo11"
$metadata = Join-Path $project "legacy_reference_metadata"
$ultralyticsConfig = Join-Path $repoRoot ".ultralytics"

if ((-not (Test-Path $python)) -or (-not (Test-Path $yolo))) { throw "Baseline environment not found. Run scripts/setup_yolo11_baseline.ps1 first." }
if (-not (Test-Path $model)) { throw "Model weight not found: $model" }
if (-not (Test-Path $source)) { throw "Prediction source not found: $source" }

New-Item -ItemType Directory -Force -Path $ultralyticsConfig | Out-Null
$env:YOLO_CONFIG_DIR = $ultralyticsConfig
New-Item -ItemType Directory -Force -Path $metadata | Out-Null
& git -C $repoRoot rev-parse HEAD | Set-Content -Encoding utf8 (Join-Path $metadata "git_commit.txt")
Get-FileHash -Algorithm SHA256 $model | Format-List | Out-File -Encoding utf8 (Join-Path $metadata "model_sha256.txt")
& $python -m pip freeze | Set-Content -Encoding utf8 (Join-Path $metadata "pip_freeze.txt")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Set-Content -Encoding utf8 (Join-Path $metadata "gpu.txt")
}

& $yolo detect predict model=$model source=$source imgsz=$ImageSize device=$Device conf=0.25 save=True save_txt=True save_conf=True project=$project name=legacy_demo exist_ok=True
if ($LASTEXITCODE -ne 0) { throw "Legacy demonstration prediction failed." }

if ([string]::IsNullOrWhiteSpace($DataYaml)) {
    Write-Warning "Demo and metadata are complete. Pass -DataYaml datasets/self-driving-cars-v6/data.yaml to run dataset validation and mAP evaluation."
    exit 0
}

$data = Join-Path $repoRoot $DataYaml
if (-not (Test-Path $data)) { throw "Dataset YAML not found: $data" }
Copy-Item -Force $data (Join-Path $metadata "data.yaml")
& $python (Join-Path $repoRoot "scripts\validate_dataset.py") --data $data --output (Join-Path $metadata "dataset_report.json")
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed. Review $metadata\dataset_report.json." }

& $yolo detect val model=$model data=$data split=val imgsz=$ImageSize batch=$Batch device=$Device project=$project name=legacy_val_416 exist_ok=True plots=True
if ($LASTEXITCODE -ne 0) { throw "Legacy validation-set evaluation failed." }

& $yolo detect val model=$model data=$data split=test imgsz=$ImageSize batch=$Batch device=$Device project=$project name=legacy_test_416 exist_ok=True plots=True
if ($LASTEXITCODE -ne 0) { throw "Legacy test-set evaluation failed." }
