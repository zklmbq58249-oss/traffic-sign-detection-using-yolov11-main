[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataYaml,
    [string]$VenvPath = ".venv-yolo11",
    [int]$ImageSize = 416,
    [int]$Batch = 16,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "$VenvPath\Scripts\python.exe"
$yolo = Join-Path $repoRoot "$VenvPath\Scripts\yolo.exe"
$data = Join-Path $repoRoot $DataYaml
$project = Join-Path $repoRoot "runs\baseline_yolo11"
$runName = "retrained_yolo11n_416_e50_b16"
$ultralyticsConfig = Join-Path $repoRoot ".ultralytics"

if ((-not (Test-Path $python)) -or (-not (Test-Path $yolo))) { throw "Baseline environment not found. Run scripts/setup_yolo11_baseline.ps1 first." }
if (-not (Test-Path $data)) { throw "Dataset YAML not found: $data" }

New-Item -ItemType Directory -Force -Path $ultralyticsConfig | Out-Null
$env:YOLO_CONFIG_DIR = $ultralyticsConfig
& $python (Join-Path $repoRoot "scripts\validate_dataset.py") --data $data --output (Join-Path $project "retrained_yolo11n_dataset_report.json")
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed. Training was not started." }

& $yolo detect train model=yolo11n.pt data=$data epochs=50 imgsz=$ImageSize batch=$Batch device=$Device seed=42 deterministic=True workers=4 project=$project name=$runName exist_ok=True
if ($LASTEXITCODE -ne 0) { throw "YOLO11 baseline training failed." }

$bestWeight = Join-Path $project "$runName\weights\best.pt"
if (-not (Test-Path $bestWeight)) { throw "Training completed without best.pt: $bestWeight" }

& $yolo detect val model=$bestWeight data=$data split=test imgsz=$ImageSize batch=$Batch device=$Device project=$project name="${runName}_test" exist_ok=True plots=True
if ($LASTEXITCODE -ne 0) { throw "Retrained YOLO11 test-set evaluation failed." }
