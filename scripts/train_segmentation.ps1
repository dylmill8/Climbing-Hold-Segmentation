param(
    [string]$Weights,
    [string]$RunName,
    [int]$ImageSize,
    [int]$Batch,
    [int]$Epochs,
    [int]$Patience,
    [string]$Device,
    [string]$DatasetConfig
)

$ErrorActionPreference = "Stop"

# Default run config. Edit these values in one place for your next training run.
$run = @{
    Weights = "yolo26m-seg.pt"
    RunName = "y26m_seg_resplit_1440"
    ImageSize = 1440
    Batch = 2
    Epochs = 200
    Patience = 15
    Device = "0"
    DatasetConfig = "data\dataset_sources.json"
}

if ($PSBoundParameters.ContainsKey("Weights")) { $run.Weights = $Weights }
if ($PSBoundParameters.ContainsKey("RunName")) { $run.RunName = $RunName }
if ($PSBoundParameters.ContainsKey("ImageSize")) { $run.ImageSize = $ImageSize }
if ($PSBoundParameters.ContainsKey("Batch")) { $run.Batch = $Batch }
if ($PSBoundParameters.ContainsKey("Epochs")) { $run.Epochs = $Epochs }
if ($PSBoundParameters.ContainsKey("Patience")) { $run.Patience = $Patience }
if ($PSBoundParameters.ContainsKey("Device")) { $run.Device = $Device }
if ($PSBoundParameters.ContainsKey("DatasetConfig")) { $run.DatasetConfig = $DatasetConfig }

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $repoRoot "ml\model.py"
$resolvedDatasetConfig = if ([System.IO.Path]::IsPathRooted($run.DatasetConfig)) {
    $run.DatasetConfig
} else {
    Join-Path $repoRoot $run.DatasetConfig
}

if (-not (Test-Path $python)) {
    throw "Python environment not found at $python"
}

if (-not (Test-Path $resolvedDatasetConfig)) {
    throw "Dataset config not found at $resolvedDatasetConfig"
}

$env:YOLO_DATASET_CONFIG = $resolvedDatasetConfig
$env:YOLO_WEIGHTS = $run.Weights
$env:YOLO_RUN_NAME = $run.RunName
$env:YOLO_IMGSZ = "$($run.ImageSize)"
$env:YOLO_BATCH = "$($run.Batch)"
$env:YOLO_EPOCHS = "$($run.Epochs)"
$env:YOLO_PATIENCE = "$($run.Patience)"
$env:YOLO_DEVICE = $run.Device
$env:YOLO_CACHE = "ram"
$env:YOLO_WORKERS = "6"

Write-Host "Starting segmentation training..."
Write-Host "  dataset:   $resolvedDatasetConfig"
Write-Host "  weights:   $($run.Weights)"
Write-Host "  run:       $($run.RunName)"
Write-Host "  imageSize: $($run.ImageSize)"
Write-Host "  batch:     $($run.Batch)"
Write-Host "  epochs:    $($run.Epochs)"
Write-Host "  patience:  $($run.Patience)"
Write-Host "  device:    $($run.Device)"

& $python $trainer
