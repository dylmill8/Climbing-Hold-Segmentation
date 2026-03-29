param(
    [string]$DatasetConfig,
    [string]$PreparedDataset
)

$ErrorActionPreference = "Stop"

# Default prep config. Edit these values in one place for your next dataset build.
$prep = @{
    DatasetConfig = "data\dataset_sources.json"
    PreparedDataset = "data\training_dataset"
}

if ($PSBoundParameters.ContainsKey("DatasetConfig")) { $prep.DatasetConfig = $DatasetConfig }
if ($PSBoundParameters.ContainsKey("PreparedDataset")) { $prep.PreparedDataset = $PreparedDataset }

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $repoRoot "ml\model.py"
$resolvedDatasetConfig = if ([System.IO.Path]::IsPathRooted($prep.DatasetConfig)) {
    $prep.DatasetConfig
} else {
    Join-Path $repoRoot $prep.DatasetConfig
}
$resolvedPreparedDataset = if ([System.IO.Path]::IsPathRooted($prep.PreparedDataset)) {
    $prep.PreparedDataset
} else {
    Join-Path $repoRoot $prep.PreparedDataset
}

if (-not (Test-Path $python)) {
    throw "Python environment not found at $python"
}

if (-not (Test-Path $resolvedDatasetConfig)) {
    throw "Dataset config not found at $resolvedDatasetConfig"
}

$env:YOLO_DATASET_CONFIG = $resolvedDatasetConfig
$env:YOLO_PREPARED_DATASET = $resolvedPreparedDataset
$env:YOLO_PREPARE_ONLY = "1"

Write-Host "Preparing segmentation dataset..."
Write-Host "  dataset:  $resolvedDatasetConfig"
Write-Host "  output:   $resolvedPreparedDataset"

& $python $trainer
