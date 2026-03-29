param(
    [string]$Model,
    [string]$Data,
    [string]$Split,
    [float]$Confidence,
    [int]$ImageSize,
    [string]$Classes,
    [string]$OutputRoot,
    [string]$RunName
)

$ErrorActionPreference = "Stop"

# Default review config. Edit these values in one place for your next manual review pass.
$review = @{
    Model = "releases\latest\best.pt"
    Data = "data\training_dataset\data.yaml"
    Split = "test"
    Confidence = 0.25
    ImageSize = 1600
    Classes = ""
    OutputRoot = "data\review_predictions"
    RunName = "latest_test_review"
}

if ($PSBoundParameters.ContainsKey("Model")) { $review.Model = $Model }
if ($PSBoundParameters.ContainsKey("Data")) { $review.Data = $Data }
if ($PSBoundParameters.ContainsKey("Split")) { $review.Split = $Split }
if ($PSBoundParameters.ContainsKey("Confidence")) { $review.Confidence = $Confidence }
if ($PSBoundParameters.ContainsKey("ImageSize")) { $review.ImageSize = $ImageSize }
if ($PSBoundParameters.ContainsKey("Classes")) { $review.Classes = $Classes }
if ($PSBoundParameters.ContainsKey("OutputRoot")) { $review.OutputRoot = $OutputRoot }
if ($PSBoundParameters.ContainsKey("RunName")) { $review.RunName = $RunName }

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$reviewScript = Join-Path $repoRoot "ml\review_predictions.py"

function Resolve-RepoPath([string]$value) {
    if ([System.IO.Path]::IsPathRooted($value)) {
        return $value
    }
    return Join-Path $repoRoot $value
}

$resolvedModel = Resolve-RepoPath $review.Model
$resolvedData = Resolve-RepoPath $review.Data
$resolvedOutputRoot = Resolve-RepoPath $review.OutputRoot

if (-not (Test-Path $python)) {
    throw "Python environment not found at $python"
}

if (-not (Test-Path $resolvedModel)) {
    throw "Model not found at $resolvedModel"
}

if (-not (Test-Path $resolvedData)) {
    throw "YOLO data.yaml not found at $resolvedData"
}

Write-Host "Reviewing segmentation predictions..."
Write-Host "  model:      $resolvedModel"
Write-Host "  data:       $resolvedData"
Write-Host "  split:      $($review.Split)"
Write-Host "  conf:       $($review.Confidence)"
Write-Host "  imageSize:  $($review.ImageSize)"
Write-Host "  classes:    $($review.Classes)"
Write-Host "  outputRoot: $resolvedOutputRoot"
Write-Host "  runName:    $($review.RunName)"

$arguments = @(
    $reviewScript
    "--model", $resolvedModel
    "--data", $resolvedData
    "--split", $review.Split
    "--conf", "$($review.Confidence)"
    "--imgsz", "$($review.ImageSize)"
    "--output-root", $resolvedOutputRoot
    "--run-name", $review.RunName
)

if ($review.Classes -ne "") {
    $arguments += @("--classes", $review.Classes)
}

& $python @arguments
