param(
    [string]$RunName,
    [string]$ReleaseName = "latest"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runsRoot = Join-Path $repoRoot "ml\model\runs"
$releasesRoot = Join-Path $repoRoot "releases"

if (-not (Test-Path $runsRoot)) {
    throw "Runs folder not found at $runsRoot"
}

if (-not $RunName) {
    $latestRun = Get-ChildItem $runsRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestRun) {
        throw "No training runs found in $runsRoot"
    }

    $RunName = $latestRun.Name
}

$runDir = Join-Path $runsRoot $RunName
$weightsDir = Join-Path $runDir "weights"
$bestPath = Join-Path $weightsDir "best.pt"
$lastPath = Join-Path $weightsDir "last.pt"
$resultsPath = Join-Path $runDir "results.csv"
$argsPath = Join-Path $runDir "args.yaml"

if (-not (Test-Path $bestPath)) {
    throw "best.pt not found for run '$RunName' at $bestPath"
}

if (-not (Test-Path $lastPath)) {
    throw "last.pt not found for run '$RunName' at $lastPath"
}

$releaseDir = Join-Path $releasesRoot $ReleaseName
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

Copy-Item -LiteralPath $bestPath -Destination (Join-Path $releaseDir "best.pt") -Force
Copy-Item -LiteralPath $lastPath -Destination (Join-Path $releaseDir "last.pt") -Force

$argsSummary = @{}
if (Test-Path $argsPath) {
    Get-Content $argsPath | ForEach-Object {
        if ($_ -match '^\s*([^:#]+):\s*(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            $argsSummary[$key] = $value
        }
    }
}

$metricsSummary = @{}
if (Test-Path $resultsPath) {
    $rows = Import-Csv $resultsPath
    if ($rows -and $rows.Count -gt 0) {
        $finalRow = $rows[-1]
        $metricsSummary = @{
            epoch = $finalRow.epoch
            time = $finalRow.time
            precision_box = $finalRow.'metrics/precision(B)'
            recall_box = $finalRow.'metrics/recall(B)'
            map50_box = $finalRow.'metrics/mAP50(B)'
            map50_95_box = $finalRow.'metrics/mAP50-95(B)'
            precision_mask = $finalRow.'metrics/precision(M)'
            recall_mask = $finalRow.'metrics/recall(M)'
            map50_mask = $finalRow.'metrics/mAP50(M)'
            map50_95_mask = $finalRow.'metrics/mAP50-95(M)'
            val_box_loss = $finalRow.'val/box_loss'
            val_seg_loss = $finalRow.'val/seg_loss'
            val_cls_loss = $finalRow.'val/cls_loss'
            val_dfl_loss = $finalRow.'val/dfl_loss'
        }
    }
}

$metricsPath = Join-Path $releaseDir "metrics.json"
Set-Content -Path $metricsPath -Value ($metricsSummary | ConvertTo-Json -Depth 5)

$manifestPath = Join-Path $releaseDir "release.json"
$manifest = @{
    release_name = $ReleaseName
    source_run = $RunName
    published_at = (Get-Date).ToString("s")
    files = @("best.pt", "last.pt", "metrics.json")
    metrics = $metricsSummary
    training = @{
        model_name = if ($argsSummary["model"]) { Split-Path $argsSummary["model"] -Leaf } else { $null }
        data_name = if ($argsSummary["data"]) { Split-Path $argsSummary["data"] -Leaf } else { $null }
        epochs = $argsSummary["epochs"]
        batch = $argsSummary["batch"]
        imgsz = $argsSummary["imgsz"]
        optimizer = $argsSummary["optimizer"]
        patience = $argsSummary["patience"]
        seed = $argsSummary["seed"]
    }
} | ConvertTo-Json -Depth 6

Set-Content -Path $manifestPath -Value $manifest

Write-Host "Published release '$ReleaseName' from run '$RunName'"
Write-Host "  best: $(Join-Path $releaseDir 'best.pt')"
Write-Host "  last: $(Join-Path $releaseDir 'last.pt')"
Write-Host "  metrics: $metricsPath"
Write-Host "  manifest: $manifestPath"
