param(
    [int]$NumGames = 7000,
    [int]$SaveEvery = 1000,
    [int]$MaxSteps = 500,
    [int]$BoardSize = 10,
    [string]$ModelType = "modular_dueling_noisy",
    [string]$Loss = "mse",
    [switch]$AdvancedLogging
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$backbones = @("classic", "resnext_snake")

foreach ($backbone in $backbones) {
    Write-Host ""
    Write-Host "=== Starting backbone: $backbone ==="

    $argsList = @(
        "run",
        "--no-capture-output",
        "-n",
        "q_learning",
        "python",
        "-u",
        "level2\classicSnake\snake_train.py",
        "--num-games",
        $NumGames,
        "--save-every",
        $SaveEvery,
        "--max-steps",
        $MaxSteps,
        "--board-size",
        $BoardSize,
        "--backbone",
        $backbone,
        "--model-type",
        $ModelType,
        "--replay-buffer",
        "uniform",
        "--loss",
        $Loss,
        "--no-priority"
    )

    if (-not $AdvancedLogging) {
        $argsList += "--no-advanced-logging"
    }

    & conda @argsList

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Run failed for backbone: $backbone"
        exit $LASTEXITCODE
    }

    Write-Host "=== Finished backbone: $backbone ==="
}

Write-Host ""
Write-Host "All backbone runs finished."
