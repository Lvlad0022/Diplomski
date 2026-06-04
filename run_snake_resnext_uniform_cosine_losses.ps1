param(
    [int]$NumGames = 7000,
    [int]$SaveEvery = 1000,
    [int]$MaxSteps = 500,
    [int]$BoardSize = 10,
    [int]$WarmupSteps = 5000,
    [int]$HoldSteps = 5000,
    [int]$DecaySteps = 500000,
    [double]$InitialLr = 1e-4,
    [double]$MaxLr = 5e-4,
    [double]$FinalLr = 1e-6,
    [switch]$AdvancedLogging
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$losses = @("huber", "mse")

foreach ($loss in $losses) {
    Write-Host ""
    Write-Host "=== Starting resnext uniform cosine run: loss=$loss ==="

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
        "resnext_snake",
        "--model-type",
        "modular_dueling_noisy",
        "--replay-buffer",
        "uniform",
        "--no-priority",
        "--loss",
        $loss,
        "--scheduler",
        "cosine_warmup_hold",
        "--scheduler-warmup-steps",
        $WarmupSteps,
        "--scheduler-hold-steps",
        $HoldSteps,
        "--scheduler-decay-steps",
        $DecaySteps,
        "--scheduler-initial-lr",
        $InitialLr,
        "--scheduler-max-lr",
        $MaxLr,
        "--scheduler-final-lr",
        $FinalLr
    )

    if (-not $AdvancedLogging) {
        $argsList += "--no-advanced-logging"
    }

    & conda @argsList

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Run failed for loss: $loss"
        exit $LASTEXITCODE
    }

    Write-Host "=== Finished resnext uniform cosine run: loss=$loss ==="
}

Write-Host ""
Write-Host "All resnext uniform cosine loss runs finished."
