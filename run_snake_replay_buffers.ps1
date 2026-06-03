param(
    [int]$NumGames = 3000,
    [int]$SaveEvery = 1000,
    [int]$MaxSteps = 500,
    [int]$BoardSize = 10,
    [string]$Backbone = "classic",
    [string]$ModelType = "modular_dueling_noisy",
    [double]$PriorityDecay = 0.995,
    [double]$TdMix = 0.5,
    [switch]$AdvancedLogging
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$buffers = @("td", "td_decay", "td_mix")

foreach ($buffer in $buffers) {
    Write-Host ""
    Write-Host "=== Starting replay buffer: $buffer ==="

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
        $Backbone,
        "--model-type",
        $ModelType,
        "--replay-buffer",
        $buffer,
        "--priority-decay",
        $PriorityDecay,
        "--td-mix",
        $TdMix
    )

    if (-not $AdvancedLogging) {
        $argsList += "--no-advanced-logging"
    }

    & conda @argsList

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Run failed for replay buffer: $buffer"
        exit $LASTEXITCODE
    }

    Write-Host "=== Finished replay buffer: $buffer ==="
}

Write-Host ""
Write-Host "All replay buffer runs finished."
