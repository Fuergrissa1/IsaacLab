# 4-hour wall-clock cap; large max_iterations so training does not stop early.
# Uses Isaac Lab launcher (isaaclab.bat -p) so isaaclab / Isaac Sim packages resolve.
param(
    [string]$IsaacLabBat = "C:\isaaclab\isaaclab.bat",
    [string]$WorkingDirectory = "C:\Users\cdlapoin\Documents\Isaac\tiltrotor_6d_lab",
    [string]$Task = "Template-Tiltrotor-6d-Hover-v0",
    [int]$NumEnvs = 256,
    [int]$MaxIterations = 10000000,
    [int]$TimeoutSeconds = 14400
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$logs = Join-Path $WorkingDirectory "logs"
$null = New-Item -ItemType Directory -Force -Path $logs
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logOut = Join-Path $logs "train_headless_4h_$stamp.log"
$logErr = Join-Path $logs "train_headless_4h_${stamp}_stderr.log"

if (-not (Test-Path $IsaacLabBat)) {
    throw "Isaac Lab launcher not found: $IsaacLabBat"
}

$p = Start-Process -FilePath $IsaacLabBat -ArgumentList @(
    "-p", "scripts/skrl/train.py",
    "--task", $Task,
    "--headless",
    "--num_envs", "$NumEnvs",
    "--max_iterations", "$MaxIterations"
) -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -NoNewWindow

Write-Host "train_pid=$($p.Id)"
Write-Host "log=$logOut"
Write-Host "stderr=$logErr"

$timeoutMs = $TimeoutSeconds * 1000
$finished = $p.WaitForExit($timeoutMs)
if (-not $finished) {
    try { $p.Kill() } catch { }
    Write-Host "Stopped after ${TimeoutSeconds}s wall-clock cap."
    exit 124
}
Write-Host "exit_code=$($p.ExitCode)"
exit $p.ExitCode
