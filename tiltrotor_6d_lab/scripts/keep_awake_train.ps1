param(
    [string]$IsaacLabBat = "C:\isaaclab\isaaclab.bat",
    [string]$TrainScript = "C:\Users\cdlapoin\Documents\Isaac\tiltrotor_6d_lab\scripts\skrl\train.py",
    [string]$Task = "Template-Tiltrotor-6d-Lab-v0",
    [int]$NumEnvs = 256,
    [int]$MaxIterations = 50000,
    [string]$Checkpoint = "",
    [string]$WorkingDirectory = "C:\Users\cdlapoin\Documents\Isaac\tiltrotor_6d_lab",
    [string]$LogsRoot = "C:\Users\cdlapoin\Documents\Isaac\tiltrotor_6d_lab\logs\keep_awake_launcher"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class SleepControl {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [uint32]::Parse("80000000", [System.Globalization.NumberStyles]::HexNumber)
$ES_SYSTEM_REQUIRED = [uint32]::Parse("00000001", [System.Globalization.NumberStyles]::HexNumber)
$ES_AWAYMODE_REQUIRED = [uint32]::Parse("00000040", [System.Globalization.NumberStyles]::HexNumber)
$keepAwakeFlags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)

$null = New-Item -ItemType Directory -Force -Path $LogsRoot
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$stdoutPath = Join-Path $LogsRoot "train_stdout_$stamp.log"
$stderrPath = Join-Path $LogsRoot "train_stderr_$stamp.log"

$arguments = @(
    "-p", $TrainScript,
    "--task", $Task,
    "--num_envs", "$NumEnvs",
    "--max_iterations", "$MaxIterations",
    "--headless"
)

if ($Checkpoint -and $Checkpoint.Trim().Length -gt 0) {
    $arguments += @("--checkpoint", $Checkpoint)
}

Write-Host "Launching headless training with keep-awake protection..."
Write-Host "stdout: $stdoutPath"
Write-Host "stderr: $stderrPath"
Write-Host "checkpoint: $Checkpoint"

$process = $null

try {
    [SleepControl]::SetThreadExecutionState($keepAwakeFlags) | Out-Null

    $process = Start-Process `
        -FilePath $IsaacLabBat `
        -ArgumentList $arguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    Write-Host "child_pid: $($process.Id)"

    while (-not $process.HasExited) {
        [SleepControl]::SetThreadExecutionState($keepAwakeFlags) | Out-Null
        Start-Sleep -Seconds 30
        $process.Refresh()
    }

    Write-Host "child_exit_code: $($process.ExitCode)"
    exit $process.ExitCode
}
finally {
    [SleepControl]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
