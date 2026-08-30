$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $PSScriptRoot 'logs\clouda-processes.json'
if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
    Write-Output 'No Clouda process file exists; nothing to stop.'
    exit 0
}

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path

function Get-CloudaDescendantProcessIds {
    param([int]$ParentPid)

    $children = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ParentProcessId -eq $ParentPid }
    )
    foreach ($child in $children) {
        if ($child.CommandLine -and $child.CommandLine.Contains($projectRoot)) {
            [int]$child.ProcessId
            Get-CloudaDescendantProcessIds -ParentPid ([int]$child.ProcessId)
        }
    }
}

function Test-SameStartedProcess {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$RecordedStarted
    )

    try {
        $actual = ([datetime]$Process.StartTime).ToUniversalTime()
        $expected = ([datetimeoffset]::Parse($RecordedStarted)).UtcDateTime
        return ([math]::Abs(($actual - $expected).TotalSeconds) -le 2)
    } catch {
        Write-Warning "Could not verify start time for PID $($Process.Id); it was not stopped."
        return $false
    }
}

$records = @(Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json)
if ($records.Count -eq 1 -and $records[0] -is [array]) {
    $records = @($records[0])
}
foreach ($record in $records) {
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if (-not $process) {
        continue
    }
    if (-not (Test-SameStartedProcess -Process $process -RecordedStarted $record.started)) {
        Write-Warning "PID $($record.pid) was reused; it was not stopped."
        continue
    }
    foreach ($childPid in @(Get-CloudaDescendantProcessIds -ParentPid ([int]$record.pid))) {
        Stop-Process -Id $childPid -ErrorAction SilentlyContinue
        Wait-Process -Id $childPid -Timeout 15 -ErrorAction SilentlyContinue
        Write-Output "Stopped Clouda child process (PID $childPid)."
    }
    Stop-Process -Id $record.pid
    Wait-Process -Id $record.pid -Timeout 15 -ErrorAction SilentlyContinue
    Write-Output "Stopped Clouda $($record.name) (PID $($record.pid))."
}

Move-Item -LiteralPath $pidFile -Destination "$pidFile.stopped-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
