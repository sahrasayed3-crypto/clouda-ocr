$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$python = Join-Path $root '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = 'python'
}

$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$poppler = Join-Path $root 'tools\poppler\Library\bin'
if (Test-Path -LiteralPath $poppler -PathType Container) {
    $env:PATH = "$poppler;$env:PATH"
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CLOUDA_PROJECT_ROOT = if ($env:CLOUDA_PROJECT_ROOT) { $env:CLOUDA_PROJECT_ROOT } else { $root }
$env:CLOUDA_STATE_HOME = if ($env:CLOUDA_STATE_HOME) { $env:CLOUDA_STATE_HOME } else { Join-Path $root '_state' }
$env:APP_ROLE = 'server'
$env:SERVER_BASE_URL = if ($env:SERVER_BASE_URL) { $env:SERVER_BASE_URL } else { 'http://127.0.0.1:8000' }
$env:LOCAL_PROCESSING_ENABLED = if ($env:LOCAL_PROCESSING_ENABLED) { $env:LOCAL_PROCESSING_ENABLED } else { 'false' }
if (-not $env:WORKER_API_KEY) {
    $env:WORKER_API_KEY = [guid]::NewGuid().ToString('N')
}

$apiOut = Join-Path $logs 'api.out.log'
$apiErr = Join-Path $logs 'api.err.log'
$uiOut = Join-Path $logs 'streamlit.out.log'
$uiErr = Join-Path $logs 'streamlit.err.log'
$workerOut = Join-Path $logs 'worker.out.log'
$workerErr = Join-Path $logs 'worker.err.log'
$pidFile = Join-Path $logs 'clouda-processes.json'
$projectRoot = (Resolve-Path -LiteralPath $root).Path

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

function Stop-StartedCloudaProcesses {
    param([object[]]$Records)
    foreach ($record in $Records) {
        $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
        if ($process) {
            foreach ($childPid in @(Get-CloudaDescendantProcessIds -ParentPid ([int]$record.pid))) {
                Stop-Process -Id $childPid -ErrorAction SilentlyContinue
                Wait-Process -Id $childPid -Timeout 15 -ErrorAction SilentlyContinue
            }
            Stop-Process -Id $record.pid -ErrorAction SilentlyContinue
            Wait-Process -Id $record.pid -Timeout 15 -ErrorAction SilentlyContinue
        }
    }
}

function Start-CloudaProcess {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$StdOut,
        [string]$StdErr
    )
    $process = Start-Process -FilePath $python `
        -ArgumentList $Arguments `
        -WorkingDirectory $root `
        -RedirectStandardOutput $StdOut `
        -RedirectStandardError $StdErr `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Milliseconds 700
    if ($process.HasExited) {
        throw "$Name failed to start. Check $StdErr"
    }
    [pscustomobject]@{
        name = $Name
        pid = $process.Id
        started = $process.StartTime.ToUniversalTime().ToString('o')
    }
}

$records = @()
try {
    $records += Start-CloudaProcess `
        -Name 'api' `
        -Arguments @('-m', 'uvicorn', 'pdfword.worker_api:app', '--host', '127.0.0.1', '--port', '8000') `
        -StdOut $apiOut `
        -StdErr $apiErr

Start-Sleep -Seconds 2

    $records += Start-CloudaProcess `
        -Name 'ui' `
        -Arguments @('-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true', '--server.fileWatcherType', 'none') `
        -StdOut $uiOut `
        -StdErr $uiErr

    $env:APP_ROLE = 'worker'
    $records += Start-CloudaProcess `
        -Name 'worker' `
        -Arguments @('-m', 'pdfword.worker') `
        -StdOut $workerOut `
        -StdErr $workerErr
    $env:APP_ROLE = 'server'

    @($records) | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidFile -Encoding UTF8
} catch {
    $env:APP_ROLE = 'server'
    Stop-StartedCloudaProcesses -Records $records
    throw
}

Write-Host 'Clouda PDF started.'
Write-Host 'UI:  http://127.0.0.1:8501'
Write-Host 'API: http://127.0.0.1:8000/health'
