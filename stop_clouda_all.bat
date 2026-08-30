@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT=%SCRIPT_DIR%stop_clouda_all.ps1"

if not exist "%SCRIPT%" (
  echo stop_clouda_all.ps1 was not found next to this BAT file.
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
exit /b %ERRORLEVEL%
