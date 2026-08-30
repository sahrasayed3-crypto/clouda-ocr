@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%CD%\.venv311\Scripts\python.exe"
set "POPLER_BIN=%CD%\tools\poppler\Library\bin"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PYTHON%" (
  echo [ERROR] Python environment is missing: "%PYTHON%"
  exit /b 1
)

if exist "%POPLER_BIN%" (
  set "PATH=%POPLER_BIN%;%PATH%"
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\start_clouda_all.ps1"
exit /b %ERRORLEVEL%
