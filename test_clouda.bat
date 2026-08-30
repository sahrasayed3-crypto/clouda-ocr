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

"%PYTHON%" -m compileall -q app.py pdfword scripts tests tools\configure_single_machine_env.py tools\generate_test_pdfs.py tools\post_benchmark_autoreview.py tools\refresh_openrouter_models.py tools\run_full_benchmark.py tools\system_rocm_info.py
if errorlevel 1 exit /b %ERRORLEVEL%

"%PYTHON%" -m pytest tests -q -p no:cacheprovider
exit /b %ERRORLEVEL%
