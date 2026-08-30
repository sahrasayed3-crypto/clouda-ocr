# Local Setup

Prerequisite: install or expose a real Python 3.10+ interpreter on PATH.

Then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m unittest discover -s tests
```

Do not install CUDA, ROCm, OCR model weights, or large AI frameworks during foundation validation.

