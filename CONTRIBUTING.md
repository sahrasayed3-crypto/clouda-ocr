# Contributing

Thank you for contributing to Clouda PDF.

## Development Setup

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Rules

- Do not commit secrets or runtime data.
- Do not add an OCR model without benchmark evidence and documentation.
- Do not claim AMD ROCm support before it is tested.
- Keep scanned-page OCR behind the generic engine interface until a model is selected.
- Run tests before submitting changes.
- Only contribute code, documentation, fixtures, or examples that you have the right to submit.
- By submitting a contribution, you agree that it may be published under the public repository license.
- Do not add restricted datasets, customer data, proprietary training recipes, private reference texts, model weights, LoRA/QLoRA adapters, checkpoints, private deployment configuration, or other non-public commercial components.
- Do not add secrets, API keys, credentials, private URLs, production cloud settings, or personal data.
- Do not change the licensing scope of components that are not part of the public repository.

## Contribution Attestation

Maintainers should consider adding a Developer Certificate of Origin sign-off or a contributor agreement before accepting broad external contributions, especially if future relicensing flexibility is important. This is a recommendation only; it has not been adopted by this repository yet.

## Test Command

```powershell
.\.venv311\Scripts\python.exe -m pytest
```
