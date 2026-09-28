# AMD ROCm Roadmap

## Current Status

ROCm support is not implemented and has not been tested.

The project is architecturally prepared for future AMD-compatible OCR integration through `pdfword.engines.ExtractionEngine`. Candidate OCR/VLM models were benchmarked on the earlier 177-page benchmark v0.1.0 (historical) and on the completed 462-page model-selection benchmark v1.0, from which model selection was made; the current project phase is selected-model development and training. No final trained OCR model exists yet, and ROCm support remains unimplemented; training and integration work remain subject to dataset licensing and written-permission verification.

## Required Before Any ROCm Claim

1. Verify any production candidate's AMD ROCm compatibility and license terms once selection criteria are met.
2. Add dependencies to `requirements-rocm.txt` with exact tested versions.
3. Add a system diagnostic report using `tools/system_rocm_info.py`.
4. Validate CPU fallback behavior.
5. Benchmark digital PDFs, scanned PDFs, Arabic, English, and mixed text.
6. Record CER/WER, throughput, memory, VRAM, driver, HIP, ROCm, and PyTorch versions.
7. Document unsupported GPUs and operating systems.

## Acceptance Criteria

- Tests pass without GPU.
- ROCm diagnostics correctly report unavailable ROCm without crashing.
- A scanned-page OCR model runs on AMD hardware.
- Accuracy and performance are reported honestly.
- README states the exact tested hardware and software stack.
