# AMD ROCm Roadmap

## Current Status

ROCm support is not implemented and has not been tested.

The project is architecturally prepared for future AMD-compatible OCR integration through `pdfword.engines.ExtractionEngine`. Candidate OCR/VLM models have been benchmarked on the published 177-page benchmark v0.1.0; final production selection remains open pending the expanded evaluation. No model training or adaptation has started; any future training would be pursued only if benchmark evidence shows a meaningful gap that existing open and self-hostable models do not adequately close, and would remain subject to dataset licensing and written-permission verification.

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
