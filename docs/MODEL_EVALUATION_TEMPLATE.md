# Model Evaluation Template

Use this template only after a candidate OCR model has been selected for integration testing.

## Model Identity

- Model name:
- Model version:
- Source/repository:
- License:
- Architecture type:
- Integration method:

## Runtime Requirements

- Required Python version:
- Required framework versions:
- Memory requirements:
- CPU requirements:
- GPU used:
- AMD GPU model:
- Driver version:
- ROCm version:
- HIP version:
- PyTorch version:

## Inference Evaluation

- Inference command or entrypoint:
- Batch size:
- Precision tested: BF16 / FP16 / FP32 / other
- Average latency per page:
- Pages per minute:
- Peak VRAM:
- CPU fallback behavior:
- Failure modes:

## Training Or Fine-Tuning Evaluation

- Training or fine-tuning supported:
- Dataset used:
- Training command or entrypoint:
- Precision tested: BF16 / FP16 / FP32 / other
- Training time:
- Peak VRAM:
- Checkpoint size:
- Resume behavior:
- Reproducibility notes:

## Accuracy

- CER:
- WER:
- Arabic old-script results:
- Arabic modern-script results:
- Medium-quality scan results:
- Low-quality scan results:
- Mixed Arabic/English text:
- Numbers:
- Tables:
- Footnotes:
- Marginalia:
- Page rotation/skew:

## Limitations And Errors

- Known limitations:
- Common errors:
- Unsupported page types:
- Unsupported languages/scripts:
- Layout issues:
- Memory issues:
- ROCm/HIP issues:
- PyTorch issues:

## Decision

- Accepted for integration:
- Reason:
- Required fixes before production:
- Documentation updates required:
