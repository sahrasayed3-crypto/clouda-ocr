# HARDWARE_VALIDATION_TODO

The experiment lifecycle is tested offline with synthetic metadata and the
mock trainer. The following claims require suitable controlled hardware and
real, approved training inputs, so they are deliberately not made or simulated:

- HunyuanOCR-1.5 real training and adapter compatibility
- actual GPU VRAM use and optimizer memory
- BF16/FP16 behavior and mixed-precision stability
- multi-GPU and distributed resume behavior
- gradient-checkpointing effectiveness
- real throughput, dataloader saturation, and batch-size ceilings
- CUDA out-of-memory recovery
- real checkpoint size, load time, and performance
- LoRA versus full fine-tune quality/cost
- real CER/WER improvement after training
- two-GPU scaling
- Blackwell-specific behavior

Before enabling a real adapter, run isolated smoke, interruption/resume,
determinism, checkpoint-integrity, precision, OOM, throughput, and evaluation
tests without granting the trainer access to the protected holdout.
