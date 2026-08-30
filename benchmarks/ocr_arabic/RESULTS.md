# Final Arabic OCR Benchmark Results

Normalized Arabic CER is the primary ranking metric; lower is better. The main
leaderboard includes only runs that completed all 177 common benchmark pages.

| Rank | Model | Pages | CER | WER | Normalized Arabic CER | GPU | Seconds/page |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | HunyuanOCR-1.5 | 177 | 0.564666 | 0.719511 | 0.391497 | NVIDIA L4 | 22.004 |
| 2 | MBZUAI/AIN-7B | 177 | 0.752905 | 0.643977 | 0.837028 | NVIDIA RTX PRO 6000 Blackwell Server Edition (95.6 GB) | 7.590 |
| 3 | Qari OCR 0.4.0 | 177 | 1.419241 | 1.638313 | 1.076260 | NVIDIA L4 | 48.601 |
| 4 | Qwen3-VL-4B-Instruct | 177 | 2.007940 | 1.529543 | 1.286567 | NVIDIA L4 | 48.088 |
| 5 | DeepSeek-OCR-2 | 177 | 1.727661 | 1.549825 | 1.486473 | NVIDIA L4 | 22.337 |
| 6 | Arabic Nougat Large | 177 | 2.391472 | 1.876031 | 2.226341 | NVIDIA L4 | 3.172 |

HunyuanOCR-1.5 ranked first on this specific 177-page distorted Arabic
benchmark by Normalized Arabic CER. This result does not establish universal
model superiority beyond this benchmark.

## Not ranked

| Model | Status | Coverage | Reason |
|---|---|---:|---|
| dots.mocr | PARTIAL | 30 / 177 | No final completion summary; excluded from the leaderboard. |
| PaddleOCR-VL-1.6 | FAILED_SMOKE | 0 / 177 full run | Failed its smoke test; a full run was not started. |

## Hardware caveat

Accuracy is comparable because all complete runs use the same 177-page
benchmark. Runtime is not a controlled cross-GPU comparison: AIN-7B ran on a
95.6 GB RTX PRO 6000 Blackwell system, while the other complete runs used an
NVIDIA L4. The timing data must not be used to claim that AIN-7B is
definitively faster than another model.
