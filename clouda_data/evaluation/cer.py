from __future__ import annotations


def levenshtein(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, r_item in enumerate(reference, 1):
        current = [i]
        for j, h_item in enumerate(hypothesis, 1):
            current.append(
                min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + (r_item != h_item))
            )
        prev = current
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)
