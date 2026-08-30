from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentIssue:
    position: int
    reference: str
    candidate: str


def first_difference(reference: str, candidate: str) -> AlignmentIssue | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return AlignmentIssue(index, left, right)
    if len(reference) != len(candidate):
        return AlignmentIssue(
            min(len(reference), len(candidate)),
            reference[min(len(reference), len(candidate)) :],
            candidate[min(len(reference), len(candidate)) :],
        )
    return None
