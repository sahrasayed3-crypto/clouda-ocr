from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .enums import TeacherRole

CONTRACT_VERSION = "teacher-contract-v1"
PROMPT_VERSION = "teacher-prompt-v1"
GROUNDING_RULES = (
    "Use only information visibly grounded in the authorized source image or supplied "
    "candidate transcription. Do not add, paraphrase, summarize, modernize, normalize "
    "names, guess spelling, drop unreadable text, fabricate characters, or complete "
    "sentences from model knowledge. Represent uncertainty with structured spans."
)


@dataclass(frozen=True)
class TeacherTask:
    role: TeacherRole
    task_type: str
    contract_version: str
    prompt_version: str
    prompt: str
    input_manifest: dict[str, Any]
    input_manifest_hash: str


ROLE_INSTRUCTIONS = {
    TeacherRole.PAGE_ANALYSIS: "Return structured page analysis only; do not transcribe the complete page.",
    TeacherRole.ARABIC_MAIN_OCR: "Return a literal transcription of the visible Arabic main-text region.",
    TeacherRole.MARGIN_AND_FOOTNOTE_OCR: "Return literal text with its margin or footnote region association.",
    TeacherRole.ENGLISH_OCR: "Return a literal transcription of the visible English or mixed-language region.",
    TeacherRole.IMAGE_GROUNDED_ARABIC_CORRECTION: "Correct only differences proved by the visible image.",
    TeacherRole.STRUCTURE_RECONSTRUCTION: "Return reading order and explicit relationships between supplied regions.",
    TeacherRole.QUALITY_JUDGE: "Return quality, omission, duplication, hallucination, and decision fields.",
    TeacherRole.DIFFICULT_PAGE_VISION: "Inspect difficult visible regions and preserve every uncertainty.",
    TeacherRole.FINAL_IMAGE_GROUNDED_REVIEW: "Return accept, review, or reject with localized visible issues.",
    TeacherRole.SUBJECT_CLASSIFICATION: "Return subject labels only from approved supplied samples; do not rewrite content.",
}


def build_task(
    role: TeacherRole,
    *,
    source_hash: str,
    region_hash: str = "",
    candidate_hash: str = "",
    metadata: dict[str, Any] | None = None,
) -> TeacherTask:
    if not source_hash:
        raise ValueError("source_hash_required")
    manifest = {
        "schema_version": 1,
        "role": role.value,
        "source_hash": source_hash,
        "region_hash": region_hash,
        "candidate_hash": candidate_hash,
        "metadata": metadata or {},
    }
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return TeacherTask(
        role=role,
        task_type=role.value,
        contract_version=CONTRACT_VERSION,
        prompt_version=PROMPT_VERSION,
        prompt=f"{GROUNDING_RULES}\n\n{ROLE_INSTRUCTIONS[role]}",
        input_manifest=manifest,
        input_manifest_hash=hashlib.sha256(encoded).hexdigest(),
    )


def contract_catalog() -> list[dict[str, str]]:
    return [
        {
            "teacher_role": role.value,
            "task_contract_version": CONTRACT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "instruction_hash": hashlib.sha256(
                ROLE_INSTRUCTIONS[role].encode("utf-8")
            ).hexdigest(),
        }
        for role in TeacherRole
    ]
