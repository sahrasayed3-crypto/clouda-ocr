from __future__ import annotations

from typing import Protocol

from clouda_contracts.ocr_observation import OCRObservation
from clouda_contracts.page_identity import PageIdentity


class ModelProvider(Protocol):
    provider_id: str

    def available(self) -> bool: ...

    def observe(
        self,
        *,
        page: PageIdentity,
        image_bytes: bytes,
    ) -> OCRObservation: ...
