"""Schema versions for persisted cross-subsystem contracts."""

PAGE_IDENTITY_VERSION = 1
DATASET_IDENTITY_VERSION = 1
OCR_OBSERVATION_VERSION = 1
MODEL_OBSERVATION_VERSION = 1


class SchemaVersion(int):
    def __new__(cls, value: int) -> "SchemaVersion":
        if value < 1:
            raise ValueError("Schema versions must be positive integers.")
        return int.__new__(cls, value)
