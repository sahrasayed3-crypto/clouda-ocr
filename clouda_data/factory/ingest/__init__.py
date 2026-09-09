"""Source ingestion: local text files, local images, HF dataset adapters.

Text: whole-file UTF-8 reading (System B semantics — strict decode, no
normalization, source bytes hashed and never modified).
Image: System A semantics — bytes ingested as-is, SHA-256 recorded, decoded
only for processing (re-encoded into the run's own output tree, never in place).
"""
