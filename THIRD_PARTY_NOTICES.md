# Third-Party Notices

This project currently ships source code, documentation, small deterministic test fixtures, and public sample fixtures. It must not include proprietary datasets, model weights, paid cloud resources, private credentials, or unlicensed book scans.

## Runtime Dependencies

Dependency declarations are maintained in:

- `requirements-base.txt`
- `requirements-dev.txt`
- `requirements-linux.txt`
- `requirements-server.txt`
- `requirements-worker.txt`
- `requirements-rocm.txt`

Primary Python packages currently include Streamlit, FastAPI, Uvicorn, Requests, python-docx, pypdf, pypdfium2, Pillow, Redis/RQ, PyMuPDF, pdfplumber, pytest, ruff, black, and mypy. Their exact installed transitive dependency set is environment-specific and should be regenerated from a clean environment before a formal release.

## External Tools

- Poppler may be used as a local external tool for PDF workflows, but Poppler binaries and libraries must not be committed to the repository.
- Tesseract, PaddleOCR, Kraken, Qwen, QARI, AtlasOCR, Baseer, or other OCR engines may be candidate or baseline engines. A primary model candidate has been selected, but no third-party OCR engine is bundled, integrated as final, or claimed as supported in the current verified release.
- AMD ROCm tooling is not bundled. ROCm support must be documented only after real hardware/software validation.

## License Notes

- Project source code is licensed under the repository `LICENSE`.
- Third-party packages remain under their own licenses.
- Generated notices for a formal release should include package names, versions, license identifiers, and source URLs from the lockfile or installed environment used for that release.

## Restrictions

- Do not redistribute model weights without a confirmed redistribution license.
- Do not redistribute datasets, book scans, email permissions, or private source material unless the permission explicitly allows it.
- Do not include API keys, cloud credentials, local `.env` files, databases, logs, uploaded PDFs, generated DOCX files, or backup archives.
