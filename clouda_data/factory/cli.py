"""Clouda Data Factory command surface.

The standalone repository exposed a separate ``clouda-data-factory``
executable. In the canonical repository the same capabilities are commands of
the unified ``clouda-data`` CLI (see ``clouda_data.pipeline.cli``):
``factory-generate``, ``factory-run``, ``factory-profiles``,
``factory-verify``, ``factory-seeds``.

This module keeps the command implementations in one place so both the
unified CLI and ``python -m clouda_data.factory`` share them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .factory import SEED_MODES, generate_run
from .profiles import ProfileError, load_profile_book
from .render import available_backends

DEFAULT_BASE_SEED = 20260831


def _default_backend() -> str:
    available = available_backends()
    return available[0] if available else "weasyprint"


def _split_profiles(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [p.strip() for p in str(value).split(",") if p.strip()]


def command_generate(args: argparse.Namespace) -> int:
    profile_names = _split_profiles(args.profiles) or ["05_old_book_medium"]
    backend = args.backend or _default_backend()
    try:
        metadata = generate_run(
            inputs=list(args.inputs),
            runs_root=args.output,
            profile_names=profile_names,
            variants=args.variants,
            base_seed=args.seed,
            seed_mode=args.seed_mode,
            backend=backend,
            workers=args.workers,
            export_pdf=not args.no_pdf,
            export_png=not args.no_png,
            max_pages=args.max_pages,
            resume=args.resume,
            run_id=args.run_id,
        )
    except (FileExistsError, FileNotFoundError, ProfileError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    from .autorun import auto_run

    try:
        summary = auto_run(
            input_path=args.input,
            output_root=args.output,
            variants=args.variants if args.variants is not None else 2,
            profile_names=_split_profiles(args.profiles),
            base_seed=args.seed,
            seed_mode=args.seed_mode,
            backend=args.backend,
            workers=args.workers,
            export_pdf=not args.no_pdf,
            export_png=not args.no_png,
            max_pages=args.max_pages,
            fresh=args.fresh,
        )
    except (FileExistsError, FileNotFoundError, ProfileError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["verification"]["passed"] else 1


def command_profiles(args: argparse.Namespace) -> int:
    book = load_profile_book()
    print(
        json.dumps(
            {
                "profiles": {
                    name: p.as_dict() for name, p in sorted(book.profiles.items())
                },
                "distortions": sorted(book.distortions),
                "backends": available_backends(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    from .manifest import read_manifest
    from .provenance.hashing import sha256_file

    manifest = args.run_dir / "manifest.jsonl"
    if not manifest.is_file():
        print(f"error: no manifest under {args.run_dir}", file=sys.stderr)
        return 2
    rows = read_manifest(manifest)
    bad: list[str] = []
    checked = 0
    for row in rows:
        rel = row.get("output_path")
        if row.get("status") == "ok" and rel:
            path = args.run_dir / rel
            checked += 1
            if not path.is_file() or sha256_file(path) != row.get("output_sha256"):
                bad.append(rel)
    print(json.dumps({"checked": checked, "mismatches": bad}, indent=2))
    return 1 if bad else 0


def command_seeds(args: argparse.Namespace) -> int:
    from .seed import derive, legacy

    source = "0" * 64
    vectors: dict[str, Any] = {
        "v1": derive.derive_seed(
            DEFAULT_BASE_SEED,
            source,
            document_id="doc",
            page_index=0,
            variant_index=0,
            profile="05_old_book_medium",
            distortion_stage="paper_degradation",
            severity="medium",
        ),
        "ocr_benchmark": legacy.derive_seed_ocr_benchmark(20260825, source),
        "arabic_scan_factory": legacy.variant_seed_arabic_scan_factory(
            source, 0, "05_old_book_medium", DEFAULT_BASE_SEED
        ),
    }
    print(json.dumps(vectors, indent=2))
    return 0


def command_version(args: argparse.Namespace) -> int:
    print(json.dumps({"factory_version": __version__}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m clouda_data.factory",
        description="Clouda Data Factory — deterministic Arabic OCR "
        "data generation (integrated Clouda OCR subsystem).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="zero-configuration one-command data factory")
    run.add_argument("input", type=Path, help="text/image/PDF file or mixed directory")
    run.add_argument(
        "output", type=Path, help="output root directory (runs are created inside)"
    )
    run.add_argument(
        "--variants", type=int, default=None, help="scan variants per page (default: 2)"
    )
    run.add_argument(
        "--profiles",
        default=None,
        help="comma-separated profile names (default: automatic sensible pair)",
    )
    run.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="base seed")
    run.add_argument("--seed-mode", choices=SEED_MODES, default="v1")
    run.add_argument(
        "--backend", default="auto", help="render backend (default: best available)"
    )
    run.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: automatic)",
    )
    run.add_argument(
        "--no-pdf", action="store_true", help="skip image-only scan PDF export"
    )
    run.add_argument("--no-png", action="store_true", help="skip distorted PNG export")
    run.add_argument("--max-pages", type=int, default=8, help="pages per text document")
    run.add_argument(
        "--fresh",
        action="store_true",
        help="force a new run instead of auto-resuming the identical run",
    )
    run.set_defaults(func=command_run)

    gen = sub.add_parser("generate", help="generate clean + synthetic scan data")
    gen.add_argument(
        "inputs", nargs="+", type=Path, help="text/image files or directories"
    )
    gen.add_argument("--output", type=Path, required=True, help="runs root directory")
    gen.add_argument(
        "--profiles",
        default="05_old_book_medium",
        help="comma-separated profile names cycled across variants",
    )
    gen.add_argument("--variants", type=int, default=1, help="scan variants per page")
    gen.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="base seed")
    gen.add_argument("--seed-mode", choices=SEED_MODES, default="v1")
    gen.add_argument(
        "--backend", default=None, help="render backend (default: first available)"
    )
    gen.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    gen.add_argument("--no-pdf", action="store_true", help="skip distorted PDF export")
    gen.add_argument("--no-png", action="store_true", help="skip distorted PNG export")
    gen.add_argument("--max-pages", type=int, default=8, help="pages per text document")
    gen.add_argument(
        "--resume", action="store_true", help="resume an existing run directory"
    )
    gen.add_argument("--run-id", default=None, help="explicit run id")
    gen.set_defaults(func=command_generate)

    sub.add_parser(
        "profiles", help="list available profiles and backends"
    ).set_defaults(func=command_profiles)

    ver = sub.add_parser("verify", help="verify hashes in a run manifest")
    ver.add_argument("run_dir", type=Path)
    ver.set_defaults(func=command_verify)

    sub.add_parser("seeds", help="show seed derivation vectors").set_defaults(
        func=command_seeds
    )
    sub.add_parser("version", help="print the factory version").set_defaults(
        func=command_version
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
