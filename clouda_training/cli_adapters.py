"""Adapters CLI helper (Wave 2B): list / inspect / preflight subcommands.

Extracted from clouda_training.cli to keep that module import-light; all
concrete adapter package imports stay lazy so the CLI works before Wave 2A's
qwen package exists. No command here ever starts training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clouda_training.adapters.registry import (
    UnknownAdapterError,
    get_default_registry,
)

# Status constants (verified upstream audits — see UPSTREAM_COMPATIBILITY.md).
_HUNYUAN_REVISION = "c55965d3da1e"
_QWEN_REVISION = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
_QWEN_TRANSFORMERS = ">=4.57.0"

_CAPABILITY_ORDER: tuple[str, ...] = (
    "supports_full_finetune",
    "supports_selective_finetune",
    "supports_lora",
    "supports_gradient_checkpointing",
    "supports_packed_sequences",
    "supports_multimodal_batches",
    "supports_local_only_loading",
    "supports_bf16",
    "supports_fp16",
    "supports_cpu_smoke",
    "supports_resume",
    "real_weights_validated",
    "gpu_validated",
)


def _adapter_command(args: argparse.Namespace) -> int:
    if args.adapters_command == "list":
        # Importing a package triggers its self-registration (idempotent,
        # guarded by is_registered inside each package).
        _import_known_adapter_packages()
        registry = get_default_registry()
        entries: list[dict[str, Any]] = []
        for adapter_type in registry.list_adapters():
            descriptor = registry.get(adapter_type)
            entries.append(
                {
                    "adapter_type": descriptor.adapter_type,
                    "adapter_version": descriptor.adapter_version,
                    "model_family": descriptor.model_family,
                    "task_family": descriptor.task_family,
                    "capabilities": descriptor.capabilities.summary(),
                    "supported_precision": list(descriptor.supported_precision),
                    "supported_devices": list(descriptor.supported_devices),
                    "supported_data_modes": list(descriptor.supported_data_modes),
                    "required_optional_dependencies": list(
                        descriptor.required_optional_dependencies
                    ),
                }
            )
        listing = {"registry": "default-model-adapters", "adapters": entries}
        if args.json:
            print(json.dumps(listing, ensure_ascii=False, indent=2))
        else:
            print(f"Registry '{listing['registry']}': {len(entries)} adapter(s)")
            for entry in entries:
                caps = entry["capabilities"]
                enabled = [k for k in _CAPABILITY_ORDER if caps.get(k)]
                print(f"- {entry['adapter_type']} (v{entry['adapter_version']})")
                print(f"    model_family={entry['model_family']}")
                print(f"    task_family={entry['task_family']}")
                print(f"    capabilities={', '.join(enabled) if enabled else '(none)'}")
        return 0

    if args.adapters_command == "inspect":
        _import_known_adapter_packages()
        registry = get_default_registry()
        try:
            descriptor = registry.get(args.adapter_type)
        except UnknownAdapterError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 2
        payload: dict[str, object] = {
            "identity": descriptor.identity_dict(),
            "capabilities": descriptor.capabilities.summary(),
            "expected_model_symbols": list(descriptor.expected_model_symbols),
            "expected_processor_symbols": list(descriptor.expected_processor_symbols),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            identity = descriptor.identity_dict()
            capabilities = descriptor.capabilities.summary()
            print(f"Adapter '{args.adapter_type}'")
            for key, value in identity.items():
                print(f"  {key}: {value}")
            print("  capabilities:")
            for key, value in capabilities.items():
                print(f"    {key}: {value}")
        return 0

    if args.adapters_command == "preflight":
        return _run_adapter_preflight(args)

    raise AssertionError(f"Unhandled adapters command: {args.adapters_command}")


def _import_known_adapter_packages() -> None:
    """Import packages that self-register adapters (best effort, tolerant)."""
    for module_name, register in (
        (
            "clouda_training.hunyuan",
            "clouda_training.hunyuan.registration:register_hunyuan_adapters",
        ),
        (
            "clouda_training.qwen",
            "clouda_training.qwen.registration:register_qwen_adapters",
        ),
    ):
        try:
            __import__(module_name)
            module_path, func_name = register.split(":")
            register_fn = getattr(
                __import__(module_path, fromlist=[func_name]), func_name
            )
            register_fn()
        except ImportError:
            # Package not present yet (e.g. qwen before Wave 2A lands) or an
            # optional dependency missing at import time — listing stays
            # truthful about whatever did register.
            continue


def _run_adapter_preflight(args: argparse.Namespace) -> int:
    """Run the adapter's preflight hook if it has one; fail cleanly if not."""
    _import_known_adapter_packages()
    registry = get_default_registry()
    adapter_type: str = args.adapter_type
    try:
        registry.get(adapter_type)
    except UnknownAdapterError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    if adapter_type == "hunyuanocr15_sft":
        from clouda_training.hunyuan.preflight import run_preflight

        result = run_preflight(
            model_path=str(args.model_path) if args.model_path else None,
        )
    else:
        # No generic preflight contract yet (qwen ships without one in this
        # wave): fail with a clear, actionable message instead of guessing.
        message = (
            f"adapter '{adapter_type}' does not provide a preflight hook; "
            "only 'hunyuanocr15_sft' has run_preflight in this build. "
            "Validate the data path manually with 'hunyuan validate-raw' / "
            "'hunyuan validate-packed' equivalents once the adapter ships "
            "validators."
        )
        print(json.dumps({"ok": False, "error": message}, indent=2))
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Preflight for '{adapter_type}': ok={result.get('ok')}")
        for check in result.get("checks", []):
            print(
                f"  [{'PASS' if check.get('passed') else 'FAIL'}] "
                f"{check.get('name')}: {check.get('detail', '')}"
            )
    return 0 if result.get("ok") else 1


def _adapters_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``adapters`` subcommand (Wave 2B). No training entry."""
    adapters = subparsers.add_parser(
        "adapters",
        help="Model adapter registry: list/inspect/preflight (no training).",
    )
    sub = adapters.add_subparsers(dest="adapters_command", required=True)

    p_list = sub.add_parser("list", help="List registered adapters + capabilities.")
    p_list.add_argument("--json", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Show one adapter descriptor.")
    p_inspect.add_argument("adapter_type")
    p_inspect.add_argument("--json", action="store_true")

    p_pre = sub.add_parser(
        "preflight",
        help="Run the adapter's local preflight hook (no training, no downloads).",
    )
    p_pre.add_argument("adapter_type")
    p_pre.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local model directory (operators supply weights; never fetched).",
    )
    p_pre.add_argument("--json", action="store_true")
