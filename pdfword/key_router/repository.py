from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from ..database import Database, utc_now
from .capabilities import ModelEndpointCapability
from .credential_refs import validate_secret_ref
from .enums import (
    AccountState,
    CapabilityVerificationStatus,
    CircuitState,
    ReservationState,
    WindowType,
)
from .errors import NormalizedProviderError
from .exceptions import FencingTokenError, QuotaExceeded
from .models import (
    AccountRuntime,
    ProviderAccount,
    RequestReservation,
    metadata_from_json,
    metadata_to_json,
)


def _bool(value: Any) -> bool:
    return bool(int(value or 0))


def _value(row: sqlite3.Row, name: str, default: Any = "") -> Any:
    return row[name] if name in row.keys() else default


def _account_from_row(row: sqlite3.Row) -> ProviderAccount:
    try:
        allowed_privacy = tuple(json.loads(_value(row, "allowed_privacy_json", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        allowed_privacy = ("public", "internal")
    return ProviderAccount(
        account_id=row["account_id"],
        provider=row["provider"],
        pool_id=row["pool_id"],
        display_name=row["display_name"],
        secret_ref=row["secret_ref"],
        enabled=_bool(row["enabled"]),
        environment=row["environment"],
        region=row["region"],
        project_id=row["project_id"],
        tenant_id=row["tenant_id"],
        priority=int(row["priority"] or 0),
        weight=int(row["weight"] or 1),
        allows_free=_bool(row["allows_free"]),
        allows_paid=_bool(row["allows_paid"]),
        allows_vision=_bool(row["allows_vision"]),
        allows_text=_bool(row["allows_text"]),
        monthly_cost_limit=float(row["monthly_cost_limit"] or 0),
        daily_cost_limit=float(row["daily_cost_limit"] or 0),
        daily_request_limit=int(row["daily_request_limit"] or 0),
        daily_token_limit=int(row["daily_token_limit"] or 0),
        max_concurrency=max(1, int(row["max_concurrency"] or 1)),
        metadata=metadata_from_json(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        version=int(row["version"] or 0),
        account_state=AccountState(
            _value(row, "account_state", AccountState.ACTIVE.value)
        ),
        organization_id=_value(row, "organization_id", ""),
        allowed_privacy=allowed_privacy,
        cooldown_until=_value(row, "cooldown_until", ""),
    )


def _runtime_from_row(row: sqlite3.Row) -> AccountRuntime:
    return AccountRuntime(
        router_id=row["router_id"],
        provider=row["provider"],
        pool_id=row["pool_id"],
        state=AccountState(row["state"]),
        active_account_id=row["active_account_id"],
        pending_account_id=row["pending_account_id"],
        account_epoch=int(row["account_epoch"] or 0),
        switch_id=row["switch_id"],
        drain_deadline=row["drain_deadline"],
        closed_at=row["closed_at"],
        switch_started_at=row["switch_started_at"],
        cooldown_until=row["cooldown_until"],
        last_error_code=row["last_error_code"],
        last_error_at=row["last_error_at"],
        version=int(row["version"] or 0),
        updated_at=row["updated_at"],
    )


def _reservation_from_row(row: sqlite3.Row) -> RequestReservation:
    raw_state = row["state"]
    legacy_states = {
        "SUCCEEDED": ReservationState.SETTLED_SUCCESS,
        "FAILED": ReservationState.SETTLED_FAILURE,
    }
    reservation_state = (
        legacy_states[raw_state]
        if raw_state in legacy_states
        else ReservationState(raw_state)
    )
    return RequestReservation(
        reservation_id=row["reservation_id"],
        idempotency_key=row["idempotency_key"],
        provider=row["provider"],
        pool_id=row["pool_id"],
        account_id=row["account_id"],
        model=row["model"],
        reserved_requests=int(row["reserved_requests"] or 1),
        estimated_input_tokens=int(row["estimated_input_tokens"] or 0),
        estimated_output_tokens=int(row["estimated_output_tokens"] or 0),
        estimated_cost=float(row["estimated_cost"] or 0),
        state=reservation_state,
        reservation_token=row["reservation_token"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        settled_at=row["settled_at"],
        actual_input_tokens=int(row["actual_input_tokens"] or 0),
        actual_output_tokens=int(row["actual_output_tokens"] or 0),
        actual_cost=float(row["actual_cost"] or 0),
        error_code=row["error_code"],
        request_id=_value(row, "request_id", ""),
        parent_request_id=_value(row, "parent_request_id", ""),
        attempt_id=_value(row, "attempt_id", ""),
        provider_model_id=_value(row, "provider_model_id", row["model"]),
        policy_id=_value(row, "policy_id", ""),
        reserved_input_tokens=int(_value(row, "reserved_input_tokens", 0) or 0),
        reserved_output_tokens=int(_value(row, "reserved_output_tokens", 0) or 0),
        reserved_cost=float(_value(row, "reserved_cost", 0) or 0),
        request_count_consumed=_bool(_value(row, "request_count_consumed", 0)),
        token_cost_consumed=_bool(_value(row, "token_cost_consumed", 0)),
        monetary_cost_consumed=_bool(_value(row, "monetary_cost_consumed", 0)),
        dispatched_at=_value(row, "dispatched_at", ""),
        sent_at=_value(row, "sent_at", ""),
        fencing_token=int(_value(row, "fencing_token", 1) or 1),
        uncertainty_reason=_value(row, "uncertainty_reason", ""),
    )


def _endpoint_from_row(row: sqlite3.Row) -> ModelEndpointCapability:
    def tuple_json(name: str) -> tuple[str, ...]:
        try:
            value = json.loads(row[name])
        except (TypeError, json.JSONDecodeError):
            return ()
        return tuple(str(item) for item in value) if isinstance(value, list) else ()

    return ModelEndpointCapability(
        endpoint_id=row["endpoint_id"],
        canonical_model_id=row["canonical_model_id"],
        provider=row["provider"],
        provider_model_id=row["provider_model_id"],
        endpoint_type=row["endpoint_type"],
        supports_text=_bool(row["supports_text"]),
        supports_vision_declared=_bool(row["supports_vision_declared"]),
        supports_vision_verified=_bool(row["supports_vision_verified"]),
        supports_base64=_bool(row["supports_base64"]),
        supports_image_url=_bool(row["supports_image_url"]),
        supports_native_pdf=_bool(row["supports_native_pdf"]),
        supports_multiple_images=_bool(row["supports_multiple_images"]),
        max_images_per_request=int(row["max_images_per_request"] or 1),
        supported_mime_types=tuple_json("supported_mime_types_json"),
        max_image_width=int(row["max_image_width"] or 0),
        max_image_height=int(row["max_image_height"] or 0),
        max_image_pixels=int(row["max_image_pixels"] or 0),
        max_payload_bytes=int(row["max_payload_bytes"] or 0),
        max_context_tokens=int(row["max_context_tokens"] or 0),
        max_output_tokens=int(row["max_output_tokens"] or 0),
        supports_structured_json=_bool(row["supports_structured_json"]),
        supports_json_schema=_bool(row["supports_json_schema"]),
        supports_streaming=_bool(row["supports_streaming"]),
        supports_system_prompt=_bool(row["supports_system_prompt"]),
        supports_temperature=_bool(row["supports_temperature"]),
        supports_seed=_bool(row["supports_seed"]),
        supports_tools=_bool(row["supports_tools"]),
        supports_parallel_tools=_bool(row["supports_parallel_tools"]),
        supports_reasoning=_bool(row["supports_reasoning"]),
        pricing_input=float(row["pricing_input"] or 0),
        pricing_output=float(row["pricing_output"] or 0),
        pricing_image=float(row["pricing_image"] or 0),
        currency=row["currency"],
        region_availability=tuple_json("region_availability_json"),
        declared_status=CapabilityVerificationStatus(row["declared_status"]),
        verified_status=CapabilityVerificationStatus(row["verified_status"]),
        declared_at=row["declared_at"],
        last_verified_at=row["last_verified_at"],
        capability_source=row["capability_source"],
        notes=row["notes"],
        enabled=_bool(row["enabled"]),
        expected_latency_ms=int(row["expected_latency_ms"] or 0),
        quality_score=float(row["quality_score"] or 0),
        supports_text_output=_bool(_value(row, "supports_text_output", 1)),
        privacy_compatibility=(
            tuple_json("privacy_compatibility_json")
            if "privacy_compatibility_json" in row.keys()
            else ()
        ),
        region_support=(
            tuple_json("region_support_json")
            if "region_support_json" in row.keys()
            else ()
        ),
        teacher_role_support=(
            tuple_json("teacher_role_support_json")
            if "teacher_role_support_json" in row.keys()
            else ()
        ),
        training_output_policy_status=_value(
            row, "training_output_policy_status", "unknown"
        ),
    )


class KeyRouterRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_account(self, account: ProviderAccount) -> None:
        now = utc_now()
        secret_ref = validate_secret_ref(account.secret_ref)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_accounts(
                    account_id, provider, pool_id, display_name, secret_ref, enabled,
                    environment, region, project_id, tenant_id, priority, weight,
                    allows_free, allows_paid, allows_vision, allows_text,
                    monthly_cost_limit, daily_cost_limit, daily_request_limit,
                    daily_token_limit, max_concurrency, metadata_json,
                    created_at, updated_at, version, account_state,
                    organization_id, allowed_privacy_json, cooldown_until
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    provider = excluded.provider,
                    pool_id = excluded.pool_id,
                    display_name = excluded.display_name,
                    secret_ref = excluded.secret_ref,
                    enabled = excluded.enabled,
                    environment = excluded.environment,
                    region = excluded.region,
                    project_id = excluded.project_id,
                    tenant_id = excluded.tenant_id,
                    priority = excluded.priority,
                    weight = excluded.weight,
                    allows_free = excluded.allows_free,
                    allows_paid = excluded.allows_paid,
                    allows_vision = excluded.allows_vision,
                    allows_text = excluded.allows_text,
                    monthly_cost_limit = excluded.monthly_cost_limit,
                    daily_cost_limit = excluded.daily_cost_limit,
                    daily_request_limit = excluded.daily_request_limit,
                    daily_token_limit = excluded.daily_token_limit,
                    max_concurrency = excluded.max_concurrency,
                    metadata_json = excluded.metadata_json,
                    account_state = excluded.account_state,
                    organization_id = excluded.organization_id,
                    allowed_privacy_json = excluded.allowed_privacy_json,
                    cooldown_until = excluded.cooldown_until,
                    updated_at = excluded.updated_at,
                    version = provider_accounts.version + 1
                """,
                (
                    account.account_id,
                    account.provider,
                    account.pool_id,
                    account.display_name,
                    secret_ref,
                    int(account.enabled),
                    account.environment,
                    account.region,
                    account.project_id,
                    account.tenant_id,
                    account.priority,
                    account.weight,
                    int(account.allows_free),
                    int(account.allows_paid),
                    int(account.allows_vision),
                    int(account.allows_text),
                    account.monthly_cost_limit,
                    account.daily_cost_limit,
                    account.daily_request_limit,
                    account.daily_token_limit,
                    account.max_concurrency,
                    metadata_to_json(account.metadata),
                    account.created_at or now,
                    now,
                    account.account_state.value,
                    account.organization_id,
                    json.dumps(list(account.allowed_privacy), sort_keys=True),
                    account.cooldown_until,
                ),
            )

    def list_accounts(self) -> list[ProviderAccount]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM provider_accounts ORDER BY provider, pool_id, account_id"
            ).fetchall()
        return [_account_from_row(row) for row in rows]

    def get_account(self, account_id: str) -> ProviderAccount | None:
        with closing(self.database.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM provider_accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        return _account_from_row(row) if row else None

    def set_account_state(
        self, account_id: str, state: AccountState, *, enabled: bool | None = None
    ) -> ProviderAccount:
        now = utc_now()
        with self.database.transaction() as connection:
            if enabled is None:
                cursor = connection.execute(
                    """
                    UPDATE provider_accounts
                    SET account_state = ?, version = version + 1, updated_at = ?
                    WHERE account_id = ?
                    """,
                    (state.value, now, account_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE provider_accounts
                    SET account_state = ?, enabled = ?, version = version + 1,
                        updated_at = ?
                    WHERE account_id = ?
                    """,
                    (state.value, int(enabled), now, account_id),
                )
            row = connection.execute(
                "SELECT * FROM provider_accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        if cursor.rowcount != 1 or row is None:
            raise KeyError("Unknown provider account")
        return _account_from_row(row)

    def set_account_cooldown(
        self, account_id: str, cooldown_until: str
    ) -> ProviderAccount:
        now = utc_now()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE provider_accounts
                SET cooldown_until = ?, version = version + 1, updated_at = ?
                WHERE account_id = ?
                """,
                (cooldown_until, now, account_id),
            )
            row = connection.execute(
                "SELECT * FROM provider_accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        if cursor.rowcount != 1 or row is None:
            raise KeyError("Unknown provider account")
        return _account_from_row(row)

    def upsert_endpoint(self, endpoint: ModelEndpointCapability) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO capability_endpoints(
                    endpoint_id, canonical_model_id, provider, provider_model_id,
                    endpoint_type, supports_text, supports_vision_declared,
                    supports_vision_verified, supports_base64, supports_image_url,
                    supports_native_pdf, supports_multiple_images,
                    max_images_per_request, supported_mime_types_json,
                    max_image_width, max_image_height, max_image_pixels,
                    max_payload_bytes, max_context_tokens, max_output_tokens,
                    supports_structured_json, supports_json_schema,
                    supports_streaming, supports_system_prompt,
                    supports_temperature, supports_seed, supports_tools,
                    supports_parallel_tools, supports_reasoning, pricing_input,
                    pricing_output, pricing_image, currency,
                    region_availability_json, declared_status, verified_status,
                    declared_at, last_verified_at, capability_source, notes,
                    enabled, expected_latency_ms, quality_score, created_at,
                    updated_at, version, supports_text_output,
                    privacy_compatibility_json, region_support_json,
                    teacher_role_support_json, training_output_policy_status
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?
                )
                ON CONFLICT(endpoint_id) DO UPDATE SET
                    canonical_model_id = excluded.canonical_model_id,
                    provider = excluded.provider,
                    provider_model_id = excluded.provider_model_id,
                    endpoint_type = excluded.endpoint_type,
                    supports_text = excluded.supports_text,
                    supports_vision_declared = excluded.supports_vision_declared,
                    supports_vision_verified = excluded.supports_vision_verified,
                    supports_base64 = excluded.supports_base64,
                    supports_image_url = excluded.supports_image_url,
                    supports_native_pdf = excluded.supports_native_pdf,
                    supports_multiple_images = excluded.supports_multiple_images,
                    max_images_per_request = excluded.max_images_per_request,
                    supported_mime_types_json = excluded.supported_mime_types_json,
                    max_image_width = excluded.max_image_width,
                    max_image_height = excluded.max_image_height,
                    max_image_pixels = excluded.max_image_pixels,
                    max_payload_bytes = excluded.max_payload_bytes,
                    max_context_tokens = excluded.max_context_tokens,
                    max_output_tokens = excluded.max_output_tokens,
                    supports_structured_json = excluded.supports_structured_json,
                    supports_json_schema = excluded.supports_json_schema,
                    supports_streaming = excluded.supports_streaming,
                    supports_system_prompt = excluded.supports_system_prompt,
                    supports_temperature = excluded.supports_temperature,
                    supports_seed = excluded.supports_seed,
                    supports_tools = excluded.supports_tools,
                    supports_parallel_tools = excluded.supports_parallel_tools,
                    supports_reasoning = excluded.supports_reasoning,
                    pricing_input = excluded.pricing_input,
                    pricing_output = excluded.pricing_output,
                    pricing_image = excluded.pricing_image,
                    currency = excluded.currency,
                    region_availability_json = excluded.region_availability_json,
                    declared_status = excluded.declared_status,
                    verified_status = excluded.verified_status,
                    declared_at = excluded.declared_at,
                    last_verified_at = excluded.last_verified_at,
                    capability_source = excluded.capability_source,
                    notes = excluded.notes,
                    enabled = excluded.enabled,
                    expected_latency_ms = excluded.expected_latency_ms,
                    quality_score = excluded.quality_score,
                    supports_text_output = excluded.supports_text_output,
                    privacy_compatibility_json =
                        excluded.privacy_compatibility_json,
                    region_support_json = excluded.region_support_json,
                    teacher_role_support_json =
                        excluded.teacher_role_support_json,
                    training_output_policy_status =
                        excluded.training_output_policy_status,
                    updated_at = excluded.updated_at,
                    version = capability_endpoints.version + 1
                """,
                (
                    endpoint.endpoint_id,
                    endpoint.canonical_model_id,
                    endpoint.provider,
                    endpoint.provider_model_id,
                    endpoint.endpoint_type,
                    int(endpoint.supports_text),
                    int(endpoint.supports_vision_declared),
                    int(endpoint.supports_vision_verified),
                    int(endpoint.supports_base64),
                    int(endpoint.supports_image_url),
                    int(endpoint.supports_native_pdf),
                    int(endpoint.supports_multiple_images),
                    endpoint.max_images_per_request,
                    json.dumps(list(endpoint.supported_mime_types), sort_keys=True),
                    endpoint.max_image_width,
                    endpoint.max_image_height,
                    endpoint.max_image_pixels,
                    endpoint.max_payload_bytes,
                    endpoint.max_context_tokens,
                    endpoint.max_output_tokens,
                    int(endpoint.supports_structured_json),
                    int(endpoint.supports_json_schema),
                    int(endpoint.supports_streaming),
                    int(endpoint.supports_system_prompt),
                    int(endpoint.supports_temperature),
                    int(endpoint.supports_seed),
                    int(endpoint.supports_tools),
                    int(endpoint.supports_parallel_tools),
                    int(endpoint.supports_reasoning),
                    endpoint.pricing_input,
                    endpoint.pricing_output,
                    endpoint.pricing_image,
                    endpoint.currency,
                    json.dumps(list(endpoint.region_availability), sort_keys=True),
                    endpoint.declared_status.value,
                    endpoint.verified_status.value,
                    endpoint.declared_at,
                    endpoint.last_verified_at,
                    endpoint.capability_source,
                    endpoint.notes,
                    int(endpoint.enabled),
                    endpoint.expected_latency_ms,
                    endpoint.quality_score,
                    now,
                    now,
                    int(endpoint.supports_text_output),
                    json.dumps(list(endpoint.privacy_compatibility), sort_keys=True),
                    json.dumps(list(endpoint.region_support), sort_keys=True),
                    json.dumps(list(endpoint.teacher_role_support), sort_keys=True),
                    endpoint.training_output_policy_status,
                ),
            )

    def list_endpoints(
        self, *, enabled_only: bool = False
    ) -> list[ModelEndpointCapability]:
        query = "SELECT * FROM capability_endpoints"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY canonical_model_id, provider, provider_model_id"
        with closing(self.database.connect()) as connection:
            rows = connection.execute(query).fetchall()
        return [_endpoint_from_row(row) for row in rows]

    def ensure_runtime(
        self,
        *,
        router_id: str,
        provider: str,
        pool_id: str,
        state: AccountState = AccountState.DISABLED,
    ) -> AccountRuntime:
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO account_runtime(
                    router_id, provider, pool_id, state, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (router_id, provider, pool_id, state.value, now),
            )
            row = connection.execute(
                """
                SELECT * FROM account_runtime
                WHERE router_id = ? AND provider = ? AND pool_id = ?
                """,
                (router_id, provider, pool_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Account runtime insert failed")
        return _runtime_from_row(row)

    def get_runtime(
        self, *, router_id: str, provider: str, pool_id: str
    ) -> AccountRuntime | None:
        with closing(self.database.connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM account_runtime
                WHERE router_id = ? AND provider = ? AND pool_id = ?
                """,
                (router_id, provider, pool_id),
            ).fetchone()
        return _runtime_from_row(row) if row else None

    def compare_and_set_runtime(
        self,
        *,
        runtime: AccountRuntime,
        new_state: AccountState,
        active_account_id: str | None = None,
        pending_account_id: str | None = None,
        switch_id: str | None = None,
        account_epoch: int | None = None,
        drain_deadline: str | None = None,
        closed_at: str | None = None,
        switch_started_at: str | None = None,
        last_error_code: str = "",
    ) -> AccountRuntime:
        now = utc_now()
        updates = {
            "state": new_state.value,
            "updated_at": now,
            "version": runtime.version + 1,
            "active_account_id": (
                active_account_id
                if active_account_id is not None
                else runtime.active_account_id
            ),
            "pending_account_id": (
                pending_account_id
                if pending_account_id is not None
                else runtime.pending_account_id
            ),
            "switch_id": switch_id if switch_id is not None else runtime.switch_id,
            "account_epoch": (
                account_epoch if account_epoch is not None else runtime.account_epoch
            ),
            "drain_deadline": (
                drain_deadline if drain_deadline is not None else runtime.drain_deadline
            ),
            "closed_at": closed_at if closed_at is not None else runtime.closed_at,
            "switch_started_at": (
                switch_started_at
                if switch_started_at is not None
                else runtime.switch_started_at
            ),
            "last_error_code": last_error_code,
            "last_error_at": now if last_error_code else runtime.last_error_at,
        }
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE account_runtime
                SET state = ?, active_account_id = ?, pending_account_id = ?,
                    account_epoch = ?, switch_id = ?, drain_deadline = ?,
                    closed_at = ?, switch_started_at = ?, last_error_code = ?,
                    last_error_at = ?, version = ?, updated_at = ?
                WHERE router_id = ? AND provider = ? AND pool_id = ? AND version = ?
                """,
                (
                    updates["state"],
                    updates["active_account_id"],
                    updates["pending_account_id"],
                    updates["account_epoch"],
                    updates["switch_id"],
                    updates["drain_deadline"],
                    updates["closed_at"],
                    updates["switch_started_at"],
                    updates["last_error_code"],
                    updates["last_error_at"],
                    updates["version"],
                    updates["updated_at"],
                    runtime.router_id,
                    runtime.provider,
                    runtime.pool_id,
                    runtime.version,
                ),
            )
            if cursor.rowcount != 1:
                raise FencingTokenError("Runtime was updated by another worker")
            row = connection.execute(
                """
                SELECT * FROM account_runtime
                WHERE router_id = ? AND provider = ? AND pool_id = ?
                """,
                (runtime.router_id, runtime.provider, runtime.pool_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Runtime disappeared after update")
        return _runtime_from_row(row)

    def active_reservation_count(self, account_id: str) -> int:
        with closing(self.database.connect()) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM request_reservations
                WHERE account_id = ?
                  AND state IN ('PENDING', 'RESERVED', 'DISPATCHING', 'SENT', 'UNCERTAIN')
                """,
                (account_id,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def reserve_quota(
        self,
        *,
        idempotency_key: str,
        provider: str,
        pool_id: str,
        account_id: str,
        model: str,
        request_limit: int,
        token_limit: int,
        cost_limit: float,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        estimated_cost: float,
        ttl_seconds: int,
        request_id: str = "",
        parent_request_id: str = "",
        attempt_id: str = "",
        provider_model_id: str = "",
        policy_id: str = "",
    ) -> RequestReservation:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=ttl_seconds)).isoformat()
        window_started_at = now_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        reservation_id = uuid.uuid4().hex
        reservation_token = uuid.uuid4().hex
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM request_reservations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return _reservation_from_row(existing)
            connection.execute(
                """
                INSERT OR IGNORE INTO quota_runtime(
                    provider, pool_id, account_id, model, window_type,
                    window_started_at, request_limit, token_limit, cost_limit,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    pool_id,
                    account_id,
                    model,
                    WindowType.DAY.value,
                    window_started_at,
                    request_limit,
                    token_limit,
                    cost_limit,
                    now,
                ),
            )
            quota = connection.execute(
                """
                SELECT * FROM quota_runtime
                WHERE provider = ? AND pool_id = ? AND account_id = ?
                  AND model = ? AND window_type = ? AND window_started_at = ?
                """,
                (
                    provider,
                    pool_id,
                    account_id,
                    model,
                    WindowType.DAY.value,
                    window_started_at,
                ),
            ).fetchone()
            if quota is None:
                raise RuntimeError("Quota runtime insert failed")
            next_requests = (
                int(quota["reserved_requests"] or 0)
                + int(quota["consumed_requests"] or 0)
                + 1
            )
            next_tokens = (
                int(quota["reserved_input_tokens"] or 0)
                + int(quota["reserved_output_tokens"] or 0)
                + int(quota["consumed_input_tokens"] or 0)
                + int(quota["consumed_output_tokens"] or 0)
                + estimated_input_tokens
                + estimated_output_tokens
            )
            next_cost = (
                float(quota["reserved_cost"] or 0)
                + float(quota["consumed_cost"] or 0)
                + estimated_cost
            )
            if request_limit and next_requests > request_limit:
                raise QuotaExceeded("request_limit_reached")
            if token_limit and next_tokens > token_limit:
                raise QuotaExceeded("token_limit_reached")
            if cost_limit and next_cost > cost_limit:
                raise QuotaExceeded("cost_limit_reached")
            connection.execute(
                """
                UPDATE quota_runtime
                SET reserved_requests = reserved_requests + 1,
                    reserved_input_tokens = reserved_input_tokens + ?,
                    reserved_output_tokens = reserved_output_tokens + ?,
                    reserved_cost = reserved_cost + ?,
                    version = version + 1,
                    updated_at = ?
                WHERE provider = ? AND pool_id = ? AND account_id = ?
                  AND model = ? AND window_type = ? AND window_started_at = ?
                """,
                (
                    estimated_input_tokens,
                    estimated_output_tokens,
                    estimated_cost,
                    now,
                    provider,
                    pool_id,
                    account_id,
                    model,
                    WindowType.DAY.value,
                    window_started_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO request_reservations(
                    reservation_id, idempotency_key, provider, pool_id, account_id,
                    model, reserved_requests, estimated_input_tokens,
                    estimated_output_tokens, estimated_cost, state,
                    reservation_token, created_at, expires_at, request_id,
                    parent_request_id, attempt_id, provider_model_id, policy_id,
                    reserved_input_tokens, reserved_output_tokens, reserved_cost,
                    fencing_token
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    reservation_id,
                    idempotency_key,
                    provider,
                    pool_id,
                    account_id,
                    model,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    estimated_cost,
                    ReservationState.RESERVED.value,
                    reservation_token,
                    now,
                    expires_at,
                    request_id,
                    parent_request_id,
                    attempt_id,
                    provider_model_id or model,
                    policy_id,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    estimated_cost,
                ),
            )
            row = connection.execute(
                "SELECT * FROM request_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if row is None:
            raise RuntimeError("Reservation insert failed")
        return _reservation_from_row(row)

    def mark_dispatching(
        self, reservation_id: str, reservation_token: str
    ) -> RequestReservation:
        return self._update_reservation_state(
            reservation_id, reservation_token, ReservationState.DISPATCHING
        )

    def mark_sent(
        self, reservation_id: str, reservation_token: str
    ) -> RequestReservation:
        return self._update_reservation_state(
            reservation_id, reservation_token, ReservationState.SENT
        )

    def mark_uncertain(
        self, reservation_id: str, reservation_token: str, error_code: str
    ) -> RequestReservation:
        return self._update_reservation_state(
            reservation_id,
            reservation_token,
            ReservationState.UNCERTAIN,
            error_code=error_code,
        )

    def _update_reservation_state(
        self,
        reservation_id: str,
        reservation_token: str,
        state: ReservationState,
        *,
        error_code: str = "",
    ) -> RequestReservation:
        with self.database.transaction() as connection:
            timestamp_column = ""
            now = utc_now()
            if state is ReservationState.DISPATCHING:
                timestamp_column = ", dispatched_at = ?"
            elif state is ReservationState.SENT:
                timestamp_column = ", sent_at = ?"
            params: list[Any] = [state.value, error_code]
            if timestamp_column:
                params.append(now)
            params.extend([reservation_id, reservation_token])
            cursor = connection.execute(
                """
                UPDATE request_reservations
                SET state = ?, error_code = ?
                """
                + timestamp_column
                + """
                WHERE reservation_id = ? AND reservation_token = ?
                  AND state IN ('PENDING', 'RESERVED', 'DISPATCHING', 'SENT', 'UNCERTAIN')
                """,
                tuple(params),
            )
            if cursor.rowcount != 1:
                raise FencingTokenError("Reservation token is stale or invalid")
            row = connection.execute(
                "SELECT * FROM request_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Reservation disappeared after update")
        return _reservation_from_row(row)

    def settle_reservation(
        self,
        *,
        reservation_id: str,
        reservation_token: str,
        state: ReservationState,
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
        actual_cost: float = 0.0,
        error_code: str = "",
        request_count_consumed: bool | None = None,
        token_cost_consumed: bool | None = None,
        monetary_cost_consumed: bool | None = None,
    ) -> RequestReservation:
        if state not in {
            ReservationState.SETTLED_SUCCESS,
            ReservationState.SETTLED_FAILURE,
            ReservationState.RELEASED,
            ReservationState.EXPIRED,
            ReservationState.CANCELLED,
        }:
            raise ValueError("Unsupported settlement state")
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM request_reservations
                WHERE reservation_id = ? AND reservation_token = ?
                  AND state IN ('PENDING', 'RESERVED', 'DISPATCHING', 'SENT', 'UNCERTAIN')
                """,
                (reservation_id, reservation_token),
            ).fetchone()
            if row is None:
                raise FencingTokenError("Reservation token is stale or invalid")
            now = utc_now()
            consume_request = (
                state is ReservationState.SETTLED_SUCCESS
                if request_count_consumed is None
                else request_count_consumed
            )
            consume_tokens = (
                state is ReservationState.SETTLED_SUCCESS
                and (actual_input_tokens > 0 or actual_output_tokens > 0)
                if token_cost_consumed is None
                else token_cost_consumed
            )
            consume_money = (
                state is ReservationState.SETTLED_SUCCESS and actual_cost > 0
                if monetary_cost_consumed is None
                else monetary_cost_consumed
            )
            connection.execute(
                """
                UPDATE request_reservations
                SET state = ?, settled_at = ?, actual_input_tokens = ?,
                    actual_output_tokens = ?, actual_cost = ?, error_code = ?,
                    request_count_consumed = ?, token_cost_consumed = ?,
                    monetary_cost_consumed = ?, fencing_token = fencing_token + 1
                WHERE reservation_id = ? AND reservation_token = ?
                """,
                (
                    state.value,
                    now,
                    actual_input_tokens,
                    actual_output_tokens,
                    actual_cost,
                    error_code,
                    int(consume_request),
                    int(consume_tokens),
                    int(consume_money),
                    reservation_id,
                    reservation_token,
                ),
            )
            connection.execute(
                """
                UPDATE quota_runtime
                SET reserved_requests = MAX(0, reserved_requests - 1),
                    reserved_input_tokens = MAX(0, reserved_input_tokens - ?),
                    reserved_output_tokens = MAX(0, reserved_output_tokens - ?),
                    reserved_cost = MAX(0, reserved_cost - ?),
                    consumed_requests = consumed_requests + ?,
                    consumed_input_tokens = consumed_input_tokens + ?,
                    consumed_output_tokens = consumed_output_tokens + ?,
                    consumed_cost = consumed_cost + ?,
                    version = version + 1,
                    updated_at = ?
                WHERE provider = ? AND pool_id = ? AND account_id = ?
                  AND model = ? AND window_type = 'day'
                """,
                (
                    int(row["estimated_input_tokens"] or 0),
                    int(row["estimated_output_tokens"] or 0),
                    float(row["estimated_cost"] or 0),
                    1 if consume_request else 0,
                    actual_input_tokens if consume_tokens else 0,
                    actual_output_tokens if consume_tokens else 0,
                    actual_cost if consume_money else 0,
                    now,
                    row["provider"],
                    row["pool_id"],
                    row["account_id"],
                    row["model"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM request_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if updated is None:
            raise RuntimeError("Reservation disappeared after settlement")
        return _reservation_from_row(updated)

    def list_reservations(self, limit: int = 100) -> list[RequestReservation]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM request_reservations
                ORDER BY created_at DESC LIMIT ?
                """,
                (max(1, min(500, limit)),),
            ).fetchall()
        return [_reservation_from_row(row) for row in rows]

    def reconcile_reservations(
        self,
        *,
        now: datetime | None = None,
        stale_after_seconds: int = 1800,
    ) -> list[dict[str, str]]:
        current = now or datetime.now(timezone.utc)
        cutoff = (current - timedelta(seconds=max(1, stale_after_seconds))).isoformat()
        decisions: list[dict[str, str]] = []
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("""
                SELECT * FROM request_reservations
                WHERE state IN ('PENDING', 'RESERVED', 'DISPATCHING', 'SENT', 'UNCERTAIN')
                ORDER BY created_at, reservation_id
                """).fetchall()
            for row in rows:
                old_state = str(row["state"])
                new_state = old_state
                reason = ""
                release = False
                if (
                    old_state in {"PENDING", "RESERVED"}
                    and row["expires_at"] <= current.isoformat()
                ):
                    new_state = ReservationState.EXPIRED.value
                    reason = "expired_before_dispatch"
                    release = True
                elif old_state in {"DISPATCHING", "SENT"}:
                    timestamp = row["dispatched_at"] or row["created_at"]
                    if timestamp <= cutoff:
                        new_state = ReservationState.UNCERTAIN.value
                        reason = "stale_after_dispatch"
                elif old_state == "UNCERTAIN":
                    timestamp = row["sent_at"] or row["created_at"]
                    if timestamp <= cutoff:
                        reason = "uncertain_requires_manual_reconciliation"
                if not reason:
                    continue
                if new_state != old_state:
                    cursor = connection.execute(
                        """
                        UPDATE request_reservations
                        SET state = ?, uncertainty_reason = ?,
                            settled_at = CASE WHEN ? = 'EXPIRED' THEN ? ELSE settled_at END,
                            fencing_token = fencing_token + 1
                        WHERE reservation_id = ? AND state = ? AND fencing_token = ?
                        """,
                        (
                            new_state,
                            reason,
                            new_state,
                            current.isoformat(),
                            row["reservation_id"],
                            old_state,
                            int(row["fencing_token"] or 1),
                        ),
                    )
                    if cursor.rowcount != 1:
                        continue
                if release:
                    connection.execute(
                        """
                        UPDATE quota_runtime
                        SET reserved_requests = MAX(0, reserved_requests - 1),
                            reserved_input_tokens = MAX(0, reserved_input_tokens - ?),
                            reserved_output_tokens = MAX(0, reserved_output_tokens - ?),
                            reserved_cost = MAX(0, reserved_cost - ?),
                            version = version + 1, updated_at = ?
                        WHERE provider = ? AND pool_id = ? AND account_id = ?
                          AND model = ? AND window_type = 'day'
                        """,
                        (
                            int(row["estimated_input_tokens"] or 0),
                            int(row["estimated_output_tokens"] or 0),
                            float(row["estimated_cost"] or 0),
                            current.isoformat(),
                            row["provider"],
                            row["pool_id"],
                            row["account_id"],
                            row["model"],
                        ),
                    )
                decisions.append(
                    {
                        "reservation_id": row["reservation_id"],
                        "old_state": old_state,
                        "new_state": new_state,
                        "reason": reason,
                    }
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        for decision in decisions:
            self.record_audit(
                {
                    "event_id": uuid.uuid4().hex,
                    "event_type": "reservation_reconciled",
                    "decision_reason": decision["reason"],
                    "old_state": decision["old_state"],
                    "new_state": decision["new_state"],
                    "created_at": current.isoformat(),
                    "metadata_json": json.dumps(
                        {"reservation_id": decision["reservation_id"]}, sort_keys=True
                    ),
                }
            )
        return decisions

    def record_normalized_error(self, error: NormalizedProviderError) -> str:
        error_id = uuid.uuid4().hex
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO key_router_errors(
                    error_id, error_type, provider, provider_account_id,
                    provider_model_id, http_status, provider_error_code,
                    sanitized_message, retryable, safe_to_failover,
                    retry_after_seconds, scope, request_may_have_been_processed,
                    raw_error_fingerprint, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error_id,
                    error.error_type.value,
                    error.provider,
                    error.provider_account_id,
                    error.provider_model_id,
                    error.http_status,
                    error.provider_error_code,
                    error.sanitized_message,
                    int(error.retryable),
                    int(error.safe_to_failover),
                    error.retry_after_seconds,
                    error.scope.value,
                    int(error.request_may_have_been_processed),
                    error.raw_error_fingerprint,
                    error.occurred_at,
                ),
            )
        return error_id

    def record_attempt(
        self,
        *,
        attempt_id: str,
        parent_request_id: str,
        request_id: str,
        attempt_number: int,
        provider: str,
        provider_model_id: str,
        account_id: str,
        reservation_id: str = "",
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO key_router_attempts(
                    attempt_id, parent_request_id, request_id, attempt_number,
                    provider, provider_model_id, account_id, reservation_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    parent_request_id,
                    request_id,
                    attempt_number,
                    provider,
                    provider_model_id,
                    account_id,
                    reservation_id,
                    utc_now(),
                ),
            )

    def complete_attempt(
        self, attempt_id: str, *, outcome: str, error_type: str = ""
    ) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE key_router_attempts
                SET outcome = ?, error_type = ?, completed_at = ?
                WHERE attempt_id = ? AND outcome = 'pending'
                """,
                (outcome, error_type, utc_now(), attempt_id),
            )
        if cursor.rowcount != 1:
            raise FencingTokenError("Attempt was already completed")

    def list_attempts(self, parent_request_id: str) -> list[dict[str, Any]]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM key_router_attempts
                WHERE parent_request_id = ? ORDER BY attempt_number
                """,
                (parent_request_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def circuit_id(
        *, level: str, provider: str, provider_model_id: str = "", account_id: str = ""
    ) -> str:
        material = f"{level}:{provider}:{provider_model_id}:{account_id}"
        return uuid.uuid5(uuid.NAMESPACE_URL, material).hex

    def record_circuit_failure(
        self,
        *,
        level: str,
        provider: str,
        provider_model_id: str = "",
        account_id: str = "",
        failure_threshold: int,
        open_seconds: int,
        failure_window_seconds: int = 60,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        circuit_id = self.circuit_id(
            level=level,
            provider=provider,
            provider_model_id=provider_model_id,
            account_id=account_id,
        )
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO key_router_circuits(
                    circuit_id, level, provider, provider_model_id, account_id,
                    state, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'CLOSED', ?)
                """,
                (
                    circuit_id,
                    level,
                    provider,
                    provider_model_id,
                    account_id,
                    current.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
            window_started_raw = str(row["window_started_at"] or "")
            try:
                window_started = datetime.fromisoformat(window_started_raw)
            except ValueError:
                window_started = current
            if window_started.tzinfo is None:
                window_started = window_started.replace(tzinfo=timezone.utc)
            reset_window = not window_started_raw or (
                current - window_started
            ).total_seconds() > max(1, failure_window_seconds)
            failures = 1 if reset_window else int(row["failure_count"] or 0) + 1
            next_window_started = (
                current.isoformat() if reset_window else window_started_raw
            )
            new_state = (
                CircuitState.OPEN.value
                if failures >= max(1, failure_threshold)
                else row["state"]
            )
            cooldown = (
                (current + timedelta(seconds=max(1, open_seconds))).isoformat()
                if new_state == CircuitState.OPEN.value
                else row["cooldown_until"]
            )
            connection.execute(
                """
                UPDATE key_router_circuits
                SET state = ?, failure_count = ?, success_count = 0,
                    window_started_at = ?,
                    opened_at = CASE WHEN ? = 'OPEN' THEN ? ELSE opened_at END,
                    cooldown_until = ?, half_open_in_flight = 0,
                    version = version + 1, updated_at = ?
                WHERE circuit_id = ?
                """,
                (
                    new_state,
                    failures,
                    next_window_started,
                    new_state,
                    current.isoformat(),
                    cooldown,
                    current.isoformat(),
                    circuit_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return dict(updated)

    def acquire_half_open_probe(
        self,
        circuit_id: str,
        *,
        probe_limit: int,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(timezone.utc)
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return True
            state = str(row["state"])
            if (
                state == CircuitState.OPEN.value
                and row["cooldown_until"] <= current.isoformat()
            ):
                connection.execute(
                    """
                    UPDATE key_router_circuits
                    SET state = 'HALF_OPEN', half_open_in_flight = 0,
                        version = version + 1, updated_at = ?
                    WHERE circuit_id = ? AND version = ?
                    """,
                    (current.isoformat(), circuit_id, row["version"]),
                )
                state = CircuitState.HALF_OPEN.value
            if state == CircuitState.CLOSED.value:
                connection.commit()
                return True
            if state != CircuitState.HALF_OPEN.value:
                connection.commit()
                return False
            cursor = connection.execute(
                """
                UPDATE key_router_circuits
                SET half_open_in_flight = half_open_in_flight + 1,
                    version = version + 1, updated_at = ?
                WHERE circuit_id = ? AND state = 'HALF_OPEN'
                  AND half_open_in_flight < ?
                """,
                (current.isoformat(), circuit_id, max(1, probe_limit)),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_circuit_success(
        self, circuit_id: str, *, success_threshold: int = 1
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Unknown circuit")
            successes = int(row["success_count"] or 0) + 1
            close = successes >= max(1, success_threshold)
            connection.execute(
                """
                UPDATE key_router_circuits
                SET state = ?, success_count = ?, failure_count = ?,
                    half_open_in_flight = MAX(0, half_open_in_flight - 1),
                    cooldown_until = CASE WHEN ? THEN '' ELSE cooldown_until END,
                    version = version + 1, updated_at = ?
                WHERE circuit_id = ?
                """,
                (
                    CircuitState.CLOSED.value if close else row["state"],
                    0 if close else successes,
                    0 if close else row["failure_count"],
                    int(close),
                    now,
                    circuit_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
        return dict(updated)

    def reset_circuit(self, circuit_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE key_router_circuits
                SET state = 'CLOSED', failure_count = 0, success_count = 0,
                    cooldown_until = '', half_open_in_flight = 0,
                    version = version + 1, updated_at = ?
                WHERE circuit_id = ?
                """,
                (now, circuit_id),
            )
            row = connection.execute(
                "SELECT * FROM key_router_circuits WHERE circuit_id = ?",
                (circuit_id,),
            ).fetchone()
        if cursor.rowcount != 1 or row is None:
            raise KeyError("Unknown circuit")
        return dict(row)

    def list_circuits(self) -> list[dict[str, Any]]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM key_router_circuits ORDER BY provider, level, circuit_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_audit(self, event: dict[str, Any]) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO key_router_audit(
                    event_id, event_type, provider, pool_id, account_id, model,
                    decision_reason, old_state, new_state, request_id, created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["event_type"],
                    event.get("provider", ""),
                    event.get("pool_id", ""),
                    event.get("account_id", ""),
                    event.get("model", ""),
                    event.get("decision_reason", ""),
                    event.get("old_state", ""),
                    event.get("new_state", ""),
                    event.get("request_id", ""),
                    event["created_at"],
                    event.get("metadata_json", "{}"),
                ),
            )

    def recent_audit(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM key_router_audit
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(100, limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def status_snapshot(self) -> dict[str, Any]:
        with closing(self.database.connect()) as connection:
            accounts = [dict(row) for row in connection.execute("""
                    SELECT provider, pool_id, account_id, display_name, enabled,
                           environment, region, priority, weight, allows_free,
                           allows_paid, allows_vision, allows_text,
                           daily_cost_limit, daily_request_limit,
                           max_concurrency, account_state, cooldown_until, version
                    FROM provider_accounts
                    ORDER BY provider, pool_id, account_id
                    """).fetchall()]
            runtimes = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM account_runtime ORDER BY provider, pool_id"
                ).fetchall()
            ]
            reservations = connection.execute("""
                SELECT COUNT(*) AS active_reservations
                FROM request_reservations
                WHERE state IN ('PENDING', 'RESERVED', 'DISPATCHING', 'SENT', 'UNCERTAIN')
                """).fetchone()
            reservation_counts = {
                row["state"]: int(row["count"]) for row in connection.execute("""
                    SELECT state, COUNT(*) AS count
                    FROM request_reservations GROUP BY state
                    """).fetchall()
            }
            endpoint_count = connection.execute(
                "SELECT COUNT(*) AS count FROM capability_endpoints WHERE enabled = 1"
            ).fetchone()
            capability_status = {
                row["verified_status"]: int(row["count"])
                for row in connection.execute("""
                    SELECT verified_status, COUNT(*) AS count
                    FROM capability_endpoints GROUP BY verified_status
                    """).fetchall()
            }
            open_circuits = connection.execute("""
                SELECT COUNT(*) AS count FROM key_router_circuits
                WHERE state IN ('OPEN', 'HALF_OPEN')
                """).fetchone()
            error_counts = {
                row["error_type"]: int(row["count"]) for row in connection.execute("""
                    SELECT error_type, COUNT(*) AS count
                    FROM key_router_errors GROUP BY error_type
                    """).fetchall()
            }
        return {
            "accounts": accounts,
            "runtimes": runtimes,
            "active_reservations": (
                int(reservations["active_reservations"] or 0) if reservations else 0
            ),
            "reservation_counts": reservation_counts,
            "enabled_endpoints": (
                int(endpoint_count["count"] or 0) if endpoint_count else 0
            ),
            "capability_status": capability_status,
            "open_circuits": int(open_circuits["count"] or 0) if open_circuits else 0,
            "error_counts": error_counts,
        }
