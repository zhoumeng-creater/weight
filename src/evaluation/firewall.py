"""Immutable information-time snapshots for v1.1 event decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class InformationBoundaryError(RuntimeError):
    """A field is unavailable or prohibited at the decision time."""


PROHIBITED_FIELDS = frozenset(
    {
        "future_trajectory",
        "future_disturbance",
        "future_shock",
        "future_feedback_missing_mask",
        "hidden_instance_label",
        "other_method_results",
        "unexecuted_candidate_outcome",
        "D2",
        "D3",
        "late_feedback",
        "future_RNG_values",
    }
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class InformationField:
    available_at: int
    value: Any


@dataclass(frozen=True)
class InformationSnapshot:
    decision_time: int
    fields: Mapping[str, InformationField]
    information_hash: str


def freeze_information(
    *,
    decision_time: int,
    fields: Mapping[str, InformationField],
    prohibited_fields: frozenset[str] = PROHIBITED_FIELDS,
) -> InformationSnapshot:
    if decision_time < 0:
        raise InformationBoundaryError("decision_time must be nonnegative")

    frozen: dict[str, InformationField] = {}
    payload: dict[str, dict[str, Any]] = {}
    for name in sorted(fields):
        field = fields[name]
        if not isinstance(field, InformationField):
            raise InformationBoundaryError(
                f"information field has invalid type: {name}"
            )
        if name in prohibited_fields:
            raise InformationBoundaryError(f"prohibited information field: {name}")
        if int(field.available_at) > int(decision_time):
            raise InformationBoundaryError(f"future information field: {name}")
        frozen_value = _freeze_value(field.value)
        frozen[name] = InformationField(
            available_at=int(field.available_at), value=frozen_value
        )
        payload[name] = {
            "available_at": int(field.available_at),
            "value": _jsonable(frozen_value),
        }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return InformationSnapshot(
        decision_time=int(decision_time),
        fields=MappingProxyType(frozen),
        information_hash=hashlib.sha256(encoded).hexdigest(),
    )


def validate_information_snapshot(
    snapshot: InformationSnapshot,
) -> InformationSnapshot:
    """Rebuild and authenticate a snapshot at a trust boundary."""

    if not isinstance(snapshot, InformationSnapshot):
        raise InformationBoundaryError("information snapshot type is invalid")
    rebuilt = freeze_information(
        decision_time=snapshot.decision_time,
        fields=snapshot.fields,
    )
    if rebuilt.information_hash != snapshot.information_hash:
        raise InformationBoundaryError("information snapshot hash is invalid")
    return rebuilt
