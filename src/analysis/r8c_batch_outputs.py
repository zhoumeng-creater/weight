"""Result-blind R8 integrity audit and authorization-gated R9 exports.

The R8 audit authenticates the frozen schedule, run/task manifests and every
committed artifact.  It reports only counts, hashes and outcome classes.  It
does not calculate, print or export an effect endpoint.

The separate R9 function requires both the exact locked run-manifest SHA-256
and a literal authorization token.  It then writes one compact endpoint row
and one cost/failure row per scheduled sequence.  Event diagnostics are
compressed and opt-in; no source or derived artifact is ever deleted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, fields
import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Final

from formal_execution.checkpoint_data import (
    CHECKPOINTS_PER_EVENT,
    EVENT_SUMMARY_MAX_RECORD_BYTES,
    WORKER_CONTROL_REPORT_MAX_BYTES,
    CheckpointDataError,
    CheckpointFile,
    file_sha256,
    read_checkpoint_file,
)
from formal_execution.schedule import (
    FormalSequenceSpec,
    canonical_json_bytes,
)

from .checkpoint_consumer import (
    CheckpointAnalysisError,
    NumericalContinuousEndpointExcluded,
    _validated_execution_observation,
    read_manifest_bound_complete_task_nhv,
)
from .checkpoint_metrics import (
    AnalyticReferenceScale,
    FormalMetricError,
    e1e2_sequence_endpoints,
    event_anytime_auc,
    event_early_auc,
    negative_transfer_rate,
)
from .reference_catalog import (
    ReferenceDerivation,
    load_reference_catalog,
)
from .reference_fronts import ReferenceArtifactError
from .r8c_failure_outcomes import (
    FailureOutcomeClosureError,
    validate_failure_outcome_closure,
)
from evaluation.contracts import TerminalCode


R8C_E1E2_SCHEDULE_ID: Final = (
    "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01"
)
R8C_E1E2_SCHEDULE_SHA256: Final = (
    "db468253fb1430749d9f816d19532e428ca1054a86f399f80b12575a5c45282d"
)
R8C_E1E2_REUSE_SHA256: Final = (
    "d235c1c53d7e504400ad37674bebba4a01145a934964039454776c9f09ba0c9e"
)
R8C_E1E2_TASK_COUNT: Final = 5_030
R8C_E1E2_TOTAL_CFE: Final = 851_000_000
R8C_E1E2_TOTAL_ATOMIC_STEPS: Final = 1_971_000_000
R8C_E1E2_REUSE_ROWS: Final = 310
R8C_E1E2_NEGATIVE_TRANSFER_PAIR_COUNT: Final = 2_330
R8C_E1E2_REFERENCE_CATALOG_SHA256: Final = (
    "c0754e503aa80fa577e3764d2cd3fe3b9ed9814efaa76b2abc55156c615ba98f"
)
R8C_E1E2_REFERENCE_CATALOG_LINES: Final = 2_294
R9_EXPORT_AUTHORIZATION: Final = (
    "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
)
_R9_README: Final = """# R9 compact readable outputs

This directory is a derived, human-readable view of one immutable raw R8C
E1+E2 root. `r9_export_manifest.json` binds every file here by byte count and
SHA-256. It contains no narrative effect conclusion.

- `task_endpoints.csv`: one row per scheduled sequence. Empty endpoint cells
  mean not computed; `endpoint_status` gives the reason. Values are never
  imputed for failed or numerically excluded tasks.
- `e2_negative_transfer.csv`: one row per frozen E2 comparator/FULL pair.
  `pair_status` identifies included and unavailable pairs.
- `failure_cost.csv`: one row per scheduled sequence, including failed and
  partial tasks. Unknown charged work remains empty and
  `charged_work_exact` remains false. Authenticated hard-kill JSONL tail
  fragments are reported by presence, byte count and SHA-256; they are never
  interpreted as completed events.
- `post_execution_hard_violation.csv`: one row per scheduled sequence.
  Its status and observation counts preserve task failure or missing
  execution observations; a rate is emitted only for a complete denominator.
- `event_diagnostics.jsonl.gz`: optional and absent by default. When requested,
  it contains one row per durably completed event, including authenticated
  prefixes from partial tasks.

The raw root is not modified or deleted by this export.
"""

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_WORKER_REPORT_MAX_BYTES = WORKER_CONTROL_REPORT_MAX_BYTES
_WORKER_REPORT_FILES = {
    "TASK_SUMMARY": "task_summary.json",
    "TASK_FAILURE": "task_failure.json",
    "SUPERVISOR_OUTCOME": "task_supervisor_outcome.json",
}
_SUMMARY_REPORT_KEYS = frozenset(
    {
        "artifact_role",
        "status",
        "task",
        "method_identity",
        "adapter_identity",
        "events",
        "total_cfe",
        "total_atomic_model_steps",
        "budget_accounting",
        "timeout_semantics",
        "runtime",
        "permissions",
        "charged_evaluation_count",
        "individual_evaluation_rows_persisted",
        "checkpoint_data_format",
        "event_summary_data_format",
    }
)
_FAILURE_REPORT_KEYS = frozenset(
    {
        "artifact_role",
        "task_id",
        "schedule_index",
        "status",
        "outcome_class",
        "task",
        "error_type",
        "reason_code",
        "algorithm_terminal_code",
        "timeout_marker",
        "accounting",
        "wall_seconds",
        "cpu_seconds",
        "attempt",
        "automatic_retries",
        "results_analysis_performed",
    }
)
_SUPERVISOR_REPORT_KEYS = frozenset(
    {
        "artifact_role",
        "status",
        "outcome_class",
        "task",
        "reason_code",
        "error_type",
        "accounting",
        "attempt",
        "automatic_retries",
        "algorithm_terminal_code",
        "results_analysis_performed",
    }
)
_TASK_SUMMARY_REPORT_STATUSES = frozenset(
    {"COMPLETE", "INCOMPLETE_RESOURCE_CEILING"}
)
_WORKER_REPORT_ACCOUNTING_KEYS = frozenset(
    {
        "scheduled_cfe",
        "scheduled_atomic_model_steps",
        "atomic_steps_per_cfe",
        "charged_cfe",
        "charged_atomic_model_steps",
        "charged_work_exact",
        "charged_work_source",
        "charged_work_recovery_error_type",
    }
)
_FORBIDDEN_WORKER_REPORT_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_ids",
        "vector",
        "vectors",
        "objectives",
        "front_objectives",
        "constraints",
        "nhv",
        "auc",
        "negative_transfer",
        "effect_size",
        "p_value",
        "error",
        "message",
        "traceback",
    }
)
_ALLOWED_WORKLOADS = {
    "E1_STATIC",
    "E1_DYNAMIC",
    "E1_ROLLING",
    "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
    "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
}
_SPEC_FIELDS = {field.name for field in fields(FormalSequenceSpec)}
_DERIVED_SCHEDULE_FIELDS = {
    "task_id",
    "total_cfe",
    "total_atomic_steps",
}
_TERMINAL_CODES: Final = frozenset(code.value for code in TerminalCode)
_OPAQUE_PARTIAL_CHECKPOINT_OUTCOME_CLASSES: Final = frozenset(
    {
        "TECHNICAL_SEQUENCE_TIMEOUT",
        "TECHNICAL_GLOBAL_TIMEOUT",
        "TECHNICAL_RESOURCE_TERMINATION",
    }
)
_EVENT_SUMMARY_FIELDS: Final = frozenset(
    {
        "event_id",
        "terminal",
        "ledger",
        "evaluation_failure_type_counts",
        "information_hash",
        "execution_feedback",
        "execution_observation",
    }
)


class R8CIntegrityError(ValueError):
    """A frozen R8C E1+E2 artifact failed authentication."""


class R9ExportAuthorizationError(PermissionError):
    """R9 export was requested without the exact prospective authorization."""


@dataclass(frozen=True)
class R8CFreeze:
    """Immutable batch identity used by the strict validator."""

    schedule_id: str
    schedule_sha256: str
    reuse_sha256: str
    task_count: int
    total_cfe: int
    total_atomic_steps: int
    reuse_rows: int


FORMAL_R8C_E1E2_FREEZE: Final = R8CFreeze(
    schedule_id=R8C_E1E2_SCHEDULE_ID,
    schedule_sha256=R8C_E1E2_SCHEDULE_SHA256,
    reuse_sha256=R8C_E1E2_REUSE_SHA256,
    task_count=R8C_E1E2_TASK_COUNT,
    total_cfe=R8C_E1E2_TOTAL_CFE,
    total_atomic_steps=R8C_E1E2_TOTAL_ATOMIC_STEPS,
    reuse_rows=R8C_E1E2_REUSE_ROWS,
)


@dataclass(frozen=True)
class EventSummaryRead:
    """Authenticated completed-event prefix and any opaque hard-kill tail."""

    rows: tuple[dict[str, Any], ...]
    trailing_fragment_present: bool
    trailing_fragment_bytes: int
    trailing_fragment_sha256: str | None


@dataclass(frozen=True)
class TaskIntegrity:
    """Control-plane facts for one authenticated task."""

    task_id: str
    status: str
    outcome_class: str | None
    event_count: int
    checkpoint_record_count: int
    charged_cfe: int | None
    evaluation_failure_count: int | None
    event_summary_trailing_fragment_present: bool
    event_summary_trailing_fragment_bytes: int
    event_summary_trailing_fragment_sha256: str | None
    task_manifest_sha256: str


@dataclass(frozen=True)
class R8CIntegrityReport:
    """Compact, effect-free output of a complete batch audit."""

    run_root: Path
    run_manifest_sha256: str
    run_status: str
    schedule_sha256: str
    scheduled_task_count: int
    authenticated_task_count: int
    completed_task_count: int
    failed_task_count: int
    authenticated_event_count: int
    authenticated_checkpoint_record_count: int
    scheduled_cfe: int
    authenticated_charged_cfe: int
    unknown_charged_cfe_task_count: int
    event_summary_trailing_fragment_task_count: int
    event_summary_trailing_fragment_total_bytes: int
    task_status_counts: Mapping[str, int]
    failure_class_counts: Mapping[str, int]
    tasks: tuple[TaskIntegrity, ...]

    def control_plane_dict(self) -> dict[str, Any]:
        """Return only non-effect facts suitable for R8 stdout/monitoring."""

        return {
            "artifact_role": (
                "R8C_E1E2_CONTROL_PLANE_INTEGRITY_NO_EFFECTS"
            ),
            "integrity_status": "PASS",
            "run_status": self.run_status,
            "run_manifest_sha256": self.run_manifest_sha256,
            "schedule_sha256": self.schedule_sha256,
            "scheduled_task_count": self.scheduled_task_count,
            "authenticated_task_count": self.authenticated_task_count,
            "completed_task_count": self.completed_task_count,
            "failed_task_count": self.failed_task_count,
            "authenticated_event_count": self.authenticated_event_count,
            "authenticated_checkpoint_record_count": (
                self.authenticated_checkpoint_record_count
            ),
            "scheduled_cfe": self.scheduled_cfe,
            "authenticated_charged_cfe": (
                self.authenticated_charged_cfe
            ),
            "unknown_charged_cfe_task_count": (
                self.unknown_charged_cfe_task_count
            ),
            "event_summary_trailing_fragment_task_count": (
                self.event_summary_trailing_fragment_task_count
            ),
            "event_summary_trailing_fragment_total_bytes": (
                self.event_summary_trailing_fragment_total_bytes
            ),
            "task_status_counts": dict(
                sorted(self.task_status_counts.items())
            ),
            "failure_class_counts": dict(
                sorted(self.failure_class_counts.items())
            ),
            "effect_values_emitted": False,
            "effect_endpoints_computed": False,
            "source_artifacts_deleted": False,
        }


def _sha256_file(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise R8CIntegrityError(f"cannot hash artifact: {path.name}") from error


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise R8CIntegrityError(f"{label} cannot be read") from error
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise R8CIntegrityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise R8CIntegrityError(f"{label} must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise R8CIntegrityError(f"{label} is not canonical JSON")
    return value


def _read_canonical_json_bytes(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise R8CIntegrityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise R8CIntegrityError(f"{label} must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise R8CIntegrityError(f"{label} is not canonical JSON")
    return value


def _read_sha256_locked_canonical_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if _SHA256_HEX.fullmatch(expected_sha256) is None:
        raise R8CIntegrityError(f"{label} SHA-256 lock is invalid")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise R8CIntegrityError(f"{label} cannot be read") from error
    if sha256(raw).hexdigest() != expected_sha256:
        raise R8CIntegrityError(f"{label} SHA-256 lock differs")
    return _read_canonical_json_bytes(raw, label=label)


def _read_committed_canonical_json(
    path: Path,
    commitment: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != {"bytes", "sha256"}
        or type(commitment["bytes"]) is not int
        or commitment["bytes"] < 0
        or not isinstance(commitment["sha256"], str)
        or _SHA256_HEX.fullmatch(commitment["sha256"]) is None
    ):
        raise R8CIntegrityError(f"{label} commitment is invalid")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise R8CIntegrityError(f"{label} cannot be read") from error
    if (
        len(raw) != commitment["bytes"]
        or sha256(raw).hexdigest() != commitment["sha256"]
    ):
        raise R8CIntegrityError(f"{label} commitment differs")
    return _read_canonical_json_bytes(raw, label=label)


def _validate_commitment(
    path: Path,
    commitment: Any,
    *,
    label: str,
) -> str:
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != {"bytes", "sha256"}
        or type(commitment["bytes"]) is not int
        or commitment["bytes"] < 0
        or not isinstance(commitment["sha256"], str)
        or _SHA256_HEX.fullmatch(commitment["sha256"]) is None
    ):
        raise R8CIntegrityError(f"{label} commitment is invalid")
    if not path.is_file() or path.stat().st_size != commitment["bytes"]:
        raise R8CIntegrityError(f"{label} byte count differs")
    observed = _sha256_file(path)
    if observed != commitment["sha256"]:
        raise R8CIntegrityError(f"{label} SHA-256 differs")
    return observed


def _spec_from_schedule_row(row: Mapping[str, Any]) -> FormalSequenceSpec:
    if set(row) != _SPEC_FIELDS | _DERIVED_SCHEDULE_FIELDS:
        raise R8CIntegrityError("schedule row schema is invalid")
    try:
        spec = FormalSequenceSpec(
            **{key: row[key] for key in _SPEC_FIELDS}
        )
        reconstructed = spec.to_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise R8CIntegrityError(
            "schedule row cannot reconstruct its formal sequence"
        ) from error
    if reconstructed != dict(row):
        raise R8CIntegrityError(
            "schedule row derived identity or accounting differs"
        )
    if spec.workload_id not in _ALLOWED_WORKLOADS:
        raise R8CIntegrityError("schedule contains a non-E1/E2 workload")
    return spec


def _read_schedule(
    path: Path,
    *,
    freeze: R8CFreeze,
) -> tuple[tuple[dict[str, Any], ...], bytes]:
    try:
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
    except (OSError, EOFError) as error:
        raise R8CIntegrityError("schedule gzip cannot be decoded") from error
    if not raw.endswith(b"\n"):
        raise R8CIntegrityError("schedule JSONL lacks its final LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise R8CIntegrityError(
                f"schedule line {line_number} is invalid JSON"
            ) from error
        if (
            not isinstance(value, dict)
            or line != canonical_json_bytes(value)
        ):
            raise R8CIntegrityError(
                f"schedule line {line_number} is not canonical"
            )
        rows.append(value)
    if len(rows) != freeze.task_count:
        raise R8CIntegrityError(
            "schedule task count differs from the frozen batch"
        )
    if sha256(raw).hexdigest() != freeze.schedule_sha256:
        raise R8CIntegrityError(
            "schedule content differs from the frozen SHA-256"
        )
    specs = tuple(_spec_from_schedule_row(row) for row in rows)
    if any(
        spec.schedule_index != index
        for index, spec in enumerate(specs)
    ):
        raise R8CIntegrityError("schedule indexes are not consecutive")
    task_ids = [spec.task_id for spec in specs]
    if len(task_ids) != len(set(task_ids)):
        raise R8CIntegrityError("schedule contains duplicate task IDs")
    if sum(spec.total_cfe for spec in specs) != freeze.total_cfe:
        raise R8CIntegrityError("schedule CFE total differs from freeze")
    if (
        sum(spec.total_atomic_steps for spec in specs)
        != freeze.total_atomic_steps
    ):
        raise R8CIntegrityError(
            "schedule atomic-model-step total differs from freeze"
        )
    return tuple(rows), raw


def _read_reuse_map(path: Path, *, freeze: R8CFreeze) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise R8CIntegrityError("E2 reuse map cannot be read") from error
    if not raw.endswith(b"\n"):
        raise R8CIntegrityError("E2 reuse map lacks its final LF")
    if sha256(raw).hexdigest() != freeze.reuse_sha256:
        raise R8CIntegrityError("E2 reuse map SHA-256 differs from freeze")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise R8CIntegrityError(
                f"E2 reuse line {line_number} is invalid JSON"
            ) from error
        if (
            not isinstance(value, dict)
            or line != canonical_json_bytes(value)
        ):
            raise R8CIntegrityError(
                f"E2 reuse line {line_number} is not canonical"
            )
        rows.append(value)
    if len(rows) != freeze.reuse_rows:
        raise R8CIntegrityError("E2 reuse row count differs from freeze")
    return tuple(rows)


def _validate_checkpoint_control(
    decoded: CheckpointFile,
    *,
    spec: FormalSequenceSpec,
    events: Sequence[Any] | None,
) -> tuple[int, int, int]:
    """Validate checkpoint structure/counts without computing endpoints."""

    if decoded.metadata.task_id != spec.task_id:
        raise R8CIntegrityError(
            "checkpoint metadata task identity differs from schedule"
        )
    records_by_event: dict[int, list[Any]] = {}
    for record in decoded.records:
        if (
            record.event_id >= spec.events
            or record.cfe_budget != spec.cfe_per_event
        ):
            raise R8CIntegrityError(
                "checkpoint event or CFE budget differs from schedule"
            )
        records_by_event.setdefault(record.event_id, []).append(record)
    if records_by_event and set(records_by_event) != set(
        range(max(records_by_event) + 1)
    ):
        raise R8CIntegrityError(
            "checkpoint event IDs are not a zero-based prefix"
        )
    for records in records_by_event.values():
        if len(records) > CHECKPOINTS_PER_EVENT:
            raise R8CIntegrityError(
                "checkpoint event exceeds 21 fixed records"
            )
    charged = sum(records[-1].cfe for records in records_by_event.values())
    if events is not None:
        if len(events) != len(records_by_event):
            raise R8CIntegrityError(
                "summary/checkpoint event counts differ"
            )
        for event_id, event in enumerate(events):
            if (
                not isinstance(event, Mapping)
                or event.get("event_id") != event_id
                or event_id not in records_by_event
            ):
                raise R8CIntegrityError(
                    "summary event identity differs from checkpoint"
                )
            ledger = event.get("ledger")
            if not isinstance(ledger, Mapping):
                raise R8CIntegrityError("summary event ledger is missing")
            last = records_by_event[event_id][-1]
            if (
                ledger.get("cfe") != last.cfe
                or ledger.get("evaluation_failures")
                != last.failure_count
                or ledger.get("atomic_model_steps")
                != last.cfe * spec.atomic_steps_per_cfe
            ):
                raise R8CIntegrityError(
                    "summary ledger differs from checkpoint accounting"
                )
    return len(records_by_event), len(decoded.records), charged


def _read_committed_event_summaries(
    path: Path,
    commitment: Any,
    *,
    spec: FormalSequenceSpec,
    allow_trailing_fragment: bool = False,
) -> EventSummaryRead:
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != {"bytes", "sha256"}
        or type(commitment["bytes"]) is not int
        or commitment["bytes"] < 0
        or not isinstance(commitment["sha256"], str)
        or _SHA256_HEX.fullmatch(commitment["sha256"]) is None
    ):
        raise R8CIntegrityError("event summaries commitment is invalid")
    maximum_committed_bytes = (
        spec.events * EVENT_SUMMARY_MAX_RECORD_BYTES
        + (
            EVENT_SUMMARY_MAX_RECORD_BYTES - 1
            if allow_trailing_fragment
            else 0
        )
    )
    if commitment["bytes"] > maximum_committed_bytes:
        raise R8CIntegrityError(
            "event summaries exceed the frozen byte bound"
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise R8CIntegrityError("event summaries cannot be read") from error
    if (
        len(raw) != commitment["bytes"]
        or sha256(raw).hexdigest() != commitment["sha256"]
    ):
        raise R8CIntegrityError("event summaries commitment differs")
    trailing_fragment = b""
    complete_raw = raw
    if raw and not raw.endswith(b"\n"):
        if not allow_trailing_fragment:
            raise R8CIntegrityError("event summaries lack the final LF")
        final_lf = raw.rfind(b"\n")
        if final_lf < 0:
            complete_raw = b""
            trailing_fragment = raw
        else:
            complete_raw = raw[: final_lf + 1]
            trailing_fragment = raw[final_lf + 1 :]
    if len(trailing_fragment) >= EVENT_SUMMARY_MAX_RECORD_BYTES:
        raise R8CIntegrityError(
            "event-summary trailing fragment exceeds the frozen byte bound"
        )

    ledger_fields = {
        "cfe",
        "objective_calls",
        "constraint_calls",
        "scenario_evaluations",
        "atomic_model_steps",
        "execution_transition_count",
        "repair_failed",
        "evaluation_failures",
    }
    rows: list[dict[str, Any]] = []
    for event_id, line in enumerate(complete_raw.splitlines()):
        if len(line) + 1 > EVENT_SUMMARY_MAX_RECORD_BYTES:
            raise R8CIntegrityError(
                "event summary line exceeds the frozen byte bound"
            )
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise R8CIntegrityError(
                f"event summary line {event_id + 1} is invalid JSON"
            ) from error
        if (
            not isinstance(value, dict)
            or line != canonical_json_bytes(value)
            or set(value) != _EVENT_SUMMARY_FIELDS
            or value.get("event_id") != event_id
        ):
            raise R8CIntegrityError(
                "event summaries schema/order is invalid"
            )
        if event_id >= spec.events:
            raise R8CIntegrityError(
                "event summaries exceed the scheduled event count"
            )
        terminal = value.get("terminal")
        if (
            not isinstance(terminal, Mapping)
            or set(terminal)
            != {"code", "reason", "candidate_available"}
            or terminal.get("code") not in _TERMINAL_CODES
            or type(terminal.get("candidate_available")) is not bool
            or (
                terminal.get("reason") is not None
                and not isinstance(terminal.get("reason"), str)
            )
        ):
            raise R8CIntegrityError(
                "event summary terminal classification is invalid"
            )
        ledger = value.get("ledger")
        if (
            not isinstance(ledger, Mapping)
            or set(ledger) != ledger_fields
            or any(type(item) is not int or item < 0 for item in ledger.values())
        ):
            raise R8CIntegrityError("event summary ledger is invalid")
        cfe = ledger["cfe"]
        if (
            cfe > spec.cfe_per_event
            or ledger["objective_calls"] != cfe
            or ledger["constraint_calls"] != cfe
            or ledger["scenario_evaluations"] != cfe
            or ledger["atomic_model_steps"]
            != cfe * spec.atomic_steps_per_cfe
            or ledger["evaluation_failures"] > cfe
        ):
            raise R8CIntegrityError(
                "event summary ledger differs from the frozen accounting"
            )
        if ledger["execution_transition_count"] not in {0, 1}:
            raise R8CIntegrityError(
                "event summary execution transition count is invalid"
            )
        failure_types = value.get("evaluation_failure_type_counts")
        if (
            not isinstance(failure_types, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count < 1
                for name, count in failure_types.items()
            )
            or sum(failure_types.values())
            != ledger["evaluation_failures"]
        ):
            raise R8CIntegrityError(
                "event summary failure types differ from the ledger"
            )
        information_hash = value.get("information_hash")
        if (
            not isinstance(information_hash, str)
            or _SHA256_HEX.fullmatch(information_hash) is None
        ):
            raise R8CIntegrityError(
                "event summary information hash is invalid"
            )
        try:
            feedback = _validated_execution_observation(
                value.get("execution_feedback"),
                event_id=event_id,
            )
            observation = _validated_execution_observation(
                value.get("execution_observation"),
                event_id=event_id,
            )
        except CheckpointAnalysisError as error:
            raise R8CIntegrityError(
                "event summary execution channel is invalid"
            ) from error
        if feedback is not None and feedback != observation:
            raise R8CIntegrityError(
                "event summary feedback differs from execution observation"
            )
        rows.append(value)
    return EventSummaryRead(
        rows=tuple(rows),
        trailing_fragment_present=bool(trailing_fragment),
        trailing_fragment_bytes=len(trailing_fragment),
        trailing_fragment_sha256=(
            sha256(trailing_fragment).hexdigest()
            if trailing_fragment
            else None
        ),
    )


def _validate_complete_summary(
    summary: Mapping[str, Any],
    *,
    spec: FormalSequenceSpec,
    decoded: CheckpointFile,
    event_summaries: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int]:
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("task") != spec.to_dict()
        or summary.get("individual_evaluation_rows_persisted") != 0
        or summary.get("permissions", {}).get(
            "results_analysis_performed"
        )
        is not False
    ):
        raise R8CIntegrityError(
            "complete task summary boundary or identity is invalid"
        )
    events = summary.get("events")
    if not isinstance(events, list) or len(events) != spec.events:
        raise R8CIntegrityError(
            "complete task event count differs from schedule"
        )
    if list(event_summaries) != events:
        raise R8CIntegrityError(
            "task summary differs from the append-only event summaries"
        )
    event_count, record_count, charged = _validate_checkpoint_control(
        decoded,
        spec=spec,
        events=events,
    )
    failures = 0
    for event in events:
        ledger = event["ledger"]
        failure_count = ledger["evaluation_failures"]
        if type(failure_count) is not int or failure_count < 0:
            raise R8CIntegrityError(
                "event numerical failure count is invalid"
            )
        failure_types = event.get("evaluation_failure_type_counts")
        if (
            not isinstance(failure_types, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count < 1
                for name, count in failure_types.items()
            )
            or sum(failure_types.values()) != failure_count
        ):
            raise R8CIntegrityError(
                "event failure-type counts differ from ledger"
            )
        terminal = event.get("terminal")
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("code") not in _TERMINAL_CODES
        ):
            raise R8CIntegrityError("event terminal code is invalid")
        failures += failure_count
    if (
        summary.get("total_cfe") != charged
        or summary.get("charged_evaluation_count") != charged
        or summary.get("total_atomic_model_steps")
        != charged * spec.atomic_steps_per_cfe
    ):
        raise R8CIntegrityError(
            "task summary totals differ from authenticated checkpoints"
        )
    return event_count, record_count, charged, failures


def _runtime_outcomes(
    runtime: Mapping[str, Any],
    *,
    task_ids: set[str],
) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    completed = runtime.get("completed")
    failures = runtime.get("failures")
    if not isinstance(completed, list) or not isinstance(failures, list):
        raise R8CIntegrityError("runtime outcome lists are missing")
    outcomes: dict[str, Mapping[str, Any]] = {}
    failure_classes: Counter[str] = Counter()
    for expected_status, values in (
        ("COMPLETE", completed),
        (None, failures),
    ):
        for value in values:
            if not isinstance(value, Mapping):
                raise R8CIntegrityError("runtime outcome is not an object")
            task_id = value.get("task_id")
            status = value.get("status")
            if (
                not isinstance(task_id, str)
                or task_id not in task_ids
                or task_id in outcomes
                or not isinstance(status, str)
                or not status
            ):
                raise R8CIntegrityError(
                    "runtime outcome identity/status is invalid or duplicate"
                )
            if expected_status is not None and status != expected_status:
                raise R8CIntegrityError(
                    "runtime completed outcome is not COMPLETE"
                )
            outcomes[task_id] = value
            if expected_status is None:
                outcome_class = value.get("outcome_class")
                if not isinstance(outcome_class, str) or not outcome_class:
                    raise R8CIntegrityError(
                        "failed outcome lacks its technical class"
                    )
                failure_classes[outcome_class] += 1
    if set(outcomes) != task_ids:
        raise R8CIntegrityError(
            "runtime outcomes contain missing or extra scheduled tasks"
        )
    if (
        runtime.get("scheduled_task_count") != len(task_ids)
        or runtime.get("recorded_outcome_count") != len(task_ids)
        or runtime.get("attempts_per_task") != 1
        or runtime.get("automatic_retries") != 0
        or runtime.get("results_analysis_performed") is not False
        or runtime.get("effect_values_read_by_supervisor") is not False
    ):
        raise R8CIntegrityError(
            "runtime count/retry/effect boundary is invalid"
        )
    return outcomes, failure_classes


def _validate_task_directory(
    directory: Path,
    *,
    spec: FormalSequenceSpec,
    expected_manifest_sha256: str,
    outcome: Mapping[str, Any],
) -> TaskIntegrity:
    entries = list(directory.iterdir())
    if any(not path.is_file() for path in entries):
        raise R8CIntegrityError(
            "task directory contains an unexpected non-file entry"
        )
    manifest_path = directory / "task_manifest.json"
    if (
        _SHA256_HEX.fullmatch(expected_manifest_sha256) is None
        or _sha256_file(manifest_path) != expected_manifest_sha256
        or outcome.get("task_manifest_sha256")
        != expected_manifest_sha256
    ):
        raise R8CIntegrityError(
            "task manifest differs from run/runtime commitments"
        )
    manifest = _read_canonical_json(
        manifest_path,
        label=f"task manifest {spec.task_id}",
    )
    status = manifest.get("status")
    if (
        manifest.get("task_id") != spec.task_id
        or not isinstance(status, str)
        or not status
        or status != outcome.get("status")
    ):
        raise R8CIntegrityError(
            "task manifest identity/status differs from schedule/runtime"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise R8CIntegrityError("task artifact commitments are missing")
    actual_files = {
        path.name
        for path in entries
        if path.name != "task_manifest.json"
    }
    if actual_files != set(artifacts):
        raise R8CIntegrityError(
            "task directory has missing or extra committed files"
        )
    for name, commitment in artifacts.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
        ):
            raise R8CIntegrityError("task artifact name is unsafe")
        _validate_commitment(
            directory / name,
            commitment,
            label=f"{spec.task_id}/{name}",
        )
    expected_binding = sha256(
        canonical_json_bytes(
            {
                "task": spec.to_dict(),
                "artifacts": dict(artifacts),
            }
        )
    ).hexdigest()
    if manifest.get("task_binding_sha256") != expected_binding:
        raise R8CIntegrityError("task binding SHA-256 differs")

    event_count = 0
    record_count = 0
    charged_cfe: int | None = None
    evaluation_failures: int | None = None
    opaque_partial_checkpoint = (
        status != "COMPLETE"
        and outcome.get("outcome_class")
        in _OPAQUE_PARTIAL_CHECKPOINT_OUTCOME_CLASSES
        and outcome.get("charged_work_exact") is False
    )
    event_summaries: tuple[dict[str, Any], ...] = ()
    event_summary_read = EventSummaryRead((), False, 0, None)
    event_summaries_path = directory / "event_summaries.jsonl"
    if "event_summaries.jsonl" in artifacts:
        event_summary_read = _read_committed_event_summaries(
            event_summaries_path,
            artifacts["event_summaries.jsonl"],
            spec=spec,
            allow_trailing_fragment=opaque_partial_checkpoint,
        )
        event_summaries = event_summary_read.rows
    elif status == "COMPLETE":
        raise R8CIntegrityError(
            "complete task lacks append-only event summaries"
        )

    checkpoint_path = directory / "checkpoint_fronts.cfe"
    decoded: CheckpointFile | None = None
    if checkpoint_path.is_file() and not opaque_partial_checkpoint:
        try:
            decoded = read_checkpoint_file(checkpoint_path)
        except (CheckpointDataError, OSError) as error:
            raise R8CIntegrityError(
                f"checkpoint file failed strict decoding: {spec.task_id}"
            ) from error
        if decoded.sha256 != artifacts["checkpoint_fronts.cfe"]["sha256"]:
            raise R8CIntegrityError("checkpoint SHA-256 differs")
    summary_path = directory / "task_summary.json"
    if status == "COMPLETE":
        if set(artifacts) != {
            "checkpoint_fronts.cfe",
            "event_summaries.jsonl",
            "task_summary.json",
        } or decoded is None:
            raise R8CIntegrityError(
                "complete task lacks its exact compact artifact set"
            )
        summary = _read_canonical_json(
            summary_path,
            label=f"task summary {spec.task_id}",
        )
        (
            event_count,
            record_count,
            charged_cfe,
            evaluation_failures,
        ) = _validate_complete_summary(
            summary,
            spec=spec,
            decoded=decoded,
            event_summaries=event_summaries,
        )
        if (
            outcome.get("total_cfe") != charged_cfe
            or outcome.get("total_atomic_model_steps")
            != charged_cfe * spec.atomic_steps_per_cfe
        ):
            raise R8CIntegrityError(
                "runtime complete accounting differs from task"
            )
    elif opaque_partial_checkpoint:
        if outcome.get("charged_cfe") is not None:
            raise R8CIntegrityError(
                "inexact hard-kill outcome fabricates charged CFE"
            )
        event_count = len(event_summaries)
        evaluation_failures = sum(
            int(event["ledger"]["evaluation_failures"])
            for event in event_summaries
        )
    elif decoded is not None:
        event_count, record_count, checkpoint_cfe = (
            _validate_checkpoint_control(
                decoded,
                spec=spec,
                events=None,
            )
        )
        charged_cfe = outcome.get("charged_cfe")
        if (
            charged_cfe is not None
            and (
                type(charged_cfe) is not int
                or charged_cfe < 0
                or charged_cfe != checkpoint_cfe
            )
        ):
            raise R8CIntegrityError(
                "failed task charged CFE differs from checkpoint"
            )
        evaluation_failures = sum(
            records[-1].failure_count
            for records in (
                [
                    record
                    for record in decoded.records
                    if record.event_id == event_id
                ]
                for event_id in range(event_count)
            )
        )
        if len(event_summaries) > event_count:
            raise R8CIntegrityError(
                "event summaries exceed strict checkpoint event prefix"
            )
        final_records = {
            record.event_id: record for record in decoded.records
        }
        for event_id, event in enumerate(event_summaries):
            record = final_records[event_id]
            ledger = event["ledger"]
            if (
                record.cfe != ledger["cfe"]
                or record.failure_count != ledger["evaluation_failures"]
            ):
                raise R8CIntegrityError(
                    "event summaries differ from strict checkpoint accounting"
                )
    else:
        charged_cfe = outcome.get("charged_cfe")
        if charged_cfe is not None and (
            type(charged_cfe) is not int
            or not 0 <= charged_cfe <= spec.total_cfe
        ):
            raise R8CIntegrityError("failed task charged CFE is invalid")
        event_count = len(event_summaries)
        evaluation_failures = sum(
            int(event["ledger"]["evaluation_failures"])
            for event in event_summaries
        )
    return TaskIntegrity(
        task_id=spec.task_id,
        status=status,
        outcome_class=(
            str(outcome["outcome_class"])
            if isinstance(outcome.get("outcome_class"), str)
            else None
        ),
        event_count=event_count,
        checkpoint_record_count=record_count,
        charged_cfe=charged_cfe,
        evaluation_failure_count=evaluation_failures,
        event_summary_trailing_fragment_present=(
            event_summary_read.trailing_fragment_present
        ),
        event_summary_trailing_fragment_bytes=(
            event_summary_read.trailing_fragment_bytes
        ),
        event_summary_trailing_fragment_sha256=(
            event_summary_read.trailing_fragment_sha256
        ),
        task_manifest_sha256=expected_manifest_sha256,
    )


def _control_artifact_paths(
    root: Path,
    commitments: Mapping[str, Any],
) -> dict[str, Path]:
    required = {
        "schedule.jsonl.gz",
        "e2_full_reuse_map.jsonl",
        "launch_binding.json",
        "runtime_report.json",
        "request_consumption_record.json",
    }
    if set(commitments) != required:
        raise R8CIntegrityError(
            "run manifest control artifact set is invalid"
        )
    paths = {name: root / name for name in required}
    launch = _read_canonical_json(
        root / "launch_binding.json",
        label="launch binding",
    )
    launch_paths = launch.get("paths")
    if not isinstance(launch_paths, Mapping):
        raise R8CIntegrityError("launch binding paths are missing")
    marker_value = launch_paths.get("request_consumption_marker")
    record_value = launch_paths.get("request_consumption_record")
    if not isinstance(marker_value, str) or not marker_value:
        raise R8CIntegrityError(
            "launch binding lacks the external consumption marker path"
        )
    if (
        not isinstance(record_value, str)
        or record_value != "request_consumption_record.json"
    ):
        raise R8CIntegrityError(
            "launch binding portable consumption record path differs"
        )
    marker = _read_canonical_json(
        root / "request_consumption_record.json",
        label="portable request consumption record",
    )
    if marker.get("consumption") != "ONE_TIME_FORMAL_SUPERVISOR_START":
        raise R8CIntegrityError(
            "portable request consumption record is invalid"
        )
    if marker.get("launch_binding_sha256") != _sha256_file(
        root / "launch_binding.json"
    ):
        raise R8CIntegrityError(
            "portable request consumption record binding differs"
        )
    return paths


def _validate_worker_log_commitments(
    root: Path,
    *,
    run_manifest: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    task_ids: set[str],
) -> None:
    run_values = run_manifest.get("worker_log_commitments")
    runtime_values = runtime_report.get("worker_log_commitments")
    if (
        not isinstance(run_values, Mapping)
        or not isinstance(runtime_values, Mapping)
        or dict(run_values) != dict(runtime_values)
        or set(run_values) != task_ids
    ):
        raise R8CIntegrityError(
            "worker-log commitments do not cover the frozen schedule"
        )
    logs_root = (root / "worker_logs").resolve()
    if not logs_root.is_dir():
        raise R8CIntegrityError("worker logs directory is missing")
    expected_present: set[str] = set()
    for task_id, bindings in run_values.items():
        if (
            not isinstance(bindings, Mapping)
            or set(bindings) != {"stdout", "stderr"}
        ):
            raise R8CIntegrityError(
                "worker-log stream commitments are invalid"
            )
        for label, suffix in (
            ("stdout", "stdout.log"),
            ("stderr", "stderr.log"),
        ):
            expected_path = (logs_root / f"{task_id}.{suffix}").resolve()
            expected_relative = (
                Path("worker_logs") / f"{task_id}.{suffix}"
            ).as_posix()
            binding = bindings[label]
            if not isinstance(binding, Mapping):
                raise R8CIntegrityError(
                    "worker-log commitment is not an object"
                )
            if binding.get("path") != expected_relative:
                raise R8CIntegrityError(
                    "worker-log path differs from the frozen task identity"
                )
            if binding.get("missing") is True:
                if set(binding) != {"path", "missing"} or (
                    expected_path.exists()
                ):
                    raise R8CIntegrityError(
                        "missing worker-log commitment differs"
                    )
                continue
            if set(binding) != {"path", "bytes", "sha256"}:
                raise R8CIntegrityError(
                    "present worker-log commitment schema is invalid"
                )
            _validate_commitment(
                expected_path,
                {
                    "bytes": binding["bytes"],
                    "sha256": binding["sha256"],
                },
                label=f"worker log {task_id}/{label}",
            )
            expected_present.add(expected_path.name)
    actual_present = {
        path.name for path in logs_root.iterdir() if path.is_file()
    }
    if actual_present != expected_present:
        raise R8CIntegrityError(
            "worker logs directory has missing or extra files"
        )
    if any(path.is_dir() for path in logs_root.iterdir()):
        raise R8CIntegrityError(
            "worker logs directory contains unexpected subdirectories"
        )


def _contains_forbidden_worker_report_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                str(key).casefold() in _FORBIDDEN_WORKER_REPORT_KEYS
                or _contains_forbidden_worker_report_key(child)
            )
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(
            _contains_forbidden_worker_report_key(child) for child in value
        )
    return False


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and value >= 0
    )


def _valid_worker_report_accounting(
    value: Any,
    *,
    spec: FormalSequenceSpec,
) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value) != _WORKER_REPORT_ACCOUNTING_KEYS
        or value.get("scheduled_cfe") != spec.total_cfe
        or value.get("scheduled_atomic_model_steps")
        != spec.total_atomic_steps
        or value.get("atomic_steps_per_cfe")
        != spec.atomic_steps_per_cfe
        or type(value.get("charged_work_exact")) is not bool
    ):
        return False
    charged_cfe = value.get("charged_cfe")
    charged_atomic = value.get("charged_atomic_model_steps")
    source = value.get("charged_work_source")
    recovery_error_type = value.get(
        "charged_work_recovery_error_type"
    )
    if source is not None and (
        type(source) is not str or not source
    ):
        return False
    if recovery_error_type is not None and (
        type(recovery_error_type) is not str
        or not recovery_error_type
    ):
        return False
    if value["charged_work_exact"]:
        return (
            type(charged_cfe) is int
            and 0 <= charged_cfe <= spec.total_cfe
            and type(charged_atomic) is int
            and charged_atomic
            == charged_cfe * spec.atomic_steps_per_cfe
            and source is not None
            and recovery_error_type is None
        )
    return (
        charged_cfe is None
        and charged_atomic is None
        and source is None
    )


def _validate_worker_control_report_commitments(
    root: Path,
    *,
    run_manifest: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    specs: Mapping[str, FormalSequenceSpec],
) -> None:
    launch = _read_canonical_json(
        root / "launch_binding.json",
        label="worker-report launch binding",
    )
    launch_paths = launch.get("paths")
    run_values = run_manifest.get("worker_control_report_commitments")
    runtime_values = runtime_report.get(
        "worker_control_report_commitments"
    )
    if (
        not isinstance(run_values, Mapping)
        or not isinstance(runtime_values, Mapping)
        or dict(run_values) != dict(runtime_values)
        or set(run_values) != set(specs)
        or run_manifest.get("raw_worker_stdout_persisted") is not False
        or run_manifest.get("raw_worker_stderr_persisted") is not False
        or runtime_report.get("raw_worker_stdout_persisted") is not False
        or runtime_report.get("raw_worker_stderr_persisted") is not False
        or "worker_log_commitments" in run_manifest
        or "worker_log_commitments" in runtime_report
        or (root / "worker_logs").exists()
        or not isinstance(launch_paths, Mapping)
        or launch_paths.get("worker_control_reports")
        != "TASK_MANIFEST_COMMITTED_TASK_ARTIFACTS"
        or "worker_logs_root" in launch_paths
    ):
        raise R8CIntegrityError(
            "bounded worker-control report boundary differs"
        )
    for task_id, binding in run_values.items():
        spec = specs[task_id]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"kind", "path", "bytes", "sha256"}
        ):
            raise R8CIntegrityError(
                "worker-control report commitment schema is invalid"
            )
        kind = binding.get("kind")
        filename = _WORKER_REPORT_FILES.get(kind)
        expected_relative = (
            Path("tasks") / task_id / str(filename)
        ).as_posix()
        if (
            filename is None
            or binding.get("path") != expected_relative
            or type(binding.get("bytes")) is not int
            or not 0 < binding["bytes"] <= _WORKER_REPORT_MAX_BYTES
        ):
            raise R8CIntegrityError(
                "worker-control report path or byte bound differs"
            )
        report_path = root / expected_relative
        _validate_commitment(
            report_path,
            {
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            },
            label=f"worker-control report {task_id}",
        )
        report = _read_canonical_json(
            report_path,
            label=f"worker-control report {task_id}",
        )
        expected_keys = {
            "TASK_SUMMARY": _SUMMARY_REPORT_KEYS,
            "TASK_FAILURE": _FAILURE_REPORT_KEYS,
            "SUPERVISOR_OUTCOME": _SUPERVISOR_REPORT_KEYS,
        }[kind]
        if (
            set(report) != expected_keys
            or report.get("task") != spec.to_dict()
            or _contains_forbidden_worker_report_key(report)
        ):
            raise R8CIntegrityError(
                "worker-control report violates its blind schema"
            )
        if kind == "TASK_SUMMARY":
            if (
                report.get("status")
                not in _TASK_SUMMARY_REPORT_STATUSES
                or report.get("permissions", {}).get(
                    "results_analysis_performed"
                )
                is not False
            ):
                raise R8CIntegrityError(
                    "task-summary worker-control report is invalid"
                )
        elif (
            type(report.get("status")) is not str
            or not report.get("status")
            or type(report.get("outcome_class")) is not str
            or not report.get("outcome_class")
            or report.get("reason_code") != report.get("outcome_class")
            or (
                report.get("error_type") is not None
                and (
                    type(report.get("error_type")) is not str
                    or not report.get("error_type")
                )
            )
            or not _valid_worker_report_accounting(
                report.get("accounting"),
                spec=spec,
            )
            or report.get("attempt") != 1
            or report.get("automatic_retries") != 0
            or report.get("results_analysis_performed") is not False
        ):
            raise R8CIntegrityError(
                "failed worker-control report is invalid"
            )
        if kind == "TASK_FAILURE" and (
            report.get("task_id") != task_id
            or report.get("schedule_index") != spec.schedule_index
            or not _is_nonnegative_finite_number(
                report.get("cpu_seconds")
            )
        ):
            raise R8CIntegrityError(
                "worker task-failure report identity is invalid"
            )
        task_manifest = _read_canonical_json(
            root / "tasks" / task_id / "task_manifest.json",
            label=f"worker report task manifest {task_id}",
        )
        artifact_binding = task_manifest.get("artifacts", {}).get(filename)
        if artifact_binding != {
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
        }:
            raise R8CIntegrityError(
                "worker report is not bound by its task manifest"
            )


def validate_r8c_e1e2_run(
    run_root: Path,
    *,
    expected_run_manifest_sha256: str | None = None,
    _freeze: R8CFreeze = FORMAL_R8C_E1E2_FREEZE,
) -> R8CIntegrityReport:
    """Authenticate one full R8C E1+E2 root without computing effects.

    ``_freeze`` exists only for small synthetic unit tests.  Public tools never
    expose it and always enforce the exact 5,030-sequence formal freeze.
    """

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise R8CIntegrityError("run root is not a directory")
    manifest_path = root / "run_manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    if expected_run_manifest_sha256 is not None and (
        _SHA256_HEX.fullmatch(expected_run_manifest_sha256) is None
        or manifest_sha != expected_run_manifest_sha256
    ):
        raise R8CIntegrityError(
            "run manifest differs from the supplied raw-lock SHA-256"
        )
    manifest = _read_canonical_json(
        manifest_path,
        label="run manifest",
    )
    if (
        manifest.get("analysis_gate")
        != "R9_RAW_LOCK_AND_ANALYSIS_NOT_YET_AUTHORIZED"
        or manifest.get("schedule", {}).get("id") != _freeze.schedule_id
        or manifest.get("schedule", {}).get("sha256")
        != _freeze.schedule_sha256
        or manifest.get("schedule", {}).get("e2_full_reuse_sha256")
        != _freeze.reuse_sha256
        or manifest.get("permissions", {}).get(
            "effect_analysis"
        )
        is not False
    ):
        raise R8CIntegrityError(
            "run manifest schedule/analysis boundary differs from freeze"
        )
    commitments = manifest.get("control_artifacts")
    if not isinstance(commitments, Mapping):
        raise R8CIntegrityError(
            "run manifest control commitments are missing"
        )
    artifact_paths = _control_artifact_paths(root, commitments)
    for name, path in artifact_paths.items():
        _validate_commitment(
            path,
            commitments[name],
            label=f"control artifact {name}",
        )

    schedule_rows, schedule_raw = _read_schedule(
        root / "schedule.jsonl.gz",
        freeze=_freeze,
    )
    schedule_hash = sha256(schedule_raw).hexdigest()
    _read_reuse_map(
        root / "e2_full_reuse_map.jsonl",
        freeze=_freeze,
    )
    runtime = _read_canonical_json(
        root / "runtime_report.json",
        label="runtime report",
    )
    schedule_specs = {
        str(row["task_id"]): _spec_from_schedule_row(row)
        for row in schedule_rows
    }
    task_ids = set(schedule_specs)
    outcomes, failure_classes = _runtime_outcomes(
        runtime,
        task_ids=task_ids,
    )
    if _freeze == FORMAL_R8C_E1E2_FREEZE:
        _validate_worker_control_report_commitments(
            root,
            run_manifest=manifest,
            runtime_report=runtime,
            specs=schedule_specs,
        )
    task_commitments = manifest.get("task_manifest_commitments")
    if (
        not isinstance(task_commitments, Mapping)
        or set(task_commitments) != task_ids
    ):
        raise R8CIntegrityError(
            "run task-manifest commitments have missing/extra task IDs"
        )
    tasks_root = root / "tasks"
    if not tasks_root.is_dir():
        raise R8CIntegrityError("run tasks directory is missing")
    actual_task_directories = {
        path.name for path in tasks_root.iterdir() if path.is_dir()
    }
    if actual_task_directories != task_ids:
        raise R8CIntegrityError(
            "tasks root has missing or extra task directories"
        )
    if any(path.is_file() for path in tasks_root.iterdir()):
        raise R8CIntegrityError("tasks root contains unexpected files")

    task_integrity: list[TaskIntegrity] = []
    for row in schedule_rows:
        spec = _spec_from_schedule_row(row)
        commitment = task_commitments[spec.task_id]
        if not isinstance(commitment, str):
            raise R8CIntegrityError(
                "task-manifest commitment is not a SHA-256 string"
            )
        task_integrity.append(
            _validate_task_directory(
                tasks_root / spec.task_id,
                spec=spec,
                expected_manifest_sha256=commitment,
                outcome=outcomes[spec.task_id],
            )
        )
    statuses = Counter(item.status for item in task_integrity)
    charged_values = [
        item.charged_cfe
        for item in task_integrity
        if item.charged_cfe is not None
    ]
    completed_count = statuses.get("COMPLETE", 0)
    manifest_schedule = manifest["schedule"]
    if (
        manifest_schedule.get("method_sequences") != _freeze.task_count
        or manifest_schedule.get("recorded_outcomes") != _freeze.task_count
        or manifest_schedule.get("completed_sequences") != completed_count
        or manifest.get("resources", {}).get("automatic_retries") != 0
    ):
        raise R8CIntegrityError(
            "run manifest task counts/retry boundary differ"
        )
    if _freeze == FORMAL_R8C_E1E2_FREEZE:
        try:
            validate_failure_outcome_closure(root)
        except FailureOutcomeClosureError as error:
            raise R8CIntegrityError(
                "run-wide technical failure/cost closure differs"
            ) from error
    return R8CIntegrityReport(
        run_root=root,
        run_manifest_sha256=manifest_sha,
        run_status=str(manifest.get("status", "")),
        schedule_sha256=schedule_hash,
        scheduled_task_count=_freeze.task_count,
        authenticated_task_count=len(task_integrity),
        completed_task_count=completed_count,
        failed_task_count=len(task_integrity) - completed_count,
        authenticated_event_count=sum(
            item.event_count for item in task_integrity
        ),
        authenticated_checkpoint_record_count=sum(
            item.checkpoint_record_count for item in task_integrity
        ),
        scheduled_cfe=_freeze.total_cfe,
        authenticated_charged_cfe=sum(charged_values),
        unknown_charged_cfe_task_count=sum(
            item.charged_cfe is None for item in task_integrity
        ),
        event_summary_trailing_fragment_task_count=sum(
            item.event_summary_trailing_fragment_present
            for item in task_integrity
        ),
        event_summary_trailing_fragment_total_bytes=sum(
            item.event_summary_trailing_fragment_bytes
            for item in task_integrity
        ),
        task_status_counts=dict(statuses),
        failure_class_counts=dict(failure_classes),
        tasks=tuple(task_integrity),
    )


def _reference_index(
    derivations: Sequence[ReferenceDerivation],
) -> dict[tuple[str, str | None, int, str | None], AnalyticReferenceScale]:
    values: dict[
        tuple[str, str | None, int, str | None],
        AnalyticReferenceScale,
    ] = {}
    for derivation in derivations:
        identity = derivation.extrema.identity
        if identity.event_id is None:
            raise R8CIntegrityError(
                "reference catalog contains an event-free identity"
            )
        key = (
            identity.problem_id,
            identity.profile,
            identity.event_id,
            identity.master_seed_u64,
        )
        if key in values:
            raise R8CIntegrityError(
                "reference catalog has duplicate analysis keys"
            )
        values[key] = AnalyticReferenceScale.from_extrema(
            minima=derivation.extrema.minima,
            maxima=derivation.extrema.maxima,
        )
    return values


def _scales_for_task(
    row: Mapping[str, Any],
    index: Mapping[
        tuple[str, str | None, int, str | None],
        AnalyticReferenceScale,
    ],
) -> dict[int, AnalyticReferenceScale]:
    problem_id = row.get("problem_id")
    if not isinstance(problem_id, str):
        raise R8CIntegrityError(
            "static/dynamic schedule row lacks problem identity"
        )
    profile = row.get("profile")
    seed = row.get("master_seed_u64") if problem_id == "CDF13" else None
    values: dict[int, AnalyticReferenceScale] = {}
    for event_id in range(int(row["events"])):
        key = (
            problem_id,
            profile if isinstance(profile, str) else None,
            event_id,
            seed if isinstance(seed, str) else None,
        )
        if key not in index:
            raise R8CIntegrityError(
                f"reference scale is missing for {problem_id} event {event_id}"
            )
        values[event_id] = index[key]
    return values


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _gzip_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    raw = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    target = io.BytesIO()
    with gzip.GzipFile(
        fileobj=target,
        mode="wb",
        filename="",
        mtime=0,
        compresslevel=6,
    ) as stream:
        stream.write(raw)
    return target.getvalue()


def _e1_full_reuse_index(
    schedule_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], str]:
    index: dict[tuple[str, str, int], str] = {}
    for row in schedule_rows:
        if (
            row.get("workload_id") not in {"E1_DYNAMIC", "E1_ROLLING"}
            or row.get("method_id") != "DT-RAMDE_TS2_FULL"
        ):
            continue
        key = (
            str(row["workload_id"]),
            str(row["unit_id"]),
            int(row["replicate_index"]),
        )
        if key in index:
            raise R8CIntegrityError(
                "E1 FULL reuse identity is duplicated"
            )
        index[key] = str(row["task_id"])
    for row in schedule_rows:
        source_workload = row.get("reused_full_workload_id")
        if source_workload is None:
            continue
        key = (
            str(source_workload),
            str(row["unit_id"]),
            int(row["replicate_index"]),
        )
        if key not in index:
            raise R8CIntegrityError(
                "E2 task lacks its paired E1 FULL source"
            )
    return index


def _negative_transfer_summary(
    *,
    full_status: str,
    comparator_status: str,
    full_curves: Sequence[Sequence[float]] | None,
    comparator_curves: Sequence[Sequence[float]] | None,
    event_count: int,
) -> tuple[str, float | str, int | str, int]:
    paired_event_count = event_count - 1
    if paired_event_count <= 0:
        raise R8CIntegrityError(
            "negative-transfer pairing requires post-initial events"
        )
    if full_status != "INCLUDED":
        return (
            f"FULL_{full_status}",
            "",
            "",
            paired_event_count,
        )
    if comparator_status != "INCLUDED":
        return (
            f"COMPARATOR_{comparator_status}",
            "",
            "",
            paired_event_count,
        )
    if full_curves is None or comparator_curves is None:
        raise R8CIntegrityError(
            "included negative-transfer pair lacks event curves"
        )
    try:
        rate = negative_transfer_rate(full_curves, comparator_curves)
    except FormalMetricError as error:
        raise R8CIntegrityError(
            "negative-transfer pair metric failed"
        ) from error
    count = int(round(rate * paired_event_count))
    if not 0 <= count <= paired_event_count:
        raise R8CIntegrityError(
            "negative-transfer event count is invalid"
        )
    return "INCLUDED", rate, count, paired_event_count


def _hard_violation_summary(
    *,
    task_status: str,
    events: Sequence[Mapping[str, Any]],
    scheduled_event_count: int,
) -> tuple[str, float | str, int, int, int, int, int]:
    durably_completed = len(events)
    if task_status == "COMPLETE" and (
        durably_completed != scheduled_event_count
    ):
        raise R8CIntegrityError(
            "complete hard-violation event count differs from schedule"
        )
    executed = 0
    available = 0
    missing = 0
    violations = 0
    for event_id, event in enumerate(events):
        transition_count = event["ledger"][
            "execution_transition_count"
        ]
        try:
            observation = _validated_execution_observation(
                event.get("execution_observation"),
                event_id=event_id,
            )
        except CheckpointAnalysisError as error:
            raise R8CIntegrityError(
                "hard-violation observation failed strict validation"
            ) from error
        if transition_count == 0:
            if observation is not None and observation["available"]:
                raise R8CIntegrityError(
                    "non-executed event has an available execution observation"
                )
            continue
        if transition_count != 1:
            raise R8CIntegrityError(
                "hard-violation execution transition count is invalid"
            )
        executed += 1
        if observation is None or not observation["available"]:
            missing += 1
            continue
        available += 1
        if observation["hard_constraint_violation"]:
            violations += 1

    if task_status != "COMPLETE":
        status = "NOT_COMPUTED_TASK_FAILURE"
        rate: float | str = ""
    elif executed == 0:
        status = "NOT_COMPUTED_NO_EXECUTED_EVENTS"
        rate = ""
    elif missing:
        status = "NOT_COMPUTED_MISSING_EXECUTION_OBSERVATION"
        rate = ""
    else:
        status = "INCLUDED"
        rate = violations / executed
    return (
        status,
        rate,
        violations,
        executed,
        available,
        missing,
        durably_completed,
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except OSError as error:
        raise R8CIntegrityError(
            f"cannot write derived artifact exclusively: {path.name}"
        ) from error


def _revalidate_r9_source_bindings(
    root: Path,
    *,
    run_manifest_sha256: str,
    runtime_commitment: Mapping[str, Any],
    task_manifest_commitments: Mapping[str, str],
) -> None:
    if _sha256_file(root / "run_manifest.json") != run_manifest_sha256:
        raise R8CIntegrityError(
            "raw manifest changed after R9 validation"
        )
    _validate_commitment(
        root / "runtime_report.json",
        runtime_commitment,
        label="R9 source runtime report",
    )
    for task_id, expected_sha256 in task_manifest_commitments.items():
        if _sha256_file(
            root / "tasks" / task_id / "task_manifest.json"
        ) != expected_sha256:
            raise R8CIntegrityError(
                f"R9 source task manifest changed: {task_id}"
            )


def export_r9_readable_outputs(
    run_root: Path,
    *,
    raw_manifest_sha256: str,
    authorization: str,
    reference_catalog_path: Path,
    reference_catalog_sha256: str,
    output_root: Path,
    include_event_diagnostics: bool = False,
    _freeze: R8CFreeze = FORMAL_R8C_E1E2_FREEZE,
) -> dict[str, Any]:
    """Create compact R9 tables after an explicit raw-lock authorization."""

    if authorization != R9_EXPORT_AUTHORIZATION:
        raise R9ExportAuthorizationError(
            "exact R9 analysis authorization token is required"
        )
    if (
        _freeze == FORMAL_R8C_E1E2_FREEZE
        and reference_catalog_sha256
        != R8C_E1E2_REFERENCE_CATALOG_SHA256
    ):
        raise R8CIntegrityError(
            "reference catalog differs from the frozen formal SHA-256"
        )
    report = validate_r8c_e1e2_run(
        run_root,
        expected_run_manifest_sha256=raw_manifest_sha256,
        _freeze=_freeze,
    )
    root = report.run_root
    locked_manifest = _read_sha256_locked_canonical_json(
        root / "run_manifest.json",
        expected_sha256=report.run_manifest_sha256,
        label="R9 raw-locked run manifest",
    )
    locked_control_commitments = locked_manifest.get(
        "control_artifacts"
    )
    if not isinstance(locked_control_commitments, Mapping):
        raise R8CIntegrityError(
            "R9 raw-locked control commitments are missing"
        )
    runtime_commitment = locked_control_commitments.get(
        "runtime_report.json"
    )
    if not isinstance(runtime_commitment, Mapping):
        raise R8CIntegrityError(
            "R9 raw-locked runtime commitment is missing"
        )
    locked_task_values = locked_manifest.get(
        "task_manifest_commitments"
    )
    if not isinstance(locked_task_values, Mapping):
        raise R8CIntegrityError(
            "R9 raw-locked task commitments are missing"
        )
    task_commitments = {
        str(task_id): str(commitment)
        for task_id, commitment in locked_task_values.items()
    }
    if task_commitments != {
        task.task_id: task.task_manifest_sha256
        for task in report.tasks
    }:
        raise R8CIntegrityError(
            "R9 raw-locked task commitments differ from initial validation"
        )
    target = Path(output_root).resolve()
    if (
        target.exists()
        or target == root
        or target.is_relative_to(root)
        or root.is_relative_to(target)
    ):
        raise R8CIntegrityError(
            "R9 output root must be new and separate from immutable raw root"
        )
    try:
        derivations = load_reference_catalog(
            Path(reference_catalog_path),
            expected_sha256=reference_catalog_sha256,
            expected_lines=(
                R8C_E1E2_REFERENCE_CATALOG_LINES
                if _freeze == FORMAL_R8C_E1E2_FREEZE
                else None
            ),
        )
    except (OSError, ReferenceArtifactError) as error:
        raise R8CIntegrityError(
            "reference catalog failed strict authentication"
        ) from error
    try:
        reference_index = _reference_index(derivations)
    except FormalMetricError as error:
        raise R8CIntegrityError(
            "reference catalog contains an invalid analytic scale"
        ) from error
    schedule_rows, _ = _read_schedule(
        root / "schedule.jsonl.gz",
        freeze=_freeze,
    )
    runtime = _read_committed_canonical_json(
        root / "runtime_report.json",
        runtime_commitment,
        label="runtime report",
    )
    outcomes, _ = _runtime_outcomes(
        runtime,
        task_ids={str(row["task_id"]) for row in schedule_rows},
    )
    task_integrity = {item.task_id: item for item in report.tasks}
    full_reuse_index = _e1_full_reuse_index(schedule_rows)

    endpoint_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    negative_transfer_rows: list[dict[str, Any]] = []
    hard_violation_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    endpoint_status_by_task: dict[str, str] = {}
    curves_by_task: dict[str, Sequence[Sequence[float]]] = {}
    for row in schedule_rows:
        task_id = str(row["task_id"])
        integrity = task_integrity[task_id]
        outcome = outcomes[task_id]
        common = {
            "task_id": task_id,
            "schedule_index": row["schedule_index"],
            "workload_id": row["workload_id"],
            "unit_id": row["unit_id"],
            "method_id": row["method_id"],
            "replicate_index": row["replicate_index"],
        }
        task_directory = root / "tasks" / task_id
        task_manifest = _read_sha256_locked_canonical_json(
            task_directory / "task_manifest.json",
            expected_sha256=task_commitments[task_id],
            label=f"R9 source task manifest {task_id}",
        )
        task_artifacts = task_manifest.get("artifacts")
        if not isinstance(task_artifacts, Mapping):
            raise R8CIntegrityError(
                f"R9 source task artifacts are missing: {task_id}"
            )
        event_summary_commitment = task_artifacts.get(
            "event_summaries.jsonl"
        )
        task_events: tuple[dict[str, Any], ...] = ()
        if event_summary_commitment is not None:
            event_summary_read = _read_committed_event_summaries(
                task_directory / "event_summaries.jsonl",
                event_summary_commitment,
                spec=_spec_from_schedule_row(row),
                allow_trailing_fragment=(
                    integrity.status != "COMPLETE"
                    and integrity.outcome_class
                    in _OPAQUE_PARTIAL_CHECKPOINT_OUTCOME_CLASSES
                    and outcome.get("charged_work_exact") is False
                ),
            )
            task_events = event_summary_read.rows
            if (
                event_summary_read.trailing_fragment_present
                != integrity.event_summary_trailing_fragment_present
                or event_summary_read.trailing_fragment_bytes
                != integrity.event_summary_trailing_fragment_bytes
                or event_summary_read.trailing_fragment_sha256
                != integrity.event_summary_trailing_fragment_sha256
            ):
                raise R8CIntegrityError(
                    f"R9 event-summary tail binding differs: {task_id}"
                )
        endpoint_status = "NOT_COMPUTED_TASK_FAILURE"
        anytime: float | str = ""
        final: float | str = ""
        transfer: float | str = ""
        timeout_events = 0
        summary: Mapping[str, Any] | None = None
        decoded_task = None
        if integrity.status == "COMPLETE":
            summary = _read_committed_canonical_json(
                task_directory / "task_summary.json",
                task_artifacts["task_summary.json"],
                label=f"task summary {task_id}",
            )
            if list(task_events) != summary.get("events"):
                raise R8CIntegrityError(
                    f"R9 event-summary binding differs: {task_id}"
                )
            rolling = row["workload_id"] in {
                "E1_ROLLING",
                "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
            }
            try:
                decoded_task = read_manifest_bound_complete_task_nhv(
                    task_directory,
                    expected_task=row,
                    expected_task_manifest_sha256=(
                        task_commitments[task_id]
                    ),
                    mode="ROLLING" if rolling else "STATIC_CDF",
                    analytic_reference_scales=(
                        None
                        if rolling
                        else _scales_for_task(row, reference_index)
                    ),
                )
                endpoints = e1e2_sequence_endpoints(
                    decoded_task.nhv_by_event,
                    include_transfer=str(row["workload_id"]).startswith(
                        "E2_"
                    ),
                )
                endpoint_status = "INCLUDED"
                anytime = endpoints.anytime_nhv_auc
                final = endpoints.final_nhv
                transfer = (
                    ""
                    if endpoints.transfer_early_auc is None
                    else endpoints.transfer_early_auc
                )
                timeout_events = len(
                    decoded_task.timeout_carried_forward_event_ids
                )
            except NumericalContinuousEndpointExcluded:
                endpoint_status = "EXCLUDED_NUMERICAL_FAILURE"
            except CheckpointAnalysisError as error:
                raise R8CIntegrityError(
                    f"R9 task endpoint join failed: {task_id}"
                ) from error
            except FormalMetricError as error:
                raise R8CIntegrityError(
                    f"R9 task metric computation failed: {task_id}"
                ) from error
        endpoint_rows.append(
            {
                **common,
                "task_status": integrity.status,
                "endpoint_status": endpoint_status,
                "anytime_nhv_auc": anytime,
                "final_nhv": final,
                "transfer_early_auc": transfer,
                "timeout_carried_forward_event_count": timeout_events,
            }
        )
        endpoint_status_by_task[task_id] = endpoint_status
        if decoded_task is not None:
            curves_by_task[task_id] = decoded_task.nhv_by_event
        source_workload = row.get("reused_full_workload_id")
        if source_workload is not None:
            source_key = (
                str(source_workload),
                str(row["unit_id"]),
                int(row["replicate_index"]),
            )
            full_task_id = full_reuse_index[source_key]
            if full_task_id not in endpoint_status_by_task:
                raise R8CIntegrityError(
                    "E2 task precedes its paired E1 FULL source"
                )
            (
                pair_status,
                negative_rate,
                negative_count,
                paired_event_count,
            ) = _negative_transfer_summary(
                full_status=endpoint_status_by_task[full_task_id],
                comparator_status=endpoint_status,
                full_curves=curves_by_task.get(full_task_id),
                comparator_curves=curves_by_task.get(task_id),
                event_count=int(row["events"]),
            )
            negative_transfer_rows.append(
                {
                    "workload_id": row["workload_id"],
                    "unit_id": row["unit_id"],
                    "replicate_index": row["replicate_index"],
                    "full_task_id": full_task_id,
                    "full_method_id": "DT-RAMDE_TS2_FULL",
                    "comparator_task_id": task_id,
                    "comparator_method_id": row["method_id"],
                    "pair_status": pair_status,
                    "negative_transfer_rate": negative_rate,
                    "negative_transfer_event_count": negative_count,
                    "paired_post_initial_event_count": (
                        paired_event_count
                    ),
                    "strict_difference_threshold": -0.01,
                }
            )

        scheduled_cfe = int(row["total_cfe"])
        charged_cfe = integrity.charged_cfe
        evaluation_failures = integrity.evaluation_failure_count
        terminal_failure_count = 0
        terminal_code_counts: Counter[str] = Counter()
        for event in task_events:
            terminal_code = str(event["terminal"]["code"])
            terminal_code_counts[terminal_code] += 1
            if terminal_code != "ACCEPTED":
                terminal_failure_count += 1
        cost_rows.append(
            {
                **common,
                "task_status": integrity.status,
                "outcome_class": integrity.outcome_class or "",
                "terminal_failure_event_count": terminal_failure_count,
                "terminal_code_counts_json": json.dumps(
                    dict(sorted(terminal_code_counts.items())),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evaluation_failure_count": (
                    "" if evaluation_failures is None else evaluation_failures
                ),
                "scheduled_cfe": scheduled_cfe,
                "charged_cfe": (
                    "" if charged_cfe is None else charged_cfe
                ),
                "unconsumed_cfe": (
                    ""
                    if charged_cfe is None
                    else scheduled_cfe - charged_cfe
                ),
                "scheduled_atomic_model_steps": row[
                    "total_atomic_steps"
                ],
                "charged_atomic_model_steps": (
                    ""
                    if charged_cfe is None
                    else charged_cfe * int(row["atomic_steps_per_cfe"])
                ),
                "charged_work_exact": (
                    charged_cfe is not None
                    and outcome.get("charged_work_exact", True) is not False
                ),
                "event_summary_trailing_fragment_present": (
                    integrity.event_summary_trailing_fragment_present
                ),
                "event_summary_trailing_fragment_bytes": (
                    integrity.event_summary_trailing_fragment_bytes
                ),
                "event_summary_trailing_fragment_sha256": (
                    integrity.event_summary_trailing_fragment_sha256 or ""
                ),
                "wall_seconds": outcome.get("wall_seconds", ""),
                "cpu_seconds": outcome.get("cpu_seconds", ""),
                "peak_rss_bytes": outcome.get("peak_rss_bytes", ""),
                "output_bytes": outcome.get("output_bytes", ""),
                "automatic_retries": outcome.get(
                    "automatic_retries",
                    0,
                ),
            }
        )
        (
            hard_status,
            hard_rate,
            hard_count,
            executed_count,
            observation_available_count,
            observation_missing_count,
            durably_completed_event_count,
        ) = _hard_violation_summary(
            task_status=integrity.status,
            events=task_events,
            scheduled_event_count=int(row["events"]),
        )
        hard_violation_rows.append(
            {
                **common,
                "task_status": integrity.status,
                "endpoint_status": hard_status,
                "post_execution_hard_violation_rate": hard_rate,
                "hard_violation_event_count": hard_count,
                "executed_event_count": executed_count,
                "execution_observation_available_event_count": (
                    observation_available_count
                ),
                "execution_observation_missing_event_count": (
                    observation_missing_count
                ),
                "durably_completed_event_count": (
                    durably_completed_event_count
                ),
            }
        )
        if include_event_diagnostics and task_events:
            timeout_ids = (
                set(decoded_task.timeout_carried_forward_event_ids)
                if decoded_task is not None
                else set()
            )
            for event_id, event in enumerate(task_events):
                curve = (
                    decoded_task.nhv_by_event[event_id]
                    if decoded_task is not None
                    else None
                )
                observation = (
                    decoded_task.execution_observations[event_id]
                    if decoded_task is not None
                    else event.get("execution_observation")
                )
                diagnostic_rows.append(
                    {
                        **common,
                        "event_id": event_id,
                        "terminal_code": event["terminal"]["code"],
                        "charged_cfe": event["ledger"]["cfe"],
                        "evaluation_failure_count": event["ledger"][
                            "evaluation_failures"
                        ],
                        "anytime_nhv_auc": (
                            event_anytime_auc(curve)
                            if curve is not None
                            else None
                        ),
                        "final_nhv": (
                            curve[-1] if curve is not None else None
                        ),
                        "early_nhv_auc": (
                            event_early_auc(curve)
                            if curve is not None
                            else None
                        ),
                        "timeout_carried_forward": event_id in timeout_ids,
                        "execution_observation": observation,
                    }
                )

    if (
        len(endpoint_rows) != _freeze.task_count
        or len(cost_rows) != _freeze.task_count
        or len(hard_violation_rows) != _freeze.task_count
    ):
        raise R8CIntegrityError(
            "derived table cardinality differs from frozen schedule"
        )
    if (
        _freeze == FORMAL_R8C_E1E2_FREEZE
        and len(negative_transfer_rows)
        != R8C_E1E2_NEGATIVE_TRANSFER_PAIR_COUNT
    ):
        raise R8CIntegrityError(
            "negative-transfer pair count differs from formal freeze"
        )
    endpoint_fields = (
        "task_id",
        "schedule_index",
        "workload_id",
        "unit_id",
        "method_id",
        "replicate_index",
        "task_status",
        "endpoint_status",
        "anytime_nhv_auc",
        "final_nhv",
        "transfer_early_auc",
        "timeout_carried_forward_event_count",
    )
    cost_fields = (
        "task_id",
        "schedule_index",
        "workload_id",
        "unit_id",
        "method_id",
        "replicate_index",
        "task_status",
        "outcome_class",
        "terminal_failure_event_count",
        "terminal_code_counts_json",
        "evaluation_failure_count",
        "scheduled_cfe",
        "charged_cfe",
        "unconsumed_cfe",
        "scheduled_atomic_model_steps",
        "charged_atomic_model_steps",
        "charged_work_exact",
        "event_summary_trailing_fragment_present",
        "event_summary_trailing_fragment_bytes",
        "event_summary_trailing_fragment_sha256",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "output_bytes",
        "automatic_retries",
    )
    negative_transfer_fields = (
        "workload_id",
        "unit_id",
        "replicate_index",
        "full_task_id",
        "full_method_id",
        "comparator_task_id",
        "comparator_method_id",
        "pair_status",
        "negative_transfer_rate",
        "negative_transfer_event_count",
        "paired_post_initial_event_count",
        "strict_difference_threshold",
    )
    hard_violation_fields = (
        "task_id",
        "schedule_index",
        "workload_id",
        "unit_id",
        "method_id",
        "replicate_index",
        "task_status",
        "endpoint_status",
        "post_execution_hard_violation_rate",
        "hard_violation_event_count",
        "executed_event_count",
        "execution_observation_available_event_count",
        "execution_observation_missing_event_count",
        "durably_completed_event_count",
    )
    payloads: dict[str, bytes] = {
        "README.md": _R9_README.encode("utf-8"),
        "task_endpoints.csv": _csv_bytes(endpoint_fields, endpoint_rows),
        "failure_cost.csv": _csv_bytes(cost_fields, cost_rows),
        "e2_negative_transfer.csv": _csv_bytes(
            negative_transfer_fields,
            negative_transfer_rows,
        ),
        "post_execution_hard_violation.csv": _csv_bytes(
            hard_violation_fields,
            hard_violation_rows,
        ),
    }
    if include_event_diagnostics:
        payloads["event_diagnostics.jsonl.gz"] = _gzip_jsonl_bytes(
            diagnostic_rows
        )
    _revalidate_r9_source_bindings(
        root,
        run_manifest_sha256=report.run_manifest_sha256,
        runtime_commitment=runtime_commitment,
        task_manifest_commitments=task_commitments,
    )
    try:
        target.mkdir(parents=False, exist_ok=False)
    except OSError as error:
        raise R8CIntegrityError(
            "R9 output root cannot be created exclusively"
        ) from error
    for name, payload in payloads.items():
        _write_exclusive(target / name, payload)
    derived_commitments = {
        name: {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    publication_report = validate_r8c_e1e2_run(
        root,
        expected_run_manifest_sha256=report.run_manifest_sha256,
        _freeze=_freeze,
    )
    if (
        publication_report.run_manifest_sha256
        != report.run_manifest_sha256
        or publication_report.tasks != report.tasks
    ):
        raise R8CIntegrityError(
            "raw integrity changed before R9 manifest publication"
        )
    export_manifest = {
        "artifact_role": "R9_AUTHORIZED_COMPACT_HUMAN_READABLE_DERIVATION",
        "authorization": R9_EXPORT_AUTHORIZATION,
        "raw_run_manifest_sha256": report.run_manifest_sha256,
        "raw_lock_revalidated_immediately_before_publication": True,
        "runtime_and_task_manifest_commitments_revalidated": True,
        "all_raw_task_artifacts_reaudited_after_derived_write": True,
        "raw_source_mutated_or_deleted": False,
        "schedule_sha256": report.schedule_sha256,
        "reference_catalog_sha256": reference_catalog_sha256,
        "task_endpoints_rows": len(endpoint_rows),
        "failure_cost_rows": len(cost_rows),
        "e2_negative_transfer_rows": len(negative_transfer_rows),
        "post_execution_hard_violation_rows": len(
            hard_violation_rows
        ),
        "event_summary_trailing_fragment_task_count": (
            report.event_summary_trailing_fragment_task_count
        ),
        "event_summary_trailing_fragment_total_bytes": (
            report.event_summary_trailing_fragment_total_bytes
        ),
        "event_diagnostics_generated": include_event_diagnostics,
        "event_diagnostics_rows": (
            len(diagnostic_rows) if include_event_diagnostics else 0
        ),
        "artifacts": derived_commitments,
    }
    manifest_payload = canonical_json_bytes(export_manifest) + b"\n"
    _write_exclusive(target / "r9_export_manifest.json", manifest_payload)
    return {
        **export_manifest,
        "r9_export_manifest_sha256": sha256(manifest_payload).hexdigest(),
        "output_root": str(target),
    }
