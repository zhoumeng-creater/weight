"""Strict R9 consumer for complete result-blind checkpoint files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from formal_execution.checkpoint_data import (
    CHECKPOINTS_PER_EVENT,
    CheckpointFile,
    CheckpointDataError,
    CheckpointRecord,
    read_checkpoint_file,
)
from formal_execution.schedule import canonical_json_bytes

from .checkpoint_metrics import (
    AnalyticReferenceScale,
    rolling_nhv,
    static_cdf_nhv,
)


class CheckpointAnalysisError(ValueError):
    """A checkpoint file cannot be interpreted under the frozen endpoints."""


class IncompleteCheckpointDataError(CheckpointAnalysisError):
    """A partial event needs external frozen terminal-status handling."""


class NumericalContinuousEndpointExcluded(CheckpointAnalysisError):
    """R5 excludes a numerically failed run from continuous VAS endpoints."""


@dataclass(frozen=True)
class TaskCheckpointNHV:
    """Complete per-event nHV curves decoded from one immutable task file."""

    task_id: str
    objective_names: tuple[str, ...]
    event_ids: tuple[int, ...]
    nhv_by_event: tuple[tuple[float, ...], ...]
    checkpoint_file_sha256: str
    task_summary_sha256: str | None = None
    task_manifest_sha256: str | None = None
    timeout_carried_forward_event_ids: tuple[int, ...] = ()
    execution_observations: tuple[Mapping[str, Any] | None, ...] = ()


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _canonical_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CheckpointAnalysisError(f"{label} cannot be read") from error
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointAnalysisError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CheckpointAnalysisError(f"{label} must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise CheckpointAnalysisError(f"{label} is not canonical JSON")
    return value


def _validate_artifact_commitment(
    *,
    path: Path,
    commitment: Any,
    observed_sha256: str | None = None,
) -> str:
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != {"bytes", "sha256"}
        or type(commitment["bytes"]) is not int
        or commitment["bytes"] < 0
        or not isinstance(commitment["sha256"], str)
        or _SHA256_HEX.fullmatch(commitment["sha256"]) is None
    ):
        raise CheckpointAnalysisError(
            f"artifact commitment is invalid: {path.name}"
        )
    if not path.is_file() or path.stat().st_size != commitment["bytes"]:
        raise CheckpointAnalysisError(
            f"artifact size differs from manifest: {path.name}"
        )
    digest = (
        observed_sha256
        if observed_sha256 is not None
        else sha256(path.read_bytes()).hexdigest()
    )
    if digest != commitment["sha256"]:
        raise CheckpointAnalysisError(
            f"artifact SHA-256 differs from manifest: {path.name}"
        )
    return digest


def _task_nhv_from_decoded(
    decoded: CheckpointFile,
    *,
    mode: Literal["STATIC_CDF", "ROLLING"],
    analytic_reference_scales: Mapping[int, AnalyticReferenceScale] | None,
    expected_event_count: int | None,
    terminal_codes: Mapping[int, str] | None = None,
    execution_observations: tuple[
        Mapping[str, Any] | None,
        ...,
    ] = (),
) -> TaskCheckpointNHV:
    """Compute task curves from one already authenticated checkpoint file."""

    checkpoint_records: dict[int, list[CheckpointRecord]] = {}
    event_order: list[int] = []
    for record in decoded.records:
        if record.event_id not in checkpoint_records:
            checkpoint_records[record.event_id] = []
            event_order.append(record.event_id)
        if record.kind == "checkpoint":
            checkpoint_records[record.event_id].append(record)

    if not event_order:
        raise CheckpointAnalysisError("checkpoint file contains no events")
    if expected_event_count is not None:
        if expected_event_count <= 0:
            raise CheckpointAnalysisError(
                "expected_event_count must be positive"
            )
        expected_ids = tuple(range(expected_event_count))
        if tuple(event_order) != expected_ids:
            raise CheckpointAnalysisError(
                "event IDs differ from the expected zero-based sequence"
            )

    if mode == "STATIC_CDF":
        if analytic_reference_scales is None:
            raise CheckpointAnalysisError(
                "STATIC_CDF analysis requires independent analytic scales"
            )
        if set(analytic_reference_scales) != set(event_order):
            raise CheckpointAnalysisError(
                "analytic scale event IDs must exactly match checkpoint events"
            )
    elif mode == "ROLLING":
        if analytic_reference_scales is not None:
            raise CheckpointAnalysisError(
                "ROLLING analysis must not receive analytic reference scales"
            )
        if len(decoded.metadata.objective_names) != 3:
            raise CheckpointAnalysisError(
                "ROLLING checkpoint data must have three objectives"
            )
    else:
        raise CheckpointAnalysisError(f"unknown nHV mode {mode!r}")

    curves: list[tuple[float, ...]] = []
    timeout_carried_forward: list[int] = []
    failed_event_ids = {
        record.event_id
        for record in decoded.records
        if record.failure_count > 0
    }
    if execution_observations and len(execution_observations) != len(
        event_order
    ):
        raise CheckpointAnalysisError(
            "execution observation count differs from checkpoint events"
        )
    for event_id in event_order:
        records = checkpoint_records[event_id]
        indexes = tuple(record.checkpoint_index for record in records)
        terminal_code = (
            None if terminal_codes is None else terminal_codes[event_id]
        )
        if terminal_code == "REJECT_NUMERICAL":
            raise NumericalContinuousEndpointExcluded(
                f"event {event_id} has a numerical terminal"
            )
        if event_id in failed_event_ids:
            raise NumericalContinuousEndpointExcluded(
                f"event {event_id} has charged numerical evaluation failures"
            )
        fronts = [record.front_objectives for record in records]
        if indexes != tuple(range(CHECKPOINTS_PER_EVENT)):
            if terminal_code != "REJECT_TIMEOUT":
                raise IncompleteCheckpointDataError(
                    f"event {event_id} does not contain all 21 checkpoints"
                )
            terminal_records = [
                record
                for record in decoded.records
                if record.event_id == event_id
                and record.kind == "terminal"
            ]
            if (
                indexes != tuple(range(len(indexes)))
                or len(terminal_records) != 1
            ):
                raise CheckpointAnalysisError(
                    f"event {event_id} timeout evidence is not consecutive"
                )
            fronts.extend(
                [terminal_records[0].front_objectives]
                * (CHECKPOINTS_PER_EVENT - len(fronts))
            )
            timeout_carried_forward.append(event_id)
        budgets = {record.cfe_budget for record in records}
        if len(budgets) != 1:
            raise CheckpointAnalysisError(
                f"event {event_id} changes CFE budget across checkpoints"
            )
        if mode == "STATIC_CDF":
            scale = analytic_reference_scales[event_id]
            if (
                scale.objective_dimension
                != len(decoded.metadata.objective_names)
            ):
                raise CheckpointAnalysisError(
                    f"event {event_id} analytic scale dimension differs"
                )
            curve = tuple(
                static_cdf_nhv(front, scale) for front in fronts
            )
        else:
            curve = tuple(rolling_nhv(front) for front in fronts)
        curves.append(curve)

    return TaskCheckpointNHV(
        task_id=decoded.metadata.task_id,
        objective_names=decoded.metadata.objective_names,
        event_ids=tuple(event_order),
        nhv_by_event=tuple(curves),
        checkpoint_file_sha256=decoded.sha256,
        timeout_carried_forward_event_ids=tuple(
            timeout_carried_forward
        ),
        execution_observations=tuple(
            None if value is None else dict(value)
            for value in execution_observations
        ),
    )


_EXECUTION_OBSERVATION_FIELDS = frozenset(
    {
        "available",
        "ell_exec",
        "ell_ref",
        "s_exec",
        "hard_constraint_violation",
        "released_at",
    }
)
_PUBLIC_MISSING_EXECUTION_OBSERVATION_FIELDS = (
    _EXECUTION_OBSERVATION_FIELDS | {"reason"}
)


def _validated_execution_observation(
    value: Any,
    *,
    event_id: int,
) -> dict[str, Any] | None:
    """Validate the result-observation channel independently of feedback."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CheckpointAnalysisError(
            "task execution observation must be an object or null"
        )
    fields = frozenset(value)
    if fields not in {
        _EXECUTION_OBSERVATION_FIELDS,
        _PUBLIC_MISSING_EXECUTION_OBSERVATION_FIELDS,
    }:
        raise CheckpointAnalysisError(
            "task execution observation fields differ from the frozen schema"
        )
    if (
        type(value["available"]) is not bool
        or type(value["released_at"]) is not int
        or value["released_at"] != event_id + 1
    ):
        raise CheckpointAnalysisError(
            "task execution observation timing is invalid"
        )
    if value["available"]:
        if fields != _EXECUTION_OBSERVATION_FIELDS:
            raise CheckpointAnalysisError(
                "available task execution observation has extra fields"
            )
        if any(
            isinstance(value[field], bool)
            or not isinstance(value[field], (int, float))
            or not math.isfinite(float(value[field]))
            for field in ("ell_exec", "ell_ref", "s_exec")
        ):
            raise CheckpointAnalysisError(
                "task execution observation values are invalid"
            )
        if (
            float(value["s_exec"]) <= 0.0
            or type(value["hard_constraint_violation"]) is not bool
        ):
            raise CheckpointAnalysisError(
                "task execution observation values are invalid"
            )
    elif any(
        value[field] is not None
        for field in (
            "ell_exec",
            "ell_ref",
            "s_exec",
            "hard_constraint_violation",
        )
    ):
        raise CheckpointAnalysisError(
            "unavailable task execution observation is invalid"
        )
    elif "reason" in value and value["reason"] != (
        "MISSING_BY_DESIGN_PUBLIC_BENCHMARK"
    ):
        raise CheckpointAnalysisError(
            "unavailable task execution observation reason is invalid"
        )
    return dict(value)


def read_complete_task_nhv(
    path: Path,
    *,
    mode: Literal["STATIC_CDF", "ROLLING"],
    analytic_reference_scales: Mapping[int, AnalyticReferenceScale] | None = None,
    expected_event_count: int | None = None,
) -> TaskCheckpointNHV:
    """Decode complete 21-point event curves without inferring missing values.

    This is the low-level decoder used by known-answer tests. Formal R9 must
    use :func:`read_manifest_bound_complete_task_nhv` so that the checkpoint,
    task summary, schedule row and run-manifest commitment are joined first.
    Partial terminal events deliberately fail closed here.
    """

    try:
        decoded = read_checkpoint_file(Path(path))
    except CheckpointDataError as error:
        raise CheckpointAnalysisError(
            f"checkpoint file failed strict decoding: {error}"
        ) from error
    return _task_nhv_from_decoded(
        decoded,
        mode=mode,
        analytic_reference_scales=analytic_reference_scales,
        expected_event_count=expected_event_count,
        terminal_codes=None,
    )


def read_manifest_bound_complete_task_nhv(
    task_directory: Path,
    *,
    expected_task: Mapping[str, Any],
    expected_task_manifest_sha256: str,
    mode: Literal["STATIC_CDF", "ROLLING"],
    analytic_reference_scales: Mapping[int, AnalyticReferenceScale] | None = None,
) -> TaskCheckpointNHV:
    """Authenticate one complete task against schedule and run commitments."""

    directory = Path(task_directory)
    if (
        not directory.is_dir()
        or _SHA256_HEX.fullmatch(expected_task_manifest_sha256) is None
    ):
        raise CheckpointAnalysisError(
            "task directory or expected manifest SHA-256 is invalid"
        )
    expected_files = {
        "checkpoint_fronts.cfe",
        "event_summaries.jsonl",
        "task_summary.json",
        "task_manifest.json",
    }
    if {path.name for path in directory.iterdir()} != expected_files:
        raise CheckpointAnalysisError(
            "completed task directory contains an unexpected file set"
        )

    manifest_path = directory / "task_manifest.json"
    summary_path = directory / "task_summary.json"
    event_summaries_path = directory / "event_summaries.jsonl"
    checkpoint_path = directory / "checkpoint_fronts.cfe"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_task_manifest_sha256:
        raise CheckpointAnalysisError(
            "task manifest differs from the run-manifest commitment"
        )
    manifest = _canonical_json_object(
        manifest_path,
        label="task manifest",
    )
    if set(manifest) != {
        "task_id",
        "status",
        "artifacts",
        "task_binding_sha256",
    }:
        raise CheckpointAnalysisError("task manifest schema is invalid")
    if manifest["status"] != "COMPLETE":
        raise CheckpointAnalysisError("task manifest is not complete")
    task_id = expected_task.get("task_id")
    if not isinstance(task_id, str) or manifest["task_id"] != task_id:
        raise CheckpointAnalysisError(
            "task identity differs from the frozen schedule"
        )
    artifacts = manifest["artifacts"]
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "checkpoint_fronts.cfe",
            "event_summaries.jsonl",
            "task_summary.json",
        }
    ):
        raise CheckpointAnalysisError(
            "task manifest artifact set is invalid"
        )

    summary = _canonical_json_object(summary_path, label="task summary")
    summary_sha256 = _validate_artifact_commitment(
        path=summary_path,
        commitment=artifacts["task_summary.json"],
    )
    _validate_artifact_commitment(
        path=event_summaries_path,
        commitment=artifacts["event_summaries.jsonl"],
    )
    try:
        event_summary_raw = event_summaries_path.read_bytes()
    except OSError as error:
        raise CheckpointAnalysisError(
            "event summaries cannot be read"
        ) from error
    if not event_summary_raw.endswith(b"\n"):
        raise CheckpointAnalysisError(
            "event summaries lack their final LF"
        )
    event_summaries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        event_summary_raw.splitlines(),
        start=1,
    ):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointAnalysisError(
                f"event summary line {line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise CheckpointAnalysisError(
                f"event summary line {line_number} is not canonical"
            )
        event_summaries.append(value)
    try:
        decoded = read_checkpoint_file(checkpoint_path)
    except CheckpointDataError as error:
        raise CheckpointAnalysisError(
            f"checkpoint file failed strict decoding: {error}"
        ) from error
    _validate_artifact_commitment(
        path=checkpoint_path,
        commitment=artifacts["checkpoint_fronts.cfe"],
        observed_sha256=decoded.sha256,
    )

    if decoded.metadata.task_id != task_id or summary.get("task") != dict(
        expected_task
    ):
        raise CheckpointAnalysisError(
            "checkpoint/summary task identity differs from schedule"
        )
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("artifact_role")
        != "R8C_E1E2_IMMUTABLE_ENDPOINT_SUFFICIENT_UNANALYZED"
        or summary.get("individual_evaluation_rows_persisted") != 0
        or summary.get("permissions", {}).get(
            "results_analysis_performed"
        )
        is not False
    ):
        raise CheckpointAnalysisError(
            "task summary role/status/permission boundary is invalid"
        )
    expected_binding = sha256(
        canonical_json_bytes(
            {
                "task": dict(expected_task),
                "artifacts": dict(artifacts),
            }
        )
    ).hexdigest()
    if manifest["task_binding_sha256"] != expected_binding:
        raise CheckpointAnalysisError("task binding SHA-256 differs")

    events = summary.get("events")
    if not isinstance(events, list) or len(events) != expected_task.get(
        "events"
    ):
        raise CheckpointAnalysisError(
            "task summary event count differs from schedule"
        )
    if event_summaries != events:
        raise CheckpointAnalysisError(
            "append-only event summaries differ from task summary events"
        )
    final_records: dict[int, CheckpointRecord] = {}
    for record in decoded.records:
        final_records[record.event_id] = record
    if set(final_records) != set(range(len(events))):
        raise CheckpointAnalysisError(
            "checkpoint and summary event identities differ"
        )
    expected_cfe_per_event = expected_task.get("cfe_per_event")
    if (
        type(expected_cfe_per_event) is not int
        or expected_cfe_per_event <= 0
        or any(
            record.cfe_budget != expected_cfe_per_event
            for record in decoded.records
        )
    ):
        raise CheckpointAnalysisError(
            "checkpoint CFE budget differs from the frozen schedule"
        )
    charged_cfe = 0
    execution_observations: list[dict[str, Any] | None] = []
    for event_id, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("event_id") != event_id:
            raise CheckpointAnalysisError(
                "task summary event ordering is invalid"
            )
        ledger = event.get("ledger")
        if not isinstance(ledger, Mapping):
            raise CheckpointAnalysisError("task event ledger is missing")
        failure_counts = event.get("evaluation_failure_type_counts")
        evaluation_failures = ledger.get("evaluation_failures")
        if (
            type(evaluation_failures) is not int
            or evaluation_failures < 0
            or not isinstance(failure_counts, Mapping)
            or any(
                not isinstance(failure_type, str)
                or not failure_type
                or type(count) is not int
                or count < 1
                for failure_type, count in failure_counts.items()
            )
            or sum(failure_counts.values())
            != evaluation_failures
        ):
            raise CheckpointAnalysisError(
                "task event failure-type counts differ from ledger"
            )
        terminal = event.get("terminal")
        if (
            not isinstance(terminal, Mapping)
            or set(terminal)
            != {"code", "reason", "candidate_available"}
            or type(terminal["candidate_available"]) is not bool
        ):
            raise CheckpointAnalysisError(
                "compact task terminal schema is invalid"
            )
        if "execution_observation" not in event:
            raise CheckpointAnalysisError(
                "task execution observation is missing"
            )
        execution_observations.append(
            _validated_execution_observation(
                event["execution_observation"],
                event_id=event_id,
            )
        )
        record = final_records[event_id]
        if (
            record.cfe != ledger.get("cfe")
            or record.failure_count != ledger.get("evaluation_failures")
        ):
            raise CheckpointAnalysisError(
                "checkpoint terminal counts differ from event ledger"
            )
        charged_cfe += record.cfe
    if (
        summary.get("charged_evaluation_count") != charged_cfe
        or summary.get("total_cfe") != charged_cfe
    ):
        raise CheckpointAnalysisError(
            "task charged-CFE totals differ across artifacts"
        )

    terminal_codes: dict[int, str] = {}
    for event_id, event in enumerate(events):
        terminal = event.get("terminal")
        terminal_code = (
            terminal.get("code")
            if isinstance(terminal, Mapping)
            else None
        )
        if not isinstance(terminal_code, str) or not terminal_code:
            raise CheckpointAnalysisError(
                "task event terminal code is missing"
            )
        terminal_codes[event_id] = terminal_code

    result = _task_nhv_from_decoded(
        decoded,
        mode=mode,
        analytic_reference_scales=analytic_reference_scales,
        expected_event_count=int(expected_task["events"]),
        terminal_codes=terminal_codes,
        execution_observations=tuple(execution_observations),
    )
    return replace(
        result,
        task_summary_sha256=summary_sha256,
        task_manifest_sha256=manifest_sha256,
    )
