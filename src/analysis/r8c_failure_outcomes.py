"""Strict result-blind validation of R8C task failure and cost outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from formal_execution.schedule import (
    FormalSequenceSpec,
    schedule_commitment,
)


TECHNICAL_OUTCOME_CLASSES = frozenset(
    {
        "TECHNICAL_SEQUENCE_TIMEOUT",
        "TECHNICAL_GLOBAL_TIMEOUT",
        "TECHNICAL_RESOURCE_TERMINATION",
        "TECHNICAL_WORKER_LAUNCH_FAILURE",
        "TECHNICAL_NOT_DISPATCHED",
    }
)
ALLOWED_FAILURE_OUTCOME_CLASSES = (
    TECHNICAL_OUTCOME_CLASSES | {"TASK_EXECUTION_FAILURE"}
)


class FailureOutcomeClosureError(RuntimeError):
    """The run-wide failure/cost evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class FailureCostOutcome:
    task_id: str
    schedule_index: int
    status: str
    outcome_class: str
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    output_bytes: int
    scheduled_cfe: int
    charged_cfe: int | None
    scheduled_atomic_model_steps: int
    charged_atomic_model_steps: int | None
    charged_work_exact: bool
    task_manifest_sha256: str

    @property
    def is_technical_failure(self) -> bool:
        return self.outcome_class in TECHNICAL_OUTCOME_CLASSES


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FailureOutcomeClosureError(
            f"{path.name} is not readable canonical JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise FailureOutcomeClosureError(
            f"{path.name} must contain a JSON object"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FailureOutcomeClosureError(f"{label} is not a SHA-256")
    return value


def _require_nonnegative_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise FailureOutcomeClosureError(
            f"{label} must be finite and nonnegative"
        )
    return float(value)


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FailureOutcomeClosureError(
            f"{label} must be a nonnegative integer"
        )
    return value


def _load_schedule(
    output_root: Path,
    run_manifest: Mapping[str, Any],
) -> tuple[Any, ...]:
    path = output_root / "schedule.jsonl.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [
                FormalSequenceSpec(
                    **{
                        key: value
                        for key, value in json.loads(line).items()
                        if key in FormalSequenceSpec.__dataclass_fields__
                    }
                )
                for line in stream
                if line.strip()
            ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise FailureOutcomeClosureError(
            "formal schedule could not be decoded"
        ) from error
    if not rows:
        raise FailureOutcomeClosureError("formal schedule is empty")
    if any(row.schedule_index != index for index, row in enumerate(rows)):
        raise FailureOutcomeClosureError(
            "formal schedule indexes are not consecutive"
        )
    task_ids = [row.task_id for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise FailureOutcomeClosureError(
            "formal schedule task IDs are not unique"
        )
    schedule_section = run_manifest.get("schedule")
    if (
        not isinstance(schedule_section, Mapping)
        or schedule_section.get("method_sequences") != len(rows)
        or schedule_section.get("sha256") != schedule_commitment(rows)
    ):
        raise FailureOutcomeClosureError(
            "run manifest schedule commitment differs"
        )
    return tuple(rows)


def _validate_control_artifact(
    output_root: Path,
    run_manifest: Mapping[str, Any],
    name: str,
) -> Path:
    bindings = run_manifest.get("control_artifacts")
    binding = bindings.get(name) if isinstance(bindings, Mapping) else None
    path = output_root / name
    if (
        not isinstance(binding, Mapping)
        or not path.is_file()
        or binding.get("bytes") != path.stat().st_size
        or binding.get("sha256") != _file_sha256(path)
    ):
        raise FailureOutcomeClosureError(
            f"{name} control commitment differs"
        )
    return path


def _validate_task_manifest(
    output_root: Path,
    *,
    task_id: str,
    expected_sha256: str,
    expected_status: str,
) -> None:
    if Path(task_id).name != task_id or "/" in task_id or "\\" in task_id:
        raise FailureOutcomeClosureError("task ID is unsafe")
    task_directory = output_root / "tasks" / task_id
    manifest_path = task_directory / "task_manifest.json"
    if (
        not manifest_path.is_file()
        or _file_sha256(manifest_path) != expected_sha256
    ):
        raise FailureOutcomeClosureError(
            f"task manifest commitment differs for {task_id}"
        )
    manifest = _read_object(manifest_path)
    if (
        manifest.get("task_id") != task_id
        or manifest.get("status") != expected_status
    ):
        raise FailureOutcomeClosureError(
            f"task manifest identity/status differs for {task_id}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FailureOutcomeClosureError(
            f"task artifacts are missing for {task_id}"
        )
    actual_names = {
        path.name
        for path in task_directory.iterdir()
        if path.is_file()
        and path.name not in {"task_manifest.json", "heartbeat"}
    }
    if set(artifacts) != actual_names:
        raise FailureOutcomeClosureError(
            f"task artifact set differs for {task_id}"
        )
    if (task_directory / "heartbeat").exists():
        raise FailureOutcomeClosureError(
            f"stale task heartbeat remains for {task_id}"
        )
    for name, binding in artifacts.items():
        artifact_path = task_directory / str(name)
        if (
            Path(str(name)).name != name
            or not isinstance(binding, Mapping)
            or not artifact_path.is_file()
            or binding.get("bytes") != artifact_path.stat().st_size
            or binding.get("sha256") != _file_sha256(artifact_path)
        ):
            raise FailureOutcomeClosureError(
                f"task artifact commitment differs for {task_id}/{name}"
            )


def _failure_cost_outcome(
    row: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_schedule_index: int,
) -> FailureCostOutcome:
    task_id = row.get("task_id")
    schedule_index = row.get("schedule_index")
    status = row.get("status")
    outcome_class = row.get("outcome_class")
    if task_id != expected_task_id or schedule_index != expected_schedule_index:
        raise FailureOutcomeClosureError(
            "failure row task identity differs from the schedule"
        )
    if type(status) is not str or not status or status == "COMPLETE":
        raise FailureOutcomeClosureError(
            f"failure status is invalid for {expected_task_id}"
        )
    if outcome_class not in ALLOWED_FAILURE_OUTCOME_CLASSES:
        raise FailureOutcomeClosureError(
            f"failure class is invalid for {expected_task_id}"
        )
    if (
        row.get("attempt") != 1
        or row.get("automatic_retries") != 0
        or row.get("algorithm_terminal_code") is not None
    ):
        raise FailureOutcomeClosureError(
            f"retry/terminal isolation differs for {expected_task_id}"
        )
    wall_seconds = _require_nonnegative_number(
        row.get("wall_seconds"),
        label=f"{expected_task_id} wall_seconds",
    )
    cpu_seconds = _require_nonnegative_number(
        row.get("cpu_seconds"),
        label=f"{expected_task_id} cpu_seconds",
    )
    peak_rss = _require_nonnegative_int(
        row.get("peak_rss_bytes"),
        label=f"{expected_task_id} peak_rss_bytes",
    )
    output_bytes = _require_nonnegative_int(
        row.get("output_bytes"),
        label=f"{expected_task_id} output_bytes",
    )
    scheduled_cfe = _require_nonnegative_int(
        row.get("scheduled_cfe"),
        label=f"{expected_task_id} scheduled_cfe",
    )
    scheduled_atomic = _require_nonnegative_int(
        row.get("scheduled_atomic_model_steps"),
        label=f"{expected_task_id} scheduled atomic steps",
    )
    if scheduled_cfe < 1 or scheduled_atomic < 1:
        raise FailureOutcomeClosureError(
            f"scheduled work is empty for {expected_task_id}"
        )
    exact = row.get("charged_work_exact")
    if type(exact) is not bool:
        raise FailureOutcomeClosureError(
            f"charged-work exactness is missing for {expected_task_id}"
        )
    charged_cfe_value = row.get("charged_cfe")
    charged_atomic_value = row.get("charged_atomic_model_steps")
    if exact:
        charged_cfe = _require_nonnegative_int(
            charged_cfe_value,
            label=f"{expected_task_id} charged_cfe",
        )
        charged_atomic = _require_nonnegative_int(
            charged_atomic_value,
            label=f"{expected_task_id} charged atomic steps",
        )
        if charged_cfe > scheduled_cfe or charged_atomic > scheduled_atomic:
            raise FailureOutcomeClosureError(
                f"charged work exceeds schedule for {expected_task_id}"
            )
        if charged_cfe * scheduled_atomic != charged_atomic * scheduled_cfe:
            raise FailureOutcomeClosureError(
                f"charged CFE/atomic ratio differs for {expected_task_id}"
            )
    else:
        if charged_cfe_value is not None or charged_atomic_value is not None:
            raise FailureOutcomeClosureError(
                f"inexact charged work fabricates a value for {expected_task_id}"
            )
        charged_cfe = None
        charged_atomic = None
    manifest_sha256 = _require_sha256(
        row.get("task_manifest_sha256"),
        label=f"{expected_task_id} task manifest",
    )
    return FailureCostOutcome(
        task_id=expected_task_id,
        schedule_index=expected_schedule_index,
        status=status,
        outcome_class=str(outcome_class),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=peak_rss,
        output_bytes=output_bytes,
        scheduled_cfe=scheduled_cfe,
        charged_cfe=charged_cfe,
        scheduled_atomic_model_steps=scheduled_atomic,
        charged_atomic_model_steps=charged_atomic,
        charged_work_exact=exact,
        task_manifest_sha256=manifest_sha256,
    )


def validate_failure_outcome_closure(
    output_root: Path,
) -> tuple[FailureCostOutcome, ...]:
    """Validate every scheduled outcome and return the failure/cost rows.

    The validator reads only schedule, resource, status and manifest data.  It
    never opens checkpoint objective payloads and never maps a technical
    process timeout to the algorithm terminal ``REJECT_TIMEOUT``.
    """

    root = Path(output_root).resolve()
    run_manifest = _read_object(root / "run_manifest.json")
    resources = run_manifest.get("resources")
    if (
        not isinstance(resources, Mapping)
        or resources.get("total_output_bytes_scope")
        != "ENTIRE_OUTPUT_ROOT_INCLUDING_THIS_RUN_MANIFEST"
        or resources.get("total_output_bytes") != _directory_bytes(root)
    ):
        raise FailureOutcomeClosureError(
            "run manifest final output byte count/scope differs"
        )
    runtime_path = _validate_control_artifact(
        root,
        run_manifest,
        "runtime_report.json",
    )
    runtime_report = _read_object(runtime_path)
    schedule = _load_schedule(root, run_manifest)
    expected_by_id = {row.task_id: row for row in schedule}
    if (
        runtime_report.get("scheduled_task_count") != len(schedule)
        or runtime_report.get("recorded_outcome_count") != len(schedule)
        or runtime_report.get("automatic_retries") != 0
        or runtime_report.get("attempts_per_task") != 1
        or runtime_report.get("effect_values_read_by_supervisor") is not False
        or runtime_report.get("results_analysis_performed") is not False
    ):
        raise FailureOutcomeClosureError(
            "runtime report run-wide control fields differ"
        )

    completed = runtime_report.get("completed")
    failures = runtime_report.get("failures")
    if not isinstance(completed, list) or not isinstance(failures, list):
        raise FailureOutcomeClosureError(
            "runtime report outcome arrays are missing"
        )
    outcome_rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for kind, rows in (("complete", completed), ("failure", failures)):
        for row in rows:
            if not isinstance(row, Mapping):
                raise FailureOutcomeClosureError(
                    "runtime outcome row is not an object"
                )
            task_id = row.get("task_id")
            if type(task_id) is not str or task_id in outcome_rows:
                raise FailureOutcomeClosureError(
                    "runtime outcome task identity is missing or duplicated"
                )
            outcome_rows[task_id] = (kind, row)
    if set(outcome_rows) != set(expected_by_id):
        raise FailureOutcomeClosureError(
            "runtime outcomes do not cover the complete schedule"
        )

    run_commitments = run_manifest.get("task_manifest_commitments")
    runtime_commitments = runtime_report.get("task_manifest_commitments")
    if (
        not isinstance(run_commitments, Mapping)
        or not isinstance(runtime_commitments, Mapping)
        or dict(run_commitments) != dict(runtime_commitments)
        or set(run_commitments) != set(expected_by_id)
    ):
        raise FailureOutcomeClosureError(
            "task manifest commitments do not cover the complete schedule"
        )

    failure_outcomes: list[FailureCostOutcome] = []
    for task_id, spec in sorted(
        expected_by_id.items(),
        key=lambda item: item[1].schedule_index,
    ):
        kind, row = outcome_rows[task_id]
        commitment = _require_sha256(
            run_commitments[task_id],
            label=f"{task_id} run task manifest",
        )
        if kind == "complete":
            if (
                row.get("status") != "COMPLETE"
                or row.get("task_manifest_sha256") != commitment
                or row.get("automatic_retries", 0) != 0
            ):
                raise FailureOutcomeClosureError(
                    f"completed outcome differs for {task_id}"
                )
            expected_status = "COMPLETE"
        else:
            outcome = _failure_cost_outcome(
                row,
                expected_task_id=task_id,
                expected_schedule_index=spec.schedule_index,
            )
            if outcome.task_manifest_sha256 != commitment:
                raise FailureOutcomeClosureError(
                    f"failure manifest differs for {task_id}"
                )
            expected_status = outcome.status
            failure_outcomes.append(outcome)
        _validate_task_manifest(
            root,
            task_id=task_id,
            expected_sha256=commitment,
            expected_status=expected_status,
        )

    unknown_count = sum(
        not outcome.charged_work_exact for outcome in failure_outcomes
    )
    if (
        runtime_report.get("failed_outcome_count")
        != len(failure_outcomes)
        or runtime_report.get("unknown_failure_accounting_count")
        != unknown_count
    ):
        raise FailureOutcomeClosureError(
            "runtime failure totals differ from validated outcomes"
        )
    return tuple(failure_outcomes)
