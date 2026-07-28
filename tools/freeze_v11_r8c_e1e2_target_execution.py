"""Freeze one result-blind, target-qualified R8C E1+E2 launch pair.

This tool never starts formal execution and never consumes the one-time
request.  It accepts only a completed control-plane qualification report,
derives the qualifying worker count without an override, validates the
candidate pair with the production schemas and runner, and then publishes
canonical JSON files with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for _import_root in (SRC_ROOT, TOOLS_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from formal_execution.host import (  # noqa: E402
    host_fingerprint_sha256,
)
from resource_pilot.e1e2_fullpath import (  # noqa: E402
    DEFAULT_DYNAMIC_EVENTS,
    DEFAULT_REPETITIONS,
    DEFAULT_WORKERS,
    DYNAMIC_CFE_PER_EVENT,
    QUALIFICATION_ID,
    ROLLING_CFE_PER_EVENT,
    STATIC_CFE_PER_EVENT,
)
import run_v11_r8_formal as formal_runner  # noqa: E402


PENDING_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_formal_execution_contract.json"
)
FORMAL_WRAPPER = "tools/run_v11_r8c_e1e2_formal.py"
CONTRACT_ID = (
    "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
    "FORMAL-EXECUTION-CONTRACT-01"
)
REQUEST_ID = (
    "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
    "EXECUTION-REQUEST-20260726-01"
)
QUALIFICATION_STATUS = (
    "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
)
GO_DECISION = "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
PROJECTION_STATUS = "TARGET_HOST_FULL_PATH_ESTIMATE"
ARTIFACT_ROLE = (
    "R8C_E1E2_RESULT_BLIND_FULL_PATH_TARGET_QUALIFICATION"
)


class FreezeError(RuntimeError):
    """A fail-closed target-execution freeze refusal."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository canonical JSON encoding without the final LF."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FreezeError("artifact cannot be encoded as canonical JSON") from error


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute_path(value: Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise FreezeError(f"{label} must be an absolute path")
    return path.resolve()


def _require_nonempty(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreezeError(f"{label} must be non-empty")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise FreezeError(f"{label} must be a positive JSON integer")
    return value


def _require_positive_number(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise FreezeError(f"{label} must be a positive JSON number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise FreezeError(f"{label} must be finite and positive")
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{label} is not readable UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FreezeError(f"{label} must contain one JSON object")
    return value


def _read_canonical_qualification_report(
    report_path: Path,
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = report_path.read_bytes()
    except OSError as error:
        raise FreezeError("qualification report is not readable") from error
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(
            "qualification report is not valid UTF-8 JSON"
        ) from error
    if not isinstance(report, dict):
        raise FreezeError("qualification report must contain one JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise FreezeError(
            "qualification report must be canonical JSON with exactly one LF"
        )
    return report, payload


def _validate_control_plane_report_header(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> tuple[int, Mapping[str, Any]]:
    if (
        report.get("artifact_role") != ARTIFACT_ROLE
        or report.get("qualification_id") != QUALIFICATION_ID
        or report.get("status") != QUALIFICATION_STATUS
        or report.get("mode") != "TARGET_QUALIFICATION"
        or report.get("target_qualification_complete") is not True
        or report.get("formal_launch_authorized") is not False
        or report.get("failed_task_count") != 0
        or report.get("automatic_retries") != 0
        or report.get("real_effect_values_persisted") is not False
        or report.get("formal_execution_started") is not False
    ):
        raise FreezeError(
            "qualification report is not a clean completed control-plane pass"
        )

    report_root = report.get("output_root")
    if (
        not isinstance(report_root, str)
        or not Path(report_root).is_absolute()
        or Path(report_root).resolve() != report_path.parent
        or report_path.name != "qualification_report.json"
    ):
        raise FreezeError(
            "qualification report path differs from its frozen output root"
        )

    source = report.get("code_identity")
    if (
        not isinstance(source, Mapping)
        or source.get("worktree_clean") is not True
    ):
        raise FreezeError("qualification report lacks a clean source identity")

    host = report.get("host_fingerprint")
    host_sha256 = report.get("host_fingerprint_sha256")
    if (
        not isinstance(host, dict)
        or not isinstance(host_sha256, str)
        or host_fingerprint_sha256(host) != host_sha256
        or host_fingerprint_sha256() != host_sha256
    ):
        raise FreezeError(
            "qualification report does not bind the current exact host"
        )

    pilot = report.get("pilot_design")
    runtime = report.get("runtime_contract")
    if not isinstance(pilot, Mapping) or not isinstance(runtime, Mapping):
        raise FreezeError("qualification design evidence is missing")
    if (
        pilot.get("worker_counts") != list(DEFAULT_WORKERS)
        or pilot.get("repetitions") != DEFAULT_REPETITIONS
        or pilot.get("static_cfe_per_event") != STATIC_CFE_PER_EVENT
        or pilot.get("dynamic_cfe_per_event") != DYNAMIC_CFE_PER_EVENT
        or pilot.get("rolling_cfe_per_event") != ROLLING_CFE_PER_EVENT
        or pilot.get("dynamic_events") != DEFAULT_DYNAMIC_EVENTS
        or pilot.get("all_task_ids_unique") is not True
    ):
        raise FreezeError("qualification report differs from the frozen design")

    recommendation = report.get("worker_recommendation")
    if not isinstance(recommendation, Mapping):
        raise FreezeError("qualification worker recommendation is missing")
    worker = recommendation.get("recommended_worker_count")
    projected_hours = recommendation.get("recommended_projected_wall_hours")
    if (
        recommendation.get("status")
        != "MEMORY_ELIGIBLE_THROUGHPUT_OPTIMUM_IDENTIFIED"
        or recommendation.get("formal_launch_authorized") is not False
        or recommendation.get("recommended_decision_classification")
        != GO_DECISION
        or type(worker) is not int
        or worker not in DEFAULT_WORKERS
        or not isinstance(projected_hours, (int, float))
        or not math.isfinite(float(projected_hours))
        or not 0.0 < float(projected_hours) <= 36.0
    ):
        raise FreezeError(
            "qualification recommendation is not a <=36h GO decision"
        )

    wall_projection = report.get("e1_e2_wall_projection")
    projections = (
        wall_projection.get("projections")
        if isinstance(wall_projection, Mapping)
        else None
    )
    if not isinstance(projections, list):
        raise FreezeError("qualification wall projections are missing")
    selected_rows = [
        row
        for row in projections
        if isinstance(row, Mapping) and row.get("workers") == worker
    ]
    if len(selected_rows) != 1:
        raise FreezeError(
            "qualification recommendation does not select one projection"
        )
    selected = selected_rows[0]
    memory = selected.get("memory_qualification")
    if (
        selected.get("status") != PROJECTION_STATUS
        or selected.get("decision_classification") != GO_DECISION
        or selected.get("projected_wall_hours") != projected_hours
        or not isinstance(memory, Mapping)
        or memory.get("eligible") is not True
    ):
        raise FreezeError(
            "selected qualification projection is not the recommended GO row"
        )
    return worker, selected


def _formal_command(
    *,
    contract_path: Path,
    request_path: Path,
    output_root: Path,
) -> str:
    arguments = [
        sys.executable,
        FORMAL_WRAPPER,
        "--contract",
        str(contract_path),
        "--request",
        str(request_path),
        "--output-root",
        str(output_root),
    ]
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def _validate_paths(
    *,
    qualification_report_path: Path,
    contract_path: Path,
    request_path: Path,
    marker_path: Path,
    output_root: Path,
) -> None:
    project_root = PROJECT_ROOT.resolve()
    qualification_root = qualification_report_path.parent
    protected_paths = {
        "contract output": contract_path,
        "request output": request_path,
        "request consumption marker": marker_path,
        "formal output root": output_root,
    }
    for label, path in protected_paths.items():
        if _is_relative_to(path, project_root):
            raise FreezeError(f"{label} must be outside the source worktree")
        if _is_relative_to(path, qualification_root):
            raise FreezeError(
                f"{label} must be outside the immutable qualification root"
            )
    if _is_relative_to(qualification_report_path, project_root):
        raise FreezeError(
            "qualification report must be outside the source worktree"
        )
    if len({contract_path, request_path, marker_path, output_root}) != 4:
        raise FreezeError("contract, request, marker and output paths must differ")
    if _is_relative_to(marker_path, output_root):
        raise FreezeError(
            "request consumption marker cannot be inside the output root"
        )
    if "onedrive" in str(output_root).casefold():
        raise FreezeError("formal output root cannot be under OneDrive")

    for label, path in (
        ("contract output", contract_path),
        ("request output", request_path),
        ("request consumption marker", marker_path),
    ):
        if path.exists():
            raise FreezeError(f"{label} already exists; overwrite is forbidden")
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            raise FreezeError(f"{label} parent must be an existing writable directory")
    if output_root.exists():
        raise FreezeError("formal output root already exists; overwrite is forbidden")
    if (
        not output_root.parent.is_dir()
        or not os.access(output_root.parent, os.W_OK)
    ):
        raise FreezeError(
            "formal output parent must be an existing writable scratch root"
        )


def _target_design(report: Mapping[str, Any]) -> dict[str, Any]:
    pilot = report["pilot_design"]
    runtime = report["runtime_contract"]
    return {
        "worker_counts": list(pilot["worker_counts"]),
        "repetitions": pilot["repetitions"],
        "static_cfe_per_event": pilot["static_cfe_per_event"],
        "dynamic_cfe_per_event": pilot["dynamic_cfe_per_event"],
        "rolling_cfe_per_event": pilot["rolling_cfe_per_event"],
        "dynamic_events": pilot["dynamic_events"],
        "workload_method_case_binding_count": (
            runtime["workload_method_case_bindings_covered"]
        ),
        "unique_representative_benchmark_case_count": (
            runtime["unique_representative_benchmark_cases_covered"]
        ),
        "formal_projection_rate_class_count": (
            runtime["formal_projection_rate_classes_covered"]
        ),
        "task_count": pilot["task_count"],
    }


def _build_contract(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    report_sha256: str,
    selected_worker: int,
    selected_projection: Mapping[str, Any],
    contract_path: Path,
    request_path: Path,
    marker_path: Path,
    output_root: Path,
    provider: str,
    instance_type: str,
    author_authorization_text: str,
    created_date: str,
) -> dict[str, Any]:
    template = _read_json_object(
        PENDING_TEMPLATE_PATH,
        label="pending R8C E1+E2 contract template",
    )
    contract = deepcopy(template)
    command = _formal_command(
        contract_path=contract_path,
        request_path=request_path,
        output_root=output_root,
    )
    contract.update(
        {
            "contract_id": CONTRACT_ID,
            "protocol_stage": "R8C_E1E2_TARGET_QUALIFIED_FORMAL_EXECUTION",
            "status": "TARGET_HOST_QUALIFIED_AND_AUTHORIZED",
            "created_date": created_date,
        }
    )
    contract["authorization"].update(
        {
            "author_text": author_authorization_text,
            "authorized_scope": (
                "FORMAL_E1_E2_PUBLIC_BENCHMARK_EFFECT_EXECUTION_ONLY"
            ),
            "formal_effect_execution_authorized": True,
            "effect_analysis_authorized": False,
            "results_writing_authorized": False,
        }
    )
    contract["target_qualification_evidence"] = {
        "qualification_report_path": str(report_path),
        "qualification_report_sha256": report_sha256,
        "qualification_id": report["qualification_id"],
        "qualification_status": report["status"],
        "source": deepcopy(report["code_identity"]),
        "host_fingerprint_sha256": report["host_fingerprint_sha256"],
        "design": _target_design(report),
        "selected_worker_count": selected_worker,
        "selected_projection": {
            "status": selected_projection["status"],
            "projected_wall_hours": selected_projection[
                "projected_wall_hours"
            ],
            "decision_classification": selected_projection[
                "decision_classification"
            ],
        },
    }
    contract["launch"].update(
        {
            "contract_path": str(contract_path),
            "request_path": str(request_path),
            "request_consumption_marker": str(marker_path),
            "output_root": str(output_root),
            "exact_command": command,
            "command_identity_frozen": True,
            "command_executable_now": True,
            "formal_launch_prohibited": False,
            "current_confirmation_state": (
                "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED"
            ),
        }
    )

    host = report["host_fingerprint"]
    effective_processors = _require_positive_int(
        host.get("effective_logical_processors"),
        label="qualified host effective_logical_processors",
    )
    memory_bytes = _require_positive_int(
        host.get("memory_bytes"),
        label="qualified host memory_bytes",
    )
    if selected_worker > effective_processors:
        raise FreezeError(
            "recommended worker count exceeds the qualified CPU allocation"
        )
    memory = selected_projection["memory_qualification"]
    max_worker_rss = _require_positive_int(
        memory.get("conservative_worker_peak_rss_bytes"),
        label="selected conservative worker RSS",
    )
    max_pool_rss = _require_positive_int(
        memory.get("conservative_pool_peak_rss_bytes"),
        label="selected conservative pool RSS",
    )
    if max_worker_rss * selected_worker != max_pool_rss:
        raise FreezeError(
            "selected conservative pool RSS differs from worker RSS times workers"
        )

    resources = contract["resources"]
    resources.pop("candidate_profile_frozen", None)
    resources.update(
        {
            "qualification_status": "TARGET_HOST_QUALIFIED",
            "selected_exact_host_frozen": True,
        }
    )
    resources["candidate_target"] = {
        "processor_reference": _require_nonempty(
            host.get("cpu_model"),
            label="qualified host CPU model",
        ),
        "offered_instance_description": (
            f"{provider.strip()} {instance_type.strip()}; "
            "exact target-qualified host"
        ),
        "normalized_compute_allocation": (
            f"{effective_processors}_EFFECTIVE_LOGICAL_PROCESSORS"
        ),
        "memory_gib": memory_bytes / (1024**3),
        "provider": provider.strip(),
        "instance_type": instance_type.strip(),
        "host_fingerprint_sha256": report["host_fingerprint_sha256"],
        "remote_measurement_completed": True,
    }
    resources["parallelism"] = {
        "max_workers": selected_worker,
        "logical_threads_per_worker": 1,
        "blas_openmp_threads_per_worker": 1,
        "max_worker_peak_rss_bytes": max_worker_rss,
        "max_pool_peak_rss_bytes": max_pool_rss,
        "worker_count_qualified_on_target": True,
    }
    resources["decision_rule"]["current_decision"] = (
        "GO_TARGET_HOST_QUALIFIED"
    )
    resources["scratch"]["required_root"] = str(output_root.parent)
    contract["permissions"]["public_benchmark_effect_execution"] = True
    contract["fail_closed_gate"] = {
        "request_id": REQUEST_ID,
        "request_status": "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED",
        "target_host_status": "TARGET_HOST_QUALIFIED",
        "formal_launch_status": "ELIGIBLE",
    }
    return contract


def _build_request(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    upstream = contract["upstream"]
    schedule = contract["schedule"]
    command = contract["launch"]["exact_command"]
    return {
        "request_id": REQUEST_ID,
        "scope": "benchmark_effect",
        "companion_scope": "benchmark_effect",
        "contracts": {
            "protocol_id": contract["protocol_id"],
            "r5_contract_id": upstream["r5"]["contract_id"],
            "r5_contract_sha256": upstream["r5"]["sha256"],
            "r5a_contract_id": upstream["r5a"]["contract_id"],
            "r5a_contract_sha256": upstream["r5a"]["sha256"],
            "corrective_protocol_id": upstream["corrective_protocol"][
                "contract_id"
            ],
            "corrective_protocol_sha256": upstream["corrective_protocol"][
                "sha256"
            ],
            "r8c_formal_contract_id": CONTRACT_ID,
            "r8c_formal_contract_sha256": contract_sha256,
            "formal_schedule_id": schedule["id"],
            "formal_schedule_sha256": schedule["sha256"],
            "source_git_commit": report["code_identity"]["git_commit"],
            "source_git_tree": report["code_identity"]["git_tree"],
        },
        "frozen_exact_command": command,
        "author_confirmation_text": command,
        "author_exact_command_confirmed": True,
        "formal_effect_execution_requested": True,
        "participant_data_requested": False,
        "hidden_generation_requested": False,
        "results_analysis_requested": False,
        "results_writing_requested": False,
        "remote_git_mutation_requested": False,
        "release_or_distribution_requested": False,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if os.name == "nt":
        flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FreezeError(f"{path} already exists; overwrite is forbidden") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runner_validate_pair(
    contract_path: Path,
    request_path: Path,
) -> tuple[Mapping[str, Any], Any]:
    contract, request, _schedule = formal_runner._load_and_validate(
        contract_path,
        request_path,
        formal_runner.CORRECTIVE_E1E2_PROFILE,
    )
    formal_runner._validate_source(request)
    return contract, request


def _validate_staged_pair(
    *,
    contract_payload: bytes,
    request_payload: bytes,
    parent: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".r8c-e1e2-target-freeze-",
        dir=parent,
    ) as temporary:
        root = Path(temporary)
        staged_contract = root / "target-qualified-contract.json"
        staged_request = root / "target-qualified-request.json"
        _write_exclusive(staged_contract, contract_payload)
        _write_exclusive(staged_request, request_payload)
        _runner_validate_pair(staged_contract, staged_request)


def _remove_created(path: Path, payload: bytes) -> None:
    try:
        if path.read_bytes() == payload:
            path.unlink()
            _fsync_directory(path.parent)
    except FileNotFoundError:
        return


def freeze_target_execution(
    *,
    qualification_report_path: Path,
    contract_path: Path,
    request_path: Path,
    request_consumption_marker: Path,
    output_root: Path,
    provider: str,
    instance_type: str,
    author_authorization_text: str,
    created_date: str | None = None,
) -> dict[str, Any]:
    """Validate and exclusively publish one target-qualified launch pair."""

    provider = _require_nonempty(provider, label="provider")
    instance_type = _require_nonempty(instance_type, label="instance type")
    author_authorization_text = _require_nonempty(
        author_authorization_text,
        label="author authorization text",
    )
    freeze_date = created_date or date.today().isoformat()
    try:
        date.fromisoformat(freeze_date)
    except ValueError as error:
        raise FreezeError("created date must be ISO YYYY-MM-DD") from error

    report_path = _absolute_path(
        qualification_report_path,
        label="qualification report",
    )
    contract_output = _absolute_path(contract_path, label="contract output")
    request_output = _absolute_path(request_path, label="request output")
    marker_output = _absolute_path(
        request_consumption_marker,
        label="request consumption marker",
    )
    formal_output = _absolute_path(output_root, label="formal output root")
    if not report_path.is_file():
        raise FreezeError("qualification report does not exist")
    _validate_paths(
        qualification_report_path=report_path,
        contract_path=contract_output,
        request_path=request_output,
        marker_path=marker_output,
        output_root=formal_output,
    )

    report, report_payload = _read_canonical_qualification_report(report_path)
    selected_worker, selected_projection = (
        _validate_control_plane_report_header(
            report,
            report_path=report_path,
        )
    )
    contract = _build_contract(
        report=report,
        report_path=report_path,
        report_sha256=sha256(report_payload).hexdigest(),
        selected_worker=selected_worker,
        selected_projection=selected_projection,
        contract_path=contract_output,
        request_path=request_output,
        marker_path=marker_output,
        output_root=formal_output,
        provider=provider,
        instance_type=instance_type,
        author_authorization_text=author_authorization_text,
        created_date=freeze_date,
    )
    contract_payload = canonical_json_bytes(contract) + b"\n"
    contract_sha256 = sha256(contract_payload).hexdigest()
    request = _build_request(
        contract=contract,
        contract_sha256=contract_sha256,
        report=report,
    )
    request_payload = canonical_json_bytes(request) + b"\n"

    _validate_staged_pair(
        contract_payload=contract_payload,
        request_payload=request_payload,
        parent=contract_output.parent,
    )
    if (
        host_fingerprint_sha256()
        != report["host_fingerprint_sha256"]
    ):
        raise FreezeError("host identity changed during target freeze")

    created_contract = False
    created_request = False
    try:
        _write_exclusive(contract_output, contract_payload)
        created_contract = True
        _fsync_directory(contract_output.parent)
        _write_exclusive(request_output, request_payload)
        created_request = True
        _fsync_directory(request_output.parent)
        _runner_validate_pair(contract_output, request_output)
        if (
            _file_sha256(contract_output) != contract_sha256
            or _file_sha256(request_output)
            != sha256(request_payload).hexdigest()
        ):
            raise FreezeError("published target-qualified pair hash drifted")
    except Exception:
        if created_request:
            _remove_created(request_output, request_payload)
        if created_contract:
            _remove_created(contract_output, contract_payload)
        raise

    return {
        "artifact_role": (
            "R8C_E1E2_TARGET_QUALIFIED_FREEZE_CONTROL_PLANE_ONLY"
        ),
        "status": "TARGET_QUALIFIED_CONTRACT_AND_REQUEST_FROZEN",
        "qualification_report_path": str(report_path),
        "qualification_report_sha256": sha256(report_payload).hexdigest(),
        "selected_worker_count": selected_worker,
        "projected_wall_hours": selected_projection[
            "projected_wall_hours"
        ],
        "contract_path": str(contract_output),
        "contract_sha256": contract_sha256,
        "request_path": str(request_output),
        "request_sha256": sha256(request_payload).hexdigest(),
        "request_consumption_marker": str(marker_output),
        "formal_output_root": str(formal_output),
        "exact_formal_command": contract["launch"]["exact_command"],
        "request_consumed": False,
        "formal_execution_started": False,
        "effect_analysis_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a result-blind target-qualified R8C E1+E2 contract and "
            "one-time request without consuming the request or starting work."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument(
        "--request-consumption-marker",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument(
        "--author-authorization-text",
        required=True,
        help=(
            "Verbatim author text authorizing only the formal E1+E2 public "
            "benchmark effect execution."
        ),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        summary = freeze_target_execution(
            qualification_report_path=args.qualification_report,
            contract_path=args.contract_output,
            request_path=args.request_output,
            request_consumption_marker=args.request_consumption_marker,
            output_root=args.output_root,
            provider=args.provider,
            instance_type=args.instance_type,
            author_authorization_text=args.author_authorization_text,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
