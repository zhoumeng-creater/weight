"""Execute the R7-frozen R8 public E1--E3 schedule without analysis."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import errno
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dt_ramde_v11.contracts import ConfigurationError
from evaluation.evaluator import ExecutionTimeoutBeforeEntry
from formal_execution.checkpoint_data import (
    WORKER_CONTROL_REPORT_MAX_BYTES,
    CheckpointDataError,
    read_checkpoint_file,
)
from formal_execution.runtime import (
    CHECKPOINT_FRONT_PERSISTENCE,
    CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    CORRECTIVE_R8C_RUNTIME_SETTINGS,
    LEGACY_R8_RUNTIME_SETTINGS,
    FormalRuntimeSettings,
    file_sha256,
    parse_r8c_request,
    parse_r8_request,
    run_task,
)
from formal_execution.host import (
    HostSamplingError,
    host_fingerprint,
    host_fingerprint_sha256,
    process_cpu_seconds as _process_cpu_seconds,
    process_rss_bytes as _process_rss_bytes,
)
from formal_execution.schedule import (
    build_corrective_e1e2_formal_schedule,
    build_corrective_formal_schedule,
    build_formal_schedule,
    build_e2_full_reuse_map,
    canonical_json_bytes,
    e2_full_reuse_commitment,
    schedule_commitment,
)


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "config" / "r7" / "r7_formal_execution_contract.json"
)


@dataclass(frozen=True)
class RunnerProfile:
    profile_id: str
    contract_id: str
    default_contract: Path
    contract_hash_binding: str
    parse_request: Any
    build_schedule: Any
    runtime_settings: FormalRuntimeSettings
    require_r5_runtime_match: bool
    artifact_stage: str
    target_qualification_required: bool = False
    required_effect_permissions: tuple[str, ...] = ()
    contract_schema: Path | None = None
    request_schema: Path | None = None
    task_artifact_worker_reports: bool = False


LEGACY_PROFILE = RunnerProfile(
    profile_id="legacy_r8",
    contract_id="WGT-V11-R7-FORMAL-EXECUTION-CONTRACT-01",
    default_contract=DEFAULT_CONTRACT,
    contract_hash_binding="r7_contract_sha256",
    parse_request=parse_r8_request,
    build_schedule=build_formal_schedule,
    runtime_settings=LEGACY_R8_RUNTIME_SETTINGS,
    require_r5_runtime_match=False,
    artifact_stage="R8",
)
CORRECTIVE_PROFILE = RunnerProfile(
    profile_id="corrective_r8c",
    contract_id="WGT-V11-R8C-FORMAL-EXECUTION-CONTRACT-01",
    default_contract=(
        PROJECT_ROOT
        / "config"
        / "r8c"
        / "r8c_formal_execution_contract.json"
    ),
    contract_hash_binding="r8c_formal_contract_sha256",
    parse_request=parse_r8c_request,
    build_schedule=build_corrective_formal_schedule,
    runtime_settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    require_r5_runtime_match=True,
    artifact_stage="R8C",
    target_qualification_required=True,
    required_effect_permissions=(
        "public_benchmark_effect_execution",
        "public_synthetic_weight_effect_execution",
    ),
)
CORRECTIVE_E1E2_PROFILE = RunnerProfile(
    profile_id="corrective_r8c_e1e2",
    contract_id=(
        "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
        "FORMAL-EXECUTION-CONTRACT-01"
    ),
    default_contract=(
        PROJECT_ROOT
        / "config"
        / "r8c_e1e2"
        / "r8c_e1e2_formal_execution_contract.json"
    ),
    contract_hash_binding="r8c_formal_contract_sha256",
    parse_request=parse_r8c_request,
    build_schedule=build_corrective_e1e2_formal_schedule,
    runtime_settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    require_r5_runtime_match=True,
    artifact_stage="R8C_E1E2",
    target_qualification_required=True,
    required_effect_permissions=("public_benchmark_effect_execution",),
    contract_schema=(
        PROJECT_ROOT
        / "config"
        / "r8c_e1e2"
        / "r8c_e1e2_target_qualified_contract.schema.json"
    ),
    request_schema=(
        PROJECT_ROOT
        / "config"
        / "r8c_e1e2"
        / "r8c_e1e2_target_qualified_execution_request.schema.json"
    ),
    task_artifact_worker_reports=True,
)
RUNNER_PROFILES = {
    profile.profile_id: profile
    for profile in (
        LEGACY_PROFILE,
        CORRECTIVE_PROFILE,
        CORRECTIVE_E1E2_PROFILE,
    )
}

LAUNCH_TOKEN_ENV = "WGT_R8_FORMAL_LAUNCH_TOKEN"
R8C_E1E2_WORKER_REPORT_MAX_BYTES = WORKER_CONTROL_REPORT_MAX_BYTES
R8C_E1E2_WORKER_REPORT_FILENAMES = (
    "task_summary.json",
    "task_failure.json",
    "task_supervisor_outcome.json",
)
CONTROL_PLANE_RESERVE_BYTES = 1 << 30
ACTIVE_TASK_WRITE_RESERVE_BYTES = 256 << 20
SUPERVISOR_MEMORY_RESERVE_BYTES = 4 << 30
R8C_E1E2_RNG_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-RNG-IMPLEMENTATION-AMENDMENT-01"
)
R8C_E1E2_RNG_AMENDMENT_SCHEMA = (
    PROJECT_ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_rng_implementation_amendment.schema.json"
)
R8C_E1E2_COMPACT_RUNTIME_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-COMPACT-RUNTIME-AMENDMENT-01"
)
R8C_E1E2_TIMEOUT_SEMANTICS_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-TIMEOUT-SEMANTICS-AMENDMENT-01"
)
R8C_E1E2_LIRCMOP_REFERENCE_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-LIRCMOP-REFERENCE-AMENDMENT-01"
)
R8C_E1E2_CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-CDF-OPERATIONAL-AUTHORITY-AMENDMENT-01"
)
R8C_E1E2_REFERENCE_CATALOG_ID = (
    "WGT-V11-R8C-E1E2-REFERENCE-CATALOG-01"
)
R8C_E1E2_CDF_OPERATIONAL_SUITE_ID = (
    "CDF-1-15-CMLSGA-1926A5A1-OPERATIONAL"
)
R8C_E1E2_CDF_OPERATIONAL_AUTHORITY_ID = (
    "WGT-V11-R8C-CDF-OPERATIONAL-AUTHORITY-1.0.0"
)
R8C_E1E2_TARGET_QUALIFICATION_ID = (
    "WGT-V11-R8C-E1E2-TARGET-QUALIFICATION-20260726-02"
)
R8C_E1E2_TARGET_QUALIFICATION_STATUS = (
    "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
)
R8C_E1E2_TARGET_QUALIFICATION_WORKERS = (1, 8, 16, 24, 32, 48, 64)
TASK_TIMEOUT_MARKER_NAME = "TASK_TIMEOUT_REQUESTED"
TASK_TIMEOUT_GRACE_SECONDS = 30.0
PROCESS_SAMPLE_EXIT_RACE_GRACE_SECONDS = 0.005
PROCESS_SAMPLE_MAX_ATTEMPTS = 5
TECHNICAL_SEQUENCE_TIMEOUT = "TECHNICAL_SEQUENCE_TIMEOUT"
TECHNICAL_GLOBAL_TIMEOUT = "TECHNICAL_GLOBAL_TIMEOUT"
TECHNICAL_RESOURCE_TERMINATION = "TECHNICAL_RESOURCE_TERMINATION"
TECHNICAL_WORKER_LAUNCH_FAILURE = "TECHNICAL_WORKER_LAUNCH_FAILURE"
TECHNICAL_NOT_DISPATCHED = "TECHNICAL_NOT_DISPATCHED"
TASK_EXECUTION_FAILURE = "TASK_EXECUTION_FAILURE"


@dataclass(frozen=True)
class PrelaunchPlan:
    source: Mapping[str, Any]
    host: Mapping[str, Any]
    output_root: Path
    marker_path: Path
    max_workers: int
    max_worker_rss: int
    max_pool_rss: int
    monitor_seconds: float
    global_timeout: float
    max_output: int
    max_cpu: float
    stop_free: int
    control_plane_reserve: int
    inflight_write_reserve_per_worker: int
    active_write_reserve: int


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path.name} must contain a JSON object")
    return value


def _validate_json_schema(
    value: Mapping[str, Any],
    schema_path: Path | None,
    *,
    label: str,
) -> None:
    if schema_path is None:
        return
    if not schema_path.is_file():
        raise ConfigurationError(
            f"{label} schema is missing: {schema_path}"
        )
    schema = _read_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: (
                tuple(str(item) for item in error.absolute_path),
                error.message,
            ),
        )
    except Exception as error:
        raise ConfigurationError(
            f"{label} schema validation could not be completed"
        ) from error
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path)
        suffix = f" at {location}" if location else ""
        raise ConfigurationError(
            f"{label} fails its frozen JSON schema{suffix}: "
            f"{error.message}"
        )


def _launch_path(value: str, *, relative_to_project: bool = True) -> Path:
    if value == "$PROJECT_ROOT":
        return PROJECT_ROOT.resolve()
    path = Path(value)
    if not path.is_absolute() and relative_to_project:
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{label} must be a positive JSON integer")
    return value


def _strict_positive_number(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ConfigurationError(f"{label} must be a positive JSON number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{label} must be finite and positive")
    return result


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _validate_source(request: Any) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ConfigurationError(
            "R8 requires the clean R7-authorized implementation worktree"
        )
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if commit != request.contracts.source_git_commit:
        raise ConfigurationError("current Git commit differs from R8 request")
    if tree != request.contracts.source_git_tree:
        raise ConfigurationError("current Git tree differs from R8 request")
    return {"git_commit": commit, "git_tree": tree, "git_dirty": False}


def _validate_project_file_binding(
    binding: Any,
    *,
    label: str,
    path_key: str = "path",
    sha256_key: str = "sha256",
    bytes_key: str | None = None,
) -> Path:
    if not isinstance(binding, Mapping):
        raise ConfigurationError(f"{label} binding is missing")
    path_value = binding.get(path_key)
    if not isinstance(path_value, str) or not path_value:
        raise ConfigurationError(f"{label} path is missing")
    file_path = _launch_path(path_value, relative_to_project=True)
    if not _is_relative_to(file_path, PROJECT_ROOT.resolve()):
        raise ConfigurationError(f"{label} must be inside the project")
    if (
        not file_path.is_file()
        or file_sha256(file_path) != binding.get(sha256_key)
    ):
        raise ConfigurationError(f"{label} drifted")
    if bytes_key is not None:
        expected_bytes = binding.get(bytes_key)
        if type(expected_bytes) is not int or (
            file_path.stat().st_size != expected_bytes
        ):
            raise ConfigurationError(f"{label} size drifted")
    return file_path


def _validate_rng_implementation_amendment(
    upstream: Mapping[str, Any],
) -> None:
    binding = upstream.get("rng_implementation_amendment")
    if not isinstance(binding, Mapping):
        raise ConfigurationError(
            "E1+E2 RNG implementation amendment binding is missing"
        )
    if binding.get("amendment_id") != R8C_E1E2_RNG_AMENDMENT_ID:
        raise ConfigurationError(
            "E1+E2 RNG implementation amendment identity differs"
        )
    amendment_path = _validate_project_file_binding(
        binding,
        label="E1+E2 RNG implementation amendment",
    )

    amendment = _read_json(amendment_path)
    _validate_json_schema(
        amendment,
        R8C_E1E2_RNG_AMENDMENT_SCHEMA,
        label="R8C E1+E2 RNG implementation amendment",
    )
    if amendment.get("amendment_id") != binding.get("amendment_id"):
        raise ConfigurationError(
            "RNG implementation amendment identity differs"
        )

    for label, file_binding, path_key, sha256_key, bytes_key in (
        (
            "current RNG implementation",
            amendment.get("current_equivalent_implementation"),
            "path",
            "sha256",
            "bytes",
        ),
        (
            "RNG byte-exact evidence",
            amendment.get("byte_exact_evidence"),
            "test_file",
            "test_file_sha256",
            None,
        ),
    ):
        _validate_project_file_binding(
            file_binding,
            label=label,
            path_key=path_key,
            sha256_key=sha256_key,
            bytes_key=bytes_key,
        )


def _validate_compact_runtime_amendment(
    upstream: Mapping[str, Any],
) -> None:
    binding = upstream.get("compact_runtime_amendment")
    if not isinstance(binding, Mapping):
        raise ConfigurationError(
            "E1+E2 compact runtime amendment binding is missing"
        )
    if (
        binding.get("amendment_id")
        != R8C_E1E2_COMPACT_RUNTIME_AMENDMENT_ID
    ):
        raise ConfigurationError(
            "E1+E2 compact runtime amendment identity differs"
        )
    amendment_path = _validate_project_file_binding(
        binding,
        label="E1+E2 compact runtime amendment",
    )
    schema_path = _validate_project_file_binding(
        binding,
        label="E1+E2 compact runtime amendment schema",
        path_key="schema_path",
        sha256_key="schema_sha256",
    )
    amendment = _read_json(amendment_path)
    _validate_json_schema(
        amendment,
        schema_path,
        label="R8C E1+E2 compact runtime amendment",
    )
    scope = amendment.get("scope")
    equivalence = amendment.get("same_seed_equivalence")
    verification = amendment.get("verification")
    projection = amendment.get("full_batch_projection")
    authority = amendment.get("authority_boundary")
    if (
        amendment.get("amendment_id") != binding.get("amendment_id")
        or amendment.get("status")
        != (
            "FROZEN_RESULT_BLIND_CORRECTIVE_RUNTIME_EVIDENCE_"
            "NOT_EXECUTION_AUTHORITY"
        )
        or not isinstance(scope, Mapping)
        or scope.get("e3_in_scope") is not False
        or scope.get("research_endpoints_changed") is not False
        or not isinstance(equivalence, Mapping)
        or equivalence.get("all_three_case_classes_passed") is not True
        or [
            item.get("case_class")
            for item in equivalence.get("cases", [])
            if isinstance(item, Mapping)
        ]
        != ["STATIC", "CDF", "ROLLING"]
        or not isinstance(verification, Mapping)
        or not isinstance(
            verification.get("focused_regression"),
            Mapping,
        )
        or verification["focused_regression"].get("status") != "PASS"
        or verification["focused_regression"].get("test_count") != 323
        or not isinstance(projection, Mapping)
        or projection.get("decision_gate_passed") is not True
        or projection.get("is_target_host_wall_time_eta") is not False
        or projection.get("may_replace_target_host_qualification")
        is not False
        or not isinstance(authority, Mapping)
        or authority.get("authorizes_formal_effect_execution") is not False
        or authority.get("authorizes_effect_analysis") is not False
    ):
        raise ConfigurationError(
            "E1+E2 compact runtime evidence or authority boundary differs"
        )


def _validate_timeout_semantics_amendment(
    upstream: Mapping[str, Any],
) -> None:
    binding = upstream.get("timeout_semantics_amendment")
    if not isinstance(binding, Mapping):
        raise ConfigurationError(
            "E1+E2 timeout semantics amendment binding is missing"
        )
    if (
        binding.get("amendment_id")
        != R8C_E1E2_TIMEOUT_SEMANTICS_AMENDMENT_ID
    ):
        raise ConfigurationError(
            "E1+E2 timeout semantics amendment identity differs"
        )
    amendment_path = _validate_project_file_binding(
        binding,
        label="E1+E2 timeout semantics amendment",
    )
    schema_path = _validate_project_file_binding(
        binding,
        label="E1+E2 timeout semantics amendment schema",
        path_key="schema_path",
        sha256_key="schema_sha256",
    )
    amendment = _read_json(amendment_path)
    _validate_json_schema(
        amendment,
        schema_path,
        label="R8C E1+E2 timeout semantics amendment",
    )
    if (
        amendment.get("amendment_id") != binding.get("amendment_id")
        or amendment.get("status")
        != "RESULT_BLIND_TIMEOUT_SEMANTICS_IMPLEMENTED_AND_VERIFIED"
        or amendment.get("scientific_event_deadlines_seconds")
        != {
            "E1_STATIC": 1800,
            "E1_DYNAMIC_CDF": 300,
            "E1_E2_ROLLING": 120,
        }
        or amendment.get("technical_sequence_hard_ceilings_seconds")
        != {
            "E1_STATIC": 3600,
            "E1_E2_DYNAMIC": 21600,
            "E1_E2_ROLLING": 10800,
            "GLOBAL_FORMAL_WALL": 172800,
        }
    ):
        raise ConfigurationError(
            "E1+E2 timeout layer identities or ceilings differ"
        )
    sources = amendment.get("source_authorities")
    r5_binding = (
        sources.get("r5_machine_contract")
        if isinstance(sources, Mapping)
        else None
    )
    _validate_project_file_binding(
        r5_binding,
        label="timeout semantics R5 machine contract",
        bytes_key="bytes",
    )
    classification = amendment.get("classification")
    scientific = (
        classification.get("scientific_event_deadline")
        if isinstance(classification, Mapping)
        else None
    )
    technical = (
        classification.get("technical_sequence_or_global_timeout")
        if isinstance(classification, Mapping)
        else None
    )
    if (
        not isinstance(scientific, Mapping)
        or scientific.get("applies_equally_to_all_methods") is not True
        or scientific.get("terminal_code") != "REJECT_TIMEOUT"
        or not isinstance(technical, Mapping)
        or technical.get("applies_equally_to_all_methods") is not True
        or technical.get("algorithm_terminal_code") is not None
        or technical.get("task_outcome_classes")
        != [
            TECHNICAL_SEQUENCE_TIMEOUT,
            TECHNICAL_GLOBAL_TIMEOUT,
        ]
    ):
        raise ConfigurationError(
            "E1+E2 scientific/technical timeout classification differs"
        )
    implementation_gate = amendment.get("implementation_gate")
    if (
        not isinstance(implementation_gate, Mapping)
        or implementation_gate.get("timeout_semantics_gate_passed")
        is not True
        or not implementation_gate.get("historical_drift_closed")
        or not implementation_gate.get("verification")
    ):
        raise ConfigurationError(
            "E1+E2 timeout implementation gate is not closed"
        )


def _validate_lircmop_reference_amendment(
    upstream: Mapping[str, Any],
) -> None:
    binding = upstream.get("lircmop_reference_amendment")
    if not isinstance(binding, Mapping):
        raise ConfigurationError(
            "E1+E2 LIR-CMOP/reference amendment binding is missing"
        )
    if (
        binding.get("amendment_id")
        != R8C_E1E2_LIRCMOP_REFERENCE_AMENDMENT_ID
    ):
        raise ConfigurationError(
            "E1+E2 LIR-CMOP/reference amendment identity differs"
        )
    amendment_path = _validate_project_file_binding(
        binding,
        label="E1+E2 LIR-CMOP/reference amendment",
    )
    schema_path = _validate_project_file_binding(
        binding,
        label="E1+E2 LIR-CMOP/reference amendment schema",
        path_key="schema_path",
        sha256_key="schema_sha256",
    )
    amendment = _read_json(amendment_path)
    _validate_json_schema(
        amendment,
        schema_path,
        label="R8C E1+E2 LIR-CMOP/reference amendment",
    )
    if amendment.get("amendment_id") != binding.get("amendment_id"):
        raise ConfigurationError(
            "LIR-CMOP/reference amendment identity differs"
        )

    historical = amendment.get("historical_freeze")
    if not isinstance(historical, Mapping):
        raise ConfigurationError(
            "LIR-CMOP/reference historical binding is missing"
        )
    for label, path_key, sha256_key in (
        ("LIR-CMOP/reference R5 contract", "r5_contract_path", "r5_contract_sha256"),
        (
            "LIR-CMOP/reference R4 benchmark registry",
            "r4_benchmark_registry_path",
            "r4_benchmark_registry_sha256",
        ),
    ):
        _validate_project_file_binding(
            historical,
            label=label,
            path_key=path_key,
            sha256_key=sha256_key,
        )

    implementation_bindings = amendment.get("implementation_bindings")
    if not isinstance(implementation_bindings, Mapping):
        raise ConfigurationError(
            "LIR-CMOP/reference implementation bindings are missing"
        )
    required_bindings = {
        "paper_evaluator",
        "static_suite_registry",
        "corrective_factory",
        "reference_derivation",
        "reference_artifacts",
        "analytic_scale",
    }
    if set(implementation_bindings) != required_bindings:
        raise ConfigurationError(
            "LIR-CMOP/reference implementation binding set differs"
        )
    for name in sorted(required_bindings):
        _validate_project_file_binding(
            implementation_bindings[name],
            label=f"LIR-CMOP/reference {name}",
            bytes_key="bytes",
        )

    _validate_project_file_binding(
        amendment.get("verification"),
        label="LIR-CMOP/reference verification evidence",
        path_key="test_file",
        sha256_key="test_file_sha256",
    )
    _validate_amendment_catalog_binding(
        amendment.get("reference_catalog_binding"),
        upstream,
        label="LIR-CMOP/reference",
    )


def _validate_reference_catalog_binding(
    upstream: Mapping[str, Any],
) -> Mapping[str, Any]:
    binding = upstream.get("reference_catalog")
    if not isinstance(binding, Mapping):
        raise ConfigurationError("E1+E2 reference catalog binding is missing")
    if binding.get("catalog_id") != R8C_E1E2_REFERENCE_CATALOG_ID:
        raise ConfigurationError("E1+E2 reference catalog identity differs")
    manifest_path = _validate_project_file_binding(
        binding,
        label="E1+E2 reference catalog manifest",
        path_key="manifest_path",
        sha256_key="manifest_sha256",
    )
    schema_path = _validate_project_file_binding(
        binding,
        label="E1+E2 reference catalog manifest schema",
        path_key="manifest_schema_path",
        sha256_key="manifest_schema_sha256",
    )
    artifact_path = _validate_project_file_binding(
        binding,
        label="E1+E2 reference catalog artifact",
        path_key="artifact_path",
        sha256_key="artifact_sha256",
        bytes_key="artifact_bytes",
    )
    expected_lines = binding.get("artifact_lines")
    if type(expected_lines) is not int or expected_lines != 2_294:
        raise ConfigurationError("E1+E2 reference catalog line binding differs")

    manifest = _read_json(manifest_path)
    _validate_json_schema(
        manifest,
        schema_path,
        label="R8C E1+E2 reference catalog manifest",
    )
    expected_artifact = {
        "path": binding.get("artifact_path"),
        "bytes": binding.get("artifact_bytes"),
        "lines": expected_lines,
        "sha256": binding.get("artifact_sha256"),
    }
    authority_amendments = manifest.get("authority_amendments")
    identity_scope = manifest.get("identity_scope")
    if (
        manifest.get("catalog_id") != binding.get("catalog_id")
        or manifest.get("status")
        != "FROZEN_RESULT_BLIND_REFERENCE_INPUT_NOT_EXECUTION_AUTHORITY"
        or not isinstance(authority_amendments, Mapping)
        or authority_amendments.get("lircmop")
        != R8C_E1E2_LIRCMOP_REFERENCE_AMENDMENT_ID
        or authority_amendments.get("cdf")
        != R8C_E1E2_CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
        or manifest.get("catalog_artifact") != expected_artifact
        or not isinstance(identity_scope, Mapping)
        or identity_scope.get("expected_total") != 2_294
        or identity_scope.get("actual_total") != 2_294
    ):
        raise ConfigurationError(
            "E1+E2 reference catalog manifest identity/scope differs"
        )

    source_bindings = manifest.get("source_bindings")
    required_local_sources = {
        "cdf_authority_audit",
        "cdf_corrective_evaluator",
        "historical_evaluator",
        "lircmop_paper_evaluator",
        "reference_derivation",
        "reference_identity_model",
        "analytic_scale",
        "generator",
    }
    if (
        not isinstance(source_bindings, Mapping)
        or not required_local_sources.issubset(source_bindings)
    ):
        raise ConfigurationError(
            "E1+E2 reference catalog source bindings are incomplete"
        )
    for name in sorted(required_local_sources):
        _validate_project_file_binding(
            source_bindings[name],
            label=f"E1+E2 reference catalog {name}",
            bytes_key="bytes",
        )

    from analysis.reference_catalog import (
        ReferenceArtifactError,
        load_reference_catalog,
    )

    try:
        derivations = load_reference_catalog(
            artifact_path,
            expected_sha256=str(binding["artifact_sha256"]),
            expected_lines=expected_lines,
        )
    except (OSError, ReferenceArtifactError, TypeError, ValueError) as error:
        raise ConfigurationError(
            "E1+E2 reference catalog records failed strict validation"
        ) from error
    if len(derivations) != 2_294:
        raise ConfigurationError(
            "E1+E2 reference catalog identity count differs"
        )
    return manifest


def _validate_amendment_catalog_binding(
    catalog_binding: Any,
    upstream: Mapping[str, Any],
    *,
    label: str,
) -> None:
    independent = upstream.get("reference_catalog")
    if not isinstance(catalog_binding, Mapping) or not isinstance(
        independent,
        Mapping,
    ):
        raise ConfigurationError(f"{label} reference catalog binding is missing")
    manifest_path = _launch_path(
        str(independent.get("manifest_path", "")),
        relative_to_project=True,
    )
    schema_path = _launch_path(
        str(independent.get("manifest_schema_path", "")),
        relative_to_project=True,
    )
    expected_file_bindings = {
        "manifest": {
            "path": independent.get("manifest_path"),
            "bytes": (
                manifest_path.stat().st_size if manifest_path.is_file() else None
            ),
            "sha256": independent.get("manifest_sha256"),
        },
        "manifest_schema": {
            "path": independent.get("manifest_schema_path"),
            "bytes": (
                schema_path.stat().st_size if schema_path.is_file() else None
            ),
            "sha256": independent.get("manifest_schema_sha256"),
        },
        "artifact": {
            "path": independent.get("artifact_path"),
            "bytes": independent.get("artifact_bytes"),
            "lines": independent.get("artifact_lines"),
            "sha256": independent.get("artifact_sha256"),
        },
    }
    if (
        catalog_binding.get("catalog_id")
        != independent.get("catalog_id")
        or any(
            catalog_binding.get(name) != expected
            for name, expected in expected_file_bindings.items()
        )
    ):
        raise ConfigurationError(
            f"{label} reference catalog binding differs from the contract"
        )


def _validate_cdf_operational_authority_amendment(
    upstream: Mapping[str, Any],
) -> None:
    binding = upstream.get("cdf_operational_authority_amendment")
    if not isinstance(binding, Mapping):
        raise ConfigurationError(
            "E1+E2 CDF operational authority amendment binding is missing"
        )
    if (
        binding.get("amendment_id")
        != R8C_E1E2_CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
    ):
        raise ConfigurationError(
            "E1+E2 CDF operational authority amendment identity differs"
        )
    amendment_path = _validate_project_file_binding(
        binding,
        label="E1+E2 CDF operational authority amendment",
    )
    schema_path = _validate_project_file_binding(
        binding,
        label="E1+E2 CDF operational authority amendment schema",
        path_key="schema_path",
        sha256_key="schema_sha256",
    )
    amendment = _read_json(amendment_path)
    _validate_json_schema(
        amendment,
        schema_path,
        label="R8C E1+E2 CDF operational authority amendment",
    )
    if amendment.get("amendment_id") != binding.get("amendment_id"):
        raise ConfigurationError(
            "CDF operational authority amendment identity differs"
        )

    historical = amendment.get("historical_freeze")
    if not isinstance(historical, Mapping):
        raise ConfigurationError(
            "CDF operational authority historical binding is missing"
        )
    for name, label in (
        ("r5_contract", "CDF operational authority R5 contract"),
        (
            "r4_benchmark_registry",
            "CDF operational authority R4 benchmark registry",
        ),
    ):
        _validate_project_file_binding(
            historical.get(name),
            label=label,
            bytes_key="bytes",
        )

    authority = amendment.get("authority_binding")
    if not isinstance(authority, Mapping):
        raise ConfigurationError(
            "CDF operational authority source binding is missing"
        )
    _validate_project_file_binding(
        authority.get("audit_document"),
        label="CDF operational authority audit",
        bytes_key="bytes",
    )
    operational = amendment.get("operational_binding")
    if (
        not isinstance(operational, Mapping)
        or operational.get("corrective_suite_id")
        != R8C_E1E2_CDF_OPERATIONAL_SUITE_ID
        or operational.get("authority_id")
        != R8C_E1E2_CDF_OPERATIONAL_AUTHORITY_ID
    ):
        raise ConfigurationError(
            "CDF operational suite/authority identity differs"
        )

    implementation_bindings = amendment.get("implementation_bindings")
    required_bindings = {
        "corrective_evaluator",
        "historical_evaluator",
        "suite_registry",
        "corrective_factory",
        "reference_derivation",
        "reference_identity_model",
        "analytic_scale",
    }
    if (
        not isinstance(implementation_bindings, Mapping)
        or set(implementation_bindings) != required_bindings
    ):
        raise ConfigurationError(
            "CDF operational implementation binding set differs"
        )
    for name in sorted(required_bindings):
        _validate_project_file_binding(
            implementation_bindings[name],
            label=f"CDF operational {name}",
            bytes_key="bytes",
        )

    verification = amendment.get("verification")
    test_files = (
        verification.get("test_files")
        if isinstance(verification, Mapping)
        else None
    )
    if not isinstance(test_files, list) or len(test_files) != 2:
        raise ConfigurationError(
            "CDF operational verification bindings are incomplete"
        )
    for index, test_binding in enumerate(test_files):
        _validate_project_file_binding(
            test_binding,
            label=f"CDF operational verification test {index + 1}",
        )

    expected_failure_policy = {
        "source_domain": "Q_EQUALS_1_MINUS_M_X1_TO_THE_H_MUST_BE_GTE_0",
        "undefined_real_domain_action": (
            "CHARGE_ONCE_AND_RAISE_CDFDomainUndefinedError"
        ),
        "ledger_failure_type": "CDFDomainUndefinedError",
        "external_terminal": "REJECT_NUMERICAL",
        "batch_preentry_action": (
            "RAISE_BatchEvaluationUnavailableBeforeEntry_WITH_ZERO_LEDGER_"
            "SIDE_EFFECTS_THEN_ORDERED_SCALAR_FALLBACK"
        ),
        "domain_extension_allowed": False,
        "clamp_allowed": False,
        "sign_extension_allowed": False,
        "bound_narrowing_allowed": False,
        "resampling_allowed": False,
        "automatic_retry_allowed": False,
    }
    if amendment.get("cdf9_failure_policy") != expected_failure_policy:
        raise ConfigurationError("CDF9 typed failure policy differs")
    _validate_amendment_catalog_binding(
        amendment.get("reference_catalog_binding"),
        upstream,
        label="CDF operational authority",
    )


def _runtime_environment_lock_evidence() -> Mapping[str, Any]:
    from resource_pilot.e1e2_fullpath import runtime_environment_lock_evidence

    return runtime_environment_lock_evidence()


def _validate_linux_runtime_lock(
    upstream: Mapping[str, Any],
    report_environment: Any,
) -> None:
    binding = upstream.get("linux_runtime_lock")
    if not isinstance(binding, Mapping):
        raise ConfigurationError("E1+E2 Linux runtime lock binding is missing")
    _validate_project_file_binding(
        binding,
        label="E1+E2 Linux runtime lock",
        bytes_key="bytes",
    )
    if (
        binding.get("locked_package_count") != 35
        or binding.get("target_interpreter") != "CPython 3.12"
        or binding.get("target_platform")
        != "Linux x86_64 manylinux2014-compatible"
    ):
        raise ConfigurationError("E1+E2 Linux runtime lock target differs")
    try:
        live_environment = _runtime_environment_lock_evidence()
    except Exception as error:
        raise ConfigurationError(
            "live Linux runtime lock evidence could not be collected"
        ) from error
    expected_binding = {
        "lock_path": binding.get("path"),
        "lock_sha256": binding.get("sha256"),
        "lock_bytes": binding.get("bytes"),
        "target_interpreter": binding.get("target_interpreter"),
        "target_platform": binding.get("target_platform"),
        "locked_package_count": binding.get("locked_package_count"),
    }
    if (
        not isinstance(live_environment, Mapping)
        or any(
            live_environment.get(name) != value
            for name, value in expected_binding.items()
        )
        or live_environment.get("target_environment_match") is not True
        or live_environment.get("interpreter_matches") is not True
        or live_environment.get("platform_matches") is not True
        or live_environment.get("all_locked_packages_match") is not True
    ):
        raise ConfigurationError(
            "live environment differs from the frozen Linux runtime lock"
        )
    if (
        not isinstance(report_environment, Mapping)
        or dict(report_environment) != dict(live_environment)
    ):
        raise ConfigurationError(
            "qualification environment differs from the live formal environment"
        )


def _validate_target_qualification_evidence(
    contract: Mapping[str, Any],
    request: Any,
) -> None:
    from resource_pilot.e1e2_fullpath import (
        CDF_OPERATIONAL_AUTHORITY_ID,
        CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID,
        CDF_OPERATIONAL_SUITE_ID,
        DEFAULT_DYNAMIC_EVENTS,
        DEFAULT_REPETITIONS,
        DYNAMIC_CFE_PER_EVENT,
        QUALIFICATION_ID,
        ROLLING_CFE_PER_EVENT,
        RSS_ROUNDING_BYTES,
        RSS_SAFETY_FACTOR,
        STATIC_CFE_PER_EVENT,
        SUPERVISOR_MEMORY_RESERVE_BYTES,
        qualification_profiles,
        formal_schedule_weights,
        r5_memory_limits,
    )

    expected_profiles = qualification_profiles(DEFAULT_DYNAMIC_EVENTS)
    expected_profile_rows = [asdict(profile) for profile in expected_profiles]
    expected_weights = [dict(row) for row in formal_schedule_weights()]
    expected_path_count = len({profile.key for profile in expected_profiles})
    expected_binding_count = len(
        {profile.case_key for profile in expected_profiles}
    )
    expected_case_count = len(
        {profile.representative_case_id for profile in expected_profiles}
    )
    expected_rate_count = len(
        {profile.rate_key for profile in expected_profiles}
    )
    expected_sweep_task_count = (
        DEFAULT_REPETITIONS * expected_binding_count
    )
    expected_total_task_count = (
        len(R8C_E1E2_TARGET_QUALIFICATION_WORKERS)
        * expected_sweep_task_count
    )
    expected_design = {
        "worker_counts": list(R8C_E1E2_TARGET_QUALIFICATION_WORKERS),
        "repetitions": DEFAULT_REPETITIONS,
        "static_cfe_per_event": STATIC_CFE_PER_EVENT,
        "dynamic_cfe_per_event": DYNAMIC_CFE_PER_EVENT,
        "rolling_cfe_per_event": ROLLING_CFE_PER_EVENT,
        "dynamic_events": DEFAULT_DYNAMIC_EVENTS,
        "workload_method_case_binding_count": expected_binding_count,
        "unique_representative_benchmark_case_count": expected_case_count,
        "formal_projection_rate_class_count": expected_rate_count,
        "task_count": expected_total_task_count,
    }
    r5_max_worker_rss_bytes, r5_max_pool_rss_bytes = r5_memory_limits()
    if QUALIFICATION_ID != R8C_E1E2_TARGET_QUALIFICATION_ID:
        raise ConfigurationError(
            "target qualification generator identity differs from runner"
        )

    evidence = contract.get("target_qualification_evidence")
    if not isinstance(evidence, Mapping):
        raise ConfigurationError(
            "target qualification evidence binding is missing"
        )
    report_path_value = evidence.get("qualification_report_path")
    if not isinstance(report_path_value, str) or not report_path_value:
        raise ConfigurationError(
            "target qualification report path is missing"
        )
    unresolved_report_path = Path(report_path_value)
    if not unresolved_report_path.is_absolute():
        raise ConfigurationError(
            "target qualification report path must be absolute"
        )
    report_path = unresolved_report_path.resolve()
    if _is_relative_to(report_path, PROJECT_ROOT.resolve()):
        raise ConfigurationError(
            "target qualification report must be outside the source worktree"
        )
    if (
        not report_path.is_file()
        or file_sha256(report_path)
        != evidence.get("qualification_report_sha256")
    ):
        raise ConfigurationError(
            "target qualification report hash drifted"
        )
    report = _read_json(report_path)
    try:
        canonical_report = canonical_json_bytes(report) + b"\n"
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "target qualification report is not canonical JSON"
        ) from error
    if report_path.read_bytes() != canonical_report:
        raise ConfigurationError(
            "target qualification report is not canonical JSON with one LF"
        )
    if (
        report.get("artifact_role")
        != "R8C_E1E2_RESULT_BLIND_FULL_PATH_TARGET_QUALIFICATION"
        or evidence.get("qualification_id")
        != R8C_E1E2_TARGET_QUALIFICATION_ID
        or evidence.get("qualification_status")
        != R8C_E1E2_TARGET_QUALIFICATION_STATUS
        or report.get("qualification_id")
        != R8C_E1E2_TARGET_QUALIFICATION_ID
        or report.get("status")
        != R8C_E1E2_TARGET_QUALIFICATION_STATUS
        or report.get("mode") != "TARGET_QUALIFICATION"
        or report.get("target_qualification_complete") is not True
        or report.get("formal_launch_authorized") is not False
        or report.get("failed_task_count") != 0
        or report.get("automatic_retries") != 0
        or report.get("real_effect_values_persisted") is not False
        or report.get("formal_execution_started") is not False
    ):
        raise ConfigurationError(
            "target qualification report is not a clean completed evidence set"
        )
    report_output_root = report.get("output_root")
    if (
        not isinstance(report_output_root, str)
        or Path(report_output_root).resolve() != report_path.parent
    ):
        raise ConfigurationError(
            "target qualification report path differs from its output root"
        )

    source = evidence.get("source")
    report_source = report.get("code_identity")
    if (
        not isinstance(source, Mapping)
        or not isinstance(report_source, Mapping)
        or dict(source) != dict(report_source)
        or source.get("worktree_clean") is not True
        or source.get("git_commit")
        != request.contracts.source_git_commit
        or source.get("git_tree") != request.contracts.source_git_tree
        or source.get("qualification_source_sha256")
        != file_sha256(
            PROJECT_ROOT / "src" / "resource_pilot" / "e1e2_fullpath.py"
        )
    ):
        raise ConfigurationError(
            "target qualification clean source differs from the formal source"
        )
    _validate_linux_runtime_lock(
        contract.get("upstream", {}),
        report.get("runtime_environment_lock"),
    )

    report_host = report.get("host_fingerprint")
    host_sha256 = evidence.get("host_fingerprint_sha256")
    candidate_target = contract.get("resources", {}).get(
        "candidate_target",
        {},
    )
    if (
        not isinstance(report_host, Mapping)
        or report.get("host_fingerprint_sha256") != host_sha256
        or host_fingerprint_sha256(report_host) != host_sha256
        or candidate_target.get("host_fingerprint_sha256") != host_sha256
    ):
        raise ConfigurationError(
            "target qualification host differs from the selected formal host"
        )

    design = evidence.get("design")
    pilot = report.get("pilot_design")
    runtime = report.get("runtime_contract")
    if (
        not isinstance(design, Mapping)
        or not isinstance(pilot, Mapping)
        or not isinstance(runtime, Mapping)
        or tuple(pilot.get("worker_counts", ()))
        != R8C_E1E2_TARGET_QUALIFICATION_WORKERS
        or pilot.get("repetitions") != DEFAULT_REPETITIONS
        or pilot.get("static_cfe_per_event") != STATIC_CFE_PER_EVENT
        or pilot.get("dynamic_cfe_per_event") != DYNAMIC_CFE_PER_EVENT
        or pilot.get("rolling_cfe_per_event") != ROLLING_CFE_PER_EVENT
        or pilot.get("dynamic_events") != DEFAULT_DYNAMIC_EVENTS
        or pilot.get("task_count") != expected_total_task_count
        or pilot.get("all_task_ids_unique") is not True
        or runtime.get("workload_method_case_bindings_covered")
        != expected_binding_count
        or runtime.get("unique_representative_benchmark_cases_covered")
        != expected_case_count
        or runtime.get("formal_projection_rate_classes_covered")
        != expected_rate_count
        or runtime.get("workload_method_paths_covered")
        != expected_path_count
        or runtime.get("cdf_operational_suite_id")
        != CDF_OPERATIONAL_SUITE_ID
        or runtime.get("cdf_operational_authority_id")
        != CDF_OPERATIONAL_AUTHORITY_ID
        or runtime.get("cdf_operational_authority_amendment_id")
        != CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
        or runtime.get("cdf9_undefined_domain_policy")
        != (
            "CHARGED_TYPED_CDFDomainUndefinedError_TO_"
            "REJECT_NUMERICAL_NO_EXTENSION"
        )
        or runtime.get("dynamic_event_ids_covered")
        != list(range(DEFAULT_DYNAMIC_EVENTS))
        or runtime.get("cdf9_max_domain_stress_event_id") != 5
        or runtime.get("cdf9_max_domain_stress_event_exercised") is not True
        or dict(design) != expected_design
    ):
        raise ConfigurationError(
            "target qualification report differs from the exact design"
        )
    profiles = pilot.get("profiles")
    weights = report.get("formal_schedule_control_weights")
    if (
        not isinstance(profiles, list)
        or profiles != expected_profile_rows
        or not isinstance(weights, list)
        or weights != expected_weights
    ):
        raise ConfigurationError(
            "target qualification representative matrix is incomplete"
        )
    case_keys = {
        (
            row.get("workload_id"),
            row.get("method_id"),
            row.get("representative_case_id"),
        )
        for row in profiles
    }
    representative_case_ids = {
        row.get("representative_case_id") for row in profiles
    }
    profile_rate_keys = {
        (
            row.get("workload_id"),
            row.get("method_id"),
            row.get("projection_rate_class"),
        )
        for row in profiles
    }
    weight_rate_keys = {
        (
            row.get("workload_id"),
            row.get("method_id"),
            row.get("projection_rate_class"),
        )
        for row in weights
    }
    if (
        len(case_keys) != expected_binding_count
        or None in representative_case_ids
        or len(representative_case_ids) != expected_case_count
        or len(profile_rate_keys) != expected_rate_count
        or profile_rate_keys != weight_rate_keys
    ):
        raise ConfigurationError(
            "target qualification case/rate identities are incomplete"
        )

    sweeps = report.get("sweeps")
    wall_projection = report.get("e1_e2_wall_projection")
    projections = (
        wall_projection.get("projections")
        if isinstance(wall_projection, Mapping)
        else None
    )
    if (
        not isinstance(sweeps, list)
        or not isinstance(projections, list)
        or not all(isinstance(row, Mapping) for row in sweeps)
        or not all(isinstance(row, Mapping) for row in projections)
        or tuple(row.get("workers_requested") for row in sweeps)
        != R8C_E1E2_TARGET_QUALIFICATION_WORKERS
        or tuple(row.get("workers") for row in projections)
        != R8C_E1E2_TARGET_QUALIFICATION_WORKERS
        or any(
            row.get("task_count") != expected_sweep_task_count
            or row.get("passed_task_count") != expected_sweep_task_count
            or row.get("failed_task_count") != 0
            for row in sweeps
        )
        or any(
            row.get("status") != "TARGET_HOST_FULL_PATH_ESTIMATE"
            or len(row.get("method_cfe_weighted_rates", ()))
            != expected_rate_count
            for row in projections
        )
    ):
        raise ConfigurationError(
            "target qualification worker sweep evidence is incomplete"
        )

    sweeps_by_worker = {
        int(row["workers_requested"]): row for row in sweeps
    }
    expected_weight_map = {
        (
            row["workload_id"],
            row["method_id"],
            row["projection_rate_class"],
        ): (
            int(row["formal_task_count"]),
            int(row["formal_cfe"]),
        )
        for row in expected_weights
    }
    host_memory_bytes = int(report_host.get("memory_bytes", 0))
    memory_eligible_projections: list[Mapping[str, Any]] = []
    for projection in projections:
        worker_count = int(projection["workers"])
        sweep = sweeps_by_worker[worker_count]
        memory = projection.get("memory_qualification")
        rate_rows = projection.get("method_cfe_weighted_rates")
        if (
            not isinstance(memory, Mapping)
            or not isinstance(rate_rows, list)
            or not all(isinstance(row, Mapping) for row in rate_rows)
        ):
            raise ConfigurationError(
                "target qualification projection details are incomplete"
            )
        measured_worker_rss = int(
            sweep.get("max_task_peak_rss_bytes", 0)
        )
        conservative_worker_rss = (
            math.ceil(
                measured_worker_rss
                * RSS_SAFETY_FACTOR
                / RSS_ROUNDING_BYTES
            )
            * RSS_ROUNDING_BYTES
        )
        conservative_pool_rss = conservative_worker_rss * worker_count
        host_pool_ceiling = max(
            0,
            host_memory_bytes - SUPERVISOR_MEMORY_RESERVE_BYTES,
        )
        memory_eligible = (
            measured_worker_rss > 0
            and conservative_worker_rss <= r5_max_worker_rss_bytes
            and conservative_pool_rss <= r5_max_pool_rss_bytes
            and conservative_pool_rss <= host_pool_ceiling
        )
        expected_memory = {
            "measured_max_worker_peak_rss_bytes": measured_worker_rss,
            "rss_safety_factor": RSS_SAFETY_FACTOR,
            "rss_rounding_bytes": RSS_ROUNDING_BYTES,
            "conservative_worker_peak_rss_bytes": (
                conservative_worker_rss
            ),
            "worker_count": worker_count,
            "conservative_pool_peak_rss_bytes": conservative_pool_rss,
            "r5_max_worker_peak_rss_bytes": r5_max_worker_rss_bytes,
            "r5_max_pool_peak_rss_bytes": r5_max_pool_rss_bytes,
            "host_memory_bytes": host_memory_bytes,
            "supervisor_memory_reserve_bytes": (
                SUPERVISOR_MEMORY_RESERVE_BYTES
            ),
            "host_pool_ceiling_bytes": host_pool_ceiling,
            "eligible": memory_eligible,
        }
        rate_map = {
            (
                row.get("workload_id"),
                row.get("method_id"),
                row.get("projection_rate_class"),
            ): (
                row.get("formal_task_count"),
                row.get("formal_cfe"),
            )
            for row in rate_rows
        }
        projected_seconds = projection.get("projected_wall_seconds")
        projected_hours = projection.get("projected_wall_hours")
        headroom_hours = projection.get(
            "projected_wall_hours_with_25_percent_headroom"
        )
        if (
            dict(memory) != expected_memory
            or rate_map != expected_weight_map
            or len(rate_map) != expected_rate_count
            or projection.get("formal_task_count") != 5_030
            or projection.get("formal_cfe") != 851_000_000
            or not _is_nonnegative_finite_number(projected_seconds)
            or not _is_nonnegative_finite_number(projected_hours)
            or not _is_nonnegative_finite_number(headroom_hours)
            or not math.isclose(
                float(projected_hours),
                float(projected_seconds) / 3_600.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(headroom_hours),
                float(projected_hours) * 1.25,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ConfigurationError(
                "target qualification time/RSS projection is inconsistent"
            )
        if not memory_eligible:
            expected_decision = "NO_GO_R5_OR_HOST_RSS_CEILING"
        elif float(projected_hours) <= 36.0:
            expected_decision = (
                "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
            )
        elif float(projected_hours) <= 48.0:
            expected_decision = "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
        else:
            expected_decision = "NO_GO_PROJECTED_OVER_48_HOURS"
        if projection.get("decision_classification") != expected_decision:
            raise ConfigurationError(
                "target qualification projection decision is inconsistent"
            )
        if memory_eligible:
            memory_eligible_projections.append(projection)

    recommended = (
        min(
            memory_eligible_projections,
            key=lambda row: (
                float(row["projected_wall_hours"]),
                int(row["workers"]),
            ),
        )
        if memory_eligible_projections
        else None
    )
    recommendation = report.get("worker_recommendation")
    expected_recommendation = {
        "status": (
            "MEMORY_ELIGIBLE_THROUGHPUT_OPTIMUM_IDENTIFIED"
            if recommended is not None
            else "NO_MEMORY_ELIGIBLE_WORKER_COUNT"
        ),
        "selection_rule": (
            "MIN_PROJECTED_WALL_HOURS_AMONG_R5_AND_HOST_RSS_ELIGIBLE_"
            "MEASURED_WORKER_COUNTS;_LOWER_WORKER_COUNT_BREAKS_EXACT_TIES"
        ),
        "measured_worker_counts": list(
            R8C_E1E2_TARGET_QUALIFICATION_WORKERS
        ),
        "memory_eligible_worker_counts": [
            int(row["workers"]) for row in memory_eligible_projections
        ],
        "recommended_worker_count": (
            None if recommended is None else int(recommended["workers"])
        ),
        "recommended_projected_wall_hours": (
            None
            if recommended is None
            else float(recommended["projected_wall_hours"])
        ),
        "recommended_projected_wall_hours_with_25_percent_headroom": (
            None
            if recommended is None
            else float(
                recommended[
                    "projected_wall_hours_with_25_percent_headroom"
                ]
            )
        ),
        "recommended_decision_classification": (
            None
            if recommended is None
            else recommended["decision_classification"]
        ),
        "formal_launch_authorized": False,
    }
    if (
        not isinstance(recommendation, Mapping)
        or dict(recommendation) != expected_recommendation
    ):
        raise ConfigurationError(
            "target qualification worker recommendation is inconsistent"
        )

    selected_worker = evidence.get("selected_worker_count")
    parallelism = contract.get("resources", {}).get("parallelism", {})
    selected = next(
        (
            row
            for row in projections
            if row.get("workers") == selected_worker
        ),
        None,
    )
    frozen_projection = evidence.get("selected_projection")
    projected_hours = (
        selected.get("projected_wall_hours")
        if isinstance(selected, Mapping)
        else None
    )
    if (
        selected_worker not in R8C_E1E2_TARGET_QUALIFICATION_WORKERS
        or recommended is None
        or selected_worker != recommended.get("workers")
        or parallelism.get("max_workers") != selected_worker
        or not isinstance(selected, Mapping)
        or not isinstance(frozen_projection, Mapping)
        or dict(frozen_projection)
        != {
            "status": selected.get("status"),
            "projected_wall_hours": projected_hours,
            "decision_classification": selected.get(
                "decision_classification"
            ),
        }
        or not _is_nonnegative_finite_number(projected_hours)
        or not 0.0 < float(projected_hours) <= 36.0
        or selected.get("decision_classification")
        != "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
    ):
        raise ConfigurationError(
            "selected worker lacks a frozen <=36h GO projection"
        )


def _load_and_validate(
    contract_path: Path,
    request_path: Path,
    profile: RunnerProfile = LEGACY_PROFILE,
) -> tuple[Mapping[str, Any], Any, list[Any]]:
    contract = _read_json(contract_path)
    if contract.get("contract_id") != profile.contract_id:
        raise ConfigurationError(
            f"unexpected {profile.artifact_stage} formal contract identity"
        )
    _validate_json_schema(
        contract,
        profile.contract_schema,
        label=f"{profile.artifact_stage} contract",
    )
    if profile.target_qualification_required:
        authorization = contract.get("authorization", {})
        launch = contract.get("launch", {})
        resources = contract.get("resources", {})
        permissions = contract.get("permissions", {})
        fail_closed = contract.get("fail_closed_gate", {})
        candidate_target = resources.get("candidate_target", {})
        parallelism = resources.get("parallelism", {})
        permissions_ready = all(
            permissions.get(name) is True
            for name in profile.required_effect_permissions
        )
        launch_ready = (
            contract.get("status")
            == "TARGET_HOST_QUALIFIED_AND_AUTHORIZED"
            and resources.get("qualification_status")
            == "TARGET_HOST_QUALIFIED"
            and resources.get("selected_exact_host_frozen") is True
            and authorization.get("formal_effect_execution_authorized")
            is True
            and launch.get("command_executable_now") is True
            and launch.get("formal_launch_prohibited") is False
            and launch.get("command_identity_frozen") is True
            and launch.get("one_time_consumption_required") is True
            and launch.get("overwrite_allowed") is False
            and candidate_target.get("remote_measurement_completed") is True
            and isinstance(candidate_target.get("provider"), str)
            and bool(candidate_target.get("provider", "").strip())
            and isinstance(candidate_target.get("instance_type"), str)
            and bool(candidate_target.get("instance_type", "").strip())
            and parallelism.get("worker_count_qualified_on_target") is True
            and permissions_ready
            and fail_closed.get("request_status")
            == "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED"
            and fail_closed.get("target_host_status")
            == "TARGET_HOST_QUALIFIED"
            and fail_closed.get("formal_launch_status") == "ELIGIBLE"
        )
        if not launch_ready:
            raise ConfigurationError(
                "R8C contract is fail-closed: exact target host is not "
                "qualified for formal launch"
            )
    request_payload = _read_json(request_path)
    _validate_json_schema(
        request_payload,
        profile.request_schema,
        label=f"{profile.artifact_stage} request",
    )
    request = profile.parse_request(request_payload)
    if profile.target_qualification_required:
        fail_closed = contract["fail_closed_gate"]
        if fail_closed.get("request_id") != request.request_id:
            raise ConfigurationError(
                "qualified contract request identity differs from request"
            )
    expected_contract_hash = getattr(
        request.contracts,
        profile.contract_hash_binding,
    )
    if file_sha256(contract_path) != expected_contract_hash:
        raise ConfigurationError(
            f"{profile.artifact_stage} contract hash differs from request"
        )
    upstream = contract["upstream"]
    for label, path, expected in (
        (
            "R5",
            PROJECT_ROOT / upstream["r5"]["path"],
            upstream["r5"]["sha256"],
        ),
        (
            "R5a",
            PROJECT_ROOT / upstream["r5a"]["path"],
            upstream["r5a"]["sha256"],
        ),
    ):
        if file_sha256(path) != expected:
            raise ConfigurationError(f"{label} contract hash drifted")
    if profile.profile_id == "corrective_r8c_e1e2":
        _validate_rng_implementation_amendment(upstream)
        _validate_compact_runtime_amendment(upstream)
        _validate_timeout_semantics_amendment(upstream)
        _validate_reference_catalog_binding(upstream)
        _validate_lircmop_reference_amendment(upstream)
        _validate_cdf_operational_authority_amendment(upstream)
        _validate_target_qualification_evidence(contract, request)
        persistence_binding = upstream.get("checkpoint_persistence")
        if not isinstance(persistence_binding, Mapping):
            raise ConfigurationError(
                "E1+E2 checkpoint persistence binding is missing"
            )
        persistence_path = _launch_path(
            persistence_binding.get("path", ""),
            relative_to_project=True,
        )
        persistence_schema_path = _launch_path(
            persistence_binding.get("schema_path", ""),
            relative_to_project=True,
        )
        if (
            not _is_relative_to(
                persistence_path,
                PROJECT_ROOT.resolve(),
            )
            or not _is_relative_to(
                persistence_schema_path,
                PROJECT_ROOT.resolve(),
            )
        ):
            raise ConfigurationError(
                "checkpoint persistence contract and schema must be "
                "inside the project"
            )
        if (
            persistence_binding.get("contract_id")
            != "WGT-V11-R8C-E1E2-CHECKPOINT-PERSISTENCE-CONTRACT-01"
            or not persistence_path.is_file()
            or not persistence_schema_path.is_file()
            or file_sha256(persistence_path)
            != persistence_binding.get("sha256")
            or file_sha256(persistence_schema_path)
            != persistence_binding.get("schema_sha256")
        ):
            raise ConfigurationError(
                "E1+E2 checkpoint persistence contract or schema drifted"
            )
        persistence = _read_json(persistence_path)
        _validate_json_schema(
            persistence,
            persistence_schema_path,
            label="checkpoint persistence contract",
        )
        if persistence.get("contract_id") != persistence_binding.get(
            "contract_id"
        ):
            raise ConfigurationError(
                "checkpoint persistence contract identity differs"
            )
        envelope = persistence.get("formal_resource_envelope", {})
        output = contract.get("resources", {}).get("output", {})
        scratch = contract.get("resources", {}).get("scratch", {})
        observed_envelope = {
            "max_total_bytes": output.get("max_total_bytes"),
            "control_plane_reserve_bytes": output.get(
                "control_plane_reserve_bytes"
            ),
            "max_inflight_write_bytes_per_worker": output.get(
                "max_inflight_write_bytes_per_worker"
            ),
            "stop_dispatch_below_free_bytes": scratch.get(
                "stop_dispatch_below_free_bytes"
            ),
            "minimum_free_bytes_at_start": scratch.get(
                "minimum_free_bytes_at_start"
            ),
            "raw_evaluations_required": output.get(
                "raw_evaluations_required"
            ),
            "silent_truncation": output.get("silent_truncation"),
        }
        if observed_envelope != dict(envelope):
            raise ConfigurationError(
                "formal resources differ from checkpoint persistence freeze"
            )
        if (
            output.get("format")
            != "WGT_CFE_CHECKPOINT_BINARY_V1_ENDPOINT_SUFFICIENT"
            or profile.runtime_settings.persistence_mode
            != CHECKPOINT_FRONT_PERSISTENCE
        ):
            raise ConfigurationError(
                "formal runtime differs from checkpoint persistence format"
            )
    r5 = _read_json(PROJECT_ROOT / upstream["r5"]["path"])
    if profile.require_r5_runtime_match:
        common = r5.get("common_configuration", {})
        runtime = contract.get("method_runtime", {})
        expected_runtime = {
            "population_size": common.get("population_size"),
            "archive_capacity": common.get("archive_capacity"),
        }
        observed_runtime = {
            "population_size": runtime.get("population_size"),
            "archive_capacity": runtime.get("archive_capacity"),
        }
        if observed_runtime != expected_runtime or observed_runtime != {
            "population_size": profile.runtime_settings.population_size,
            "archive_capacity": profile.runtime_settings.archive_capacity,
        }:
            raise ConfigurationError(
                "formal runtime population/archive differ from R5"
            )
    schedule = profile.build_schedule(r5)
    if schedule_commitment(schedule) != contract["schedule"]["sha256"]:
        raise ConfigurationError(
            "expanded schedule differs from formal freeze"
        )
    if schedule_commitment(schedule) != (
        request.contracts.formal_schedule_sha256
    ):
        raise ConfigurationError(
            "request schedule differs from formal freeze"
        )
    if e2_full_reuse_commitment(schedule) != (
        contract["schedule"]["e2_full_reuse_sha256"]
    ):
        raise ConfigurationError(
            "E2 FULL reuse map differs from formal freeze"
        )
    expected = contract["schedule"]["totals"]
    observed = {
        "method_sequences": len(schedule),
        "CFE": sum(row.total_cfe for row in schedule),
        "atomic_model_steps": sum(
            row.total_atomic_steps for row in schedule
        ),
    }
    if observed != expected:
        raise ConfigurationError("schedule accounting differs from contract")
    return contract, request, schedule


def _validate_invocation(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    request: Any,
) -> Path:
    launch = contract["launch"]
    if Path.cwd().resolve() != _launch_path(
        launch["working_directory"],
        relative_to_project=True,
    ):
        raise ConfigurationError("R8 working directory differs from R7")
    if request.frozen_exact_command != launch["exact_command"]:
        raise ConfigurationError("R8 request command differs from R7")
    contract_path = Path(args.contract).resolve()
    request_path = Path(args.request).resolve()
    if contract_path != _launch_path(
        launch["contract_path"],
        relative_to_project=True,
    ):
        raise ConfigurationError("R8 contract argument differs from R7")
    if request_path != _launch_path(
        launch["request_path"],
        relative_to_project=True,
    ):
        raise ConfigurationError("R8 request argument differs from R7")
    output_root = Path(args.output_root).resolve()
    if output_root != _launch_path(
        launch["output_root"],
        relative_to_project=False,
    ):
        raise ConfigurationError("R8 output root differs from R7")
    if output_root.exists():
        raise ConfigurationError("R8 output root already exists; overwrite denied")
    scratch = contract["resources"]["scratch"]
    if (
        scratch.get("onedrive_path_allowed") is False
        and "onedrive" in str(output_root).casefold()
    ):
        raise ConfigurationError("R8 output root cannot be under OneDrive")
    required_parent = _launch_path(
        scratch["required_root"],
        relative_to_project=False,
    )
    if output_root.parent != required_parent:
        raise ConfigurationError("R8 output root parent differs from R7")
    if not required_parent.is_dir():
        raise ConfigurationError("R8 required scratch root is not a directory")
    return output_root


def _validate_prelaunch(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    request: Any,
    schedule: Sequence[Any],
) -> PrelaunchPlan:
    source = _validate_source(request)
    output_root = _validate_invocation(args, contract, request)
    host = host_fingerprint()
    host_sha256 = host_fingerprint_sha256(host)
    target = contract["resources"]["candidate_target"]
    if target.get("host_fingerprint_sha256") != host_sha256:
        raise ConfigurationError(
            "runtime host fingerprint differs from the target-qualified "
            "contract"
        )
    effective_processors = _strict_positive_int(
        host.get("effective_logical_processors"),
        label="host effective_logical_processors",
    )
    memory_bytes = _strict_positive_int(
        host.get("memory_bytes"),
        label="host effective memory_bytes",
    )

    resources = contract["resources"]
    parallelism = resources["parallelism"]
    forbidden_candidate_fields = {
        "candidate_workers",
        "candidate_pool_peak_rss_bytes",
    }
    present_forbidden = sorted(
        forbidden_candidate_fields.intersection(parallelism)
    )
    if present_forbidden:
        raise ConfigurationError(
            "qualified parallelism retains candidate-only fields: "
            + ", ".join(present_forbidden)
        )
    required_qualified_fields = {
        "max_workers",
        "logical_threads_per_worker",
        "blas_openmp_threads_per_worker",
        "max_worker_peak_rss_bytes",
        "max_pool_peak_rss_bytes",
        "worker_count_qualified_on_target",
    }
    missing = sorted(required_qualified_fields.difference(parallelism))
    if missing:
        raise ConfigurationError(
            "qualified parallelism fields are missing: " + ", ".join(missing)
        )
    if parallelism["worker_count_qualified_on_target"] is not True:
        raise ConfigurationError(
            "formal worker count is not qualified on the exact target"
        )
    max_workers = _strict_positive_int(
        parallelism["max_workers"],
        label="resources.parallelism.max_workers",
    )
    max_worker_rss = _strict_positive_int(
        parallelism["max_worker_peak_rss_bytes"],
        label="resources.parallelism.max_worker_peak_rss_bytes",
    )
    max_pool_rss = _strict_positive_int(
        parallelism["max_pool_peak_rss_bytes"],
        label="resources.parallelism.max_pool_peak_rss_bytes",
    )
    if max_workers > effective_processors:
        raise ConfigurationError(
            "formal max_workers exceeds the cgroup/affinity effective CPU "
            "allocation"
        )
    if (
        parallelism["logical_threads_per_worker"] != 1
        or parallelism["blas_openmp_threads_per_worker"] != 1
        or contract["method_runtime"]["logical_threads_per_worker"] != 1
        or contract["method_runtime"]["blas_openmp_threads_per_worker"] != 1
    ):
        raise ConfigurationError(
            "formal workers and numerical libraries must remain single-threaded"
        )
    if max_workers * max_worker_rss > max_pool_rss:
        raise ConfigurationError(
            "qualified pool RSS cap cannot cover all qualified workers"
        )
    if max_pool_rss + SUPERVISOR_MEMORY_RESERVE_BYTES > memory_bytes:
        raise ConfigurationError(
            "qualified pool RSS plus supervisor reserve exceeds effective "
            "host memory"
        )

    monitor_seconds = _strict_positive_number(
        resources["monitor"]["interval_seconds"],
        label="resources.monitor.interval_seconds",
    )
    global_timeout = _strict_positive_number(
        resources["timeouts_seconds"]["global_formal_wall"],
        label="resources.timeouts_seconds.global_formal_wall",
    )
    output_contract = resources["output"]
    max_output = _strict_positive_int(
        output_contract["max_total_bytes"],
        label="resources.output.max_total_bytes",
    )
    max_cpu = _strict_positive_number(
        resources["max_total_cpu_seconds"],
        label="resources.max_total_cpu_seconds",
    )
    scratch = resources["scratch"]
    stop_free = _strict_positive_int(
        scratch["stop_dispatch_below_free_bytes"],
        label="resources.scratch.stop_dispatch_below_free_bytes",
    )
    minimum_free = _strict_positive_int(
        scratch["minimum_free_bytes_at_start"],
        label="resources.scratch.minimum_free_bytes_at_start",
    )
    endpoint_sufficient_e1e2 = contract.get("contract_id") == (
        "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
        "FORMAL-EXECUTION-CONTRACT-01"
    )
    if endpoint_sufficient_e1e2:
        if {
            "control_plane_reserve_bytes",
            "max_inflight_write_bytes_per_worker",
        } - set(output_contract):
            raise ConfigurationError(
                "E1+E2 output contract lacks frozen write reserves"
            )
    control_plane_reserve = _strict_positive_int(
        output_contract.get(
            "control_plane_reserve_bytes",
            CONTROL_PLANE_RESERVE_BYTES,
        ),
        label="resources.output.control_plane_reserve_bytes",
    )
    inflight_per_worker = _strict_positive_int(
        output_contract.get(
            "max_inflight_write_bytes_per_worker",
            ACTIVE_TASK_WRITE_RESERVE_BYTES,
        ),
        label=(
            "resources.output.max_inflight_write_bytes_per_worker"
        ),
    )
    active_write_reserve = (
        control_plane_reserve
        + max_workers * inflight_per_worker
    )
    required_start_free = max_output + stop_free + active_write_reserve
    if minimum_free < required_start_free:
        raise ConfigurationError(
            "declared start free-space gate cannot cover output ceiling, "
            "stop floor and in-flight/control reserve"
        )
    actual_free = shutil.disk_usage(output_root.parent).free
    if actual_free < minimum_free:
        raise ConfigurationError("R8 scratch free space is below start gate")

    marker_path = _launch_path(
        contract["launch"]["request_consumption_marker"],
        relative_to_project=True,
    )
    if marker_path.exists():
        raise ConfigurationError(
            "formal request was already consumed or marker path is occupied"
        )
    if _is_relative_to(marker_path, output_root):
        raise ConfigurationError(
            "request consumption marker cannot be inside the output root"
        )
    if _is_relative_to(marker_path, PROJECT_ROOT.resolve()):
        raise ConfigurationError(
            "request consumption marker must be outside the source worktree"
        )
    if not marker_path.parent.is_dir() or not os.access(
        marker_path.parent,
        os.W_OK,
    ):
        raise ConfigurationError(
            "request consumption marker parent is not a writable directory"
        )
    if not os.access(output_root.parent, os.W_OK):
        raise ConfigurationError("formal output parent is not writable")
    if any(row.workload_id == "E3" for row in schedule):
        if (
            contract.get("schedule", {}).get("e3_dispatched") is False
            or contract.get("method_runtime", {}).get(
                "gpu_allowed_for_this_formal_scope"
            )
            is False
        ):
            raise ConfigurationError(
                "E3 appears in a schedule whose formal scope prohibits E3"
            )

    return PrelaunchPlan(
        source=source,
        host=host,
        output_root=output_root,
        marker_path=marker_path,
        max_workers=max_workers,
        max_worker_rss=max_worker_rss,
        max_pool_rss=max_pool_rss,
        monitor_seconds=monitor_seconds,
        global_timeout=global_timeout,
        max_output=max_output,
        max_cpu=max_cpu,
        stop_free=stop_free,
        control_plane_reserve=control_plane_reserve,
        inflight_write_reserve_per_worker=inflight_per_worker,
        active_write_reserve=active_write_reserve,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    maximum_bytes: int | None = None,
) -> None:
    if maximum_bytes is not None and len(payload) > maximum_bytes:
        raise ConfigurationError(
            "R8C E1/E2 worker control report exceeds its strict byte bound"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if os.name == "nt":
        flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _advance_worker_timeout(
    item: dict[str, Any],
    *,
    now: float,
    runtime_seconds: float,
    timeout_seconds: float,
) -> str | None:
    """Request a typed timeout, then hard-kill only after a grace period."""

    if runtime_seconds < timeout_seconds:
        return None
    process = item["process"]
    if not item.get("timeout_requested"):
        task_directory = Path(item["task_directory"])
        if not task_directory.is_dir():
            process.kill()
            item["hard_timed_out"] = True
            return "HARD_TIMEOUT_BEFORE_TASK_DIRECTORY"
        marker = task_directory / TASK_TIMEOUT_MARKER_NAME
        try:
            _write_bytes_exclusive(
                marker,
                b"FORMAL_TASK_TIMEOUT_BEFORE_NEXT_CFE\n",
            )
        except FileExistsError:
            pass
        item["timeout_requested"] = True
        item["timeout_requested_at"] = now
        item["timeout_marker_path"] = marker
        return "SOFT_TIMEOUT_REQUESTED"
    requested_at = float(item["timeout_requested_at"])
    if (
        now - requested_at >= TASK_TIMEOUT_GRACE_SECONDS
        and process.poll() is None
    ):
        process.kill()
        item["hard_timed_out"] = True
        return "HARD_TIMEOUT_AFTER_GRACE"
    return None


def _sample_live_worker_resources(
    process: Any,
    process_id: int,
) -> tuple[int, float] | None:
    """Sample a worker, tolerating only the bounded Linux exit race.

    A process can leave ``/proc/<pid>/status`` before ``Popen.poll()`` has
    reaped it.  Use a short, bounded retry window so that a normal worker exit
    is not escalated into a host-wide sampling failure.  A process that remains
    live and still cannot be sampled continues to fail closed.
    """

    last_error: HostSamplingError | None = None
    for attempt in range(PROCESS_SAMPLE_MAX_ATTEMPTS):
        if attempt:
            time.sleep(PROCESS_SAMPLE_EXIT_RACE_GRACE_SECONDS)
        if process.poll() is not None:
            return None
        try:
            return (
                _process_rss_bytes(process_id),
                _process_cpu_seconds(process_id),
            )
        except HostSamplingError as error:
            last_error = error
            if process.poll() is not None:
                return None
    assert last_error is not None
    raise last_error


def _write_json_exclusive(
    path: Path,
    value: Any,
    *,
    maximum_bytes: int | None = None,
) -> None:
    _write_bytes_exclusive(
        path,
        canonical_json_bytes(value) + b"\n",
        maximum_bytes=maximum_bytes,
    )


def _run_manifest_payload_with_final_output_bytes(
    manifest: dict[str, Any],
    *,
    output_bytes_before_manifest: int,
) -> tuple[bytes, int]:
    """Solve the manifest-size fixed point for an exact final byte count."""

    if output_bytes_before_manifest < 0:
        raise ValueError("pre-manifest output bytes must be nonnegative")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ConfigurationError("run manifest resources must be mutable")
    resources["total_output_bytes_scope"] = (
        "ENTIRE_OUTPUT_ROOT_INCLUDING_THIS_RUN_MANIFEST"
    )
    candidate_total = output_bytes_before_manifest
    for _ in range(32):
        resources["total_output_bytes"] = candidate_total
        payload = canonical_json_bytes(manifest) + b"\n"
        exact_total = output_bytes_before_manifest + len(payload)
        if exact_total == candidate_total:
            return payload, exact_total
        candidate_total = exact_total
    raise ConfigurationError(
        "run manifest final-output byte fixed point did not converge"
    )


def _consume_request_once(
    *,
    marker_path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically publish one complete, fsynced one-time marker."""

    temporary = marker_path.parent / (
        f".{marker_path.name}.tmp-{secrets.token_hex(16)}"
    )
    try:
        _write_json_exclusive(temporary, payload)
        os.link(temporary, marker_path)
        _fsync_directory(marker_path.parent)
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(
                f"formal request consumption marker exists: {marker_path}"
            ) from error
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return marker_path


def _worker_environment(launch_token: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            LAUNCH_TOKEN_ENV: launch_token,
        }
    )
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not current
        else os.pathsep.join((str(SRC_ROOT), current))
    )
    return environment


def _write_json(
    path: Path,
    value: Any,
    *,
    maximum_bytes: int | None = None,
) -> None:
    _write_json_exclusive(
        path,
        value,
        maximum_bytes=maximum_bytes,
    )


def _write_schedule(path: Path, schedule: Sequence[Any]) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            for row in schedule:
                compressed.write(
                    canonical_json_bytes(row.to_dict()) + b"\n"
                )
        raw.flush()
        os.fsync(raw.fileno())


def _write_reuse_map(path: Path, schedule: Sequence[Any]) -> None:
    with path.open("xb") as raw:
        for row in build_e2_full_reuse_map(schedule):
            raw.write(canonical_json_bytes(row) + b"\n")
        raw.flush()
        os.fsync(raw.fileno())


def _directory_bytes(root: Path) -> int:
    total = 0
    try:
        paths = root.rglob("*")
        for path in paths:
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return total
    return total


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except FileNotFoundError:
        return 0


@dataclass(frozen=True)
class _WorkerOutputScope:
    task_directory: Path
    stdout_path: Path | None
    stderr_path: Path | None


def _worker_output_bytes(scope: _WorkerOutputScope) -> int:
    """Return one worker's observed task and log bytes.

    An active worker may create or remove a file while this observation is in
    progress.  The caller must therefore retain the frozen per-worker
    in-flight write reserve until the worker process has exited and this scope
    has been committed by a final scan.
    """

    return (
        _directory_bytes(scope.task_directory)
        + (
            _file_bytes(scope.stdout_path)
            if scope.stdout_path is not None
            else 0
        )
        + (
            _file_bytes(scope.stderr_path)
            if scope.stderr_path is not None
            else 0
        )
    )


class _IncrementalOutputAccounting:
    """Bound recurring scans to active worker output scopes.

    The immutable control-plane artifacts are scanned once before dispatch.
    Each exited worker is scanned once more and transferred to the committed
    total.  Control-plane files created after startup remain covered by the
    separately frozen control-plane reserve.  A final whole-root scan is still
    required when the run manifest is assembled.
    """

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root).resolve()
        self.startup_baseline_bytes = _directory_bytes(self.output_root)
        self.committed_worker_bytes = 0
        self._active: dict[str, _WorkerOutputScope] = {}
        self._known_task_ids: set[str] = set()
        self._uncertain_launch_failure_reserves = 0

    def begin_worker(
        self,
        task_id: str,
        *,
        task_directory: Path,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> None:
        if task_id in self._known_task_ids:
            raise ConfigurationError(
                "output accounting received a duplicate formal task ID"
            )
        scope = _WorkerOutputScope(
            task_directory=Path(task_directory).resolve(),
            stdout_path=(
                None if stdout_path is None else Path(stdout_path).resolve()
            ),
            stderr_path=(
                None if stderr_path is None else Path(stderr_path).resolve()
            ),
        )
        if not all(
            _is_relative_to(path, self.output_root)
            for path in (
                scope.task_directory,
                scope.stdout_path,
                scope.stderr_path,
            )
            if path is not None
        ):
            raise ConfigurationError(
                "worker output accounting scope escaped the formal output root"
            )
        if (scope.stdout_path is None) != (scope.stderr_path is None):
            raise ConfigurationError(
                "worker output accounting requires both legacy log paths"
            )
        if scope.stdout_path is not None and scope.stderr_path is not None and (
            scope.stdout_path == scope.stderr_path
            or _is_relative_to(scope.stdout_path, scope.task_directory)
            or _is_relative_to(scope.stderr_path, scope.task_directory)
        ):
            raise ConfigurationError(
                "worker output accounting scopes must not overlap"
            )
        self._known_task_ids.add(task_id)
        self._active[task_id] = scope

    def current_bytes(self) -> int:
        return (
            self.startup_baseline_bytes
            + self.committed_worker_bytes
            + sum(
                _worker_output_bytes(scope)
                for scope in self._active.values()
            )
        )

    def finish_worker(
        self,
        task_id: str,
        *,
        retain_inflight_reserve: bool = False,
    ) -> int:
        try:
            scope = self._active.pop(task_id)
        except KeyError as error:
            raise ConfigurationError(
                "output accounting cannot finish an inactive formal task"
            ) from error
        final_bytes = _worker_output_bytes(scope)
        self.committed_worker_bytes += final_bytes
        if retain_inflight_reserve:
            self._uncertain_launch_failure_reserves += 1
        return final_bytes

    @property
    def reserve_scope_count(self) -> int:
        return (
            len(self._active)
            + self._uncertain_launch_failure_reserves
        )

    def reserve_bytes(
        self,
        *,
        control_plane_reserve: int,
        inflight_write_reserve_per_worker: int,
        additional_workers: int = 0,
    ) -> int:
        if additional_workers < 0:
            raise ValueError("additional_workers must be nonnegative")
        return (
            control_plane_reserve
            + (
                self.reserve_scope_count
                + additional_workers
            )
            * inflight_write_reserve_per_worker
        )


def _file_commitment(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat_result.st_size,
        "sha256": file_sha256(path),
    }


def _root_relative_file_commitment(
    path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    if not _is_relative_to(resolved_path, resolved_root):
        raise ConfigurationError(
            "portable file commitment escaped its artifact root"
        )
    relative_path = resolved_path.relative_to(resolved_root).as_posix()
    if not resolved_path.is_file():
        return {
            "path": relative_path,
            "missing": True,
        }
    stat_result = resolved_path.stat()
    return {
        "path": relative_path,
        "bytes": stat_result.st_size,
        "sha256": file_sha256(resolved_path),
    }


def _read_last_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


_R8C_E1E2_SUMMARY_REPORT_KEYS = frozenset(
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
_R8C_E1E2_FAILURE_REPORT_KEYS = frozenset(
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
_R8C_E1E2_SUPERVISOR_REPORT_KEYS = frozenset(
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
_R8C_E1E2_TASK_SUMMARY_STATUSES = frozenset(
    {"COMPLETE", "INCOMPLETE_RESOURCE_CEILING"}
)
_R8C_E1E2_WORKER_ACCOUNTING_KEYS = frozenset(
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
_R8C_E1E2_FORBIDDEN_REPORT_KEYS = frozenset(
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


def _contains_forbidden_worker_report_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                str(key).casefold() in _R8C_E1E2_FORBIDDEN_REPORT_KEYS
                or _contains_forbidden_worker_report_key(child)
            )
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(
            _contains_forbidden_worker_report_key(child) for child in value
        )
    return False


def _valid_worker_report_accounting(
    value: Any,
    *,
    spec: Any,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != (
        _R8C_E1E2_WORKER_ACCOUNTING_KEYS
    ):
        return False
    scheduled_cfe = value.get("scheduled_cfe")
    scheduled_atomic = value.get("scheduled_atomic_model_steps")
    atomic_per_cfe = value.get("atomic_steps_per_cfe")
    exact = value.get("charged_work_exact")
    if (
        type(scheduled_cfe) is not int
        or scheduled_cfe <= 0
        or type(scheduled_atomic) is not int
        or scheduled_atomic <= 0
        or type(atomic_per_cfe) is not int
        or atomic_per_cfe <= 0
        or scheduled_cfe * atomic_per_cfe != scheduled_atomic
        or scheduled_cfe != getattr(spec, "total_cfe", None)
        or scheduled_atomic != getattr(spec, "total_atomic_steps", None)
        or atomic_per_cfe
        != getattr(spec, "atomic_steps_per_cfe", None)
        or type(exact) is not bool
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
    if exact:
        return (
            type(charged_cfe) is int
            and 0 <= charged_cfe <= scheduled_cfe
            and type(charged_atomic) is int
            and charged_atomic == charged_cfe * atomic_per_cfe
            and source is not None
            and recovery_error_type is None
        )
    return (
        charged_cfe is None
        and charged_atomic is None
        and source is None
    )


def _read_canonical_worker_report(
    path: Path,
    *,
    report_kind: str,
    spec: Any,
) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            "R8C E1/E2 worker control report cannot be read"
        ) from error
    if not payload or len(payload) > R8C_E1E2_WORKER_REPORT_MAX_BYTES:
        raise ConfigurationError(
            "R8C E1/E2 worker control report exceeds its strict byte bound"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            "R8C E1/E2 worker control report is not canonical JSON"
        ) from error
    if (
        not isinstance(value, Mapping)
        or payload != canonical_json_bytes(dict(value)) + b"\n"
        or _contains_forbidden_worker_report_key(value)
    ):
        raise ConfigurationError(
            "R8C E1/E2 worker control report violates its blind schema"
        )
    expected_task = spec.to_dict()
    expected_keys = {
        "TASK_SUMMARY": _R8C_E1E2_SUMMARY_REPORT_KEYS,
        "TASK_FAILURE": _R8C_E1E2_FAILURE_REPORT_KEYS,
        "SUPERVISOR_OUTCOME": _R8C_E1E2_SUPERVISOR_REPORT_KEYS,
    }.get(report_kind)
    if expected_keys is None or set(value) != expected_keys:
        raise ConfigurationError(
            "R8C E1/E2 worker control report fields differ"
        )
    if report_kind == "TASK_SUMMARY":
        if (
            value.get("artifact_role")
            != "R8C_E1E2_IMMUTABLE_ENDPOINT_SUFFICIENT_UNANALYZED"
            or value.get("status")
            not in _R8C_E1E2_TASK_SUMMARY_STATUSES
            or value.get("task") != expected_task
            or value.get("permissions", {}).get(
                "results_analysis_performed"
            )
            is not False
            or not _is_nonnegative_finite_number(
                value.get("runtime", {}).get("cpu_seconds")
            )
        ):
            raise ConfigurationError(
                "R8C E1/E2 task-summary worker report identity differs"
            )
    else:
        if (
            value.get("task") != expected_task
            or type(value.get("status")) is not str
            or not value.get("status")
            or type(value.get("outcome_class")) is not str
            or not value.get("outcome_class")
            or value.get("reason_code") != value.get("outcome_class")
            or (
                value.get("error_type") is not None
                and (
                    type(value.get("error_type")) is not str
                    or not value.get("error_type")
                )
            )
            or not _valid_worker_report_accounting(
                value.get("accounting"),
                spec=spec,
            )
            or value.get("attempt") != 1
            or value.get("automatic_retries") != 0
            or value.get("results_analysis_performed") is not False
        ):
            raise ConfigurationError(
                "R8C E1/E2 failed worker report identity differs"
            )
        if report_kind == "TASK_FAILURE" and (
            value.get("task_id") != spec.task_id
            or value.get("schedule_index")
            != expected_task.get("schedule_index")
            or not _is_nonnegative_finite_number(value.get("cpu_seconds"))
        ):
            raise ConfigurationError(
                "R8C E1/E2 task-failure report identity differs"
            )
    return value


def _r8c_e1e2_worker_report(
    task_directory: Path,
    *,
    spec: Any,
) -> tuple[str, Path, Mapping[str, Any]] | None:
    choices = (
        ("TASK_FAILURE", task_directory / "task_failure.json"),
        (
            "SUPERVISOR_OUTCOME",
            task_directory / "task_supervisor_outcome.json",
        ),
        ("TASK_SUMMARY", task_directory / "task_summary.json"),
    )
    for report_kind, path in choices:
        if path.is_file():
            return (
                report_kind,
                path,
                _read_canonical_worker_report(
                    path,
                    report_kind=report_kind,
                    spec=spec,
                ),
            )
    return None


def _normalized_worker_payload_from_report(
    report: tuple[str, Path, Mapping[str, Any]] | None,
    *,
    task_directory: Path,
) -> Mapping[str, Any] | None:
    if report is None:
        return None
    report_kind, _, value = report
    manifest_path = task_directory / "task_manifest.json"
    manifest_sha256 = (
        file_sha256(manifest_path) if manifest_path.is_file() else None
    )
    if report_kind == "TASK_SUMMARY":
        runtime = value["runtime"]
        return {
            "task_id": value["task"]["task_id"],
            "status": value["status"],
            "total_cfe": value["total_cfe"],
            "total_atomic_model_steps": value["total_atomic_model_steps"],
            "cpu_seconds": runtime["cpu_seconds"],
            "task_manifest_sha256": manifest_sha256,
        }
    return {
        "task_id": value["task"]["task_id"],
        "status": value["status"],
        "outcome_class": value["outcome_class"],
        "error_type": value.get("error_type"),
        "accounting": dict(value["accounting"]),
        "cpu_seconds": (
            value.get("cpu_seconds")
            if report_kind == "TASK_FAILURE"
            else None
        ),
        "task_manifest_sha256": manifest_sha256,
    }


def _worker_report_commitment(
    report: tuple[str, Path, Mapping[str, Any]],
    *,
    output_root: Path,
) -> dict[str, Any]:
    report_kind, path, _ = report
    return {
        "kind": report_kind,
        **_root_relative_file_commitment(path, root=output_root),
    }


def _scheduled_task_accounting(spec: Any) -> dict[str, int | None]:
    scheduled_cfe = getattr(spec, "total_cfe", None)
    scheduled_atomic = getattr(spec, "total_atomic_steps", None)
    atomic_per_cfe = getattr(spec, "atomic_steps_per_cfe", None)
    return {
        "scheduled_cfe": (
            int(scheduled_cfe) if scheduled_cfe is not None else None
        ),
        "scheduled_atomic_model_steps": (
            int(scheduled_atomic) if scheduled_atomic is not None else None
        ),
        "atomic_steps_per_cfe": (
            int(atomic_per_cfe) if atomic_per_cfe is not None else None
        ),
    }


def _recover_failed_task_accounting(
    task_directory: Path,
    spec: Any,
) -> dict[str, Any]:
    """Recover charged-work counts in the worker, without computing effects.

    A cooperative timeout unwinds through the task writer, which closes the
    active event with one terminal checkpoint.  That makes the charged CFE
    count exact.  A supervisor hard kill can interrupt before this worker-side
    recovery runs; the supervisor then records explicit unknown charged work
    instead of inventing an algorithm terminal or a cost.
    """

    scheduled = _scheduled_task_accounting(spec)
    charged_cfe: int | None = None
    source: str | None = None
    error_type: str | None = None

    summary_path = task_directory / "task_summary.json"
    if summary_path.is_file():
        try:
            summary = _read_json(summary_path)
            charged_cfe = int(summary["total_cfe"])
            charged_atomic = int(summary["total_atomic_model_steps"])
            source = "TASK_SUMMARY"
            return {
                **scheduled,
                "charged_cfe": charged_cfe,
                "charged_atomic_model_steps": charged_atomic,
                "charged_work_exact": True,
                "charged_work_source": source,
                "charged_work_recovery_error_type": None,
            }
        except (KeyError, TypeError, ValueError, ConfigurationError) as error:
            error_type = type(error).__name__

    checkpoint_path = task_directory / "checkpoint_fronts.cfe"
    if checkpoint_path.is_file():
        try:
            checkpoint = read_checkpoint_file(checkpoint_path)
            last_by_event: dict[int, Any] = {}
            for record in checkpoint.records:
                last_by_event[record.event_id] = record
            charged_cfe = sum(record.cfe for record in last_by_event.values())
            source = "STRICT_CHECKPOINT_TERMINALS"
        except (CheckpointDataError, OSError, ValueError) as error:
            error_type = type(error).__name__

    raw_path = task_directory / "raw_evaluations.jsonl.gz"
    if charged_cfe is None and raw_path.is_file():
        try:
            with gzip.open(raw_path, "rb") as stream:
                charged_cfe = sum(1 for line in stream if line.strip())
            source = "COMPLETE_GZIP_RECORD_COUNT"
        except (OSError, EOFError) as error:
            error_type = type(error).__name__

    atomic_per_cfe = scheduled["atomic_steps_per_cfe"]
    charged_atomic = (
        charged_cfe * atomic_per_cfe
        if charged_cfe is not None and atomic_per_cfe is not None
        else None
    )
    return {
        **scheduled,
        "charged_cfe": charged_cfe,
        "charged_atomic_model_steps": charged_atomic,
        "charged_work_exact": charged_cfe is not None,
        "charged_work_source": source,
        "charged_work_recovery_error_type": (
            None if charged_cfe is not None else error_type
        ),
    }


def _worker_failure_class(
    item: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
) -> str:
    forced_reason = item.get("forced_termination_reason")
    if item.get("timeout_requested"):
        return TECHNICAL_SEQUENCE_TIMEOUT
    if forced_reason == "GLOBAL_HARD_TIMEOUT":
        return TECHNICAL_GLOBAL_TIMEOUT
    if forced_reason is not None:
        return TECHNICAL_RESOURCE_TERMINATION
    if (
        isinstance(payload, Mapping)
        and payload.get("status") == "INCOMPLETE_RESOURCE_CEILING"
    ):
        return TECHNICAL_RESOURCE_TERMINATION
    if (
        isinstance(payload, Mapping)
        and isinstance(payload.get("outcome_class"), str)
    ):
        return str(payload["outcome_class"])
    return TASK_EXECUTION_FAILURE


def _worker_failure_status(
    *,
    failure_class: str,
    payload: Mapping[str, Any] | None,
) -> str:
    if failure_class in {
        TECHNICAL_SEQUENCE_TIMEOUT,
        TECHNICAL_GLOBAL_TIMEOUT,
    }:
        return "PARTIAL_TECHNICAL_TIMEOUT_NO_RETRY"
    if (
        isinstance(payload, Mapping)
        and isinstance(payload.get("status"), str)
        and payload.get("status") != "COMPLETE"
    ):
        return str(payload["status"])
    if failure_class == TECHNICAL_RESOURCE_TERMINATION:
        return "PARTIAL_TECHNICAL_FAILURE_NO_RETRY"
    return "TASK_FAILED_NO_RETRY"


def _task_artifacts(task_directory: Path) -> dict[str, dict[str, Any]]:
    artifact_paths = tuple(
        sorted(
            (
                path
                for path in task_directory.iterdir()
                if path.is_file()
                and path.name not in {
                    "task_manifest.json",
                    "heartbeat",
                }
            ),
            key=lambda path: path.name,
        )
    )
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }


def _validate_existing_task_manifest(
    task_directory: Path,
    *,
    task_id: str,
) -> tuple[Mapping[str, Any], str]:
    manifest_path = task_directory / "task_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("task_id") != task_id:
        raise ConfigurationError(
            "task manifest identity differs from the scheduled task"
        )
    if manifest.get("artifacts") != _task_artifacts(task_directory):
        raise ConfigurationError(
            "task manifest artifacts differ from immutable task files"
        )
    status = manifest.get("status")
    if type(status) is not str or not status:
        raise ConfigurationError("task manifest status is missing")
    return manifest, file_sha256(manifest_path)


def _materialize_supervisor_task_outcome(
    *,
    profile: RunnerProfile,
    spec: Any,
    task_directory: Path,
    status: str,
    failure_class: str,
    reason: str,
    error_type: str | None = None,
    accounting: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Commit a task outcome when no worker manifest survived.

    Existing worker artifacts are never overwritten.  This covers hard-kill,
    launch-failure and not-dispatched outcomes, so the run manifest can commit
    every frozen schedule row rather than silently omitting technical failures.
    """

    task_directory.mkdir(parents=False, exist_ok=True)
    (task_directory / "heartbeat").unlink(missing_ok=True)
    manifest_path = task_directory / "task_manifest.json"
    if manifest_path.is_file():
        return _validate_existing_task_manifest(
            task_directory,
            task_id=spec.task_id,
        )
    outcome_path = task_directory / "task_supervisor_outcome.json"
    if not outcome_path.exists():
        if accounting is None:
            scheduled = _scheduled_task_accounting(spec)
            accounting = {
                **scheduled,
                "charged_cfe": None,
                "charged_atomic_model_steps": None,
                "charged_work_exact": False,
                "charged_work_source": None,
                "charged_work_recovery_error_type": None,
            }
        outcome: dict[str, Any] = {
            "artifact_role": (
                f"{profile.artifact_stage}_IMMUTABLE_UNANALYZED"
            ),
            "status": status,
            "outcome_class": failure_class,
            "task": spec.to_dict(),
            "attempt": 1,
            "automatic_retries": 0,
            "algorithm_terminal_code": None,
            "results_analysis_performed": False,
        }
        if profile.task_artifact_worker_reports:
            outcome.update(
                {
                    "reason_code": failure_class,
                    "error_type": error_type,
                    "accounting": dict(accounting),
                }
            )
        else:
            outcome["reason"] = reason
        _write_json_exclusive(
            outcome_path,
            outcome,
            maximum_bytes=(
                R8C_E1E2_WORKER_REPORT_MAX_BYTES
                if profile.task_artifact_worker_reports
                else None
            ),
        )
    artifacts = _task_artifacts(task_directory)
    manifest = {
        "task_id": spec.task_id,
        "status": status,
        "outcome_class": failure_class,
        "artifacts": artifacts,
        "task_binding_sha256": sha256(
            canonical_json_bytes(
                {
                    "task": spec.to_dict(),
                    "artifacts": artifacts,
                }
            )
        ).hexdigest(),
    }
    _write_json_exclusive(manifest_path, manifest)
    return manifest, file_sha256(manifest_path)


def _validate_worker_launch(
    *,
    args: argparse.Namespace,
    profile: RunnerProfile,
    contract: Mapping[str, Any],
    request: Any,
    schedule: Sequence[Any],
    spec: Any,
) -> None:
    launch_token = os.environ.get(LAUNCH_TOKEN_ENV)
    if not launch_token:
        raise ConfigurationError(
            "formal worker requires a supervisor-only launch token"
        )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise ConfigurationError(
                f"formal worker thread binding differs for {name}"
            )

    source = _validate_source(request)
    host = host_fingerprint()
    host_sha256 = host_fingerprint_sha256(host)
    expected_host_sha256 = contract["resources"]["candidate_target"].get(
        "host_fingerprint_sha256"
    )
    if host_sha256 != expected_host_sha256:
        raise ConfigurationError(
            "formal worker host differs from the qualified target"
        )

    output_root = _launch_path(
        contract["launch"]["output_root"],
        relative_to_project=False,
    )
    tasks_root = (output_root / "tasks").resolve()
    logs_root = (output_root / "worker_logs").resolve()
    task_directory = Path(args.task_directory).resolve()
    stop_path = Path(args.stop_path).resolve()
    if (
        task_directory != (tasks_root / spec.task_id).resolve()
        or task_directory.parent != tasks_root
    ):
        raise ConfigurationError(
            "formal worker task directory differs from its output binding"
        )
    if stop_path != (output_root / "STOP_DISPATCH").resolve():
        raise ConfigurationError(
            "formal worker stop path differs from its output binding"
        )
    if task_directory.exists():
        raise ConfigurationError(
            "formal worker task directory is already occupied"
        )
    if not tasks_root.is_dir() or (
        not profile.task_artifact_worker_reports
        and not logs_root.is_dir()
    ):
        raise ConfigurationError(
            "formal worker output control directories are incomplete"
        )
    if profile.task_artifact_worker_reports and logs_root.exists():
        raise ConfigurationError(
            "R8C E1/E2 formal root must not contain raw worker logs"
        )

    binding_path = output_root / "launch_binding.json"
    binding = _read_json(binding_path)
    token_sha256 = sha256(launch_token.encode("utf-8")).hexdigest()
    contract_path = Path(args.contract).resolve()
    request_path = Path(args.request).resolve()
    schedule_path = output_root / "schedule.jsonl.gz"
    reuse_path = output_root / "e2_full_reuse_map.jsonl"
    critical_paths = {
        "output_root": str(output_root),
        "tasks_root": str(tasks_root),
        "stop_path": str(stop_path),
        "request_consumption_marker": str(
            _launch_path(
                contract["launch"]["request_consumption_marker"],
                relative_to_project=True,
            )
        ),
        "request_consumption_record": (
            "request_consumption_record.json"
        ),
    }
    if profile.task_artifact_worker_reports:
        critical_paths["worker_control_reports"] = (
            "TASK_MANIFEST_COMMITTED_TASK_ARTIFACTS"
        )
    else:
        critical_paths["worker_logs_root"] = str(logs_root)
    critical_binding = {
        "contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "contract_id": profile.contract_id,
        },
        "request": {
            "path": str(request_path),
            "sha256": file_sha256(request_path),
            "request_id": request.request_id,
        },
        "source": source,
        "host": {
            "fingerprint": host,
            "fingerprint_sha256": host_sha256,
        },
        "paths": critical_paths,
        "schedule": {
            "id": contract["schedule"]["id"],
            "sha256": schedule_commitment(schedule),
            "file": _file_commitment(schedule_path),
            "e2_full_reuse_sha256": e2_full_reuse_commitment(schedule),
            "e2_full_reuse_file": _file_commitment(reuse_path),
        },
        "launch_token_sha256": token_sha256,
    }
    for key, expected in critical_binding.items():
        if binding.get(key) != expected:
            raise ConfigurationError(
                f"formal worker launch binding differs for {key}"
            )

    marker_path = Path(
        critical_binding["paths"]["request_consumption_marker"]
    )
    marker = _read_json(marker_path)
    expected_marker = {
        "schema_version": "WGT-R8-REQUEST-CONSUMPTION-1.0",
        "request_id": request.request_id,
        "request_sha256": file_sha256(request_path),
        "contract_id": profile.contract_id,
        "contract_sha256": file_sha256(contract_path),
        "source": source,
        "host_fingerprint_sha256": host_sha256,
        "output_root": str(output_root),
        "tasks_root": str(tasks_root),
        "stop_path": str(stop_path),
        "launch_binding_sha256": file_sha256(binding_path),
        "launch_token_sha256": token_sha256,
        "schedule_sha256": schedule_commitment(schedule),
        "schedule_file_sha256": file_sha256(schedule_path),
        "consumption": "ONE_TIME_FORMAL_SUPERVISOR_START",
    }
    if marker != expected_marker:
        raise ConfigurationError(
            "formal worker requires the complete atomic request marker"
        )


def _run_worker(args: argparse.Namespace) -> int:
    profile = RUNNER_PROFILES[args.execution_profile]
    contract_path = Path(args.contract).resolve()
    request_path = Path(args.request).resolve()
    contract, request, schedule = _load_and_validate(
        contract_path,
        request_path,
        profile,
    )
    if args.schedule_index not in range(len(schedule)):
        raise ConfigurationError("worker schedule index is out of range")
    spec = schedule[args.schedule_index]
    if spec.task_id != args.task_id:
        raise ConfigurationError("worker task identity differs from schedule")
    task_directory = Path(args.task_directory).resolve()
    stop_path = Path(args.stop_path).resolve()
    _validate_worker_launch(
        args=args,
        profile=profile,
        contract=contract,
        request=request,
        schedule=schedule,
        spec=spec,
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    try:
        result = run_task(
            spec=spec,
            request=request,
            task_directory=task_directory,
            stop_path=stop_path,
            settings=profile.runtime_settings,
        )
    except Exception as error:
        task_directory.mkdir(parents=False, exist_ok=True)
        (task_directory / "heartbeat").unlink(missing_ok=True)
        timeout_marker_path = (
            task_directory / TASK_TIMEOUT_MARKER_NAME
        )
        technical_timeout = (
            isinstance(error, ExecutionTimeoutBeforeEntry)
            or timeout_marker_path.is_file()
        )
        failure_class = (
            TECHNICAL_SEQUENCE_TIMEOUT
            if technical_timeout
            else TASK_EXECUTION_FAILURE
        )
        status = (
            "PARTIAL_TECHNICAL_TIMEOUT_NO_RETRY"
            if technical_timeout
            else "TASK_FAILED_NO_RETRY"
        )
        accounting = _recover_failed_task_accounting(
            task_directory,
            spec,
        )
        failure = {
            "artifact_role": (
                f"{profile.artifact_stage}_IMMUTABLE_UNANALYZED"
            ),
            "task_id": spec.task_id,
            "schedule_index": getattr(
                spec,
                "schedule_index",
                spec.to_dict().get("schedule_index"),
            ),
            "status": status,
            "outcome_class": failure_class,
            "task": spec.to_dict(),
            "error_type": type(error).__name__,
            "algorithm_terminal_code": None,
            "timeout_marker": (
                {
                    "bytes": timeout_marker_path.stat().st_size,
                    "sha256": file_sha256(timeout_marker_path),
                }
                if timeout_marker_path.is_file()
                else None
            ),
            "accounting": accounting,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "attempt": 1,
            "automatic_retries": 0,
            "results_analysis_performed": False,
        }
        if profile.task_artifact_worker_reports:
            failure["reason_code"] = failure_class
        else:
            failure["error"] = str(error)
        failure_path = task_directory / "task_failure.json"
        _write_json(
            failure_path,
            failure,
            maximum_bytes=(
                R8C_E1E2_WORKER_REPORT_MAX_BYTES
                if profile.task_artifact_worker_reports
                else None
            ),
        )
        artifacts = _task_artifacts(task_directory)
        manifest = {
            "task_id": spec.task_id,
            "status": failure["status"],
            "outcome_class": failure_class,
            "artifacts": artifacts,
            "task_binding_sha256": sha256(
                canonical_json_bytes(
                    {
                        "task": spec.to_dict(),
                        "artifacts": artifacts,
                    }
                )
            ).hexdigest(),
        }
        manifest_path = task_directory / "task_manifest.json"
        _write_json(manifest_path, manifest)
        failure["task_manifest_sha256"] = file_sha256(manifest_path)
        failure["output_bytes"] = _directory_bytes(task_directory)
        if profile.task_artifact_worker_reports:
            report = _r8c_e1e2_worker_report(
                task_directory,
                spec=spec,
            )
            if report is None or report[0] != "TASK_FAILURE":
                raise ConfigurationError(
                    "R8C E1/E2 task failure lacks its bounded control report"
                )
            return 3
        print(canonical_json_bytes(failure).decode("utf-8"))
        return 3
    if profile.task_artifact_worker_reports:
        report = _r8c_e1e2_worker_report(
            task_directory,
            spec=spec,
        )
        if report is None or report[0] != "TASK_SUMMARY":
            raise ConfigurationError(
                "R8C E1/E2 completed task lacks its bounded control report"
            )
        return 0
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


def _run_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    profile = RUNNER_PROFILES[args.execution_profile]
    contract_path = Path(args.contract).resolve()
    request_path = Path(args.request).resolve()
    contract, request, schedule = _load_and_validate(
        contract_path,
        request_path,
        profile,
    )
    plan = _validate_prelaunch(args, contract, request, schedule)
    source = plan.source
    output_root = plan.output_root
    resources = contract["resources"]
    output_root.mkdir(parents=False, exist_ok=False)
    tasks_root = output_root / "tasks"
    tasks_root.mkdir()
    logs_root = output_root / "worker_logs"
    if not profile.task_artifact_worker_reports:
        logs_root.mkdir()
    stop_path = output_root / "STOP_DISPATCH"
    schedule_path = output_root / "schedule.jsonl.gz"
    reuse_path = output_root / "e2_full_reuse_map.jsonl"
    binding_path = output_root / "launch_binding.json"
    consumption_record_path = (
        output_root / "request_consumption_record.json"
    )
    _write_schedule(schedule_path, schedule)
    _write_reuse_map(reuse_path, schedule)
    launch_token = secrets.token_urlsafe(48)
    launch_token_sha256 = sha256(
        launch_token.encode("utf-8")
    ).hexdigest()
    launch_paths = {
        "output_root": str(output_root),
        "tasks_root": str(tasks_root.resolve()),
        "stop_path": str(stop_path.resolve()),
        "request_consumption_marker": str(plan.marker_path),
        "request_consumption_record": (
            "request_consumption_record.json"
        ),
    }
    if profile.task_artifact_worker_reports:
        launch_paths["worker_control_reports"] = (
            "TASK_MANIFEST_COMMITTED_TASK_ARTIFACTS"
        )
    else:
        launch_paths["worker_logs_root"] = str(logs_root.resolve())
    launch_binding = {
        "schema_version": "WGT-R8-FORMAL-LAUNCH-BINDING-1.0",
        "artifact_role": (
            f"{profile.artifact_stage}_IMMUTABLE_CONTROL_PLANE"
        ),
        "contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "contract_id": profile.contract_id,
        },
        "request": {
            "path": str(request_path),
            "sha256": file_sha256(request_path),
            "request_id": request.request_id,
        },
        "source": source,
        "host": {
            "fingerprint": plan.host,
            "fingerprint_sha256": host_fingerprint_sha256(plan.host),
        },
        "paths": launch_paths,
        "schedule": {
            "id": contract["schedule"]["id"],
            "sha256": schedule_commitment(schedule),
            "file": _file_commitment(schedule_path),
            "e2_full_reuse_sha256": e2_full_reuse_commitment(schedule),
            "e2_full_reuse_file": _file_commitment(reuse_path),
        },
        "launch_token_sha256": launch_token_sha256,
        "permissions": contract["permissions"],
    }
    _write_json(
        binding_path,
        launch_binding,
    )
    _fsync_directory(output_root)
    if _validate_source(request) != source:
        raise ConfigurationError(
            "source identity changed while preparing formal control artifacts"
        )
    current_host = host_fingerprint()
    if current_host != plan.host:
        raise ConfigurationError(
            "host identity changed while preparing formal control artifacts"
        )
    if (
        shutil.disk_usage(output_root.parent).free
        < int(resources["scratch"]["minimum_free_bytes_at_start"])
    ):
        raise ConfigurationError(
            "scratch free space fell below the start gate before consumption"
        )
    marker_payload = {
        "schema_version": "WGT-R8-REQUEST-CONSUMPTION-1.0",
        "request_id": request.request_id,
        "request_sha256": file_sha256(request_path),
        "contract_id": profile.contract_id,
        "contract_sha256": file_sha256(contract_path),
        "source": source,
        "host_fingerprint_sha256": host_fingerprint_sha256(plan.host),
        "output_root": str(output_root),
        "tasks_root": str(tasks_root.resolve()),
        "stop_path": str(stop_path.resolve()),
        "launch_binding_sha256": file_sha256(binding_path),
        "launch_token_sha256": launch_token_sha256,
        "schedule_sha256": schedule_commitment(schedule),
        "schedule_file_sha256": file_sha256(schedule_path),
        "consumption": "ONE_TIME_FORMAL_SUPERVISOR_START",
    }
    consumption_marker = _consume_request_once(
        marker_path=plan.marker_path,
        payload=marker_payload,
    )
    _write_json(consumption_record_path, marker_payload)
    if (
        file_sha256(consumption_record_path)
        != file_sha256(consumption_marker)
    ):
        raise ConfigurationError(
            "portable consumption record differs from one-time marker"
        )
    _fsync_directory(output_root)
    output_accounting = _IncrementalOutputAccounting(output_root)

    started = time.monotonic()
    active: dict[int, dict[str, Any]] = {}
    next_index = 0
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []
    worker_log_commitments: dict[str, Any] = {}
    worker_control_report_commitments: dict[str, Any] = {}
    task_manifest_commitments: dict[str, str] = {}
    peak_pool_rss = 0
    dispatch_stopped_reason: str | None = None

    def stop_dispatch(reason: str) -> None:
        nonlocal dispatch_stopped_reason
        if dispatch_stopped_reason is None:
            dispatch_stopped_reason = reason
            _write_bytes_exclusive(
                stop_path,
                (reason + "\n").encode("utf-8"),
            )
            _fsync_directory(output_root)

    def kill_active(reason: str) -> None:
        for item in active.values():
            item.setdefault("forced_termination_reason", reason)
            try:
                item["process"].kill()
            except OSError:
                pass

    def accounted_cpu_seconds() -> float:
        return (
            sum(float(item["cpu_seconds"]) for item in completed)
            + sum(float(item["cpu_seconds"]) for item in failures)
            + sum(float(item["last_cpu"]) for item in active.values())
        )

    def safe_progress_mtimes(task_directory: Path) -> list[float]:
        mtimes: list[float] = []
        for path in (task_directory / "heartbeat",):
            try:
                mtimes.append(path.stat().st_mtime)
            except FileNotFoundError:
                continue
        return mtimes

    def commit_worker_logs(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "stdout": _root_relative_file_commitment(
                item["stdout_path"],
                root=output_root,
            ),
            "stderr": _root_relative_file_commitment(
                item["stderr_path"],
                root=output_root,
            ),
        }

    try:
        while active or (
            next_index < len(schedule)
            and dispatch_stopped_reason is None
        ):
            elapsed = time.monotonic() - started
            if elapsed >= plan.global_timeout:
                stop_dispatch("GLOBAL_HARD_TIMEOUT")
                kill_active("GLOBAL_HARD_TIMEOUT")
            current_output = output_accounting.current_bytes()
            free_bytes = shutil.disk_usage(output_root.parent).free
            current_reserve = output_accounting.reserve_bytes(
                control_plane_reserve=plan.control_plane_reserve,
                inflight_write_reserve_per_worker=(
                    plan.inflight_write_reserve_per_worker
                ),
            )
            hard_resource_breach = False
            if current_output >= plan.max_output:
                stop_dispatch("OUTPUT_RESOURCE_CEILING")
                hard_resource_breach = True
            elif (
                current_output
                + current_reserve
                >= plan.max_output
            ):
                stop_dispatch("OUTPUT_IN_FLIGHT_RESERVE")
            if free_bytes <= plan.stop_free:
                stop_dispatch("SCRATCH_FREE_SPACE_FLOOR")
                hard_resource_breach = True
            elif (
                free_bytes
                < plan.stop_free
                + current_reserve
            ):
                stop_dispatch("SCRATCH_IN_FLIGHT_RESERVE")
            if hard_resource_breach:
                kill_active(dispatch_stopped_reason or "RESOURCE_CEILING")

            pool_rss = 0
            finished: list[int] = []
            now = time.monotonic()
            sampling_failure: HostSamplingError | None = None
            for process_id, item in list(active.items()):
                process = item["process"]
                if process.poll() is not None:
                    finished.append(process_id)
                    continue
                try:
                    sample = _sample_live_worker_resources(
                        process,
                        process_id,
                    )
                except HostSamplingError as error:
                    sampling_failure = error
                    break
                if sample is None:
                    finished.append(process_id)
                    continue
                rss, cpu_seconds = sample
                pool_rss += rss
                item["peak_rss"] = max(item["peak_rss"], rss)
                item["last_cpu"] = max(item["last_cpu"], cpu_seconds)
                item["last_sampled_at"] = now
                if (
                    rss > plan.max_worker_rss
                    and not item.get("rss_reported")
                ):
                    item["rss_reported"] = True
                    advisories.append(
                        {
                            "type": "WORKER_RSS_RESOURCE_ANOMALY",
                            "task_id": item["spec"].task_id,
                            "rss_bytes": rss,
                        }
                    )
                    stop_dispatch("WORKER_RSS_RESOURCE_CEILING")
                runtime = now - item["started"]
                timeout_action = _advance_worker_timeout(
                    item,
                    now=now,
                    runtime_seconds=runtime,
                    timeout_seconds=float(item["spec"].timeout_seconds),
                )
                if timeout_action is not None:
                    advisories.append(
                        {
                            "type": timeout_action,
                            "task_id": item["spec"].task_id,
                            "runtime_seconds": runtime,
                            "grace_seconds": TASK_TIMEOUT_GRACE_SECONDS,
                        }
                    )
                task_dir = item["task_directory"]
                mtimes = safe_progress_mtimes(task_dir)
                if (
                    runtime >= resources["monitor"]["stall_suspected_seconds"]
                    and not item["stall_reported"]
                    and (
                        not mtimes
                        or time.time() - max(mtimes)
                        >= resources["monitor"]["stall_suspected_seconds"]
                    )
                ):
                    item["stall_reported"] = True
                    advisories.append(
                        {
                            "type": "STALL_SUSPECTED_ADVISORY_ONLY",
                            "task_id": item["spec"].task_id,
                            "runtime_seconds": runtime,
                        }
                    )
            if sampling_failure is not None:
                sampling_advisory = {
                    "type": "HOST_RESOURCE_SAMPLING_FAILED",
                    "error_type": type(sampling_failure).__name__,
                }
                if not profile.task_artifact_worker_reports:
                    sampling_advisory["error"] = str(sampling_failure)
                advisories.append(sampling_advisory)
                stop_dispatch("HOST_RESOURCE_SAMPLING_FAILED")
                kill_active("HOST_RESOURCE_SAMPLING_FAILED")
            peak_pool_rss = max(peak_pool_rss, pool_rss)
            if pool_rss > plan.max_pool_rss:
                advisories.append(
                    {
                        "type": "POOL_RSS_RESOURCE_ANOMALY",
                        "rss_bytes": pool_rss,
                    }
                )
                stop_dispatch("POOL_RSS_RESOURCE_CEILING")
            if accounted_cpu_seconds() >= plan.max_cpu:
                stop_dispatch("CPU_RESOURCE_CEILING")
                kill_active("CPU_RESOURCE_CEILING")

            for process_id in finished:
                item = active.pop(process_id)
                process = item["process"]
                spec = item["spec"]
                output_accounting.finish_worker(spec.task_id)
                if profile.task_artifact_worker_reports:
                    report = _r8c_e1e2_worker_report(
                        item["task_directory"],
                        spec=spec,
                    )
                    payload = _normalized_worker_payload_from_report(
                        report,
                        task_directory=item["task_directory"],
                    )
                    log_commitments = None
                else:
                    report = None
                    log_commitments = commit_worker_logs(item)
                    worker_log_commitments[spec.task_id] = log_commitments
                    payload = _read_last_json_object(item["stdout_path"])
                runtime = time.monotonic() - item["started"]
                payload_cpu = (
                    payload.get("cpu_seconds")
                    if isinstance(payload, Mapping)
                    else None
                )
                payload_cpu_valid = _is_nonnegative_finite_number(
                    payload_cpu
                )
                conservative_cpu = max(
                    float(item["last_cpu"]),
                    (
                        float(payload_cpu)
                        if payload_cpu_valid
                        else float(item["last_cpu"])
                        + max(
                            0.0,
                            time.monotonic() - item["last_sampled_at"],
                        )
                    ),
                )
                worker_reported_complete = (
                    process.returncode == 0
                    and isinstance(payload, Mapping)
                    and payload.get("status") == "COMPLETE"
                    and payload.get("task_id") == spec.task_id
                    and payload_cpu_valid
                )
                if (
                    worker_reported_complete
                    and item.get("timeout_requested")
                    and not item.get("hard_timed_out")
                ):
                    marker_path = item.get("timeout_marker_path")
                    if marker_path is not None:
                        Path(marker_path).unlink(missing_ok=True)
                    advisories.append(
                        {
                            "type": (
                                "WORKER_COMPLETION_WON_TIMEOUT_REAP_RACE"
                            ),
                            "task_id": spec.task_id,
                        }
                    )
                valid_success = (
                    worker_reported_complete
                    and not item.get("hard_timed_out")
                )
                if valid_success:
                    _, task_manifest_sha256 = (
                        _validate_existing_task_manifest(
                            item["task_directory"],
                            task_id=spec.task_id,
                        )
                    )
                    if (
                        payload.get("task_manifest_sha256")
                        != task_manifest_sha256
                    ):
                        raise ConfigurationError(
                            "worker success task-manifest commitment differs"
                        )
                    task_manifest_commitments[
                        spec.task_id
                    ] = task_manifest_sha256
                    if profile.task_artifact_worker_reports:
                        if report is None:
                            raise ConfigurationError(
                                "completed R8C E1/E2 task lacks its report"
                            )
                        worker_control_report_commitments[spec.task_id] = (
                            _worker_report_commitment(
                                report,
                                output_root=output_root,
                            )
                        )
                    result_payload = dict(payload)
                    result_payload["peak_rss_bytes"] = item["peak_rss"]
                    result_payload["wall_seconds"] = runtime
                    result_payload["output_bytes"] = _directory_bytes(
                        item["task_directory"]
                    )
                    completed.append(result_payload)
                else:
                    if conservative_cpu <= 0.0:
                        conservative_cpu = max(runtime, 0.0)
                    forced_reason = item.get("forced_termination_reason")
                    failure_class = _worker_failure_class(
                        item,
                        payload,
                    )
                    status = _worker_failure_status(
                        failure_class=failure_class,
                        payload=payload,
                    )
                    accounting_payload = (
                        payload.get("accounting")
                        if isinstance(payload, Mapping)
                        and isinstance(payload.get("accounting"), Mapping)
                        else {}
                    )
                    if (
                        not accounting_payload
                        and isinstance(payload, Mapping)
                        and type(payload.get("total_cfe")) is int
                        and type(
                            payload.get("total_atomic_model_steps")
                        )
                        is int
                    ):
                        accounting_payload = {
                            "charged_cfe": payload["total_cfe"],
                            "charged_atomic_model_steps": payload[
                                "total_atomic_model_steps"
                            ],
                            "charged_work_exact": True,
                            "charged_work_source": (
                                "WORKER_FINAL_REPORT"
                            ),
                        }
                    scheduled_accounting = _scheduled_task_accounting(spec)
                    complete_accounting = {
                        **scheduled_accounting,
                        "scheduled_cfe": accounting_payload.get(
                            "scheduled_cfe",
                            scheduled_accounting["scheduled_cfe"],
                        ),
                        "charged_cfe": accounting_payload.get("charged_cfe"),
                        "scheduled_atomic_model_steps": (
                            accounting_payload.get(
                                "scheduled_atomic_model_steps",
                                scheduled_accounting[
                                    "scheduled_atomic_model_steps"
                                ],
                            )
                        ),
                        "charged_atomic_model_steps": accounting_payload.get(
                            "charged_atomic_model_steps"
                        ),
                        "charged_work_exact": bool(
                            accounting_payload.get(
                                "charged_work_exact",
                                False,
                            )
                        ),
                        "charged_work_source": accounting_payload.get(
                            "charged_work_source"
                        ),
                        "charged_work_recovery_error_type": (
                            accounting_payload.get(
                                "charged_work_recovery_error_type"
                            )
                        ),
                    }
                    worker_error_type = (
                        payload.get("error_type")
                        if isinstance(payload, Mapping)
                        and type(payload.get("error_type")) is str
                        else None
                    )
                    _, task_manifest_sha256 = (
                        _materialize_supervisor_task_outcome(
                            profile=profile,
                            spec=spec,
                            task_directory=item["task_directory"],
                            status=status,
                            failure_class=failure_class,
                            reason=(
                                failure_class
                                if profile.task_artifact_worker_reports
                                else (
                                    str(forced_reason)
                                    if forced_reason is not None
                                    else (
                                        str(payload.get("error"))
                                        if isinstance(payload, Mapping)
                                        and payload.get("error") is not None
                                        else (
                                            "worker exited without a valid "
                                            "success"
                                        )
                                    )
                                )
                            ),
                            error_type=worker_error_type,
                            accounting=complete_accounting,
                        )
                    )
                    task_manifest_commitments[
                        spec.task_id
                    ] = task_manifest_sha256
                    if profile.task_artifact_worker_reports:
                        report = _r8c_e1e2_worker_report(
                            item["task_directory"],
                            spec=spec,
                        )
                        if report is None:
                            raise ConfigurationError(
                                "failed R8C E1/E2 task lacks its report"
                            )
                        worker_control_report_commitments[spec.task_id] = (
                            _worker_report_commitment(
                                report,
                                output_root=output_root,
                            )
                        )
                    failure_row = {
                            "task_id": spec.task_id,
                            "schedule_index": spec.schedule_index,
                            "status": status,
                            "outcome_class": failure_class,
                            "return_code": process.returncode,
                            "hard_timed_out": bool(
                                item.get("hard_timed_out")
                            ),
                            "timeout_requested": bool(
                                item.get("timeout_requested")
                            ),
                            "timeout_marker": (
                                _file_commitment(
                                    item["timeout_marker_path"]
                                )
                                if item.get("timeout_marker_path")
                                is not None
                                and Path(
                                    item["timeout_marker_path"]
                                ).is_file()
                                else None
                            ),
                            "attempt": 1,
                            "automatic_retries": 0,
                            "wall_seconds": runtime,
                            "cpu_seconds": conservative_cpu,
                            "peak_rss_bytes": int(item["peak_rss"]),
                            "output_bytes": _directory_bytes(
                                item["task_directory"]
                            ),
                            "scheduled_cfe": complete_accounting[
                                "scheduled_cfe"
                            ],
                            "charged_cfe": complete_accounting["charged_cfe"],
                            "scheduled_atomic_model_steps": (
                                complete_accounting[
                                    "scheduled_atomic_model_steps"
                                ]
                            ),
                            "charged_atomic_model_steps": (
                                complete_accounting[
                                    "charged_atomic_model_steps"
                                ]
                            ),
                            "charged_work_exact": complete_accounting[
                                "charged_work_exact"
                            ],
                            "charged_work_source": complete_accounting[
                                "charged_work_source"
                            ],
                            "task_manifest_sha256": task_manifest_sha256,
                            "algorithm_terminal_code": None,
                            "error_type": worker_error_type,
                            "worker_reported_status": (
                                payload.get("status")
                                if isinstance(payload, Mapping)
                                else None
                            ),
                        }
                    if not profile.task_artifact_worker_reports:
                        failure_row["logs"] = log_commitments
                    failures.append(failure_row)
            if accounted_cpu_seconds() >= plan.max_cpu:
                stop_dispatch("CPU_RESOURCE_CEILING")
                kill_active("CPU_RESOURCE_CEILING")

            while (
                dispatch_stopped_reason is None
                and next_index < len(schedule)
                and len(active) < plan.max_workers
            ):
                current_output = output_accounting.current_bytes()
                free_bytes = shutil.disk_usage(output_root.parent).free
                projected_reserve = output_accounting.reserve_bytes(
                    control_plane_reserve=plan.control_plane_reserve,
                    inflight_write_reserve_per_worker=(
                        plan.inflight_write_reserve_per_worker
                    ),
                    additional_workers=1,
                )
                if (
                    current_output + projected_reserve
                    >= plan.max_output
                ):
                    stop_dispatch("OUTPUT_IN_FLIGHT_RESERVE")
                    break
                if free_bytes < plan.stop_free + projected_reserve:
                    stop_dispatch("SCRATCH_IN_FLIGHT_RESERVE")
                    break
                if accounted_cpu_seconds() >= plan.max_cpu:
                    stop_dispatch("CPU_RESOURCE_CEILING")
                    break

                spec = schedule[next_index]
                if (
                    Path(spec.task_id).name != spec.task_id
                    or any(
                        separator in spec.task_id
                        for separator in ("/", "\\")
                    )
                ):
                    raise ConfigurationError(
                        "formal task ID is unsafe for task/log paths"
                    )
                task_directory = tasks_root / spec.task_id
                stdout_path = (
                    None
                    if profile.task_artifact_worker_reports
                    else logs_root / f"{spec.task_id}.stdout.log"
                )
                stderr_path = (
                    None
                    if profile.task_artifact_worker_reports
                    else logs_root / f"{spec.task_id}.stderr.log"
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--contract",
                    str(contract_path),
                    "--request",
                    str(request_path),
                    "--execution-profile",
                    profile.profile_id,
                    "--worker",
                    "--schedule-index",
                    str(next_index),
                    "--task-id",
                    spec.task_id,
                    "--task-directory",
                    str(task_directory),
                    "--stop-path",
                    str(stop_path),
                ]
                stdout_stream = None
                stderr_stream = None
                output_accounting.begin_worker(
                    spec.task_id,
                    task_directory=task_directory,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                try:
                    if stdout_path is not None and stderr_path is not None:
                        stdout_stream = stdout_path.open("xb")
                        stderr_stream = stderr_path.open("xb")
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=_worker_environment(launch_token),
                        stdout=(
                            subprocess.DEVNULL
                            if profile.task_artifact_worker_reports
                            else stdout_stream
                        ),
                        stderr=(
                            subprocess.DEVNULL
                            if profile.task_artifact_worker_reports
                            else stderr_stream
                        ),
                    )
                except Exception as error:
                    if stdout_stream is not None:
                        stdout_stream.close()
                    if stderr_stream is not None:
                        stderr_stream.close()
                    if profile.task_artifact_worker_reports:
                        log_commitments = None
                    else:
                        assert stdout_path is not None
                        assert stderr_path is not None
                        log_commitments = {
                            label: _root_relative_file_commitment(
                                path,
                                root=output_root,
                            )
                            for label, path in (
                                ("stdout", stdout_path),
                                ("stderr", stderr_path),
                            )
                        }
                        worker_log_commitments[
                            spec.task_id
                        ] = log_commitments
                    output_accounting.finish_worker(
                        spec.task_id,
                        retain_inflight_reserve=True,
                    )
                    status = "NOT_STARTED_TECHNICAL_FAILURE_NO_RETRY"
                    scheduled_accounting = _scheduled_task_accounting(spec)
                    launch_accounting = {
                        **scheduled_accounting,
                        "charged_cfe": 0,
                        "charged_atomic_model_steps": 0,
                        "charged_work_exact": True,
                        "charged_work_source": "NOT_STARTED",
                        "charged_work_recovery_error_type": None,
                    }
                    _, task_manifest_sha256 = (
                        _materialize_supervisor_task_outcome(
                            profile=profile,
                            spec=spec,
                            task_directory=task_directory,
                            status=status,
                            failure_class=(
                                TECHNICAL_WORKER_LAUNCH_FAILURE
                            ),
                            reason=(
                                TECHNICAL_WORKER_LAUNCH_FAILURE
                                if profile.task_artifact_worker_reports
                                else f"{type(error).__name__}: {error}"
                            ),
                            error_type=type(error).__name__,
                            accounting=launch_accounting,
                        )
                    )
                    task_manifest_commitments[
                        spec.task_id
                    ] = task_manifest_sha256
                    if profile.task_artifact_worker_reports:
                        report = _r8c_e1e2_worker_report(
                            task_directory,
                            spec=spec,
                        )
                        if report is None:
                            raise ConfigurationError(
                                "launch-failed R8C E1/E2 task lacks its report"
                            )
                        worker_control_report_commitments[spec.task_id] = (
                            _worker_report_commitment(
                                report,
                                output_root=output_root,
                            )
                        )
                    failure_row = {
                            "task_id": spec.task_id,
                            "schedule_index": spec.schedule_index,
                            "status": status,
                            "outcome_class": (
                                TECHNICAL_WORKER_LAUNCH_FAILURE
                            ),
                            "return_code": None,
                            "hard_timed_out": False,
                            "timeout_requested": False,
                            "timeout_marker": None,
                            "attempt": 1,
                            "automatic_retries": 0,
                            "wall_seconds": 0.0,
                            "cpu_seconds": 0.0,
                            "peak_rss_bytes": 0,
                            "output_bytes": _directory_bytes(
                                task_directory
                            ),
                            "scheduled_cfe": scheduled_accounting[
                                "scheduled_cfe"
                            ],
                            "charged_cfe": 0,
                            "scheduled_atomic_model_steps": (
                                scheduled_accounting[
                                    "scheduled_atomic_model_steps"
                                ]
                            ),
                            "charged_atomic_model_steps": 0,
                            "charged_work_exact": True,
                            "charged_work_source": "NOT_STARTED",
                            "task_manifest_sha256": task_manifest_sha256,
                            "algorithm_terminal_code": None,
                            "worker_reported_status": None,
                            "error_type": type(error).__name__,
                        }
                    if not profile.task_artifact_worker_reports:
                        failure_row["error"] = str(error)
                        failure_row["logs"] = log_commitments
                    failures.append(failure_row)
                    next_index += 1
                    stop_dispatch("WORKER_LAUNCH_FAILURE")
                    break
                finally:
                    if stdout_stream is not None and not stdout_stream.closed:
                        stdout_stream.close()
                    if stderr_stream is not None and not stderr_stream.closed:
                        stderr_stream.close()
                active[process.pid] = {
                    "process": process,
                    "spec": spec,
                    "started": time.monotonic(),
                    "peak_rss": 0,
                    "last_cpu": 0.0,
                    "last_sampled_at": time.monotonic(),
                    "stall_reported": False,
                    "task_directory": task_directory,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }
                next_index += 1
            if active or (
                next_index < len(schedule)
                and dispatch_stopped_reason is None
            ):
                time.sleep(min(plan.monitor_seconds, 30.0))
    except BaseException:
        kill_active("SUPERVISOR_EXCEPTION")
        raise

    dispatched_task_count = next_index
    if next_index < len(schedule):
        undispatched_reason = (
            dispatch_stopped_reason or "SUPERVISOR_STOPPED_DISPATCH"
        )
        for spec in schedule[next_index:]:
            task_directory = tasks_root / spec.task_id
            status = "NOT_DISPATCHED_TECHNICAL_STOP_NO_RETRY"
            scheduled_accounting = _scheduled_task_accounting(spec)
            undispatched_accounting = {
                **scheduled_accounting,
                "charged_cfe": 0,
                "charged_atomic_model_steps": 0,
                "charged_work_exact": True,
                "charged_work_source": "NOT_DISPATCHED",
                "charged_work_recovery_error_type": None,
            }
            _, task_manifest_sha256 = _materialize_supervisor_task_outcome(
                profile=profile,
                spec=spec,
                task_directory=task_directory,
                status=status,
                failure_class=TECHNICAL_NOT_DISPATCHED,
                reason=undispatched_reason,
                error_type=None,
                accounting=undispatched_accounting,
            )
            task_manifest_commitments[
                spec.task_id
            ] = task_manifest_sha256
            if profile.task_artifact_worker_reports:
                log_commitments = None
                report = _r8c_e1e2_worker_report(
                    task_directory,
                    spec=spec,
                )
                if report is None:
                    raise ConfigurationError(
                        "undispatched R8C E1/E2 task lacks its report"
                    )
                worker_control_report_commitments[spec.task_id] = (
                    _worker_report_commitment(
                        report,
                        output_root=output_root,
                    )
                )
            else:
                log_commitments = {
                    label: _root_relative_file_commitment(
                        path,
                        root=output_root,
                    )
                    for label, path in (
                        (
                            "stdout",
                            logs_root / f"{spec.task_id}.stdout.log",
                        ),
                        (
                            "stderr",
                            logs_root / f"{spec.task_id}.stderr.log",
                        ),
                    )
                }
                worker_log_commitments[spec.task_id] = log_commitments
            failure_row = {
                    "task_id": spec.task_id,
                    "schedule_index": spec.schedule_index,
                    "status": status,
                    "outcome_class": TECHNICAL_NOT_DISPATCHED,
                    "return_code": None,
                    "hard_timed_out": False,
                    "timeout_requested": False,
                    "timeout_marker": None,
                    "attempt": 1,
                    "automatic_retries": 0,
                    "wall_seconds": 0.0,
                    "cpu_seconds": 0.0,
                    "peak_rss_bytes": 0,
                    "output_bytes": _directory_bytes(task_directory),
                    "scheduled_cfe": scheduled_accounting[
                        "scheduled_cfe"
                    ],
                    "charged_cfe": 0,
                    "scheduled_atomic_model_steps": (
                        scheduled_accounting[
                            "scheduled_atomic_model_steps"
                        ]
                    ),
                    "charged_atomic_model_steps": 0,
                    "charged_work_exact": True,
                    "charged_work_source": "NOT_DISPATCHED",
                    "task_manifest_sha256": task_manifest_sha256,
                    "algorithm_terminal_code": None,
                    "worker_reported_status": None,
                    "error_type": None,
                }
            if not profile.task_artifact_worker_reports:
                failure_row["error"] = undispatched_reason
                failure_row["logs"] = log_commitments
            failures.append(failure_row)
    if len(task_manifest_commitments) != len(schedule):
        raise ConfigurationError(
            "formal supervisor did not commit every scheduled task outcome"
        )
    if profile.task_artifact_worker_reports:
        if (
            len(worker_control_report_commitments) != len(schedule)
            or worker_log_commitments
            or logs_root.exists()
        ):
            raise ConfigurationError(
                "R8C E1/E2 worker reports/log exclusion are incomplete"
            )
    elif len(worker_log_commitments) != len(schedule):
        raise ConfigurationError(
            "formal supervisor did not commit every worker log outcome"
        )

    status = (
        "COMPLETE_UNANALYZED"
        if (
            dispatch_stopped_reason is None
            and not failures
            and len(completed) == len(schedule)
            and all(item["status"] == "COMPLETE" for item in completed)
        )
        else (
            "INCOMPLETE_RESOURCE_CEILING"
            if dispatch_stopped_reason is not None
            else "INCOMPLETE_TASK_FAILURE"
        )
    )
    failure_cfe_values = [
        int(item["charged_cfe"])
        for item in failures
        if item.get("charged_cfe") is not None
    ]
    failure_atomic_values = [
        int(item["charged_atomic_model_steps"])
        for item in failures
        if item.get("charged_atomic_model_steps") is not None
    ]
    unknown_failure_accounting_count = sum(
        item.get("charged_cfe") is None
        or item.get("charged_atomic_model_steps") is None
        for item in failures
    )
    runtime_report = {
        "artifact_role": (
            f"{profile.artifact_stage}_REDACTED_CONTROL_PLANE_NO_EFFECTS"
        ),
        "status": status,
        "scheduled_task_count": len(schedule),
        "dispatched_task_count": dispatched_task_count,
        "recorded_outcome_count": len(completed) + len(failures),
        "completed_process_count": len(completed),
        "failed_process_count": sum(
            item.get("charged_work_source")
            not in {"NOT_STARTED", "NOT_DISPATCHED"}
            for item in failures
        ),
        "failed_outcome_count": len(failures),
        "not_dispatched_outcome_count": sum(
            item.get("charged_work_source") == "NOT_DISPATCHED"
            for item in failures
        ),
        "unknown_failure_accounting_count": (
            unknown_failure_accounting_count
        ),
        "dispatch_stopped_reason": dispatch_stopped_reason,
        "max_parallel_workers": plan.max_workers,
        "attempts_per_task": 1,
        "automatic_retries": 0,
        "total_wall_seconds": time.monotonic() - started,
        "reported_cpu_seconds": sum(
            item["cpu_seconds"] for item in completed
        )
        + sum(item["cpu_seconds"] for item in failures),
        "peak_pool_rss_bytes": peak_pool_rss,
        "completed": completed,
        "failures": failures,
        "task_manifest_commitments": task_manifest_commitments,
        "advisories": advisories,
        "effect_values_read_by_supervisor": False,
        "results_analysis_performed": False,
    }
    if profile.task_artifact_worker_reports:
        runtime_report.update(
            {
                "worker_control_report_commitments": (
                    worker_control_report_commitments
                ),
                "raw_worker_stdout_persisted": False,
                "raw_worker_stderr_persisted": False,
            }
        )
    else:
        runtime_report["worker_log_commitments"] = worker_log_commitments
    runtime_path = output_root / "runtime_report.json"
    _write_json(runtime_path, runtime_report)
    manifest = {
        "artifact_role": (
            f"{profile.artifact_stage}_IMMUTABLE_UNANALYZED"
        ),
        "status": status,
        "contract_sha256": file_sha256(contract_path),
        "request_sha256": file_sha256(request_path),
        "source": source,
        "schedule": {
            "id": contract["schedule"]["id"],
            "sha256": schedule_commitment(schedule),
            "e2_full_reuse_sha256": e2_full_reuse_commitment(schedule),
            "method_sequences": len(schedule),
            "completed_sequences": len(completed),
            "recorded_outcomes": len(completed) + len(failures),
            "reported_cfe": (
                sum(item["total_cfe"] for item in completed)
                + sum(failure_cfe_values)
            ),
            "reported_atomic_model_steps": sum(
                item["total_atomic_model_steps"] for item in completed
            )
            + sum(failure_atomic_values),
            "unknown_failure_accounting_count": (
                unknown_failure_accounting_count
            ),
        },
        "resources": {
            "reported_cpu_seconds": runtime_report["reported_cpu_seconds"],
            "total_wall_seconds": runtime_report["total_wall_seconds"],
            "peak_pool_rss_bytes": peak_pool_rss,
            "automatic_retries": 0,
        },
        "control_artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (
                schedule_path,
                reuse_path,
                binding_path,
                runtime_path,
                consumption_record_path,
            )
        },
        "task_manifest_commitments": task_manifest_commitments,
        "permissions": contract["permissions"],
        "analysis_gate": "R9_RAW_LOCK_AND_ANALYSIS_NOT_YET_AUTHORIZED",
    }
    if profile.task_artifact_worker_reports:
        manifest.update(
            {
                "worker_control_report_commitments": (
                    worker_control_report_commitments
                ),
                "raw_worker_stdout_persisted": False,
                "raw_worker_stderr_persisted": False,
            }
        )
    else:
        manifest["worker_log_commitments"] = worker_log_commitments
    manifest_path = output_root / "run_manifest.json"
    output_bytes_before_manifest = _directory_bytes(output_root)
    manifest_payload, final_total_output_bytes = (
        _run_manifest_payload_with_final_output_bytes(
            manifest,
            output_bytes_before_manifest=output_bytes_before_manifest,
        )
    )
    if final_total_output_bytes > plan.max_output:
        raise ConfigurationError(
            "final run manifest would exceed the output resource ceiling"
        )
    _write_bytes_exclusive(manifest_path, manifest_payload)
    observed_final_output_bytes = _directory_bytes(output_root)
    if observed_final_output_bytes != final_total_output_bytes:
        raise ConfigurationError(
            "final output byte count differs after run-manifest publication"
        )
    if observed_final_output_bytes > plan.max_output:
        raise ConfigurationError(
            "final output tree exceeds the output resource ceiling"
        )
    return {
        "artifact_role": manifest["artifact_role"],
        "status": status,
        "output_root": str(output_root),
        "completed_sequences": len(completed),
        "failed_sequences": len(failures),
        "automatic_retries": 0,
        "results_analysis_performed": False,
        "run_manifest_sha256": file_sha256(manifest_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one contract-bound R8/R8C formal execution profile",
        allow_abbrev=False,
    )
    parser.add_argument("--contract")
    parser.add_argument(
        "--execution-profile",
        choices=tuple(RUNNER_PROFILES),
        default=LEGACY_PROFILE.profile_id,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-root", required=False)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source, contract, request, host and paths without launch",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--schedule-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--task-id", help=argparse.SUPPRESS)
    parser.add_argument("--task-directory", help=argparse.SUPPRESS)
    parser.add_argument("--stop-path", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = RUNNER_PROFILES[args.execution_profile]
    if args.contract is None:
        args.contract = str(profile.default_contract)
    try:
        if args.worker:
            if not os.environ.get(LAUNCH_TOKEN_ENV):
                raise ConfigurationError(
                    "formal worker mode is supervisor-internal and requires "
                    "its launch token"
                )
            if args.preflight_only:
                raise ConfigurationError(
                    "formal worker cannot enter preflight-only mode"
                )
            if (
                args.schedule_index is None
                or args.task_id is None
                or args.task_directory is None
                or args.stop_path is None
            ):
                raise ConfigurationError("incomplete R8 worker invocation")
            return _run_worker(args)
        if args.output_root is None:
            raise ConfigurationError("R8 supervisor requires --output-root")
        if args.preflight_only:
            profile = RUNNER_PROFILES[args.execution_profile]
            contract_path = Path(args.contract).resolve()
            request_path = Path(args.request).resolve()
            contract, request, schedule = _load_and_validate(
                contract_path,
                request_path,
                profile,
            )
            plan = _validate_prelaunch(
                args,
                contract,
                request,
                schedule,
            )
            summary = {
                "artifact_role": (
                    f"{profile.artifact_stage}_PREFLIGHT_CONTROL_PLANE_ONLY"
                ),
                "status": "PREFLIGHT_PASS",
                "source": plan.source,
                "host_fingerprint_sha256": (
                    host_fingerprint_sha256(plan.host)
                ),
                "schedule_sequences": len(schedule),
                "output_root": str(plan.output_root),
                "request_consumption_marker": str(plan.marker_path),
                "max_workers": plan.max_workers,
                "max_pool_peak_rss_bytes": plan.max_pool_rss,
                "max_total_output_bytes": plan.max_output,
                "active_and_control_write_reserve_bytes": (
                    plan.active_write_reserve
                ),
                "request_consumed": False,
                "formal_execution_started": False,
            }
        else:
            summary = _run_supervisor(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
