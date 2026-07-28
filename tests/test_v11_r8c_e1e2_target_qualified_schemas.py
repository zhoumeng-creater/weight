from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from formal_execution.host import host_fingerprint, host_fingerprint_sha256
from resource_pilot.e1e2_fullpath import (
    CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID,
    CDF_OPERATIONAL_AUTHORITY_ID,
    CDF_OPERATIONAL_SUITE_ID,
    DEFAULT_DYNAMIC_EVENTS,
    DEFAULT_REPETITIONS,
    DEFAULT_WORKERS,
    DYNAMIC_CFE_PER_EVENT,
    QUALIFICATION_ID,
    ROLLING_CFE_PER_EVENT,
    RSS_ROUNDING_BYTES,
    RSS_SAFETY_FACTOR,
    STATIC_CFE_PER_EVENT,
    SUPERVISOR_MEMORY_RESERVE_BYTES,
    formal_schedule_weights,
    qualification_profiles,
    r5_memory_limits,
    runtime_environment_lock_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "r8c_e1e2"
CONTRACT_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_target_qualified_contract.schema.json"
)
REQUEST_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_target_qualified_execution_request.schema.json"
)
CONTRACT_ID = (
    "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
    "FORMAL-EXECUTION-CONTRACT-01"
)
REQUEST_ID = (
    "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
    "EXECUTION-REQUEST-20260726-01"
)
SCHEDULE_SHA256 = (
    "db468253fb1430749d9f816d19532e428ca1054a86f399f80b12575a5c45282d"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(path: Path) -> Draft202012Validator:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _target_runtime_environment() -> dict[str, Any]:
    evidence = runtime_environment_lock_evidence()
    evidence.update(
        {
            "actual_python_implementation": "CPython",
            "actual_python_version": "3.12.0",
            "actual_system": "Linux",
            "actual_machine": "x86_64",
            "installed_package_versions": dict(
                evidence["locked_package_versions"]
            ),
            "missing_locked_packages": [],
            "version_mismatches": {},
            "interpreter_matches": True,
            "platform_matches": True,
            "all_locked_packages_match": True,
            "target_environment_match": True,
        }
    )
    return evidence


def _qualified_contract(tmp_path: Path) -> dict[str, Any]:
    contract = _read_json(
        CONFIG / "r8c_e1e2_formal_execution_contract.json"
    )
    host = host_fingerprint()
    command = "python test-only-target-qualified-launch.py"
    contract.update(
        {
            "contract_id": CONTRACT_ID,
            "protocol_stage": "R8C_E1E2_TARGET_QUALIFIED_FORMAL_EXECUTION",
            "status": "TARGET_HOST_QUALIFIED_AND_AUTHORIZED",
        }
    )
    contract["authorization"].update(
        {
            "author_text": command,
            "authorized_scope": (
                "FORMAL_E1_E2_PUBLIC_BENCHMARK_EFFECT_EXECUTION_ONLY"
            ),
            "formal_effect_execution_authorized": True,
        }
    )
    contract["launch"].update(
        {
            "contract_path": str(tmp_path / "qualified-contract.json"),
            "request_path": str(tmp_path / "qualified-request.json"),
            "request_consumption_marker": str(
                tmp_path / "qualified-request.consumed"
            ),
            "output_root": str(tmp_path / "formal-output"),
            "exact_command": command,
            "command_identity_frozen": True,
            "command_executable_now": True,
            "formal_launch_prohibited": False,
            "current_confirmation_state": (
                "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED"
            ),
        }
    )
    resources = contract["resources"]
    resources.pop("candidate_profile_frozen")
    resources.update(
        {
            "qualification_status": "TARGET_HOST_QUALIFIED",
            "selected_exact_host_frozen": True,
        }
    )
    resources["candidate_target"].update(
        {
            "processor_reference": host["cpu_model"],
            "offered_instance_description": "dynamic local test fixture",
            "normalized_compute_allocation": (
                f"{host['effective_logical_processors']}_LOGICAL_PROCESSORS"
            ),
            "memory_gib": host["memory_bytes"] / (1024**3),
            "provider": "LOCAL_TEST_FIXTURE",
            "instance_type": "DYNAMIC_CURRENT_HOST",
            "host_fingerprint_sha256": host_fingerprint_sha256(host),
            "remote_measurement_completed": True,
        }
    )
    resources["parallelism"] = {
        "max_workers": 1,
        "logical_threads_per_worker": 1,
        "blas_openmp_threads_per_worker": 1,
        "max_worker_peak_rss_bytes": 536870912,
        "max_pool_peak_rss_bytes": 536870912,
        "worker_count_qualified_on_target": True,
    }
    resources["decision_rule"]["current_decision"] = (
        "GO_TARGET_HOST_QUALIFIED"
    )
    resources["scratch"].update(
        {
            "required_root": str(tmp_path),
            "minimum_free_bytes_at_start": 34359738368,
            "stop_dispatch_below_free_bytes": 8589934592,
        }
    )
    resources["output"] = {
        "max_total_bytes": 12884901888,
        "control_plane_reserve_bytes": 67108864,
        "max_inflight_write_bytes_per_worker": 8388608,
        "raw_evaluations_required": False,
        "format": "WGT_CFE_CHECKPOINT_BINARY_V1_ENDPOINT_SUFFICIENT",
        "silent_truncation": False,
    }
    contract["persistence"]["data_plane"] = (
        "Immutable endpoint-sufficient checkpoint fronts; no per-evaluation "
        "raw rows."
    )
    contract["permissions"]["public_benchmark_effect_execution"] = True
    contract["fail_closed_gate"] = {
        "request_id": REQUEST_ID,
        "request_status": "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED",
        "target_host_status": "TARGET_HOST_QUALIFIED",
        "formal_launch_status": "ELIGIBLE",
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    profiles = [asdict(profile) for profile in qualification_profiles()]
    weights = [dict(row) for row in formal_schedule_weights()]
    path_count = len(
        {
            (row["workload_id"], row["method_id"])
            for row in profiles
        }
    )
    binding_count = len(profiles)
    representative_case_count = len(
        {row["representative_case_id"] for row in profiles}
    )
    rate_count = len(
        {
            (
                row["workload_id"],
                row["method_id"],
                row["projection_rate_class"],
            )
            for row in profiles
        }
    )
    sweep_task_count = DEFAULT_REPETITIONS * binding_count
    total_task_count = len(DEFAULT_WORKERS) * sweep_task_count
    qualification_root = (
        tmp_path / "r8c-e1e2-qualification-target-evidence"
    )
    qualification_root.mkdir(exist_ok=True)
    report_path = qualification_root / "qualification_report.json"
    measured_worker_rss = 64 * 1024 * 1024
    conservative_worker_rss = 80 * 1024 * 1024
    r5_max_worker_rss, r5_max_pool_rss = r5_memory_limits()
    host_memory_bytes = int(host["memory_bytes"])
    runtime_environment = _target_runtime_environment()
    projections = [
        {
            "workers": workers,
            "status": "TARGET_HOST_FULL_PATH_ESTIMATE",
            "formal_task_count": 5_030,
            "formal_cfe": 851_000_000,
            "method_cfe_weighted_rates": weights,
            "projected_wall_seconds": 43_200.0,
            "projected_wall_hours": 12.0,
            "projected_wall_hours_with_25_percent_headroom": 15.0,
            "memory_qualification": {
                "measured_max_worker_peak_rss_bytes": measured_worker_rss,
                "rss_safety_factor": RSS_SAFETY_FACTOR,
                "rss_rounding_bytes": RSS_ROUNDING_BYTES,
                "conservative_worker_peak_rss_bytes": (
                    conservative_worker_rss
                ),
                "worker_count": workers,
                "conservative_pool_peak_rss_bytes": (
                    conservative_worker_rss * workers
                ),
                "r5_max_worker_peak_rss_bytes": r5_max_worker_rss,
                "r5_max_pool_peak_rss_bytes": r5_max_pool_rss,
                "host_memory_bytes": host_memory_bytes,
                "supervisor_memory_reserve_bytes": (
                    SUPERVISOR_MEMORY_RESERVE_BYTES
                ),
                "host_pool_ceiling_bytes": (
                    host_memory_bytes - SUPERVISOR_MEMORY_RESERVE_BYTES
                ),
                "eligible": True,
            },
            "decision_classification": (
                "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
            ),
        }
        for workers in DEFAULT_WORKERS
    ]
    report = {
        "artifact_role": (
            "R8C_E1E2_RESULT_BLIND_FULL_PATH_TARGET_QUALIFICATION"
        ),
        "qualification_id": QUALIFICATION_ID,
        "status": "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE",
        "mode": "TARGET_QUALIFICATION",
        "output_root": str(qualification_root.resolve()),
        "code_identity": {
            "git_commit": commit,
            "git_tree": tree,
            "worktree_clean": True,
            "qualification_source_sha256": sha256(
                (
                    ROOT
                    / "src"
                    / "resource_pilot"
                    / "e1e2_fullpath.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "host_fingerprint": host,
        "host_fingerprint_sha256": host_fingerprint_sha256(host),
        "runtime_environment_lock": runtime_environment,
        "runtime_contract": {
            "workload_method_paths_covered": path_count,
            "workload_method_case_bindings_covered": binding_count,
            "unique_representative_benchmark_cases_covered": (
                representative_case_count
            ),
            "formal_projection_rate_classes_covered": rate_count,
            "cdf_operational_suite_id": CDF_OPERATIONAL_SUITE_ID,
            "cdf_operational_authority_id": CDF_OPERATIONAL_AUTHORITY_ID,
            "cdf_operational_authority_amendment_id": (
                CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
            ),
            "cdf9_undefined_domain_policy": (
                "CHARGED_TYPED_CDFDomainUndefinedError_TO_"
                "REJECT_NUMERICAL_NO_EXTENSION"
            ),
            "dynamic_event_ids_covered": list(
                range(DEFAULT_DYNAMIC_EVENTS)
            ),
            "cdf9_max_domain_stress_event_id": 5,
            "cdf9_max_domain_stress_event_exercised": True,
        },
        "pilot_design": {
            "worker_counts": list(DEFAULT_WORKERS),
            "repetitions": DEFAULT_REPETITIONS,
            "static_cfe_per_event": STATIC_CFE_PER_EVENT,
            "dynamic_cfe_per_event": DYNAMIC_CFE_PER_EVENT,
            "rolling_cfe_per_event": ROLLING_CFE_PER_EVENT,
            "dynamic_events": DEFAULT_DYNAMIC_EVENTS,
            "profiles": profiles,
            "task_count": total_task_count,
            "all_task_ids_unique": True,
        },
        "formal_schedule_control_weights": weights,
        "sweeps": [
            {
                "workers_requested": workers,
                "task_count": sweep_task_count,
                "passed_task_count": sweep_task_count,
                "failed_task_count": 0,
                "max_task_peak_rss_bytes": measured_worker_rss,
            }
            for workers in DEFAULT_WORKERS
        ],
        "e1_e2_wall_projection": {"projections": projections},
        "worker_recommendation": {
            "status": (
                "MEMORY_ELIGIBLE_THROUGHPUT_OPTIMUM_IDENTIFIED"
            ),
            "selection_rule": (
                "MIN_PROJECTED_WALL_HOURS_AMONG_R5_AND_HOST_RSS_ELIGIBLE_"
                "MEASURED_WORKER_COUNTS;_LOWER_WORKER_COUNT_BREAKS_EXACT_TIES"
            ),
            "measured_worker_counts": list(DEFAULT_WORKERS),
            "memory_eligible_worker_counts": list(DEFAULT_WORKERS),
            "recommended_worker_count": 1,
            "recommended_projected_wall_hours": 12.0,
            "recommended_projected_wall_hours_with_25_percent_headroom": (
                15.0
            ),
            "recommended_decision_classification": (
                "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
            ),
            "formal_launch_authorized": False,
        },
        "target_qualification_complete": True,
        "formal_launch_authorized": False,
        "failed_task_count": 0,
        "automatic_retries": 0,
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    report_path.write_bytes(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    report_sha256 = sha256(report_path.read_bytes()).hexdigest()
    contract["target_qualification_evidence"] = {
        "qualification_report_path": str(report_path.resolve()),
        "qualification_report_sha256": report_sha256,
        "qualification_id": report["qualification_id"],
        "qualification_status": report["status"],
        "source": report["code_identity"],
        "host_fingerprint_sha256": report["host_fingerprint_sha256"],
        "design": {
            "worker_counts": list(DEFAULT_WORKERS),
            "repetitions": DEFAULT_REPETITIONS,
            "static_cfe_per_event": STATIC_CFE_PER_EVENT,
            "dynamic_cfe_per_event": DYNAMIC_CFE_PER_EVENT,
            "rolling_cfe_per_event": ROLLING_CFE_PER_EVENT,
            "dynamic_events": DEFAULT_DYNAMIC_EVENTS,
            "workload_method_case_binding_count": binding_count,
            "unique_representative_benchmark_case_count": (
                representative_case_count
            ),
            "formal_projection_rate_class_count": rate_count,
            "task_count": total_task_count,
        },
        "selected_worker_count": 1,
        "selected_projection": {
            "status": "TARGET_HOST_FULL_PATH_ESTIMATE",
            "projected_wall_hours": 12.0,
            "decision_classification": (
                "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
            ),
        },
    }
    return contract


def _qualified_request(
    contract_sha256: str,
    *,
    command: str,
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "request_id": REQUEST_ID,
        "scope": "benchmark_effect",
        "companion_scope": "benchmark_effect",
        "contracts": {
            "protocol_id": "WGT-JOURNAL-2026-01",
            "r5_contract_id": (
                "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
            ),
            "r5_contract_sha256": (
                "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
            ),
            "r5a_contract_id": "WGT-V11-R5A-E3-INPUT-CONTRACT-01",
            "r5a_contract_sha256": (
                "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
            ),
            "corrective_protocol_id": (
                "WGT-V11-R8C-RESULT-BLIND-CORRECTIVE-PROTOCOL-01"
            ),
            "corrective_protocol_sha256": (
                "dfe74d041f36b12fd13cb86e1fa2bba5483bbd871a7749b2c98e09160ee39b43"
            ),
            "r8c_formal_contract_id": CONTRACT_ID,
            "r8c_formal_contract_sha256": contract_sha256,
            "formal_schedule_id": (
                "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01"
            ),
            "formal_schedule_sha256": SCHEDULE_SHA256,
            "source_git_commit": commit,
            "source_git_tree": tree,
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


def _load_runner() -> Any:
    module_name = "_test_target_qualified_schema_runner"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "tools" / "run_v11_r8_formal.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_target_schema_design_counts_follow_qualification_generator() -> None:
    profiles = qualification_profiles()
    design_properties = _read_json(CONTRACT_SCHEMA_PATH)["properties"][
        "target_qualification_evidence"
    ]["properties"]["design"]["properties"]
    assert design_properties[
        "workload_method_case_binding_count"
    ]["const"] == len(profiles)
    assert design_properties[
        "unique_representative_benchmark_case_count"
    ]["const"] == len(
        {profile.representative_case_id for profile in profiles}
    )
    assert design_properties[
        "formal_projection_rate_class_count"
    ]["const"] == len({profile.rate_key for profile in profiles})
    assert design_properties["task_count"]["const"] == (
        len(DEFAULT_WORKERS) * DEFAULT_REPETITIONS * len(profiles)
    )


def test_dynamic_local_qualified_pair_passes_schemas_and_runner_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _qualified_contract(tmp_path)
    contract_path = tmp_path / "qualified-contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    contract_sha256 = sha256(contract_path.read_bytes()).hexdigest()
    request = _qualified_request(
        contract_sha256,
        command=contract["launch"]["exact_command"],
    )
    request_path = tmp_path / "qualified-request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    _validator(CONTRACT_SCHEMA_PATH).validate(contract)
    _validator(REQUEST_SCHEMA_PATH).validate(request)
    runner = _load_runner()
    report_environment = _target_runtime_environment()
    monkeypatch.setattr(
        runner,
        "_runtime_environment_lock_evidence",
        lambda: deepcopy(report_environment),
    )
    loaded_contract, loaded_request, schedule = runner._load_and_validate(
        contract_path,
        request_path,
        runner.CORRECTIVE_E1E2_PROFILE,
    )

    assert loaded_contract == contract
    assert loaded_request.request_id == REQUEST_ID
    assert len(schedule) == 5030


def test_runner_reloads_canonical_target_qualification_evidence_and_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _qualified_contract(tmp_path)
    source = contract["target_qualification_evidence"]["source"]
    request = SimpleNamespace(
        contracts=SimpleNamespace(
            source_git_commit=source["git_commit"],
            source_git_tree=source["git_tree"],
        )
    )
    runner = _load_runner()
    report_environment = json.loads(
        Path(
            contract["target_qualification_evidence"][
                "qualification_report_path"
            ]
        ).read_text(encoding="utf-8")
    )["runtime_environment_lock"]
    monkeypatch.setattr(
        runner,
        "_runtime_environment_lock_evidence",
        lambda: deepcopy(report_environment),
    )
    runner._validate_target_qualification_evidence(contract, request)

    report_path = Path(
        contract["target_qualification_evidence"][
            "qualification_report_path"
        ]
    )
    original = report_path.read_bytes()
    nonoptimal = deepcopy(contract)
    nonoptimal["target_qualification_evidence"][
        "selected_worker_count"
    ] = 32
    nonoptimal["resources"]["parallelism"]["max_workers"] = 32
    with pytest.raises(
        runner.ConfigurationError,
        match="selected worker lacks a frozen <=36h GO projection",
    ):
        runner._validate_target_qualification_evidence(nonoptimal, request)

    report_path.write_bytes(original + b" ")
    with pytest.raises(
        runner.ConfigurationError,
        match="qualification report hash drifted",
    ):
        runner._validate_target_qualification_evidence(contract, request)

    report = json.loads(original)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contract["target_qualification_evidence"][
        "qualification_report_sha256"
    ] = sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(
        runner.ConfigurationError,
        match="not canonical JSON with one LF",
    ):
        runner._validate_target_qualification_evidence(contract, request)

    for index, projection in enumerate(
        report["e1_e2_wall_projection"]["projections"]
    ):
        projected_hours = 37.0 + index
        projection["projected_wall_seconds"] = projected_hours * 3_600.0
        projection["projected_wall_hours"] = projected_hours
        projection[
            "projected_wall_hours_with_25_percent_headroom"
        ] = projected_hours * 1.25
        projection["decision_classification"] = (
            "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
        )
    report["worker_recommendation"].update(
        {
            "recommended_projected_wall_hours": 37.0,
            "recommended_projected_wall_hours_with_25_percent_headroom": (
                46.25
            ),
            "recommended_decision_classification": (
                "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
            ),
        }
    )
    contract["target_qualification_evidence"]["selected_projection"] = {
        "status": "TARGET_HOST_FULL_PATH_ESTIMATE",
        "projected_wall_hours": 37.0,
        "decision_classification": "HOLD_OPTIMIZE_CONTENTION_AND_RETEST",
    }
    report_path.write_bytes(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    contract["target_qualification_evidence"][
        "qualification_report_sha256"
    ] = sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(
        runner.ConfigurationError,
        match="selected worker lacks a frozen <=36h GO projection",
    ):
        runner._validate_target_qualification_evidence(contract, request)


def test_contract_schema_rejects_wrong_status_extra_and_old_raw_output(
    tmp_path: Path,
) -> None:
    validator = _validator(CONTRACT_SCHEMA_PATH)
    valid = _qualified_contract(tmp_path)

    wrong_status = deepcopy(valid)
    wrong_status["status"] = "TARGET_HOST_QUALIFIED"
    assert list(validator.iter_errors(wrong_status))

    extra = deepcopy(valid)
    extra["resources"]["parallelism"]["candidate_workers"] = 32
    assert list(validator.iter_errors(extra))

    old_raw = deepcopy(valid)
    old_raw["resources"]["output"] = {
        "max_total_bytes": 193273528320,
        "raw_evaluations_required": True,
        "format": "deterministic gzip UTF-8 canonical JSONL task chunks",
        "silent_truncation_allowed": False,
    }
    assert list(validator.iter_errors(old_raw))

    missing_qualification = deepcopy(valid)
    missing_qualification.pop("target_qualification_evidence")
    assert list(validator.iter_errors(missing_qualification))

    shortened_design = deepcopy(valid)
    shortened_design["target_qualification_evidence"]["design"][
        "static_cfe_per_event"
    ] = 5_000
    assert list(validator.iter_errors(shortened_design))


def test_request_schema_rejects_nonbenchmark_and_recursive_extra(
    tmp_path: Path,
) -> None:
    contract = _qualified_contract(tmp_path)
    request = _qualified_request(
        "a" * 64,
        command=contract["launch"]["exact_command"],
    )
    validator = _validator(REQUEST_SCHEMA_PATH)
    validator.validate(request)

    nonbenchmark = deepcopy(request)
    nonbenchmark["companion_scope"] = "weight_effect"
    assert list(validator.iter_errors(nonbenchmark))

    extra = deepcopy(request)
    extra["contracts"]["unfrozen_contract"] = "not allowed"
    assert list(validator.iter_errors(extra))
