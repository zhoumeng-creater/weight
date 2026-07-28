from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from dt_ramde_v11.contracts import (
    ConfigurationError,
    ExecutionScope,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
)
from formal_execution.runtime import (
    CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    RAW_GZIP_COMPRESSLEVEL,
    parse_r8c_request,
)
from formal_execution.host import (
    host_fingerprint,
    host_fingerprint_sha256,
    process_cpu_seconds,
    process_rss_bytes,
)
from formal_execution.schedule import (
    build_corrective_e1e2_formal_schedule,
    build_corrective_formal_schedule,
    e2_full_reuse_commitment,
    schedule_commitment,
)


E1E2_SCHEDULE_SHA256 = (
    "db468253fb1430749d9f816d19532e428ca1054a86f399f80b12575a5c45282d"
)
REUSE_SHA256 = (
    "d235c1c53d7e504400ad37674bebba4a01145a934964039454776c9f09ba0c9e"
)


def _r5() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "r5"
        / "r5_freeze_contract.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _bindings() -> R8CCorrectiveContractBindings:
    return R8CCorrectiveContractBindings(
        protocol_id="WGT-JOURNAL-2026-01",
        r5_contract_id=(
            "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
        ),
        r5_contract_sha256=(
            "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
        ),
        r5a_contract_id="WGT-V11-R5A-E3-INPUT-CONTRACT-01",
        r5a_contract_sha256=(
            "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
        ),
        corrective_protocol_id=(
            "WGT-V11-R8C-RESULT-BLIND-CORRECTIVE-PROTOCOL-01"
        ),
        corrective_protocol_sha256=(
            "dfe74d041f36b12fd13cb86e1fa2bba5483bbd871a7749b2c98e09160ee39b43"
        ),
        r8c_formal_contract_id=(
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "FORMAL-EXECUTION-CONTRACT-01"
        ),
        r8c_formal_contract_sha256="a" * 64,
        formal_schedule_id="WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01",
        formal_schedule_sha256=E1E2_SCHEDULE_SHA256,
        source_git_commit="b" * 40,
        source_git_tree="c" * 40,
    )


def _request(
    *,
    scope: ExecutionScope = ExecutionScope.BENCHMARK_EFFECT,
    companion_scope: ExecutionScope = ExecutionScope.BENCHMARK_EFFECT,
) -> R8CCorrectiveExecutionRequest:
    command = "test-only-e1e2-command"
    return R8CCorrectiveExecutionRequest(
        scope=scope,
        companion_scope=companion_scope,
        contracts=_bindings(),
        request_id=(
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "EXECUTION-REQUEST-20260726-01"
        ),
        frozen_exact_command=command,
        author_confirmation_text=command,
        author_exact_command_confirmed=True,
    )


def test_e1e2_schedule_is_exact_corrective_prefix() -> None:
    full = build_corrective_formal_schedule(_r5())
    staged = build_corrective_e1e2_formal_schedule(_r5())

    assert staged == full[:5030]
    assert not any(row.workload_id == "E3" for row in staged)
    assert schedule_commitment(staged) == E1E2_SCHEDULE_SHA256
    assert e2_full_reuse_commitment(staged) == REUSE_SHA256
    assert sum(row.total_cfe for row in staged) == 851_000_000
    assert (
        sum(row.total_atomic_steps for row in staged)
        == 1_971_000_000
    )


def test_e1e2_runtime_and_request_are_benchmark_only() -> None:
    assert CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS.population_size == 100
    assert CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS.archive_capacity == 100
    assert CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS.corrective is True
    assert CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS.artifact_stage == (
        "R8C_E1E2"
    )
    _request().validate()

    with pytest.raises(
        ConfigurationError,
        match="permits benchmark_effect scope only",
    ):
        _request(
            companion_scope=ExecutionScope.WEIGHT_EFFECT,
        ).validate()


def test_e1e2_pending_contract_schema_and_runner_fail_closed(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "config" / "r8c_e1e2"
    contract = json.loads(
        (config / "r8c_e1e2_formal_execution_contract.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (
            config
            / "r8c_e1e2_formal_execution_contract.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(contract)

    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "run_v11_r8c_e1e2_formal.py"),
            "--contract",
            str(
                config
                / "r8c_e1e2_formal_execution_contract.json"
            ),
            "--request",
            str(tmp_path / "request-must-not-be-read.json"),
            "--output-root",
            str(output_root),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "identity" in result.stderr
    assert not output_root.exists()


def test_host_attestation_and_process_sampling_are_live() -> None:
    fingerprint = host_fingerprint()
    assert fingerprint["visible_logical_processors"] >= 1
    assert fingerprint["memory_bytes"] > 0
    assert len(host_fingerprint_sha256(fingerprint)) == 64
    assert process_rss_bytes(os.getpid()) > 0
    assert process_cpu_seconds(os.getpid()) >= 0.0


def test_formal_raw_gzip_level_is_frozen_for_fast_writes() -> None:
    assert RAW_GZIP_COMPRESSLEVEL == 1


def test_r8c_request_parser_rejects_boolean_coercion_and_extra_fields() -> None:
    payload = {
        "request_id": (
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "EXECUTION-REQUEST-20260726-01"
        ),
        "scope": "benchmark_effect",
        "companion_scope": "benchmark_effect",
        "contracts": {
            key: value
            for key, value in vars(_bindings()).items()
        },
        "frozen_exact_command": "test-only-e1e2-command",
        "author_confirmation_text": "test-only-e1e2-command",
        "author_exact_command_confirmed": True,
        "formal_effect_execution_requested": True,
        "participant_data_requested": False,
        "hidden_generation_requested": False,
        "results_analysis_requested": False,
        "results_writing_requested": False,
        "remote_git_mutation_requested": False,
        "release_or_distribution_requested": False,
    }
    parse_r8c_request(payload)

    coerced = dict(payload)
    coerced["author_exact_command_confirmed"] = "false"
    with pytest.raises(ConfigurationError, match="JSON boolean"):
        parse_r8c_request(coerced)

    unexpected = {**payload, "unfrozen_field": False}
    with pytest.raises(ConfigurationError, match="unexpected"):
        parse_r8c_request(unexpected)


def test_target_qualified_contract_loads_and_request_is_consumed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "config"
        / "r8c_e1e2"
        / "r8c_e1e2_formal_execution_contract.json"
    )
    contract = json.loads(source.read_text(encoding="utf-8"))
    contract["contract_id"] = (
        "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
        "FORMAL-EXECUTION-CONTRACT-01"
    )
    command = "test-only-qualified-command"
    contract["protocol_stage"] = (
        "R8C_E1E2_TARGET_QUALIFIED_FORMAL_EXECUTION"
    )
    contract["status"] = "TARGET_HOST_QUALIFIED_AND_AUTHORIZED"
    contract["authorization"].update(
        {
            "authorized_scope": (
                "FORMAL_E1_E2_PUBLIC_BENCHMARK_EFFECT_EXECUTION_ONLY"
            ),
            "formal_effect_execution_authorized": True,
        }
    )
    contract["launch"].update(
        {
            "request_path": str(tmp_path / "request.json"),
            "request_consumption_marker": str(tmp_path / "request.consumed"),
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
    contract["resources"].pop("candidate_profile_frozen")
    contract["resources"]["qualification_status"] = (
        "TARGET_HOST_QUALIFIED"
    )
    contract["resources"]["selected_exact_host_frozen"] = True
    contract["resources"]["candidate_target"].update(
        {
            "provider": "LOCAL_TEST_FIXTURE",
            "instance_type": "DYNAMIC_CURRENT_HOST",
            "host_fingerprint_sha256": host_fingerprint_sha256(),
            "remote_measurement_completed": True,
        }
    )
    contract["resources"]["parallelism"] = {
        "max_workers": 1,
        "logical_threads_per_worker": 1,
        "blas_openmp_threads_per_worker": 1,
        "max_worker_peak_rss_bytes": 2_147_483_648,
        "max_pool_peak_rss_bytes": 2_147_483_648,
        "worker_count_qualified_on_target": True,
    }
    contract["resources"]["output"] = {
        "max_total_bytes": 12_884_901_888,
        "control_plane_reserve_bytes": 67_108_864,
        "max_inflight_write_bytes_per_worker": 8_388_608,
        "raw_evaluations_required": False,
        "format": "WGT_CFE_CHECKPOINT_BINARY_V1_ENDPOINT_SUFFICIENT",
        "silent_truncation": False,
    }
    contract["resources"]["scratch"]["required_root"] = str(tmp_path)
    contract["permissions"]["public_benchmark_effect_execution"] = True
    contract["fail_closed_gate"] = {
        "request_id": (
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "EXECUTION-REQUEST-20260726-01"
        ),
        "request_status": "ONE_TIME_SOURCE_BOUND_VERBATIM_CONFIRMED",
        "target_host_status": "TARGET_HOST_QUALIFIED",
        "formal_launch_status": "ELIGIBLE",
    }
    contract["target_qualification_evidence"] = {
        "qualification_report_path": str(
            (tmp_path / "qualification_report.json").resolve()
        ),
        "qualification_report_sha256": "a" * 64,
        "qualification_id": (
            "WGT-V11-R8C-E1E2-TARGET-QUALIFICATION-20260726-02"
        ),
        "qualification_status": (
            "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
        ),
        "source": {
            "git_commit": "b" * 40,
            "git_tree": "c" * 40,
            "worktree_clean": True,
            "qualification_source_sha256": "d" * 64,
        },
        "host_fingerprint_sha256": host_fingerprint_sha256(),
        "design": {
            "worker_counts": [1, 8, 16, 24, 32, 48, 64],
            "repetitions": 2,
            "static_cfe_per_event": 50_000,
            "dynamic_cfe_per_event": 5_000,
            "rolling_cfe_per_event": 5_000,
            "dynamic_events": 6,
            "workload_method_case_binding_count": 84,
            "unique_representative_benchmark_case_count": 8,
            "formal_projection_rate_class_count": 39,
            "task_count": 1176,
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
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    request = {
        "request_id": (
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "EXECUTION-REQUEST-20260726-01"
        ),
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
            "r8c_formal_contract_id": (
                "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
                "FORMAL-EXECUTION-CONTRACT-01"
            ),
            "r8c_formal_contract_sha256": contract_hash,
            "formal_schedule_id": (
                "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01"
            ),
            "formal_schedule_sha256": E1E2_SCHEDULE_SHA256,
            "source_git_commit": "b" * 40,
            "source_git_tree": "c" * 40,
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
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    module_spec = importlib.util.spec_from_file_location(
        "_test_run_v11_r8_formal",
        root / "tools" / "run_v11_r8_formal.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = runner
    module_spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner,
        "_validate_target_qualification_evidence",
        lambda contract, request: None,
    )
    loaded_contract, loaded_request, schedule = runner._load_and_validate(
        contract_path,
        request_path,
        runner.CORRECTIVE_E1E2_PROFILE,
    )
    assert len(schedule) == 5030

    marker_path = Path(loaded_contract["launch"]["request_consumption_marker"])
    marker_payload = {
        "request_id": loaded_request.request_id,
        "request_sha256": hashlib.sha256(
            request_path.read_bytes()
        ).hexdigest(),
    }
    marker = runner._consume_request_once(
        marker_path=marker_path,
        payload=marker_payload,
    )
    assert marker.is_file()
    with pytest.raises(FileExistsError):
        runner._consume_request_once(
            marker_path=marker_path,
            payload=marker_payload,
        )
