from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dt_ramde_v11.contracts import (
    ExecutionScope,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
)
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry
from formal_execution import runtime as runtime_module
from formal_execution.adapters import (
    FormalR8CCDFAdapter,
    FormalR8CWGTRRAdapter,
)
from formal_execution.public_rolling import generate_public_instance
from formal_execution.runtime import (
    CORRECTIVE_R8C_RUNTIME_SETTINGS,
    run_task,
)
from formal_execution.schedule import (
    CORRECTIVE_E3_SCENARIOS,
    FormalSequenceSpec,
    build_corrective_formal_schedule,
    build_formal_schedule,
    e2_full_reuse_commitment,
    schedule_commitment,
)


OLD_SCHEDULE_SHA256 = (
    "40ea633532a3ba2c461ae47925a91ccae305bafac397e6246ced1951fa6e8969"
)
CORRECTIVE_SCHEDULE_SHA256 = (
    "734ee0b20daf7855e566e6747ab61b74373db386ed9d5dec117f0612845ef8ba"
)
CORRECTIVE_REUSE_SHA256 = (
    "d235c1c53d7e504400ad37674bebba4a01145a934964039454776c9f09ba0c9e"
)


def _r5() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config"
            / "r5"
            / "r5_freeze_contract.json"
        ).read_text(encoding="utf-8")
    )


def _request() -> R8CCorrectiveExecutionRequest:
    command = "test-only-corrective-command"
    return R8CCorrectiveExecutionRequest(
        scope=ExecutionScope.BENCHMARK_EFFECT,
        companion_scope=ExecutionScope.WEIGHT_EFFECT,
        contracts=R8CCorrectiveContractBindings(
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
                "WGT-V11-R8C-FORMAL-EXECUTION-CONTRACT-01"
            ),
            r8c_formal_contract_sha256="a" * 64,
            formal_schedule_id="WGT-V11-R8C-FORMAL-SCHEDULE-01",
            formal_schedule_sha256=CORRECTIVE_SCHEDULE_SHA256,
            source_git_commit="b" * 40,
            source_git_tree="c" * 40,
        ),
        request_id="WGT-V11-R8C-EXECUTION-REQUEST-20260726-01",
        frozen_exact_command=command,
        author_confirmation_text=command,
        author_exact_command_confirmed=True,
    )


def test_corrective_schedule_preserves_old_identity_and_freezes_scope() -> None:
    old = build_formal_schedule(_r5())
    corrective = build_corrective_formal_schedule(_r5())

    assert schedule_commitment(old) == OLD_SCHEDULE_SHA256
    assert len(corrective) == 5702
    assert sum(row.total_cfe for row in corrective) == 925_882_496
    assert (
        sum(row.total_atomic_steps for row in corrective)
        == 2_420_294_976
    )
    assert schedule_commitment(corrective) == CORRECTIVE_SCHEDULE_SHA256
    assert (
        e2_full_reuse_commitment(corrective)
        == CORRECTIVE_REUSE_SHA256
    )
    assert all(row.task_id.startswith("r8c-") for row in corrective)
    assert {
        row.scenario_id
        for row in corrective
        if row.workload_id == "E3"
    } == set(CORRECTIVE_E3_SCENARIOS)


def test_corrective_runtime_is_explicitly_100_by_100() -> None:
    assert CORRECTIVE_R8C_RUNTIME_SETTINGS.population_size == 100
    assert CORRECTIVE_R8C_RUNTIME_SETTINGS.archive_capacity == 100
    _request().validate()


def test_unmeasured_target_contract_cannot_enter_formal_runner(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "r8c-formal-must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "run_v11_r8c_formal.py"),
            "--contract",
            str(
                root
                / "config"
                / "r8c"
                / "r8c_formal_execution_contract.json"
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
    assert "contract is fail-closed" in result.stderr
    assert not output_root.exists()


def test_corrective_batch_and_scalar_raw_artifacts_are_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_DYNAMIC",
        unit_id="CDF1/CDF-HARSH",
        method_id="MATCHED_FIXED_DE_PARETO",
        replicate_index=0,
        master_seed_u64="20260726",
        events=1,
        cfe_per_event=200,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=1,
        problem_id="CDF1",
        profile="CDF-HARSH",
        task_namespace="r8c",
    )
    monkeypatch.setattr(
        runtime_module.time, "perf_counter", lambda: 123.0
    )
    monkeypatch.setattr(
        runtime_module.time, "process_time", lambda: 45.0
    )
    batch_dir = tmp_path / "batch"
    scalar_dir = tmp_path / "scalar"
    stop_path = tmp_path / "STOP"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=batch_dir,
        stop_path=stop_path,
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    def unavailable(self, vectors, event_id, ledger, candidate_ids):
        del self, vectors, event_id, ledger, candidate_ids
        raise BatchEvaluationUnavailableBeforeEntry("forced scalar reference")

    monkeypatch.setattr(
        FormalR8CCDFAdapter,
        "evaluate_batch",
        unavailable,
    )
    run_task(
        spec=spec,
        request=_request(),
        task_directory=scalar_dir,
        stop_path=stop_path,
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    for name in (
        "raw_evaluations.jsonl.gz",
        "task_summary.json",
        "task_manifest.json",
    ):
        assert (batch_dir / name).read_bytes() == (
            scalar_dir / name
        ).read_bytes()
    with gzip.open(
        batch_dir / "raw_evaluations.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        rows = [json.loads(line) for line in stream]
    candidate_ids = [row["candidate_id"] for row in rows]
    assert len(candidate_ids) == 200
    assert len(set(candidate_ids)) == 200
    with gzip.open(
        batch_dir / "raw_evaluations.jsonl.gz", "rb"
    ) as stream:
        assert len(stream.readlines()) == 200


def test_jmetal_generation_batches_are_byte_exact_to_scalar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=1,
        workload_id="E1_DYNAMIC",
        unit_id="CDF2/CDF-MILD",
        method_id="JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
        replicate_index=0,
        master_seed_u64="20260727",
        events=1,
        cfe_per_event=300,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=2,
        problem_id="CDF2",
        profile="CDF-MILD",
        task_namespace="r8c",
    )
    monkeypatch.setattr(
        runtime_module.time, "perf_counter", lambda: 123.0
    )
    monkeypatch.setattr(
        runtime_module.time, "process_time", lambda: 45.0
    )
    batch_dir = tmp_path / "batch"
    scalar_dir = tmp_path / "scalar"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=batch_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    def unavailable(self, vectors, event_id, ledger, candidate_ids):
        del self, vectors, event_id, ledger, candidate_ids
        raise BatchEvaluationUnavailableBeforeEntry("forced scalar reference")

    monkeypatch.setattr(
        FormalR8CCDFAdapter,
        "evaluate_batch",
        unavailable,
    )
    run_task(
        spec=spec,
        request=_request(),
        task_directory=scalar_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )
    for name in (
        "raw_evaluations.jsonl.gz",
        "task_summary.json",
        "task_manifest.json",
    ):
        assert (batch_dir / name).read_bytes() == (
            scalar_dir / name
        ).read_bytes()
    with gzip.open(
        batch_dir / "raw_evaluations.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        rows = [json.loads(line) for line in stream]
    candidate_ids = [row["candidate_id"] for row in rows]
    assert len(candidate_ids) == 300
    assert len(set(candidate_ids)) == 300


def test_dt_ramde_rolling_batches_are_byte_exact_to_scalar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = generate_public_instance("RR-SMOOTH", 0)
    spec = FormalSequenceSpec(
        schedule_index=2,
        workload_id="E1_ROLLING",
        unit_id="RR-SMOOTH/0",
        method_id="DT-RAMDE_TS2_FULL",
        replicate_index=0,
        master_seed_u64="20260728",
        events=2,
        cfe_per_event=200,
        atomic_steps_per_cfe=6,
        timeout_seconds=3600,
        rolling_template="RR-SMOOTH",
        rolling_index=0,
        rolling_seed_u64=str(instance["derived_seed_u64"]),
        task_namespace="r8c",
    )
    monkeypatch.setattr(
        runtime_module.time, "perf_counter", lambda: 123.0
    )
    monkeypatch.setattr(
        runtime_module.time, "process_time", lambda: 45.0
    )
    batch_dir = tmp_path / "batch"
    scalar_dir = tmp_path / "scalar"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=batch_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    def unavailable(self, vectors, event_id, ledger, candidate_ids):
        del self, vectors, event_id, ledger, candidate_ids
        raise BatchEvaluationUnavailableBeforeEntry("forced scalar reference")

    monkeypatch.setattr(
        FormalR8CWGTRRAdapter,
        "evaluate_batch",
        unavailable,
    )
    run_task(
        spec=spec,
        request=_request(),
        task_directory=scalar_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )
    for name in (
        "raw_evaluations.jsonl.gz",
        "task_summary.json",
        "task_manifest.json",
    ):
        assert (batch_dir / name).read_bytes() == (
            scalar_dir / name
        ).read_bytes()
