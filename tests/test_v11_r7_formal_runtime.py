from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from dt_ramde_v11.contracts import (
    ConfigurationError,
    ExecutionScope,
    R8ContractBindings,
    R8ExecutionRequest,
)
from evaluation.contracts import EvaluationResult
from formal_execution.adapters import (
    make_formal_cdf_adapter,
    make_formal_lircmop_adapter,
    make_formal_wgt_rr_adapter,
)
from formal_execution.runtime import RawEvaluationWriter, run_task
from formal_execution.schedule import FormalSequenceSpec
from formal_execution.schedule import (
    build_e2_full_reuse_map,
    build_formal_schedule,
    e2_full_reuse_commitment,
)


CONTRACT_SHA = (
    "43bc9d137b7ffa5eb1a7a9649a29f7873eb115d0f2607e37c42e4f610b931201"
)
COMMAND = (
    "python tools/run_v11_r8_formal.py --contract "
    "config/r7/r7_formal_execution_contract.json --request "
    "config/r7/r7_execution_request.json "
    "--output-root runs/r8-formal"
)


@pytest.fixture
def r8_request() -> R8ExecutionRequest:
    return R8ExecutionRequest(
        scope=ExecutionScope.BENCHMARK_EFFECT,
        companion_scope=ExecutionScope.WEIGHT_EFFECT,
        contracts=R8ContractBindings(
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
            r7_contract_id="WGT-V11-R7-FORMAL-EXECUTION-CONTRACT-01",
            r7_contract_sha256=CONTRACT_SHA,
            formal_schedule_id="WGT-V11-R7-FORMAL-SCHEDULE-01",
            formal_schedule_sha256=(
                "40ea633532a3ba2c461ae47925a91ccae305bafac397e6246ced1951fa6e8969"
            ),
            source_git_commit="0" * 40,
            source_git_tree="1" * 40,
        ),
        request_id="WGT-V11-R8-EXECUTION-REQUEST-20260725-01",
        frozen_exact_command=COMMAND,
        author_confirmation_text=COMMAND,
        author_exact_command_confirmed=True,
    )


def _raw_line_count(task_directory: Path) -> int:
    with gzip.open(
        task_directory / "raw_evaluations.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        return sum(1 for _ in stream)


def test_raw_evaluation_writer_buffers_without_reordering(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl.gz"
    writer = RawEvaluationWriter(
        raw_path,
        "buffer-test",
        buffer_size=256,
    )
    with writer:
        for index in range(5):
            candidate_id = f"candidate-{index}"
            writer.write(
                event_id=0,
                vector=(float(index), float(index + 1)),
                result=EvaluationResult(
                    candidate_id=candidate_id,
                    objectives=(float(index),),
                    objective_names=("objective",),
                    constraints=(-1.0,),
                    constraint_names=("constraint",),
                ),
            )

    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert writer.count == 5
    assert [record["candidate_id"] for record in records] == [
        f"candidate-{index}" for index in range(5)
    ]


def test_formal_public_adapter_wrappers_are_narrowly_effect_enabled() -> None:
    adapters = (
        make_formal_lircmop_adapter(1),
        make_formal_cdf_adapter(
            1,
            profile="CDF-MILD",
            environment_seed=123,
        ),
        make_formal_wgt_rr_adapter("RR-SMOOTH", 1),
    )
    for adapter in adapters:
        identity = adapter.identity()
        assert identity["registered_effect_instance"] is True
        assert identity["formal_effect_execution_allowed"] is True
        assert "R7_CONTRACT" in identity["execution_authority"]


def test_e2_full_reuse_is_exactly_bound_without_redispatch() -> None:
    root = Path(__file__).resolve().parents[1]
    r5 = json.loads(
        (root / "config" / "r5" / "r5_freeze_contract.json").read_text(
            encoding="utf-8"
        )
    )
    schedule = build_formal_schedule(r5)
    reuse = build_e2_full_reuse_map(schedule)
    assert len(reuse) == 310
    assert len({row["reused_task_id"] for row in reuse}) == 310
    assert e2_full_reuse_commitment(schedule) == (
        "bd6e92a96c0b7bfa899b319db1e9d8fef50c9a4cbaab08b14187ff739eb90fe8"
    )


def test_r8_request_fails_closed_without_verbatim_confirmation(
    r8_request: R8ExecutionRequest,
) -> None:
    pending = R8ExecutionRequest(
        **{
            **r8_request.__dict__,
            "author_confirmation_text": "",
            "author_exact_command_confirmed": False,
        }
    )
    with pytest.raises(ConfigurationError, match="verbatim author confirmation"):
        pending.validate()


def test_r8_request_requires_both_public_effect_scopes(
    r8_request: R8ExecutionRequest,
) -> None:
    invalid = R8ExecutionRequest(
        **{
            **r8_request.__dict__,
            "companion_scope": ExecutionScope.BENCHMARK_EFFECT,
        }
    )
    with pytest.raises(ConfigurationError, match="exactly benchmark_effect"):
        invalid.validate()


def test_small_dt_ramde_task_persists_exact_raw_cfe(
    tmp_path: Path,
    r8_request: R8ExecutionRequest,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="F22_MG_STATIC",
        replicate_index=0,
        master_seed_u64="123",
        events=1,
        cfe_per_event=20,
        atomic_steps_per_cfe=1,
        timeout_seconds=60,
        problem_index=1,
        problem_id="LIRCMOP1",
    )
    task_directory = tmp_path / "dt"
    result = run_task(
        spec=spec,
        request=r8_request,
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
    )
    assert result["status"] == "COMPLETE"
    assert result["total_cfe"] == 20
    assert _raw_line_count(task_directory) == 20


def test_small_matched_comparator_uses_same_task_contract(
    tmp_path: Path,
    r8_request: R8ExecutionRequest,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=1,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="MATCHED_FIXED_DE_PARETO",
        replicate_index=0,
        master_seed_u64="123",
        events=1,
        cfe_per_event=20,
        atomic_steps_per_cfe=1,
        timeout_seconds=60,
        problem_index=1,
        problem_id="LIRCMOP1",
    )
    task_directory = tmp_path / "matched"
    result = run_task(
        spec=spec,
        request=r8_request,
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
    )
    assert result["status"] == "COMPLETE"
    assert result["total_cfe"] == 20
    assert _raw_line_count(task_directory) == 20


def test_static_nsgaii_stably_deduplicates_repeated_final_solutions(
    tmp_path: Path,
    r8_request: R8ExecutionRequest,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=3,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="JMETALPY_1_7_NSGAII_STATIC_CMOEA",
        replicate_index=2,
        master_seed_u64="2013063862857590834",
        events=1,
        cfe_per_event=2000,
        atomic_steps_per_cfe=1,
        timeout_seconds=60,
        problem_index=1,
        problem_id="LIRCMOP1",
    )
    task_directory = tmp_path / "nsgaii"
    result = run_task(
        spec=spec,
        request=r8_request,
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
    )

    assert result["status"] == "COMPLETE"
    assert result["total_cfe"] == 2000
    assert _raw_line_count(task_directory) == 2000


def test_fixed_e3_policy_is_unreplicated_one_cfe_per_event(
    tmp_path: Path,
    r8_request: R8ExecutionRequest,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=2,
        workload_id="E3",
        unit_id="VS-000/NOMINAL",
        method_id="FIXED_ENERGY_DEFICIT_POLICY",
        replicate_index=0,
        master_seed_u64="123",
        events=2,
        cfe_per_event=1,
        atomic_steps_per_cfe=6,
        timeout_seconds=60,
        subject_id="VS-000",
        subject_seed_u64="2040978301928374650",
        scenario_id="NOMINAL",
    )
    task_directory = tmp_path / "fixed"
    result = run_task(
        spec=spec,
        request=r8_request,
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
    )
    assert result["status"] == "COMPLETE"
    assert result["total_cfe"] == 2
    assert result["total_atomic_model_steps"] == 12
    assert _raw_line_count(task_directory) == 2
