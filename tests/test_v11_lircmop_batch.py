from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark_adapters.r4_evaluators import (
    LIRCMOPEvaluator,
    R4EvaluatorBindingError,
)
from dt_ramde_v11.contracts import (
    ExecutionScope,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
)
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry
from formal_execution import runtime as runtime_module
from formal_execution.adapters import FormalR8CStaticAdapter
from formal_execution.runtime import (
    CORRECTIVE_R8C_RUNTIME_SETTINGS,
    run_task,
)
from formal_execution.schedule import FormalSequenceSpec


def _request() -> R8CCorrectiveExecutionRequest:
    command = "test-only-lircmop-batch-command"
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
            formal_schedule_sha256="b" * 64,
            source_git_commit="c" * 40,
            source_git_tree="d" * 40,
        ),
        request_id="WGT-V11-R8C-EXECUTION-REQUEST-20260726-01",
        frozen_exact_command=command,
        author_confirmation_text=command,
        author_exact_command_confirmed=True,
    )


@pytest.mark.parametrize("problem_index", range(1, 15))
def test_lircmop_ordered_batch_is_elementwise_bit_exact(
    problem_index: int,
) -> None:
    evaluator = LIRCMOPEvaluator(problem_index)
    lower = np.asarray(evaluator.lower_bounds, dtype=float)
    upper = np.asarray(evaluator.upper_bounds, dtype=float)
    rng = np.random.Generator(np.random.PCG64(2026072600 + problem_index))
    matrix = rng.uniform(lower, upper, size=(100, lower.size))

    scalar = tuple(evaluator(row, 0) for row in matrix)
    batched = evaluator.evaluate_batch(matrix, 0)
    reversed_batch = evaluator.evaluate_batch(matrix[::-1], 0)

    assert batched == scalar
    assert reversed_batch == tuple(reversed(scalar))


def test_lircmop_batch_preserves_empty_and_exception_contracts() -> None:
    evaluator = LIRCMOPEvaluator(1)

    assert evaluator.evaluate_batch((), 0) == ()
    with pytest.raises(R4EvaluatorBindingError, match="TS1"):
        evaluator.evaluate_batch((), 1)
    with pytest.raises(R4EvaluatorBindingError, match="wrong dimension"):
        evaluator.evaluate_batch(((0.5,) * 29,), 0)


@pytest.mark.parametrize(
    "method_id",
    [
        "F22_MG_STATIC",
        "MATCHED_FIXED_DE_PARETO",
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
        "JMETALPY_1_7_NSGAII_STATIC_CMOEA",
    ],
)
def test_lircmop_100_by_100_batch_matches_forced_scalar_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_id: str,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id=method_id,
        replicate_index=0,
        master_seed_u64="20260726",
        events=1,
        cfe_per_event=200,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=1,
        problem_id="LIRCMOP1",
        task_namespace="r8c",
    )
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: 123.0)
    monkeypatch.setattr(runtime_module.time, "process_time", lambda: 45.0)
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
        FormalR8CStaticAdapter,
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
