from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from dt_ramde_v11 import core as core_module
from dt_ramde_v11.contracts import (
    ExecutionScope,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
)
from dt_ramde_v11.core import (
    Candidate,
    assign_rank_and_crowding,
    constrained_sort_key,
    dominates,
)
from evaluation.contracts import EvaluationResult
from formal_execution import runtime as runtime_module
from formal_execution.runtime import (
    CORRECTIVE_R8C_RUNTIME_SETTINGS,
    run_task,
)
from formal_execution.schedule import FormalSequenceSpec


def _candidate(
    candidate_id: str,
    objectives: Sequence[float],
    constraint: float,
) -> Candidate:
    objective_values = tuple(float(value) for value in objectives)
    return Candidate(
        vector=np.zeros(4, dtype=float),
        evaluation=EvaluationResult(
            candidate_id=candidate_id,
            objectives=objective_values,
            objective_names=tuple(
                f"f{index}" for index in range(len(objective_values))
            ),
            constraints=(float(constraint),),
            constraint_names=("c0",),
        ),
        lineage_node_id=candidate_id,
    )


def _clone(candidates: Sequence[Candidate]) -> list[Candidate]:
    return [
        _candidate(
            candidate.candidate_id,
            candidate.objectives,
            candidate.constraints[0],
        )
        for candidate in candidates
    ]


def _reference_fronts(
    feasible: Sequence[Candidate],
) -> list[list[Candidate]]:
    counts = {candidate.candidate_id: 0 for candidate in feasible}
    dominated: dict[str, list[Candidate]] = {
        candidate.candidate_id: [] for candidate in feasible
    }
    fronts: list[list[Candidate]] = [[]]
    for left in feasible:
        for right in feasible:
            if left is right:
                continue
            if dominates(left, right):
                dominated[left.candidate_id].append(right)
            elif dominates(right, left):
                counts[left.candidate_id] += 1
        if counts[left.candidate_id] == 0:
            left.rank = 0
            fronts[0].append(left)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[Candidate] = []
        for left in fronts[front_index]:
            for right in dominated[left.candidate_id]:
                counts[right.candidate_id] -= 1
                if counts[right.candidate_id] == 0:
                    right.rank = front_index + 1
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts


def _reference_crowding(front: Sequence[Candidate]) -> None:
    if not front:
        return
    for candidate in front:
        candidate.crowding = 0.0
    if len(front) <= 2:
        for candidate in front:
            candidate.crowding = math.inf
        return
    for objective_index in range(len(front[0].objectives)):
        ordered = sorted(
            front,
            key=lambda candidate: (
                candidate.objectives[objective_index],
                candidate.candidate_id,
            ),
        )
        ordered[0].crowding = math.inf
        ordered[-1].crowding = math.inf
        span = (
            ordered[-1].objectives[objective_index]
            - ordered[0].objectives[objective_index]
        )
        if span <= 0.0:
            continue
        for position in range(1, len(ordered) - 1):
            candidate = ordered[position]
            if math.isfinite(candidate.crowding):
                candidate.crowding += (
                    ordered[position + 1].objectives[objective_index]
                    - ordered[position - 1].objectives[objective_index]
                ) / span


def _reference_assign(candidates: Sequence[Candidate]) -> None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    for front in _reference_fronts(feasible):
        _reference_crowding(front)
    for candidate in candidates:
        if not candidate.feasible:
            candidate.rank = 10**6
            candidate.crowding = 0.0


def _rank_crowding_rows(
    candidates: Sequence[Candidate],
) -> tuple[tuple[str, int, float], ...]:
    return tuple(
        (candidate.candidate_id, candidate.rank, candidate.crowding)
        for candidate in candidates
    )


@pytest.mark.parametrize("objective_count", [2, 3, 5, 8])
@pytest.mark.parametrize("seed", [3, 11, 29])
def test_rank_crowding_and_order_match_ordered_pair_reference(
    objective_count: int,
    seed: int,
) -> None:
    rng = np.random.Generator(
        np.random.PCG64(2026072600 + 100 * objective_count + seed)
    )
    candidates = [
        _candidate(
            f"candidate-{index:03d}",
            np.round(rng.normal(size=objective_count), decimals=2),
            -1.0 if index % 7 else 0.25,
        )
        for index in range(80)
    ]
    expected = _clone(candidates)
    actual = _clone(candidates)

    _reference_assign(expected)
    assign_rank_and_crowding(actual)

    assert _rank_crowding_rows(actual) == _rank_crowding_rows(expected)
    assert [
        candidate.candidate_id
        for candidate in sorted(
            actual,
            key=lambda candidate: constrained_sort_key(candidate, (1.0,)),
        )
    ] == [
        candidate.candidate_id
        for candidate in sorted(
            expected,
            key=lambda candidate: constrained_sort_key(candidate, (1.0,)),
        )
    ]


def test_duplicate_objectives_and_infeasible_rows_preserve_exact_order() -> None:
    candidates = [
        _candidate("a", (1.0, 1.0, 1.0), -1.0),
        _candidate("b", (1.0, 1.0, 1.0), -1.0),
        _candidate("c", (0.0, 2.0, 2.0), -1.0),
        _candidate("d", (2.0, 0.0, 2.0), -1.0),
        _candidate("e", (2.0, 2.0, 0.0), -1.0),
        _candidate("f", (3.0, 3.0, 3.0), -1.0),
        _candidate("g", (-10.0, -10.0, -10.0), 0.5),
        _candidate("h", (0.5, 0.5, 0.5), 0.25),
    ]
    expected = _clone(candidates)
    actual = _clone(candidates)

    _reference_assign(expected)
    assign_rank_and_crowding(actual)

    assert _rank_crowding_rows(actual) == _rank_crowding_rows(expected)


@pytest.mark.parametrize("objective_count", [3, 5, 8])
def test_quadratic_front_members_keep_reference_insertion_order(
    objective_count: int,
) -> None:
    rng = np.random.Generator(np.random.PCG64(2026072699 + objective_count))
    candidates = [
        _candidate(
            f"candidate-{index:03d}",
            np.round(rng.uniform(-2.0, 2.0, size=objective_count), 1),
            -1.0,
        )
        for index in range(100)
    ]
    expected = _reference_fronts(_clone(candidates))
    actual = core_module._quadratic_nondominated_fronts(_clone(candidates))

    assert [
        [candidate.candidate_id for candidate in front]
        for front in actual
    ] == [
        [candidate.candidate_id for candidate in front]
        for front in expected
    ]


def _request() -> R8CCorrectiveExecutionRequest:
    command = "test-only-quadratic-pairwise-command"
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


def test_lircmop13_pairwise_and_ordered_reference_artifacts_are_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP13",
        method_id="F22_MG_STATIC",
        replicate_index=0,
        master_seed_u64="20260726",
        events=1,
        cfe_per_event=300,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=13,
        problem_id="LIRCMOP13",
        task_namespace="r8c",
    )
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: 123.0)
    monkeypatch.setattr(runtime_module.time, "process_time", lambda: 45.0)
    optimized_dir = tmp_path / "optimized"
    reference_dir = tmp_path / "reference"
    stop_path = tmp_path / "STOP"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=optimized_dir,
        stop_path=stop_path,
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    monkeypatch.setattr(
        core_module,
        "_quadratic_nondominated_fronts",
        _reference_fronts,
    )
    run_task(
        spec=spec,
        request=_request(),
        task_directory=reference_dir,
        stop_path=stop_path,
        settings=CORRECTIVE_R8C_RUNTIME_SETTINGS,
    )

    for name in (
        "raw_evaluations.jsonl.gz",
        "task_summary.json",
        "task_manifest.json",
    ):
        assert (optimized_dir / name).read_bytes() == (
            reference_dir / name
        ).read_bytes()
