from __future__ import annotations

import random

import numpy as np
import pytest

import comparators.common as common_module
from comparators.common import ExactNondominatedAccumulator
from dt_ramde_v11.core import Candidate
from evaluation.contracts import EvaluationResult


def _candidate(
    candidate_id: str,
    objectives: tuple[float, float, float],
    *,
    feasible: bool = True,
) -> Candidate:
    constraints = (-1.0,) if feasible else (1.0,)
    return Candidate(
        vector=np.asarray([float(len(candidate_id))]),
        evaluation=EvaluationResult(
            candidate_id=candidate_id,
            objectives=objectives,
            objective_names=("f0", "f1", "f2"),
            constraints=constraints,
            constraint_names=("c0",),
        ),
        lineage_node_id=f"lineage-{candidate_id}",
    )


def _strictly_dominates(left: Candidate, right: Candidate) -> bool:
    return all(
        left_value <= right_value
        for left_value, right_value in zip(
            left.objectives,
            right.objectives,
            strict=True,
        )
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(
            left.objectives,
            right.objectives,
            strict=True,
        )
    )


def _naive_first_front(candidates: list[Candidate]) -> tuple[Candidate, ...]:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    return tuple(
        candidate
        for candidate in feasible
        if not any(
            other is not candidate and _strictly_dominates(other, candidate)
            for other in feasible
        )
    )


@pytest.mark.parametrize("seed", range(5))
def test_three_objective_random_front_matches_naive_reference(seed: int) -> None:
    rng = random.Random(seed)
    history = [
        _candidate(
            f"candidate-{index}",
            tuple(float(rng.randrange(20)) for _ in range(3)),
            feasible=index % 11 != 0,
        )
        for index in range(180)
    ]
    accumulator = ExactNondominatedAccumulator()
    for candidate in history:
        accumulator.add(candidate)

    assert accumulator.snapshot() == _naive_first_front(history)


def test_three_objective_duplicate_groups_preserve_every_exact_duplicate() -> None:
    history = [
        _candidate("tradeoff-left", (0.0, 3.0, 3.0)),
        _candidate("duplicate-a", (1.0, 1.0, 1.0)),
        _candidate("dominated", (2.0, 2.0, 2.0)),
        _candidate("duplicate-b", (1.0, 1.0, 1.0)),
        _candidate("tradeoff-right", (3.0, 0.0, 0.0)),
        _candidate("duplicate-c", (1.0, 1.0, 1.0)),
    ]
    accumulator = ExactNondominatedAccumulator()
    for candidate in history:
        accumulator.add(candidate)

    assert accumulator.snapshot() == _naive_first_front(history)
    assert [
        candidate.candidate_id for candidate in accumulator.snapshot()
    ] == [
        "tradeoff-left",
        "duplicate-a",
        "duplicate-b",
        "tradeoff-right",
        "duplicate-c",
    ]


def test_three_objective_dominated_index_entries_remain_exact() -> None:
    history = [
        _candidate("first-dominator", (0.0, 0.0, 0.0)),
        _candidate("retained-dominated-index-entry", (2.0, 2.0, 2.0)),
        _candidate("later-tradeoff", (3.0, -1.0, 1.0)),
        _candidate("later-dominated", (4.0, 2.0, 2.0)),
        _candidate("later-other-tradeoff", (-1.0, 4.0, 4.0)),
    ]
    accumulator = ExactNondominatedAccumulator()
    for index, candidate in enumerate(history, start=1):
        accumulator.add(candidate)
        assert accumulator.snapshot() == _naive_first_front(history[:index])


def test_three_objective_snapshot_preserves_insertion_order() -> None:
    history = [
        _candidate("middle", (2.0, 2.0, 2.0)),
        _candidate("high-first", (4.0, 0.0, 4.0)),
        _candidate("low-first", (0.0, 4.0, 4.0)),
        _candidate("high-third", (4.0, 4.0, 0.0)),
    ]
    accumulator = ExactNondominatedAccumulator()
    for candidate in history:
        accumulator.add(candidate)

    assert accumulator.snapshot() == tuple(history)


def test_three_objective_50000_point_front_completes_without_pairwise_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = ExactNondominatedAccumulator()
    size = 50_000
    for index in range(size):
        accumulator.add(
            _candidate(
                f"candidate-{index:05d}",
                (float(index), float(size - index), 0.0),
            )
        )

    def forbidden_pairwise_scan(
        left: Candidate,
        right: Candidate,
    ) -> bool:
        del left, right
        raise AssertionError("generic pairwise scan used for a 3-D front")

    monkeypatch.setattr(common_module, "dominates", forbidden_pairwise_scan)
    snapshot = accumulator.snapshot()

    assert len(snapshot) == size
    assert snapshot[0].candidate_id == "candidate-00000"
    assert snapshot[-1].candidate_id == "candidate-49999"
