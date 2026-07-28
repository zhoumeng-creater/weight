from __future__ import annotations

import math

import numpy as np
import pytest

from dt_ramde_v11.core import (
    Candidate,
    _assign_crowding,
    _quadratic_nondominated_fronts,
    assign_rank_and_crowding,
    constrained_sort_key,
    environmental_select,
)
from evaluation.contracts import EvaluationResult


def _candidate(
    candidate_id: str,
    objectives: tuple[float, ...],
    *,
    feasible: bool = True,
) -> Candidate:
    return Candidate(
        vector=np.asarray([float(len(candidate_id))]),
        evaluation=EvaluationResult(
            candidate_id=candidate_id,
            objectives=objectives,
            objective_names=tuple(f"f{index}" for index in range(len(objectives))),
            constraints=(-1.0 if feasible else 1.0,),
            constraint_names=("g0",),
        ),
        lineage_node_id=f"lineage-{candidate_id}",
    )


def _clone(candidates: list[Candidate]) -> list[Candidate]:
    return [
        _candidate(
            candidate.candidate_id,
            candidate.objectives,
            feasible=candidate.feasible,
        )
        for candidate in candidates
    ]


def _assign_quadratic_reference(candidates: list[Candidate]) -> None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    fronts = _quadratic_nondominated_fronts(feasible)
    for front in fronts:
        _assign_crowding(front)
    for candidate in candidates:
        if not candidate.feasible:
            candidate.rank = 10**6
            candidate.crowding = 0.0


def _assert_assignments_equal(
    actual: list[Candidate],
    expected: list[Candidate],
) -> None:
    actual_by_id = {candidate.candidate_id: candidate for candidate in actual}
    expected_by_id = {candidate.candidate_id: candidate for candidate in expected}
    assert actual_by_id.keys() == expected_by_id.keys()
    for candidate_id, actual_candidate in actual_by_id.items():
        expected_candidate = expected_by_id[candidate_id]
        assert actual_candidate.rank == expected_candidate.rank
        if math.isinf(expected_candidate.crowding):
            assert math.isinf(actual_candidate.crowding)
        else:
            assert actual_candidate.crowding == expected_candidate.crowding


def _ordered_ids(candidates: list[Candidate]) -> list[str]:
    return [
        candidate.candidate_id
        for candidate in sorted(
            candidates,
            key=lambda candidate: constrained_sort_key(candidate, (1.0,)),
        )
    ]


def test_fast_two_objective_sort_matches_quadratic_adversarial_cases() -> None:
    source = [
        _candidate("a", (0.0, 5.0)),
        _candidate("b", (1.0, 4.0)),
        _candidate("c", (2.0, 3.0)),
        _candidate("d", (3.0, 2.0)),
        _candidate("e", (4.0, 1.0)),
        _candidate("f", (5.0, 0.0)),
        _candidate("g", (1.0, 4.0)),
        _candidate("h", (1.0, 5.0)),
        _candidate("i", (2.0, 4.0)),
        _candidate("j", (2.0, 5.0)),
        _candidate("k", (3.0, 3.0)),
        _candidate("l", (4.0, 4.0)),
        _candidate("m", (-0.0, 5.0)),
        _candidate("n", (0.0, 0.0), feasible=False),
    ]
    actual = _clone(source)
    expected = _clone(source)

    assign_rank_and_crowding(actual)
    _assign_quadratic_reference(expected)

    _assert_assignments_equal(actual, expected)
    assert _ordered_ids(actual) == _ordered_ids(expected)


@pytest.mark.parametrize("seed", range(40))
def test_fast_two_objective_sort_matches_quadratic_randomized(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 161))
    objective_rows = rng.integers(-8, 9, size=(size, 2))
    source = [
        _candidate(
            f"c{index:04d}",
            (float(row[0]), float(row[1])),
            feasible=bool(rng.integers(0, 5)),
        )
        for index, row in enumerate(objective_rows)
    ]
    rng.shuffle(source)
    actual = _clone(source)
    expected = _clone(source)

    assign_rank_and_crowding(actual)
    _assign_quadratic_reference(expected)

    _assert_assignments_equal(actual, expected)
    assert _ordered_ids(actual) == _ordered_ids(expected)

    population_size = int(rng.integers(1, size + 1))
    selected = environmental_select(
        _clone(source),
        population_size=population_size,
        constraint_scales=(1.0,),
    )
    reference_pool = _clone(source)
    _assign_quadratic_reference(reference_pool)
    expected_selected = sorted(
        reference_pool,
        key=lambda candidate: constrained_sort_key(candidate, (1.0,)),
    )[:population_size]
    assert [candidate.candidate_id for candidate in selected] == [
        candidate.candidate_id for candidate in expected_selected
    ]


def test_fast_two_objective_sort_is_input_order_invariant() -> None:
    source = [
        _candidate(f"c{index:03d}", (float(index % 7), float(index % 11)))
        for index in range(100)
    ]
    forward = _clone(source)
    reverse = list(reversed(_clone(source)))

    assign_rank_and_crowding(forward)
    assign_rank_and_crowding(reverse)

    _assert_assignments_equal(forward, reverse)
    assert _ordered_ids(forward) == _ordered_ids(reverse)


def test_non_two_objective_sort_keeps_quadratic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [
        _candidate("a", (0.0, 1.0, 2.0)),
        _candidate("b", (1.0, 0.0, 2.0)),
        _candidate("c", (1.0, 1.0, 1.0)),
        _candidate("d", (2.0, 2.0, 2.0)),
    ]

    def fail_if_called(_candidates: object) -> object:
        raise AssertionError("two-objective path must not handle 3-D input")

    monkeypatch.setattr(
        "dt_ramde_v11.core._two_objective_nondominated_fronts",
        fail_if_called,
    )
    assign_rank_and_crowding(source)

    assert {candidate.candidate_id: candidate.rank for candidate in source} == {
        "a": 0,
        "b": 0,
        "c": 0,
        "d": 1,
    }
