from __future__ import annotations

import numpy as np
import pytest

import comparators.common as common_module
from comparators.common import (
    ExactNondominatedAccumulator,
    finalize_nondominated_candidates,
)
from dt_ramde_v11.core import (
    Candidate,
    CoreContractViolation,
    maintain_nondominated_archive,
    order_known_nondominated_archive,
)
from evaluation.contracts import EvaluationResult, TerminalCode


def _candidate(
    candidate_id: str,
    objectives: tuple[float, ...],
    constraints: tuple[float, ...] = (-1.0,),
) -> Candidate:
    return Candidate(
        vector=np.asarray([float(len(candidate_id))]),
        evaluation=EvaluationResult(
            candidate_id=candidate_id,
            objectives=objectives,
            objective_names=tuple(
                f"f{index}" for index in range(len(objectives))
            ),
            constraints=constraints,
            constraint_names=tuple(
                f"c{index}" for index in range(len(constraints))
            ),
        ),
        lineage_node_id=f"lineage-{candidate_id}",
    )


@pytest.mark.parametrize("capacity", [1, 3, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_incremental_accumulator_matches_all_history_finalization(
    capacity: int,
    reverse: bool,
) -> None:
    history = [
        _candidate("c1", (1.0, 4.0)),
        _candidate("c2", (2.0, 2.0)),
        _candidate("c3", (4.0, 1.0)),
        _candidate("c4", (3.0, 3.0)),
        _candidate("c5", (0.0, 0.0), (0.1,)),
        _candidate("c6", (2.0, 2.0)),
        _candidate("c7", (0.5, 5.0)),
    ]
    ordered = list(reversed(history)) if reverse else history
    accumulator = ExactNondominatedAccumulator()
    for candidate in ordered:
        accumulator.add(candidate)

    expected = maintain_nondominated_archive(
        ordered,
        capacity=capacity,
        constraint_scales=(1.0,),
    )
    actual = maintain_nondominated_archive(
        accumulator.snapshot(),
        capacity=capacity,
        constraint_scales=(1.0,),
    )

    assert [item.candidate_id for item in actual] == [
        item.candidate_id for item in expected
    ]


def test_incremental_accumulator_preserves_duplicate_id_contract() -> None:
    accumulator = ExactNondominatedAccumulator()
    accumulator.add(_candidate("same", (1.0,), (0.1,)))

    with pytest.raises(
        CoreContractViolation,
        match="duplicate candidate_id",
    ):
        accumulator.add(_candidate("same", (0.0,)))


def test_incremental_finalizer_preserves_all_infeasible_terminal_code() -> None:
    accumulator = ExactNondominatedAccumulator()
    accumulator.add(_candidate("infeasible", (0.0,), (0.1,)))

    result = finalize_nondominated_candidates(
        type("Problem", (), {"constraint_scales": (1.0,)})(),
        event_id=0,
        candidates=accumulator.snapshot(),
        had_finite_candidates=accumulator.has_candidates,
        archive_capacity=10,
        budget_exhausted=True,
    )

    assert result.terminal.code is TerminalCode.REJECT_BUDGET_NO_FEASIBLE


@pytest.mark.parametrize("capacity", [1, 3, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_known_nondominated_order_matches_generic_archive(
    capacity: int,
    reverse: bool,
) -> None:
    front = [
        _candidate("c1", (1.0, 4.0)),
        _candidate("c2", (2.0, 2.0)),
        _candidate("c3", (4.0, 1.0)),
        _candidate("c6", (2.0, 2.0)),
        _candidate("c7", (0.5, 5.0)),
    ]
    ordered = list(reversed(front)) if reverse else front

    expected = maintain_nondominated_archive(
        ordered,
        capacity=capacity,
        constraint_scales=(1.0,),
    )
    actual = order_known_nondominated_archive(
        ordered,
        capacity=capacity,
        constraint_scales=(1.0,),
    )

    assert [
        (item.candidate_id, item.rank, item.crowding) for item in actual
    ] == [
        (item.candidate_id, item.rank, item.crowding) for item in expected
    ]


def test_two_objective_accumulator_avoids_generic_pairwise_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [
        _candidate("c1", (1.0, 4.0)),
        _candidate("c2", (2.0, 3.0)),
        _candidate("c3", (3.0, 2.0)),
        _candidate("c4", (4.0, 1.0)),
        _candidate("c5", (2.5, 3.5)),
        _candidate("c6", (2.5, 1.5)),
        _candidate("c7", (2.5, 1.5)),
        _candidate("c8", (2.5, 2.5)),
    ]
    accumulator = ExactNondominatedAccumulator()
    accumulator.add(history[0])

    def forbidden_pairwise_scan(
        left: Candidate,
        right: Candidate,
    ) -> bool:
        del left, right
        raise AssertionError("generic pairwise scan used for a 2-D front")

    monkeypatch.setattr(
        common_module,
        "dominates",
        forbidden_pairwise_scan,
    )
    for candidate in history[1:]:
        accumulator.add(candidate)

    expected = maintain_nondominated_archive(
        history,
        capacity=100,
        constraint_scales=(1.0,),
    )
    actual = order_known_nondominated_archive(
        accumulator.snapshot(),
        capacity=100,
        constraint_scales=(1.0,),
    )

    assert [item.candidate_id for item in actual] == [
        item.candidate_id for item in expected
    ]
