from __future__ import annotations

from typing import Sequence

import pytest

from evaluation.contracts import NumericalEvaluationError
from evaluation.evaluator import RepairFailed, SharedEvaluator
from evaluation.firewall import (
    InformationBoundaryError,
    InformationField,
    freeze_information,
)
from evaluation.ledger import BudgetExceeded, EvaluationLedger, LedgerIntegrityError


def _evaluator(call_log: list[tuple[float, ...]]) -> SharedEvaluator:
    def joint_evaluate(
        vector: Sequence[float], _snapshot: object
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = tuple(float(value) for value in vector)
        call_log.append(values)
        if values == (99.0,):
            raise FloatingPointError("synthetic numerical failure")
        return (values[0] ** 2,), (values[0] - 1.0,)

    return SharedEvaluator(
        objective_names=("squared",),
        constraint_names=("upper_bound",),
        evaluate_joint=joint_evaluate,
    )


def _snapshot() -> object:
    return freeze_information(
        decision_time=2,
        fields={
            "current_state": InformationField(available_at=2, value={"x": 1.0}),
            "past_observation": InformationField(available_at=1, value=0.5),
        },
    )


def test_joint_evaluator_charges_cached_failed_and_warm_start_calls() -> None:
    calls: list[tuple[float, ...]] = []
    evaluator = _evaluator(calls)
    ledger = EvaluationLedger(max_cfe=4)
    snapshot = _snapshot()

    first = evaluator.evaluate(
        vector=(0.5,),
        event_id=0,
        candidate_id="random-1",
        information=snapshot,
        ledger=ledger,
        atomic_steps=6,
        cache_key="same-point",
        origin="random_initialization",
    )
    cached = evaluator.evaluate(
        vector=(0.5,),
        event_id=0,
        candidate_id="warm-1",
        information=snapshot,
        ledger=ledger,
        atomic_steps=6,
        cache_key="same-point",
        origin="warm_start",
    )
    with pytest.raises(NumericalEvaluationError, match="nonrecoverable"):
        evaluator.evaluate(
            vector=(99.0,),
            event_id=0,
            candidate_id="failed-1",
            information=snapshot,
            ledger=ledger,
            atomic_steps=6,
            origin="trial",
        )

    assert first.objectives == cached.objectives
    assert calls == [(0.5,), (99.0,)]
    assert ledger.snapshot() == {
        "cfe": 3,
        "objective_calls": 3,
        "constraint_calls": 3,
        "scenario_evaluations": 3,
        "atomic_model_steps": 18,
        "execution_transition_count": 0,
        "repair_failed": 0,
        "evaluation_failures": 1,
    }
    assert [record.origin for record in ledger.evaluations] == [
        "random_initialization",
        "warm_start",
        "trial",
    ]
    assert ledger.evaluations[1].cached is True


def test_budget_is_checked_before_a_new_joint_evaluation() -> None:
    calls: list[tuple[float, ...]] = []
    evaluator = _evaluator(calls)
    ledger = EvaluationLedger(max_cfe=1)
    snapshot = _snapshot()

    evaluator.evaluate(
        vector=(0.0,),
        event_id=0,
        candidate_id="c-1",
        information=snapshot,
        ledger=ledger,
        atomic_steps=2,
    )
    with pytest.raises(BudgetExceeded, match="CFE"):
        evaluator.evaluate(
            vector=(0.2,),
            event_id=0,
            candidate_id="c-2",
            information=snapshot,
            ledger=ledger,
            atomic_steps=2,
        )

    assert calls == [(0.0,)]
    assert ledger.snapshot()["cfe"] == 1


def test_ledger_records_are_immutable_and_candidate_ids_are_single_charge() -> None:
    ledger = EvaluationLedger(max_cfe=2)
    ledger.charge_candidate(
        candidate_id="c-1",
        event_id=0,
        atomic_steps=1,
        metadata={"information_hash": "abc"},
    )
    with pytest.raises(TypeError):
        ledger.evaluations[0].metadata["information_hash"] = "mutated"  # type: ignore[index]
    with pytest.raises(LedgerIntegrityError, match="more than once"):
        ledger.charge_candidate(
            candidate_id="c-1",
            event_id=0,
            atomic_steps=1,
        )
    assert ledger.snapshot()["cfe"] == 1
    assert ledger.cfe == 1
    assert ledger.atomic_steps == 1
    assert ledger.repair_failure_count == 0
    assert ledger.evaluation_failure_count == 0
    assert ledger.execution_transition_count == 0


def test_invalid_joint_result_is_not_admitted_to_cache() -> None:
    calls = 0

    def invalid(
        _vector: Sequence[float], _snapshot: object
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        nonlocal calls
        calls += 1
        return (float("nan"),), (-1.0,)

    evaluator = SharedEvaluator(
        objective_names=("quality",),
        constraint_names=("capacity",),
        evaluate_joint=invalid,
    )
    ledger = EvaluationLedger(max_cfe=2)
    for candidate_id in ("bad-1", "bad-2"):
        with pytest.raises(NumericalEvaluationError, match="non-finite"):
            evaluator.evaluate(
                vector=(0.0,),
                event_id=0,
                candidate_id=candidate_id,
                information=_snapshot(),
                ledger=ledger,
                atomic_steps=1,
                cache_key="invalid",
            )
    assert calls == 2
    assert ledger.snapshot()["cfe"] == 2
    assert ledger.snapshot()["evaluation_failures"] == 2


def test_repair_failure_is_a_typed_zero_cfe_validation_failure() -> None:
    calls: list[tuple[float, ...]] = []
    evaluator = _evaluator(calls)
    ledger = EvaluationLedger(max_cfe=1)

    with pytest.raises(RepairFailed, match="repair"):
        evaluator.evaluate_after_repair(
            raw_vector=(2.0,),
            target_vector=(0.5,),
            repair=lambda _raw, _target: None,
            event_id=0,
            candidate_id="repair-failed",
            information=_snapshot(),
            ledger=ledger,
            atomic_steps=6,
        )

    assert calls == []
    assert ledger.snapshot()["cfe"] == 0
    assert ledger.snapshot()["repair_failed"] == 1
    assert ledger.validation_failures[0].candidate_id == "repair-failed"


def test_repair_exception_is_also_a_typed_zero_cfe_validation_failure() -> None:
    evaluator = _evaluator([])
    ledger = EvaluationLedger(max_cfe=1)

    def broken_repair(
        _raw: Sequence[float], _target: Sequence[float]
    ) -> Sequence[float] | None:
        raise ArithmeticError("synthetic repair error")

    with pytest.raises(RepairFailed, match="synthetic repair error"):
        evaluator.evaluate_after_repair(
            raw_vector=(2.0,),
            target_vector=(0.5,),
            repair=broken_repair,
            event_id=0,
            candidate_id="repair-exception",
            information=_snapshot(),
            ledger=ledger,
            atomic_steps=6,
        )

    assert ledger.snapshot()["cfe"] == 0
    assert ledger.snapshot()["repair_failed"] == 1


def test_information_firewall_is_deterministic_immutable_and_time_bounded() -> None:
    fields = {
        "b": InformationField(available_at=2, value={"z": 2}),
        "a": InformationField(available_at=1, value=[1, 2]),
    }
    first = freeze_information(decision_time=2, fields=fields)
    second = freeze_information(
        decision_time=2,
        fields={"a": fields["a"], "b": fields["b"]},
    )

    assert first.information_hash == second.information_hash
    assert tuple(first.fields) == ("a", "b")
    with pytest.raises(TypeError):
        first.fields["new"] = InformationField(available_at=2, value=3)  # type: ignore[index]

    with pytest.raises(InformationBoundaryError, match="future"):
        freeze_information(
            decision_time=2,
            fields={"future": InformationField(available_at=3, value=1)},
        )
    with pytest.raises(InformationBoundaryError, match="prohibited"):
        freeze_information(
            decision_time=2,
            fields={
                "other_method_results": InformationField(
                    available_at=1, value={"FULL": 1}
                )
            },
        )
