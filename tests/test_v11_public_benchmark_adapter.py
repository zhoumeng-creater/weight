from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from benchmark_adapters.public_cmop import (
    CDFPublicAdapter,
    PublicAdapterContractError,
    StaticCMOPPublicAdapter,
)
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ExecutionScope,
    R2ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from dt_ramde_v11.interfaces import EventProblemAdapter
from evaluation.contracts import (
    EvaluationResult,
    NumericalEvaluationError,
)
from evaluation.firewall import InformationBoundaryError
from evaluation.ledger import EvaluationLedger


EVALUATOR_HASH = "a" * 64


def _public_evaluator(
    vector: Sequence[float], event_id: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.asarray(vector, dtype=float)
    return (
        (
            float(np.sum(values**2)),
            float(np.sum((values - event_id / 10.0) ** 2)),
        ),
        (float(np.sum(values) - 1.0),),
    )


def _static_adapter(
    *,
    evaluator: Any = _public_evaluator,
) -> StaticCMOPPublicAdapter:
    return StaticCMOPPublicAdapter(
        suite_id="DAS-CMOP-PLATEMO-4.15",
        problem_id="DASCMOP1",
        evaluator_version="STATIC-CMOP-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 30,
        upper=(1.0,) * 30,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=evaluator,
    )


class _Selector:
    selector_id = "fixture.first-feasible"
    selector_version = "1"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
        }

    def select(self, archive: Sequence[EvaluationResult]) -> str | None:
        return min((candidate.candidate_id for candidate in archive), default=None)


def test_static_public_bridge_uses_shared_interface_ledger_and_hash_identity() -> None:
    adapter = _static_adapter()
    assert isinstance(adapter, EventProblemAdapter)

    information = adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    result = adapter.evaluate((0.0,) * 30, 0, ledger, "candidate-1")

    assert result.objectives == pytest.approx((0.0, 0.0))
    assert result.constraints == pytest.approx((-1.0,))
    assert result.feasible is True
    assert ledger.snapshot()["cfe"] == 1
    assert information.decision_time == 0
    identity = adapter.identity()
    assert identity["fixture_evaluator_sha256"] == EVALUATOR_HASH
    assert identity["split"] == "r2_public_bridge_correctness_fixture"
    assert identity["target_registered_split"] == "public_fixed_confirmatory"
    assert identity["registered_effect_instance"] is False
    assert identity["formal_effect_execution_allowed"] is False
    assert identity["evaluator_interface_version"] == "STATIC-CMOP-EVAL-1.0.0"
    assert adapter.adapter_version == "1.0.0-r2-fixture"
    assert adapter.constraint_scales == (1.0,)


def test_static_public_bridge_rejects_unregistered_identity_and_non_sha_hash() -> None:
    values = {
        "suite_id": "NOT-REGISTERED",
        "problem_id": "DASCMOP1",
        "evaluator_version": "STATIC-CMOP-EVAL-1.0.0",
        "fixture_evaluator_sha256": EVALUATOR_HASH,
        "lower": (0.0,) * 30,
        "upper": (1.0,) * 30,
        "objective_names": ("f1", "f2"),
        "constraint_names": ("g1",),
        "evaluator": _public_evaluator,
    }
    with pytest.raises(PublicAdapterContractError, match="not registered"):
        StaticCMOPPublicAdapter(**values)

    values["suite_id"] = "DAS-CMOP-PLATEMO-4.15"
    values["fixture_evaluator_sha256"] = "not-a-sha"
    with pytest.raises(PublicAdapterContractError, match="SHA-256"):
        StaticCMOPPublicAdapter(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"problem_id": "DASCMOP10"}, "problem"),
        ({"lower": (0.0,) * 2, "upper": (1.0,) * 2}, "dimension"),
        ({"evaluator_version": "fixture-1"}, "evaluator interface"),
    ],
)
def test_static_bridge_binds_f23_problem_dimension_and_interface_version(
    changes: Mapping[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "suite_id": "DAS-CMOP-PLATEMO-4.15",
        "problem_id": "DASCMOP1",
        "evaluator_version": "STATIC-CMOP-EVAL-1.0.0",
        "fixture_evaluator_sha256": EVALUATOR_HASH,
        "lower": (0.0,) * 30,
        "upper": (1.0,) * 30,
        "objective_names": ("f1", "f2"),
        "constraint_names": ("g1",),
        "evaluator": _public_evaluator,
    }
    values.update(changes)
    with pytest.raises(PublicAdapterContractError, match=message):
        StaticCMOPPublicAdapter(**values)


def test_static_public_bridge_is_ts1_and_has_no_execution_feedback() -> None:
    adapter = _static_adapter()
    with pytest.raises(PublicAdapterContractError, match="TS1"):
        adapter.freeze_information(1, None)
    with pytest.raises(PublicAdapterContractError, match="TS1"):
        adapter.freeze_information(0, {"available": False})

    ledger = EvaluationLedger(max_cfe=1)
    feedback = adapter.execute((0.0,) * 30, 0, True, ledger)
    assert feedback == {
        "available": False,
        "ell_exec": None,
        "ell_ref": None,
        "s_exec": None,
        "hard_constraint_violation": None,
        "released_at": 1,
        "reason": "MISSING_BY_DESIGN_PUBLIC_BENCHMARK",
    }
    assert ledger.snapshot()["execution_transition_count"] == 0


def test_external_public_evaluator_failure_is_charged_and_typed() -> None:
    def broken(
        _vector: Sequence[float], _event_id: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        raise FloatingPointError("public evaluator failed")

    adapter = _static_adapter(evaluator=broken)
    adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    with pytest.raises(NumericalEvaluationError):
        adapter.evaluate((0.0,) * 30, 0, ledger, "broken")
    assert ledger.snapshot() == {
        "cfe": 1,
        "objective_calls": 1,
        "constraint_calls": 1,
        "scenario_evaluations": 1,
        "atomic_model_steps": 1,
        "execution_transition_count": 0,
        "repair_failed": 0,
        "evaluation_failures": 1,
    }


def test_cdf_bridge_releases_only_current_information_and_records_transition() -> None:
    released_events: list[int] = []

    def release_metadata(event_id: int) -> Mapping[str, Any]:
        released_events.append(event_id)
        return {"environment_severity": event_id / 59.0}

    adapter = CDFPublicAdapter(
        suite_id="CDF-1-15",
        problem_id="CDF1",
        profile="CDF-MILD",
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 10,
        upper=(1.0,) * 10,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_public_evaluator,
        release_metadata=release_metadata,
    )
    snapshot = adapter.freeze_information(3, None)
    assert snapshot.decision_time == 3
    assert released_events == [3]

    ledger = EvaluationLedger(max_cfe=1)
    feedback = adapter.execute((0.0,) * 10, 3, True, ledger)
    assert feedback["available"] is False
    assert feedback["reason"] == "MISSING_BY_DESIGN_PUBLIC_BENCHMARK"
    assert ledger.snapshot()["execution_transition_count"] == 1


def test_cdf_bridge_rejects_future_or_prohibited_release_fields() -> None:
    adapter = CDFPublicAdapter(
        suite_id="CDF-1-15",
        problem_id="CDF15",
        profile="CDF-HARSH",
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 10,
        upper=(1.0,) * 10,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_public_evaluator,
        release_metadata=lambda _event: {"future_trajectory": [1, 2, 3]},
    )
    with pytest.raises(InformationBoundaryError, match="prohibited"):
        adapter.freeze_information(0, None)
    with pytest.raises(PublicAdapterContractError, match="0..59"):
        adapter.freeze_information(60, None)


def test_cdf_bridge_rejects_prohibited_keys_inside_prior_feedback() -> None:
    adapter = CDFPublicAdapter(
        suite_id="CDF-1-15",
        problem_id="CDF2",
        profile="CDF-MILD",
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 10,
        upper=(1.0,) * 10,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_public_evaluator,
        release_metadata=lambda event: {"environment_severity": event},
    )
    with pytest.raises(InformationBoundaryError, match="prohibited"):
        adapter.freeze_information(
            1,
            {
                "available": False,
                "reason": "MISSING_BY_DESIGN_PUBLIC_BENCHMARK",
                "released_at": 1,
                "future_trajectory": [1, 2, 3],
            },
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"ell_exec": 0.0},
        {"ell_ref": 0.0},
        {"s_exec": 0.0},
        {"hard_constraint_violation": False},
        {"released_at": 1.9},
    ],
)
def test_cdf_missing_feedback_requires_null_payload_and_integer_release(
    changes: Mapping[str, Any],
) -> None:
    adapter = CDFPublicAdapter(
        suite_id="CDF-1-15",
        problem_id="CDF2",
        profile="CDF-MILD",
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 10,
        upper=(1.0,) * 10,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_public_evaluator,
        release_metadata=lambda event: {"environment_severity": event},
    )
    feedback: dict[str, Any] = {
        "available": False,
        "reason": "MISSING_BY_DESIGN_PUBLIC_BENCHMARK",
        "released_at": 1,
        "ell_exec": None,
        "ell_ref": None,
        "s_exec": None,
        "hard_constraint_violation": None,
    }
    feedback.update(changes)
    with pytest.raises(PublicAdapterContractError, match="missing by design"):
        adapter.freeze_information(1, feedback)


def test_cdf_prohibited_field_fails_at_firewall_before_initial_feedback_rule() -> None:
    adapter = CDFPublicAdapter(
        suite_id="CDF-1-15",
        problem_id="CDF1",
        profile="CDF-MILD",
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=EVALUATOR_HASH,
        lower=(0.0,) * 10,
        upper=(1.0,) * 10,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_public_evaluator,
        release_metadata=lambda event: {"environment_severity": event},
    )
    with pytest.raises(InformationBoundaryError, match="prohibited"):
        adapter.freeze_information(
            0,
            {
                "future_trajectory": [1, 2, 3],
            },
        )


def test_static_public_bridge_runs_as_r2_ts1_correctness_fixture() -> None:
    def feasible(
        vector: Sequence[float], event_id: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        objectives, _ = _public_evaluator(vector, event_id)
        return objectives, (-1.0,)

    adapter = _static_adapter(evaluator=feasible)
    config = AlgorithmConfig(
        variant="NO_CROSS_EVENT_MEMORY",
        population_size=4,
        cfe_per_event=4,
        algorithm_seed=7,
        max_events=1,
        timing_mode="TS1_single_event",
        method_label="F22_MG_STATIC",
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        selector_id=_Selector.selector_id,
        selector_version=_Selector.selector_version,
        atomic_steps_per_evaluation=1,
        event_time_limit_seconds=10.0,
        configuration_evidence_id="PUBLIC_CORRECTNESS_FIXTURE",
        execution_request=R2ExecutionRequest(
            scope=ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE
        ),
    )

    result = DTRAMDE(config).run_sequence(adapter, selector=_Selector())
    assert len(result.events) == 1
    assert result.events[0].ledger["cfe"] == 4
    assert result.events[0].execution_feedback["available"] is False
    assert result.effect_estimation_performed is False
    assert result.hidden_seed_or_instance_generated is False
