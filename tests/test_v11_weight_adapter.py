from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ExecutionScope,
    R2ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from dt_ramde_v11.interfaces import EventProblemAdapter
from dt_ramde_v11.state import MemoryState, PendingCredit, resolve_pending
from evaluation.contracts import EvaluationResult, TerminalCode
from evaluation.firewall import InformationBoundaryError
from evaluation.ledger import EvaluationLedger
from weight_application.adapter import SyntheticWeightAdapter
from weight_application.state import (
    SyntheticWeightModel,
    SyntheticWeightState,
    WeightStateError,
)


def _state() -> SyntheticWeightState:
    return SyntheticWeightState(
        event_id=0,
        fat_mass_kg=24.0,
        lean_mass_kg=56.0,
        cumulative_energy_imbalance_kcal=0.0,
    )


def _adapter() -> SyntheticWeightAdapter:
    return SyntheticWeightAdapter(
        initial_state=_state(),
        target_mass_kg=77.0,
        model=SyntheticWeightModel(
            event_days=7.0,
            energy_density_kcal_per_kg=7700.0,
            fat_mass_change_fraction=0.75,
        ),
    )


class _Selector:
    selector_id = "fixture.minimum-first-objective"
    selector_version = "1"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
        }

    def select(self, archive: Sequence[EvaluationResult]) -> str | None:
        if not archive:
            return None
        return min(
            archive,
            key=lambda result: (result.objectives[0], result.candidate_id),
        ).candidate_id


def test_weight_state_requires_nonnegative_compartments_and_mass_consistency() -> None:
    with pytest.raises(WeightStateError, match="nonnegative"):
        SyntheticWeightState(
            event_id=0,
            fat_mass_kg=-1.0,
            lean_mass_kg=56.0,
            cumulative_energy_imbalance_kcal=0.0,
        )
    state = _state()
    assert state.body_mass_kg == pytest.approx(80.0)
    assert state.fat_mass_kg + state.lean_mass_kg == pytest.approx(
        state.body_mass_kg
    )


@pytest.mark.parametrize(
    ("action", "direction"),
    [
        ((-500.0, 0.0), -1),
        ((500.0, 0.0), 1),
        ((500.0, 500.0), 0),
    ],
)
def test_energy_deficit_surplus_and_zero_balance_propagate_mass(
    action: tuple[float, float], direction: int
) -> None:
    adapter = _adapter()
    before = adapter.state
    ledger = EvaluationLedger(max_cfe=1)
    feedback = adapter.execute(action, 0, True, ledger)
    after = adapter.state

    observed = after.body_mass_kg - before.body_mass_kg
    if direction < 0:
        assert observed < 0.0
    elif direction > 0:
        assert observed > 0.0
    else:
        assert observed == pytest.approx(0.0)
    assert after.fat_mass_kg + after.lean_mass_kg == pytest.approx(
        after.body_mass_kg
    )
    assert after.cumulative_energy_imbalance_kcal == pytest.approx(
        7.0 * (action[0] - action[1])
    )
    assert feedback["energy_imbalance_kcal"] == pytest.approx(
        7.0 * (action[0] - action[1])
    )
    assert ledger.snapshot()["execution_transition_count"] == 1


def test_weight_evaluation_uses_shared_ledger_without_mutating_outer_state() -> None:
    adapter = _adapter()
    assert isinstance(adapter, EventProblemAdapter)
    assert adapter.constraint_scales == (1.0, 1500.0, 1.0, 1.0)
    snapshot = adapter.freeze_information(0, None)
    before = adapter.state
    ledger = EvaluationLedger(max_cfe=2)

    deficit = adapter.evaluate((-500.0, 0.0), 0, ledger, "deficit")
    surplus = adapter.evaluate((500.0, 0.0), 0, ledger, "surplus")

    assert adapter.state == before
    assert snapshot.decision_time == 0
    assert ledger.snapshot()["cfe"] == 2
    assert deficit.objectives[0] < surplus.objectives[0]
    assert deficit.objective_names == (
        "target_mass_error_kg",
        "intervention_burden_fraction",
    )
    assert deficit.constraint_names == (
        "minimum_body_mass",
        "maximum_daily_energy_imbalance",
        "nonnegative_fat_mass",
        "nonnegative_lean_mass",
    )


def test_finite_candidate_with_negative_projected_compartment_is_infeasible() -> None:
    adapter = SyntheticWeightAdapter(
        initial_state=SyntheticWeightState(
            event_id=0,
            fat_mass_kg=0.1,
            lean_mass_kg=79.9,
            cumulative_energy_imbalance_kcal=0.0,
        ),
        target_mass_kg=77.0,
        model=SyntheticWeightModel(
            event_days=7.0,
            energy_density_kcal_per_kg=7700.0,
            fat_mass_change_fraction=0.75,
        ),
    )
    adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    result = adapter.evaluate(
        (-1000.0, 1000.0),
        0,
        ledger,
        "finite-infeasible",
    )

    constraint = dict(
        zip(result.constraint_names, result.constraints, strict=True)
    )
    assert result.feasible is False
    assert constraint["nonnegative_fat_mass"] > 0.0
    assert ledger.snapshot()["cfe"] == 1
    assert ledger.snapshot()["evaluation_failures"] == 0


def test_weight_information_snapshot_rejects_future_fields() -> None:
    adapter = _adapter()
    with pytest.raises(InformationBoundaryError, match="prohibited"):
        adapter.freeze_information(
            0,
            {
                "available": True,
                "future_trajectory": [79.0, 78.0],
            },
        )


@pytest.mark.parametrize(
    "feedback",
    [
        {"available": True, "released_at": 1.9},
        {"available": True},
    ],
)
def test_weight_feedback_requires_explicit_integer_current_release(
    feedback: Mapping[str, Any],
) -> None:
    adapter = _adapter()
    ledger = EvaluationLedger(max_cfe=1)
    adapter.execute(adapter.fallback_action(0), 0, False, ledger)

    with pytest.raises(InformationBoundaryError, match="released"):
        adapter.freeze_information(1, feedback)


def test_weight_adapter_requires_strict_event_order_and_single_execution() -> None:
    adapter = _adapter()
    ledger = EvaluationLedger(max_cfe=1)
    with pytest.raises(WeightStateError, match="current state"):
        adapter.freeze_information(1, None)

    adapter.freeze_information(0, None)
    adapter.execute((-200.0, 0.0), 0, True, ledger)
    with pytest.raises(WeightStateError, match="current state"):
        adapter.execute((-200.0, 0.0), 0, True, ledger)
    next_snapshot = adapter.freeze_information(
        1,
        {
            "available": True,
            "ell_exec": 1.0,
            "ell_ref": 2.0,
            "s_exec": 0.1,
            "hard_constraint_violation": False,
            "released_at": 1,
        },
    )
    assert next_snapshot.decision_time == 1


def test_weight_adapter_fallback_is_neutral_and_feedback_is_typed() -> None:
    adapter = _adapter()
    assert tuple(adapter.fallback_action(0)) == (0.0, 0.0)
    ledger = EvaluationLedger(max_cfe=1)
    feedback = adapter.execute(adapter.fallback_action(0), 0, False, ledger)

    assert feedback["available"] is True
    assert feedback["committed"] is False
    assert feedback["ell_exec"] == pytest.approx(feedback["ell_ref"])
    assert feedback["s_exec"] == pytest.approx(7.0 * 1500.0 / 7700.0)
    assert feedback["hard_constraint_violation"] is False


def test_zero_action_uses_prefrozen_positive_scale_at_next_event() -> None:
    adapter = _adapter()
    snapshot = adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    feedback = adapter.execute((0.0, 0.0), 0, True, ledger)

    expected_scale = 7.0 * 1500.0 / 7700.0
    assert feedback["s_exec"] == pytest.approx(expected_scale)
    assert (
        snapshot.fields["execution_credit_contract"].value["s_exec_kg"]
        == pytest.approx(expected_scale)
    )

    memory = MemoryState(
        pending_credit=PendingCredit(
            pending_id="zero-action",
            source_event=0,
            terminal_code=TerminalCode.ACCEPTED,
            lineage_weights=(("n", 1.0),),
            parameter_values={"n": (0.4, 0.6)},
            information_hash=snapshot.information_hash,
            adapter_version=adapter.adapter_version,
        )
    )
    atoms, q_value, status = resolve_pending(
        memory,
        variant="FULL",
        feedback=feedback,
    )

    assert status == "EXECUTION_Q_RESOLVED"
    assert q_value == pytest.approx(0.0)
    assert atoms[0].signed_credit == pytest.approx(0.25)
    assert memory.invalidated is False


def test_tiny_action_keeps_prefrozen_scale_after_pending_restore() -> None:
    adapter = _adapter()
    snapshot = adapter.freeze_information(0, None)
    feedback = adapter.execute(
        (1e-12, 0.0),
        0,
        True,
        EvaluationLedger(max_cfe=1),
    )
    pending = PendingCredit(
        pending_id="tiny-action",
        source_event=0,
        terminal_code=TerminalCode.ACCEPTED,
        lineage_weights=(("n", 1.0),),
        parameter_values={"n": (0.4, 0.6)},
        information_hash=snapshot.information_hash,
        adapter_version=adapter.adapter_version,
    )
    restored = MemoryState.from_dict(
        MemoryState(pending_credit=pending).to_dict()
    )

    atoms, q_value, status = resolve_pending(
        restored,
        variant="FULL",
        feedback=feedback,
    )

    assert feedback["s_exec"] == pytest.approx(7.0 * 1500.0 / 7700.0)
    assert feedback["s_exec"] > 0.0
    assert status == "EXECUTION_Q_RESOLVED"
    assert q_value == pytest.approx(0.0, abs=1e-12)
    assert atoms[0].signed_credit == pytest.approx(0.25, abs=1e-12)
    assert restored.pending_credit is None


def test_weight_adapter_runs_two_r2_correctness_events_without_effect_estimation() -> None:
    adapter = _adapter()
    config = AlgorithmConfig(
        variant="FULL",
        population_size=4,
        cfe_per_event=8,
        algorithm_seed=11,
        max_events=2,
        timing_mode="TS2_fixed_periodic_replanning",
        method_label="DT-RAMDE_TS2_FULL",
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        selector_id=_Selector.selector_id,
        selector_version=_Selector.selector_version,
        atomic_steps_per_evaluation=1,
        event_time_limit_seconds=10.0,
        configuration_evidence_id="UNIT_TEST_FIXTURE",
        execution_request=R2ExecutionRequest(
            scope=ExecutionScope.UNIT_TEST_FIXTURE
        ),
    )

    result = DTRAMDE(config).run_sequence(adapter, selector=_Selector())
    assert len(result.events) == 2
    assert adapter.state.event_id == 2
    assert adapter.state.fat_mass_kg + adapter.state.lean_mass_kg == pytest.approx(
        adapter.state.body_mass_kg
    )
    assert [event.ledger["cfe"] for event in result.events] == [8, 8]
    assert [
        event.ledger["execution_transition_count"] for event in result.events
    ] == [1, 1]
    assert result.effect_estimation_performed is False
    assert result.hidden_seed_or_instance_generated is False
