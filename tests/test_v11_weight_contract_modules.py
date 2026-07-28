from __future__ import annotations

import pytest

from weight_application.adapter import SyntheticWeightAdapter
from weight_application.constraints import (
    SYNTHETIC_E0_CONSTRAINTS,
    evaluate_weight_constraints,
)
from weight_application.decisions import (
    DecisionContractError,
    SYNTHETIC_E0_DECISIONS,
)
from weight_application.model_roles import (
    E0_SYNTHETIC_PLANNING_BINDING,
    ModelRole,
    RoleViolation,
    assert_information_release,
    assert_optimizer_role_access,
)
from weight_application.objectives import (
    SYNTHETIC_E0_OBJECTIVES,
    evaluate_weight_objectives,
)


def test_weight_decision_contract_owns_bounds_repair_and_neutral_action() -> None:
    contract = SYNTHETIC_E0_DECISIONS
    assert contract.names == (
        "intake_adjustment_kcal_per_day",
        "activity_expenditure_adjustment_kcal_per_day",
    )
    assert contract.lower_bounds == (-1000.0, 0.0)
    assert contract.upper_bounds == (1000.0, 1000.0)
    assert tuple(contract.neutral_action()) == (0.0, 0.0)
    assert tuple(contract.repair((-2000.0, 1500.0))) == (-1000.0, 1000.0)
    assert contract.validate((-500.0, 250.0)) == (-500.0, 250.0)

    with pytest.raises(DecisionContractError, match="finite"):
        contract.validate((float("nan"), 0.0))
    with pytest.raises(DecisionContractError, match="bounds"):
        contract.validate((-1000.1, 0.0))


def test_weight_objective_and_constraint_registries_have_known_answers() -> None:
    objectives = evaluate_weight_objectives(
        predicted_body_mass_kg=78.0,
        target_mass_kg=77.0,
        intake_adjustment_kcal_per_day=-500.0,
        activity_expenditure_adjustment_kcal_per_day=250.0,
    )
    constraints = evaluate_weight_constraints(
        predicted_body_mass_kg=78.0,
        predicted_fat_mass_kg=23.0,
        predicted_lean_mass_kg=55.0,
        minimum_body_mass_kg=40.0,
        daily_energy_imbalance_kcal=-750.0,
        maximum_daily_energy_imbalance_kcal=1500.0,
    )

    assert SYNTHETIC_E0_OBJECTIVES.names == (
        "target_mass_error_kg",
        "intervention_burden_fraction",
    )
    assert objectives == pytest.approx((1.0, 0.375))
    assert SYNTHETIC_E0_CONSTRAINTS.names == (
        "minimum_body_mass",
        "maximum_daily_energy_imbalance",
        "nonnegative_fat_mass",
        "nonnegative_lean_mass",
    )
    assert SYNTHETIC_E0_CONSTRAINTS.feasibility_rule == "c_i <= 0"
    assert SYNTHETIC_E0_CONSTRAINTS.clinical_safety_claim is False
    assert constraints == pytest.approx((-38.0, -750.0, -23.0, -55.0))


def test_e0_model_role_is_explicit_and_scientific_roles_remain_unbound() -> None:
    binding = E0_SYNTHETIC_PLANNING_BINDING
    assert binding.role is ModelRole.PLANNING
    assert binding.qualification_status == "NOT_QUALIFIED_E0_CORRECTNESS_ONLY"
    assert binding.participant_data_allowed is False
    assert binding.effect_estimation_allowed is False
    assert_optimizer_role_access(binding.role)

    with pytest.raises(RoleViolation, match="planning"):
        assert_optimizer_role_access(ModelRole.EVALUATION_FORM)
    with pytest.raises(RoleViolation, match="future"):
        assert_information_release(observation_event=2, decision_event=1)


def test_weight_adapter_uses_registered_role_decisions_objectives_and_constraints() -> None:
    assert SyntheticWeightAdapter.lower_bounds == (
        SYNTHETIC_E0_DECISIONS.lower_bounds
    )
    assert SyntheticWeightAdapter.upper_bounds == (
        SYNTHETIC_E0_DECISIONS.upper_bounds
    )
    assert SyntheticWeightAdapter.objective_names == (
        SYNTHETIC_E0_OBJECTIVES.names
    )
    assert SyntheticWeightAdapter.constraint_names == (
        SYNTHETIC_E0_CONSTRAINTS.names
    )
    assert (
        SyntheticWeightAdapter.model_role_binding
        is E0_SYNTHETIC_PLANNING_BINDING
    )
