from __future__ import annotations

from decimal import Decimal

import pytest

from e3_inputs.contract import generate_subject_parameters
from evaluation.ledger import EvaluationLedger
from weight_application.formal_e3_adapter import FormalHallE3Adapter


SUBJECT = generate_subject_parameters("VS-000", 2_040_978_301_928_374_650)
MASTER_SEED = 15_223_742_352_718_460_915
ACTION = (-500.0, 300.0)


def _adapter(scenario: str) -> FormalHallE3Adapter:
    return FormalHallE3Adapter(
        subject=SUBJECT,
        scenario=scenario,
        replicate_index=0,
        paired_master_seed_u64=MASTER_SEED,
    )


def test_identity_is_formal_public_synthetic_and_claim_bounded() -> None:
    identity = _adapter("NOMINAL").identity()
    assert identity["formal_subject_generator_used"] is True
    assert identity["participant_data_used"] is False
    assert identity["hidden_instance_used"] is False
    assert identity["benchmark_effect_evidence"] is True
    assert identity["participant_or_clinical_effect_evidence"] is False
    assert identity["planning_scenario_transformations"] == "PROHIBITED"


@pytest.mark.parametrize(
    "scenario",
    [
        "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
        "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION",
        "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
        "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
    ],
)
def test_scenarios_do_not_leak_into_planning_evaluation(
    scenario: str,
) -> None:
    nominal = _adapter("NOMINAL")
    stressed = _adapter(scenario)
    nominal.freeze_information(0, None)
    stressed.freeze_information(0, None)
    nominal_result = nominal.evaluate(
        ACTION,
        0,
        EvaluationLedger(max_cfe=1),
        "nominal",
    )
    stressed_result = stressed.evaluate(
        ACTION,
        0,
        EvaluationLedger(max_cfe=1),
        "stressed",
    )
    assert stressed_result.objectives == nominal_result.objectives
    assert stressed_result.constraints == nominal_result.constraints


def test_infeasible_rule_changes_only_the_fifth_planning_constraint() -> None:
    nominal = _adapter("NOMINAL")
    infeasible = _adapter(
        "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
    )
    nominal.freeze_information(0, None)
    infeasible.freeze_information(0, None)
    nominal_result = nominal.evaluate(
        ACTION,
        0,
        EvaluationLedger(max_cfe=1),
        "nominal",
    )
    infeasible_result = infeasible.evaluate(
        ACTION,
        0,
        EvaluationLedger(max_cfe=1),
        "infeasible",
    )
    assert infeasible_result.objectives == nominal_result.objectives
    assert infeasible_result.constraints[:4] == nominal_result.constraints[:4]
    assert nominal_result.constraints[4] == -1.0
    assert infeasible_result.constraints[4] == 1000.0


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (
            "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
            (-375.0, 225.0),
        ),
        ("ENERGY_SURPLUS_PLUS_250_KCAL_DAY", (-250.0, 300.0)),
        (
            "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY",
            (0.0, 0.0),
        ),
    ],
)
def test_execution_transform_is_applied_only_at_execution(
    scenario: str,
    expected: tuple[float, float],
) -> None:
    adapter = _adapter(scenario)
    adapter.freeze_information(0, None)
    feedback = adapter.execute(
        ACTION,
        0,
        True,
        EvaluationLedger(max_cfe=1),
    )
    executed = feedback["executed_action"]
    assert (
        executed["intake_adjustment_kcal_day"],
        executed["activity_adjustment_kcal_day"],
    ) == expected
    assert feedback["planned_action"] == {
        "intake_adjustment_kcal_day": -500.0,
        "activity_adjustment_kcal_day": 300.0,
    }


def test_evaluation_model_mismatches_change_execution_not_planning() -> None:
    nominal = _adapter("NOMINAL")
    parameter = _adapter("PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT")
    form = _adapter("MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION")
    for adapter in (nominal, parameter, form):
        adapter.freeze_information(0, None)
    nominal_after = nominal.execute(
        ACTION, 0, True, EvaluationLedger(max_cfe=1)
    )
    parameter_after = parameter.execute(
        ACTION, 0, True, EvaluationLedger(max_cfe=1)
    )
    form_after = form.execute(
        ACTION, 0, True, EvaluationLedger(max_cfe=1)
    )
    nominal_mass = sum(nominal_after["formal_state_after"]["values"][:2])
    parameter_mass = sum(parameter_after["formal_state_after"]["values"][:2])
    form_mass = sum(form_after["formal_state_after"]["values"][:2])
    assert parameter_mass < nominal_mass
    assert form_mass != pytest.approx(nominal_mass)


def test_noise_is_exactly_paired_for_identical_seed_subject_and_replicate() -> None:
    first = _adapter("OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG")
    second = _adapter("OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG")
    first_info = first.freeze_information(0, None)
    second_info = second.freeze_information(0, None)
    first_observed = first_info.fields[
        "current_public_synthetic_observation"
    ].value
    second_observed = second_info.fields[
        "current_public_synthetic_observation"
    ].value
    assert first_observed == second_observed


def test_missing_observation_uses_locf_and_preserves_source_week() -> None:
    adapter = _adapter("MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK")
    ledger = EvaluationLedger(max_cfe=1)
    feedback = None
    week_three_observation = None
    for week in range(4):
        information = adapter.freeze_information(week, feedback)
        observation = information.fields[
            "current_public_synthetic_observation"
        ].value
        if week == 3:
            week_three_observation = observation
        feedback = adapter.execute(
            (0.0, 0.0),
            week,
            False,
            ledger,
        )
    missing = adapter.freeze_information(4, feedback).fields[
        "current_public_synthetic_observation"
    ].value
    assert missing["observation_available"] is False
    assert missing["source_week"] == 3
    assert missing["values"] == week_three_observation["values"]


def test_out_of_domain_initial_override_precedes_target_rule() -> None:
    adapter = _adapter("OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG")
    assert adapter.state.values == (50.0, 35.0, 0.0)
    assert Decimal(str(adapter.target_mass_kg)) == Decimal("80.75")
