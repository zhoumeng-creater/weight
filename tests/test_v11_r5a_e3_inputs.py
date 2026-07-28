from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest

from e3_inputs.contract import (
    DECISION_WEEKS,
    E3_SCENARIOS,
    E3InputContractError,
    apply_execution_transform,
    baseline_equilibrium_ee0_kcal_day,
    generate_subject_parameters,
    generate_subject_table,
    observation_is_missing,
    paired_observation_noise_kg,
    parameter_mismatch_ee_offset_kcal_day,
    required_intake_deficit_constraint,
    scenario_rule_rows,
    target_mass_kg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_r5a_e3_inputs import (  # noqa: E402
    DEFAULT_CONTRACT,
    R5AValidationError,
    _subject_rows,
    _validate_permissions,
    _validate_scenarios,
    load_json,
    validate_r5a,
)


R5_PATH = PROJECT_ROOT / "config" / "r5" / "r5_freeze_contract.json"


def _contract() -> dict:
    return load_json(DEFAULT_CONTRACT)


def _r5() -> dict:
    return json.loads(R5_PATH.read_text(encoding="utf-8"))


def _subjects() -> tuple:
    return generate_subject_table(_r5()["seed_contract"]["e3_public_subjects"])


def test_r5a_validator_passes_without_running_methods_or_effects() -> None:
    assert validate_r5a() == {
        "validator": "WGT-V11-R5A-E3-INPUT-VALIDATOR-01",
        "status": "PASS",
        "public_subjects": 32,
        "formal_scenarios": 9,
        "target_rows": 288,
        "paired_noise_rows": 2496,
        "method_id_in_paired_disturbance_key": False,
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "r7_authorized": False,
    }


def test_seed_to_subject_known_answers_and_context_are_exact() -> None:
    subjects = _subjects()
    assert subjects[0].canonical_row() == {
        "subject_id": "VS-000",
        "seed_u64": "1198773463880775798",
        "age_year": "54",
        "height_cm": "166.1",
        "bmi_kg_m2": "44.5",
        "weight_kg": "122.771984",
        "background_pal": "2.301",
        "sex_model_branch": "female",
        "adult_nonpregnant_nonlactating": True,
    }
    assert subjects[15].canonical_row()["weight_kg"] == "89.892032"
    assert subjects[-1].canonical_row()["weight_kg"] == "72.321256"
    assert len({tuple(item.canonical_row().values()) for item in subjects}) == 32
    for subject in subjects:
        subject.validate()
        subject.to_baseline().validate()


def test_generator_rejects_invalid_u64_and_does_not_filter_frozen_seeds() -> None:
    with pytest.raises(E3InputContractError, match="unsigned 64-bit"):
        generate_subject_parameters("VS-000", -1)
    with pytest.raises(E3InputContractError, match="unsigned 64-bit"):
        generate_subject_parameters("VS-000", 1 << 64)
    contract = _contract()
    assert contract["subject_generator"]["seed_filtering_allowed"] is False
    assert contract["subject_generator"]["replacement_allowed"] is False
    assert (
        contract["subject_generator"]["empirical_distribution_claim_allowed"]
        is False
    )


def test_target_is_frozen_after_ood_override_and_never_method_dependent() -> None:
    subject = _subjects()[0]
    assert target_mass_kg(subject, "NOMINAL") == Decimal("116.633385")
    assert target_mass_kg(
        subject,
        "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG",
    ) == Decimal("80.750000")
    target = _contract()["target_rule"]
    assert target["recalculation_allowed"] is False
    assert target["clinical_recommendation_claim_allowed"] is False
    assert target["individual_reachability_claim_allowed"] is False


def test_all_scenarios_keep_nominal_planning_and_assign_one_effect_layer() -> None:
    rows = scenario_rule_rows()
    assert tuple(row["scenario_id"] for row in rows) == E3_SCENARIOS
    assert all(
        row["planning_model_layer"] == "M_P_HALL_NONLINEAR_NOMINAL"
        for row in rows
    )
    parameter = rows[1]
    form = rows[2]
    assert parameter["execution_layer"] == "IDENTITY"
    assert "ADDITIONAL_EXPENDITURE_0_10_EE0" in parameter[
        "evaluation_model_layer"
    ]
    assert form["execution_layer"] == "IDENTITY"
    assert "HALL_LINEARIZED" in form["evaluation_model_layer"]


def test_paired_noise_known_answers_are_method_free_and_deterministic() -> None:
    r5 = _r5()
    subject = _subjects()[0]
    seed = int(r5["seed_contract"]["paired_master_seeds_u64"][0])
    kwargs = {
        "paired_master_seed_u64": seed,
        "subject_seed_u64": subject.seed_u64,
        "scenario_id": "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG",
        "replicate_index": 0,
        "decision_week": 4,
    }
    assert paired_observation_noise_kg(**kwargs) == Decimal("-1.071536038365")
    assert paired_observation_noise_kg(**kwargs) == (
        paired_observation_noise_kg(**kwargs)
    )
    paired = _contract()["paired_disturbance_contract"]
    assert paired["method_id_in_key"] is False
    assert "method" not in paired["message_bytes"].lower()


def test_missingness_execution_and_infeasibility_rules_are_exact() -> None:
    missing_scenario = "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
    assert {
        week
        for week in DECISION_WEEKS
        if observation_is_missing(missing_scenario, week)
    } == {4, 8, 12, 16, 20, 24}

    planned = {
        "intake_adjustment_kcal_day": Decimal("-500"),
        "activity_adjustment_kcal_day": Decimal("250"),
    }
    assert apply_execution_transform(
        "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
        **planned,
    ) == (Decimal("-375"), Decimal("187.50"))
    assert apply_execution_transform(
        "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
        **planned,
    ) == (Decimal("-250"), Decimal("250"))

    infeasible = (
        "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
    )
    for intake in (-1000, -500, 0, 500, 1000):
        assert required_intake_deficit_constraint(
            infeasible,
            Decimal(intake),
        ) > 0
    assert apply_execution_transform(infeasible, **planned) == (
        Decimal(0),
        Decimal(0),
    )


def test_parameter_mismatch_uses_exact_baseline_equilibrium_ee0() -> None:
    subject = _subjects()[0]
    assert baseline_equilibrium_ee0_kcal_day(subject) == Decimal("4221.977977")
    assert parameter_mismatch_ee_offset_kcal_day(
        subject,
        "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
    ) == Decimal("422.197798")
    assert parameter_mismatch_ee_offset_kcal_day(
        subject,
        "NOMINAL",
    ) == Decimal(0)


def test_validator_fails_closed_on_subject_scenario_or_permission_drift() -> None:
    r5 = _r5()
    subjects = _subjects()

    subject_drift = deepcopy(_contract())
    subject_drift["subject_generator"]["public_subject_parameters"][0][
        "age_year"
    ] = "55"
    with pytest.raises(R5AValidationError, match="subject parameter table"):
        _subject_rows(subject_drift, r5)

    scenario_drift = deepcopy(_contract())
    scenario_drift["scenario_contract"]["rules"][1][
        "planning_model_layer"
    ] = "M_P_RESULT_AWARE"
    with pytest.raises(R5AValidationError, match="scenario-rule table"):
        _validate_scenarios(scenario_drift, r5, subjects)

    permission_drift = deepcopy(_contract())
    permission_drift["permissions"]["effect_estimation_allowed"] = True
    with pytest.raises(R5AValidationError, match="permission differs"):
        _validate_permissions(permission_drift)


def test_r5a_validation_import_surface_excludes_runners_and_adapters() -> None:
    source = (
        PROJECT_ROOT / "src" / "e3_inputs" / "contract.py"
    ).read_text(encoding="utf-8")
    package_init = (
        PROJECT_ROOT / "src" / "e3_inputs" / "__init__.py"
    ).read_text(encoding="utf-8")
    validator = (
        PROJECT_ROOT / "tools" / "validate_r5a_e3_inputs.py"
    ).read_text(encoding="utf-8")
    combined = source + package_init + validator
    assert "illustrative_adapter" not in combined
    assert "run_v11_experiment" not in combined
    assert "run_v11_r6_pilot" not in combined
    assert "benchmark_adapters" not in combined
    assert "comparators" not in combined
