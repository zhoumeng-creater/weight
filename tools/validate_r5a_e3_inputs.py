"""Validate the result-blind R5a E3 input contract without running methods."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from e3_inputs.contract import (
    DECISION_WEEKS,
    E3_SCENARIOS,
    MISSING_OBSERVATION_WEEKS,
    apply_execution_transform,
    generate_subject_table,
    initial_mass_kg,
    observation_is_missing,
    paired_observation_noise_kg,
    parameter_mismatch_ee_offset_kcal_day,
    required_intake_deficit_constraint,
    scenario_rule_rows,
    target_mass_kg,
)


DEFAULT_CONTRACT = ROOT / "config" / "r5a" / "e3_input_contract.json"
SCHEMA_PATH = ROOT / "config" / "r5a" / "e3_input_contract.schema.json"
R5_PATH = ROOT / "config" / "r5" / "r5_freeze_contract.json"
R6_PATH = ROOT / "config" / "r6" / "r6_pilot_contract.json"
NOISE_SCENARIO = "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG"


class R5AValidationError(RuntimeError):
    """The R5a contract is inconsistent, result-aware, or incomplete."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(contract_path: Path) -> dict[str, Any]:
    contract = load_json(contract_path)
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    return contract


def _validate_upstream(contract: Mapping[str, Any]) -> tuple[dict, dict]:
    upstream = contract["upstream_bindings"]
    for name in (
        "r5_contract",
        "r6_contract",
        "scientific_models",
        "decision_contract",
    ):
        binding = upstream[name]
        path = ROOT / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise R5AValidationError(f"upstream hash differs: {name}")
    for name in ("generator_source", "generator_package_init"):
        generator = contract["validation_contract"][name]
        generator_path = ROOT / generator["path"]
        if (
            not generator_path.is_file()
            or file_sha256(generator_path) != generator["sha256"]
        ):
            raise R5AValidationError(f"{name} hash differs")

    r5 = load_json(R5_PATH)
    r6 = load_json(R6_PATH)
    if r5["contract_id"] != "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01":
        raise R5AValidationError("R5 identity differs")
    if r6["contract_id"] != (
        "WGT-V11-R6-ISOLATED-RESULT-BLIND-ENGINEERING-PILOT-01"
    ):
        raise R5AValidationError("R6 identity differs")
    if r6["formal_input_gap"] != {
        "r5_seed_to_subject_parameter_generator_frozen": False,
        "r5_formal_target_rule_frozen": False,
        "r6_may_invent_formal_rules": False,
        "required_next_action": (
            "R5A_RESULT_BLIND_E3_SUBJECT_GENERATOR_AND_TARGET_RULE_FREEZE"
        ),
    }:
        raise R5AValidationError("historical R6 formal-input gap was rewritten")
    return r5, r6


def _subject_rows(
    contract: Mapping[str, Any],
    r5: Mapping[str, Any],
) -> tuple[Any, ...]:
    subjects = generate_subject_table(
        r5["seed_contract"]["e3_public_subjects"]
    )
    rows = [subject.canonical_row() for subject in subjects]
    generator = contract["subject_generator"]
    if rows != generator["public_subject_parameters"]:
        raise R5AValidationError("subject parameter table differs")
    if canonical_sha256(rows) != generator["parameter_table_sha256"]:
        raise R5AValidationError("subject parameter commitment differs")
    known = contract["validation_contract"]["known_subject_rows"]
    if known != [rows[0], rows[15], rows[31]]:
        raise R5AValidationError("known subject rows differ")
    if len(subjects) != 32:
        raise R5AValidationError("subject count differs")
    if len({tuple(row.values()) for row in rows}) != 32:
        raise R5AValidationError("generated subject rows are not unique")
    return subjects


def _target_rows(
    subjects: Sequence[Any],
) -> list[dict[str, str]]:
    return [
        {
            "subject_id": subject.subject_id,
            "scenario_id": scenario_id,
            "initial_mass_kg": format(
                initial_mass_kg(subject, scenario_id),
                ".6f",
            ),
            "target_mass_kg": format(
                target_mass_kg(subject, scenario_id),
                ".6f",
            ),
        }
        for subject in subjects
        for scenario_id in E3_SCENARIOS
    ]


def _validate_target(
    contract: Mapping[str, Any],
    subjects: Sequence[Any],
) -> None:
    rows = _target_rows(subjects)
    frozen = contract["target_rule"]
    if len(rows) != frozen["target_table_rows"]:
        raise R5AValidationError("target row count differs")
    if canonical_sha256(rows) != frozen["target_table_sha256"]:
        raise R5AValidationError("target table commitment differs")
    known = contract["validation_contract"]["known_target_rows"]
    if known != [rows[0], rows[7], rows[-1]]:
        raise R5AValidationError("known target rows differ")
    ood = [
        row
        for row in rows
        if row["scenario_id"]
        == "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG"
    ]
    if {
        (row["initial_mass_kg"], row["target_mass_kg"]) for row in ood
    } != {("85.000000", "80.750000")}:
        raise R5AValidationError("OOD target is not frozen after state override")


def _noise_rows(
    subjects: Sequence[Any],
    master_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "subject_id": subject.subject_id,
            "replicate_index": replicate_index,
            "decision_week": decision_week,
            "noise_kg": format(
                paired_observation_noise_kg(
                    paired_master_seed_u64=master_seeds[replicate_index],
                    subject_seed_u64=subject.seed_u64,
                    scenario_id=NOISE_SCENARIO,
                    replicate_index=replicate_index,
                    decision_week=decision_week,
                ),
                ".12f",
            ),
        }
        for subject in subjects
        for replicate_index in range(3)
        for decision_week in DECISION_WEEKS
    ]


def _validate_noise(
    contract: Mapping[str, Any],
    r5: Mapping[str, Any],
    subjects: Sequence[Any],
) -> None:
    paired = contract["paired_disturbance_contract"]
    r5_seeds = [
        int(value)
        for value in r5["seed_contract"]["paired_master_seeds_u64"][:3]
    ]
    if [str(value) for value in r5_seeds] != paired[
        "paired_master_seeds_u64"
    ]:
        raise R5AValidationError("paired master-seed prefix differs")
    rows = _noise_rows(subjects, r5_seeds)
    if len(rows) != paired["noise_table_rows"]:
        raise R5AValidationError("noise row count differs")
    if canonical_sha256(rows) != paired["noise_table_sha256"]:
        raise R5AValidationError("noise table commitment differs")
    indices = (0, 4, 25, 26, len(rows) - 1)
    if paired["known_answers"] != [rows[index] for index in indices]:
        raise R5AValidationError("noise known answers differ")
    if paired["method_id_in_key"] is not False:
        raise R5AValidationError("paired disturbance key contains method_id")


def _validate_scenarios(
    contract: Mapping[str, Any],
    r5: Mapping[str, Any],
    subjects: Sequence[Any],
) -> None:
    frozen = contract["scenario_contract"]
    rows = scenario_rule_rows()
    if rows != frozen["rules"]:
        raise R5AValidationError("formal scenario-rule table differs")
    if canonical_sha256(rows) != frozen["rule_table_sha256"]:
        raise R5AValidationError("scenario-rule commitment differs")
    scenario_ids = [row["scenario_id"] for row in rows]
    if tuple(scenario_ids) != E3_SCENARIOS:
        raise R5AValidationError("module scenario order differs")
    if scenario_ids != r5["experiment_design"]["E3"]["scenarios"]:
        raise R5AValidationError("R5 scenario order differs")
    if any(
        row["planning_model_layer"] != "M_P_HALL_NONLINEAR_NOMINAL"
        for row in rows
    ):
        raise R5AValidationError("a scenario changed the planning model")
    streams = {
        row["scenario_id"]: row["paired_random_streams"] for row in rows
    }
    if streams != {
        scenario_id: (
            ["observation_mass_gaussian"]
            if scenario_id == NOISE_SCENARIO
            else []
        )
        for scenario_id in E3_SCENARIOS
    }:
        raise R5AValidationError("scenario paired-stream assignment differs")
    missing = {
        week
        for week in DECISION_WEEKS
        if observation_is_missing(
            "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK",
            week,
        )
    }
    if missing != set(MISSING_OBSERVATION_WEEKS):
        raise R5AValidationError("missingness weeks differ")

    planned = {
        "intake_adjustment_kcal_day": Decimal("-500"),
        "activity_adjustment_kcal_day": Decimal("250"),
    }
    if apply_execution_transform(
        "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
        **planned,
    ) != (Decimal("-375"), Decimal("187.50")):
        raise R5AValidationError("75-percent execution transform differs")
    if apply_execution_transform(
        "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
        **planned,
    ) != (Decimal("-250"), Decimal("250")):
        raise R5AValidationError("energy-surplus execution transform differs")
    if required_intake_deficit_constraint(
        (
            "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_"
            "1000_KCAL_DAY"
        ),
        Decimal("-1000"),
    ) != Decimal("500"):
        raise R5AValidationError("infeasible required-deficit rule differs")
    if parameter_mismatch_ee_offset_kcal_day(
        subjects[0],
        "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
    ) != Decimal("422.197798"):
        raise R5AValidationError("parameter-mismatch EE known answer differs")


def _validate_permissions(contract: Mapping[str, Any]) -> None:
    allowed_true = {
        "r5a_contract_validation_allowed",
        "public_subject_input_generation_allowed",
        "synthetic_known_answer_tests_allowed",
    }
    for key, value in contract["permissions"].items():
        if value is not (key in allowed_true):
            raise R5AValidationError(f"permission differs: {key}")
    if contract["next_gate"] != {
        "id": "V11_R7_FORMAL_EXECUTION_AUTHORIZATION",
        "authorized": False,
        "separate_author_confirmation_required": True,
        "formal_effect_execution_allowed": False,
    }:
        raise R5AValidationError("R5a automatically authorized R7")
    if set(contract["result_awareness_boundary"].values()) != {False}:
        raise R5AValidationError("result-awareness boundary was crossed")


def validate_r5a(
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    contract = _validate_schema(contract_path)
    r5, _r6 = _validate_upstream(contract)
    subjects = _subject_rows(contract, r5)
    _validate_target(contract, subjects)
    _validate_noise(contract, r5, subjects)
    _validate_scenarios(contract, r5, subjects)
    _validate_permissions(contract)
    return {
        "validator": "WGT-V11-R5A-E3-INPUT-VALIDATOR-01",
        "status": "PASS",
        "public_subjects": len(subjects),
        "formal_scenarios": len(E3_SCENARIOS),
        "target_rows": contract["target_rule"]["target_table_rows"],
        "paired_noise_rows": contract["paired_disturbance_contract"][
            "noise_table_rows"
        ],
        "method_id_in_paired_disturbance_key": False,
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "r7_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = validate_r5a(Path(args.contract).resolve())
    except Exception as error:
        print(str(error))
        return 2
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
