"""Result-blind R5a input generation and scenario semantics for E3.

This package module generates public synthetic inputs only.  It does not run an
optimizer, evaluate a method, estimate an effect, or read participant data.
Every pseudorandom-looking value is derived with an explicitly domain-separated
SHA-256 construction so the contract does not depend on a PRNG implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
from math import cos, log, sqrt, tau
from typing import Any, Iterable, Mapping


U64_MAX = (1 << 64) - 1
SUBJECT_PARAMETER_DOMAIN = b"WGT-V11-R5A-E3-SUBJECT-PARAMETER-v1\0"
PAIRED_DISTURBANCE_DOMAIN = b"WGT-V11-R5A-E3-PAIRED-DISTURBANCE-v1\0"
OBSERVATION_STREAM_LABEL = b"observation_mass_gaussian"
MASS_QUANTUM_KG = Decimal("0.000001")
NOISE_QUANTUM_KG = Decimal("0.000000000001")
TARGET_FRACTION = Decimal("0.95")
OOD_FAT_MASS_KG = Decimal("50")
OOD_LEAN_MASS_KG = Decimal("35")
OOD_ADAPTIVE_THERMOGENESIS_KCAL_DAY = Decimal("0")
DECISION_WEEKS = tuple(range(26))
STATE_WEEKS = tuple(range(27))
MISSING_OBSERVATION_WEEKS = (4, 8, 12, 16, 20, 24)

E3_SCENARIOS = (
    "NOMINAL",
    "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
    "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION",
    "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG",
    "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK",
    "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
    "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
    "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG",
    "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY",
)

FORMAL_SCENARIO_RULES: tuple[Mapping[str, Any], ...] = (
    {
        "scenario_id": "NOMINAL",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
    {
        "scenario_id": "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": (
            "M_E_HALL_NONLINEAR_WITH_ADDITIONAL_EXPENDITURE_0_10_EE0"
        ),
        "paired_random_streams": [],
    },
    {
        "scenario_id": "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": (
            "M_E_HALL_LINEARIZED_NET_ENERGY_INTAKE_MINUS_ACTIVITY"
        ),
        "paired_random_streams": [],
    },
    {
        "scenario_id": "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": (
            "PAST_ONLY_TOTAL_MASS_PLUS_PAIRED_GAUSSIAN_SD_0_5_KG"
        ),
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": ["observation_mass_gaussian"],
    },
    {
        "scenario_id": "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": (
            "LOCF_WITH_SOURCE_WEEK_FLAG_AT_DECISION_WEEKS_4_8_12_16_20_24"
        ),
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
    {
        "scenario_id": (
            "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY"
        ),
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": (
            "DETERMINISTIC_WEEKLY_MEAN_0_75_INTAKE_CHANGE_AND_ACTIVITY"
        ),
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
    {
        "scenario_id": "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "ADD_250_KCAL_DAY_TO_EXECUTED_INTAKE_CHANGE",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
    {
        "scenario_id": "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG",
        "initial_state_layer": "OVERRIDE_FAT_50_LEAN_35_AT_0",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": "STANDARD_HARD_CONSTRAINTS",
        "execution_layer": "IDENTITY",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
    {
        "scenario_id": (
            "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
        ),
        "initial_state_layer": "GENERATED_HALL_INITIAL_STATE",
        "observation_layer": "EXACT_PAST_ONLY",
        "planning_model_layer": "M_P_HALL_NONLINEAR_NOMINAL",
        "feasibility_layer": (
            "G_REQUIRED_INTAKE_ADJUSTMENT_PLUS_1500_LE_ZERO"
        ),
        "execution_layer": "NEUTRAL_FALLBACK_AFTER_NO_FEASIBLE",
        "evaluation_model_layer": "M_E_HALL_NONLINEAR_NOMINAL",
        "paired_random_streams": [],
    },
)


class E3InputContractError(ValueError):
    """An E3 input or rule falls outside the frozen R5a contract."""


def _validate_u64(value: int, label: str) -> None:
    if type(value) is not int or not (0 <= value <= U64_MAX):
        raise E3InputContractError(f"{label} must be an unsigned 64-bit integer")


def _field_u64(seed_u64: int, field_label: str) -> int:
    _validate_u64(seed_u64, "seed_u64")
    if not field_label or not field_label.isascii():
        raise E3InputContractError("field label must be nonempty ASCII")
    digest = sha256(
        SUBJECT_PARAMETER_DOMAIN
        + seed_u64.to_bytes(8, "big")
        + b"\0"
        + field_label.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class E3SubjectParameters:
    """Exact public inputs for one illustrative adult-female virtual subject."""

    subject_id: str
    seed_u64: int
    age_year: int
    height_cm: Decimal
    bmi_kg_m2: Decimal
    weight_kg: Decimal
    background_pal: Decimal
    sex_model_branch: str = "female"
    adult_nonpregnant_nonlactating: bool = True

    def validate(self) -> None:
        if (
            len(self.subject_id) != 6
            or not self.subject_id.startswith("VS-")
            or not self.subject_id[3:].isdigit()
        ):
            raise E3InputContractError("subject_id must match VS-NNN")
        _validate_u64(self.seed_u64, "seed_u64")
        if self.sex_model_branch != "female":
            raise E3InputContractError("R5a freezes the female model branch")
        expected_weight = (
            self.bmi_kg_m2 * (self.height_cm / Decimal(100)) ** 2
        ).quantize(MASS_QUANTUM_KG, rounding=ROUND_HALF_EVEN)
        if self.weight_kg != expected_weight:
            raise E3InputContractError("weight differs from BMI-height identity")
        if not (
            30 <= self.age_year <= 78
            and Decimal("150.0") <= self.height_cm <= Decimal("180.0")
            and Decimal("25.0") <= self.bmi_kg_m2 <= Decimal("50.0")
            and Decimal("45.0") <= self.weight_kg <= Decimal("250.0")
            and Decimal("1.0") <= self.background_pal <= Decimal("2.5")
            and self.adult_nonpregnant_nonlactating
        ):
            raise E3InputContractError(
                "subject is outside the frozen adult-female context"
            )

    def to_baseline(self) -> Any:
        from weight_application.scientific_models import AdultFemaleBaseline

        return AdultFemaleBaseline(
            age_year=float(self.age_year),
            height_cm=float(self.height_cm),
            weight_kg=float(self.weight_kg),
            background_pal=float(self.background_pal),
            adult_nonpregnant_nonlactating=(
                self.adult_nonpregnant_nonlactating
            ),
        )

    def canonical_row(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "seed_u64": str(self.seed_u64),
            "age_year": str(self.age_year),
            "height_cm": format(self.height_cm, ".1f"),
            "bmi_kg_m2": format(self.bmi_kg_m2, ".1f"),
            "weight_kg": format(self.weight_kg, ".6f"),
            "background_pal": format(self.background_pal, ".3f"),
            "sex_model_branch": self.sex_model_branch,
            "adult_nonpregnant_nonlactating": (
                self.adult_nonpregnant_nonlactating
            ),
        }


def generate_subject_parameters(
    subject_id: str,
    seed_u64: int,
) -> E3SubjectParameters:
    """Map one frozen public seed to exact, grid-quantized subject inputs."""

    age_year = 30 + _field_u64(seed_u64, "age_year") % 49
    height_cm = Decimal(
        1500 + _field_u64(seed_u64, "height_cm_tenths") % 301
    ) / Decimal(10)
    bmi_kg_m2 = Decimal(
        250 + _field_u64(seed_u64, "bmi_kg_m2_tenths") % 251
    ) / Decimal(10)
    background_pal = Decimal(
        1400
        + _field_u64(seed_u64, "background_pal_thousandths") % 1101
    ) / Decimal(1000)
    weight_kg = (
        bmi_kg_m2 * (height_cm / Decimal(100)) ** 2
    ).quantize(MASS_QUANTUM_KG, rounding=ROUND_HALF_EVEN)
    subject = E3SubjectParameters(
        subject_id=subject_id,
        seed_u64=seed_u64,
        age_year=age_year,
        height_cm=height_cm,
        bmi_kg_m2=bmi_kg_m2,
        weight_kg=weight_kg,
        background_pal=background_pal,
    )
    subject.validate()
    return subject


def generate_subject_table(
    seed_rows: Iterable[Mapping[str, Any]],
) -> tuple[E3SubjectParameters, ...]:
    subjects = tuple(
        generate_subject_parameters(
            str(row["subject_id"]),
            int(row["seed_u64"]),
        )
        for row in seed_rows
    )
    if len({subject.subject_id for subject in subjects}) != len(subjects):
        raise E3InputContractError("subject ids must be unique")
    if len({subject.seed_u64 for subject in subjects}) != len(subjects):
        raise E3InputContractError("subject seeds must be unique")
    return subjects


def baseline_equilibrium_ee0_kcal_day(
    subject: E3SubjectParameters,
) -> Decimal:
    """Mifflin female RMR times PAL, equal to Hall baseline equilibrium intake."""

    subject.validate()
    rmr = (
        Decimal(10) * subject.weight_kg
        + Decimal("6.25") * subject.height_cm
        - Decimal(5) * Decimal(subject.age_year)
        - Decimal(161)
    )
    return (subject.background_pal * rmr).quantize(
        MASS_QUANTUM_KG,
        rounding=ROUND_HALF_EVEN,
    )


def _validate_scenario(scenario_id: str) -> None:
    if scenario_id not in E3_SCENARIOS:
        raise E3InputContractError("scenario is outside the frozen R5a table")


def initial_mass_kg(
    subject: E3SubjectParameters,
    scenario_id: str,
) -> Decimal:
    subject.validate()
    _validate_scenario(scenario_id)
    if scenario_id == "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG":
        return OOD_FAT_MASS_KG + OOD_LEAN_MASS_KG
    return subject.weight_kg


def target_mass_kg(
    subject: E3SubjectParameters,
    scenario_id: str,
) -> Decimal:
    """Freeze 95% of the scenario-specific week-0 mass to 1e-6 kg."""

    return (initial_mass_kg(subject, scenario_id) * TARGET_FRACTION).quantize(
        MASS_QUANTUM_KG,
        rounding=ROUND_HALF_EVEN,
    )


def observation_is_missing(scenario_id: str, decision_week: int) -> bool:
    _validate_scenario(scenario_id)
    if decision_week not in DECISION_WEEKS:
        raise E3InputContractError("decision week must be in 0..25")
    return (
        scenario_id == "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
        and decision_week in MISSING_OBSERVATION_WEEKS
    )


def paired_observation_noise_kg(
    *,
    paired_master_seed_u64: int,
    subject_seed_u64: int,
    scenario_id: str,
    replicate_index: int,
    decision_week: int,
) -> Decimal:
    """Return the method-independent Box--Muller observation perturbation."""

    _validate_u64(paired_master_seed_u64, "paired_master_seed_u64")
    _validate_u64(subject_seed_u64, "subject_seed_u64")
    if scenario_id != "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG":
        raise E3InputContractError("noise is defined only for the noise scenario")
    if type(replicate_index) is not int or replicate_index not in range(3):
        raise E3InputContractError("E3 replicate_index must be in 0..2")
    if decision_week not in DECISION_WEEKS:
        raise E3InputContractError("decision week must be in 0..25")
    digest = sha256(
        PAIRED_DISTURBANCE_DOMAIN
        + paired_master_seed_u64.to_bytes(8, "big")
        + subject_seed_u64.to_bytes(8, "big")
        + replicate_index.to_bytes(4, "big")
        + decision_week.to_bytes(4, "big")
        + scenario_id.encode("ascii")
        + b"\0"
        + OBSERVATION_STREAM_LABEL
    ).digest()
    denominator = float(1 << 64)
    u1 = (int.from_bytes(digest[:8], "big") + 0.5) / denominator
    u2 = (int.from_bytes(digest[8:16], "big") + 0.5) / denominator
    normal = sqrt(-2.0 * log(u1)) * cos(tau * u2)
    return Decimal.from_float(0.5 * normal).quantize(
        NOISE_QUANTUM_KG,
        rounding=ROUND_HALF_EVEN,
    )


def apply_execution_transform(
    scenario_id: str,
    *,
    intake_adjustment_kcal_day: Decimal,
    activity_adjustment_kcal_day: Decimal,
) -> tuple[Decimal, Decimal]:
    """Apply only the scenario's execution-layer action transformation."""

    _validate_scenario(scenario_id)
    intake = Decimal(intake_adjustment_kcal_day)
    activity = Decimal(activity_adjustment_kcal_day)
    if not (
        Decimal("-1000") <= intake <= Decimal("1000")
        and Decimal("0") <= activity <= Decimal("1000")
    ):
        raise E3InputContractError("planned action is outside frozen bounds")
    if scenario_id == (
        "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY"
    ):
        return Decimal("0.75") * intake, Decimal("0.75") * activity
    if scenario_id == "ENERGY_SURPLUS_PLUS_250_KCAL_DAY":
        return intake + Decimal(250), activity
    if scenario_id == (
        "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
    ):
        return Decimal(0), Decimal(0)
    return intake, activity


def parameter_mismatch_ee_offset_kcal_day(
    subject: E3SubjectParameters,
    scenario_id: str,
) -> Decimal:
    _validate_scenario(scenario_id)
    if scenario_id == "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT":
        return (
            Decimal("0.10") * baseline_equilibrium_ee0_kcal_day(subject)
        ).quantize(MASS_QUANTUM_KG, rounding=ROUND_HALF_EVEN)
    return Decimal(0)


def required_intake_deficit_constraint(
    scenario_id: str,
    intake_adjustment_kcal_day: Decimal,
) -> Decimal:
    """Return g(x), where feasibility requires g <= 0."""

    _validate_scenario(scenario_id)
    intake = Decimal(intake_adjustment_kcal_day)
    if scenario_id == (
        "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
    ):
        return intake + Decimal(1500)
    return Decimal(-1)


def scenario_rule_rows() -> list[dict[str, Any]]:
    """Return fresh JSON-compatible rows in the frozen R5 order."""

    return [
        {
            key: list(value) if isinstance(value, list) else value
            for key, value in row.items()
        }
        for row in FORMAL_SCENARIO_RULES
    ]
