"""Result-blind V11-MQ1 model-qualification builder and evaluator.

This module contains no fitting, tuning, algorithm selection, effect
estimation, or automatic participant-data execution. Archive loading is
available only to the separately guarded command-line runner.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite, sqrt
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np

from .model_roles import ModelRole, RoleViolation
from .scientific_models import (
    ActivityEnergyMap,
    AdultFemaleBaseline,
    DirectEnergyExposure,
    HallLongTermModel,
)


CONTRACT_ID = "WGT-V11-MQ1-MODEL-QUALIFICATION-01"
SOURCE_TABLE = "full0_18nih.sas7bdat"
REQUIRED_SOURCE_COLUMNS = (
    "ID",
    "NVISIT",
    "RAGE",
    "HEIGHT",
    "WEIGHT0",
    "WEIGHT",
    "DT_KCAL",
    "kcal",
)
EXPECTED_VISIT_MONTHS = (0.0, 6.0, 12.0, 18.0)
BACKGROUND_PAL = 1.5
MODEL_STEP_DAY = 0.25
ID_DOMAIN = b"WGT-V11-MQ1-ID-V1\x00"
UQ_STATUS = "NOT_QUALIFIED_NO_INDEPENDENT_CALIBRATION_SET"
PASS_CASE_NAME = "qualified physiology-informed simulated case study"
NONPASS_CASE_NAME = "illustrative mechanistic simulation"


class QualificationInputError(ValueError):
    """A structural protocol violation that invalidates V11-MQ1 input."""


@dataclass(frozen=True)
class QualificationRecord:
    participant_id: str
    visit_month: float
    baseline_weight_kg: float
    observed_weight_kg: float
    predicted_weight_kg: float


@dataclass(frozen=True)
class QualificationThresholds:
    minimum_participants: int = 140
    minimum_postbaseline_visits_per_participant: int = 2
    mae_percent_max: float = 2.5
    rmse_percent_max: float = 3.5
    absolute_bias_percent_max: float = 1.0
    bias_ci_absolute_bound_percent: float = 2.0
    trajectory_niae_percent_max: float = 2.5
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20_260_722


@dataclass(frozen=True)
class ParticipantBaselineExposure:
    participant_id: str
    age_year: float
    height_cm: float
    weight_kg: float
    diet_kcal_day: float
    activity_kcal_week: float


@dataclass(frozen=True)
class IntervalExposure:
    visit_month: float
    diet_kcal_day: float
    activity_kcal_week: float


@dataclass(frozen=True)
class BuilderOutput:
    records: tuple[QualificationRecord, ...]
    audit: Mapping[str, object]
    canonical_input_sha256: str


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def pseudonymize_identifier(value: object) -> str:
    """Hash IDs without decoding bytes or exposing values in error messages."""

    if isinstance(value, bytes):
        payload = b"B\x00" + bytes(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise QualificationInputError("participant key is missing")
        payload = b"S\x00" + normalized.encode("utf-8", errors="strict")
    else:
        if value is None:
            raise QualificationInputError("participant key is missing")
        try:
            missing = bool(np.isscalar(value) and np.isnan(value))
        except TypeError:
            missing = False
        if missing:
            raise QualificationInputError("participant key is missing")
        normalized = str(value).strip()
        if not normalized:
            raise QualificationInputError("participant key is missing")
        payload = b"S\x00" + normalized.encode("utf-8", errors="strict")
    if payload in {b"B\x00", b"S\x00"}:
        raise QualificationInputError("participant key is missing")
    return hashlib.sha256(ID_DOMAIN + payload).hexdigest()


def month_to_model_day(month: float) -> float:
    raw_steps = (
        Decimal(str(month))
        * Decimal("365.25")
        / Decimal("12")
        / Decimal(str(MODEL_STEP_DAY))
    )
    steps = raw_steps.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(steps * Decimal(str(MODEL_STEP_DAY)))


def _predict_trajectory(
    baseline: ParticipantBaselineExposure,
    exposures: Sequence[IntervalExposure],
) -> dict[float, float]:
    scientific_baseline = AdultFemaleBaseline(
        age_year=baseline.age_year,
        height_cm=baseline.height_cm,
        weight_kg=baseline.weight_kg,
        background_pal=BACKGROUND_PAL,
        adult_nonpregnant_nonlactating=True,
    )
    model = HallLongTermModel(
        ModelRole.EVALUATION_PARAMETER,
        scientific_baseline,
        ActivityEnergyMap(0.0, 0.0, "V11_MQ1_DIRECT_EXPOSURE_ONLY"),
        integration_step_day=MODEL_STEP_DAY,
    )
    state = model.initial_state()
    previous_day = 0.0
    predictions: dict[float, float] = {}
    for interval in sorted(exposures, key=lambda item: item.visit_month):
        target_day = month_to_model_day(interval.visit_month)
        if target_day <= previous_day:
            raise QualificationInputError(
                "source-native visit months must increase strictly"
            )
        intake = model.baseline_energy_intake_kcal_day + (
            interval.diet_kcal_day - baseline.diet_kcal_day
        )
        activity_change = (
            interval.activity_kcal_week - baseline.activity_kcal_week
        ) / 7.0
        exposure = DirectEnergyExposure(intake, activity_change)
        state = model.advance_days_exposure(
            state,
            exposure,
            target_day - previous_day,
        )
        predictions[interval.visit_month] = model.weight_kg(state)
        previous_day = target_day
    return predictions


def _canonical_input_hash(
    records: Sequence[QualificationRecord],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "participant_id",
            "visit_month",
            "baseline_weight_kg",
            "observed_weight_kg",
            "predicted_weight_kg",
        )
    )
    for record in sorted(
        records,
        key=lambda item: (item.participant_id, item.visit_month),
    ):
        writer.writerow(
            (
                record.participant_id,
                format(record.visit_month, ".12g"),
                format(record.baseline_weight_kg, ".17g"),
                format(record.observed_weight_kg, ".17g"),
                format(record.predicted_weight_kg, ".17g"),
            )
        )
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def build_qualification_records_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> BuilderOutput:
    """Build prediction records without serializing raw participant IDs."""

    materialized = tuple(rows)
    if not materialized:
        raise QualificationInputError("source table is empty")
    missing_columns = sorted(
        set(REQUIRED_SOURCE_COLUMNS) - set(materialized[0])
    )
    if missing_columns:
        raise QualificationInputError(
            "source table lacks frozen required columns"
        )

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in materialized:
        participant_id = pseudonymize_identifier(row.get("ID"))
        grouped[participant_id].append(row)

    exclusions: Counter[str] = Counter()
    output: list[QualificationRecord] = []
    for participant_id, participant_rows in grouped.items():
        by_visit: dict[float, Mapping[str, object]] = {}
        invalid_visit = False
        for row in participant_rows:
            visit = _finite_float(row.get("NVISIT"))
            if visit not in EXPECTED_VISIT_MONTHS or visit in by_visit:
                invalid_visit = True
                break
            assert visit is not None
            by_visit[visit] = row
        if invalid_visit or 0.0 not in by_visit:
            exclusions["invalid_or_missing_visit_structure"] += 1
            continue

        base = by_visit[0.0]
        age = _finite_float(base.get("RAGE"))
        height_m = _finite_float(base.get("HEIGHT"))
        weight0 = _finite_float(base.get("WEIGHT0"))
        diet0 = _finite_float(base.get("DT_KCAL"))
        activity0 = _finite_float(base.get("kcal"))
        if None in (age, height_m, weight0, diet0, activity0):
            exclusions["missing_baseline_or_baseline_exposure"] += 1
            continue
        assert age is not None
        assert height_m is not None
        assert weight0 is not None
        assert diet0 is not None
        assert activity0 is not None
        baseline = ParticipantBaselineExposure(
            participant_id,
            age,
            height_m * 100.0,
            weight0,
            diet0,
            activity0,
        )

        intervals: list[IntervalExposure] = []
        outcomes: dict[float, float] = {}
        stopped_for_missing_exposure = False
        for visit in EXPECTED_VISIT_MONTHS[1:]:
            row = by_visit.get(visit)
            if row is None:
                stopped_for_missing_exposure = True
                break
            diet = _finite_float(row.get("DT_KCAL"))
            activity = _finite_float(row.get("kcal"))
            if diet is None or activity is None:
                stopped_for_missing_exposure = True
                break
            intervals.append(IntervalExposure(visit, diet, activity))
            observed = _finite_float(row.get("WEIGHT"))
            if observed is not None and observed > 0.0:
                outcomes[visit] = observed

        try:
            predictions = _predict_trajectory(baseline, intervals)
        except (RoleViolation, QualificationInputError):
            exclusions["outside_model_domain_or_invalid_exposure"] += 1
            continue
        participant_records = [
            QualificationRecord(
                participant_id,
                visit,
                weight0,
                outcomes[visit],
                predictions[visit],
            )
            for visit in sorted(set(predictions) & set(outcomes))
        ]
        if len(participant_records) < 2:
            reason = (
                "missing_exposure_stopped_trajectory_before_two_visits"
                if stopped_for_missing_exposure
                else "fewer_than_two_evaluable_postbaseline_visits"
            )
            exclusions[reason] += 1
            continue
        output.extend(participant_records)

    audit = {
        "source_rows": len(materialized),
        "source_participants": len(grouped),
        "eligible_participants": len(
            {record.participant_id for record in output}
        ),
        "eligible_postbaseline_records": len(output),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "postbaseline_outcome_used_for_prediction": False,
        "calibration_performed": False,
        "model_selection_performed": False,
        "threshold_changed": False,
        "raw_identifier_serialized": False,
        "prediction_interval_status": UQ_STATUS,
    }
    records = tuple(output)
    return BuilderOutput(
        records,
        audit,
        _canonical_input_hash(records),
    )


def load_pride_archive(
    archive_path: Path,
    *,
    read_sas: Callable[..., object] | None = None,
) -> BuilderOutput:
    """Load the frozen SAS member with bytes-preserving identifier semantics."""

    if read_sas is None:
        import pandas as pd

        read_sas = pd.read_sas
    with ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names.count(SOURCE_TABLE) != 1:
            raise QualificationInputError(
                "frozen PRIDE SAS table is missing or duplicated"
            )
        payload = archive.read(SOURCE_TABLE)
    frame = read_sas(
        io.BytesIO(payload),
        format="sas7bdat",
        encoding=None,
    )
    columns = getattr(frame, "columns", ())
    if not set(REQUIRED_SOURCE_COLUMNS).issubset(columns):
        raise QualificationInputError(
            "SAS table differs from the frozen column contract"
        )
    rows = frame.loc[:, list(REQUIRED_SOURCE_COLUMNS)].to_dict(
        orient="records"
    )
    return build_qualification_records_from_rows(rows)


def _group_and_validate(
    records: Sequence[QualificationRecord],
    thresholds: QualificationThresholds,
) -> dict[str, list[QualificationRecord]]:
    grouped: dict[str, list[QualificationRecord]] = {}
    for record in records:
        numeric = (
            record.visit_month,
            record.baseline_weight_kg,
            record.observed_weight_kg,
            record.predicted_weight_kg,
        )
        if not record.participant_id or not all(
            isfinite(value) for value in numeric
        ):
            raise QualificationInputError(
                "participant id and required numeric fields must be finite"
            )
        if record.visit_month <= 0.0:
            raise QualificationInputError(
                "V11-MQ1 accepts post-baseline visits only"
            )
        if min(
            record.baseline_weight_kg,
            record.observed_weight_kg,
            record.predicted_weight_kg,
        ) <= 0.0:
            raise QualificationInputError("weights must be positive")
        grouped.setdefault(record.participant_id, []).append(record)

    for rows in grouped.values():
        rows.sort(key=lambda item: item.visit_month)
        if len({item.visit_month for item in rows}) != len(rows):
            raise QualificationInputError(
                "duplicate visit month within participant"
            )
        baseline = rows[0].baseline_weight_kg
        if any(
            abs(item.baseline_weight_kg - baseline) > 1e-9
            for item in rows
        ):
            raise QualificationInputError(
                "baseline weight drifts within participant"
            )

    return {
        participant: rows
        for participant, rows in grouped.items()
        if len(rows)
        >= thresholds.minimum_postbaseline_visits_per_participant
    }


def _bootstrap_bias_ci(
    participant_bias: np.ndarray,
    thresholds: QualificationThresholds,
) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(thresholds.bootstrap_seed))
    sample_size = participant_bias.size
    estimates = np.empty(thresholds.bootstrap_replicates, dtype=float)
    for index in range(thresholds.bootstrap_replicates):
        estimates[index] = np.mean(
            participant_bias[
                rng.integers(0, sample_size, size=sample_size)
            ]
        )
    lower, upper = np.quantile(
        estimates,
        (0.025, 0.975),
        method="linear",
    )
    return float(lower), float(upper)


def _trapezoidal_integral(
    values: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """NumPy-1.26-compatible explicit trapezoidal integration."""

    widths = np.diff(coordinates)
    return float(np.sum(widths * (values[:-1] + values[1:]) / 2.0))


def evaluate_model_qualification(
    records: Sequence[QualificationRecord],
    thresholds: QualificationThresholds = QualificationThresholds(),
) -> dict[str, object]:
    """Return the deterministic simultaneous V11-MQ1 point-model decision."""

    try:
        eligible = _group_and_validate(records, thresholds)
    except QualificationInputError as exc:
        return {
            "decision": "QUALIFICATION_INPUT_INVALID",
            "pass": False,
            "case_name": NONPASS_CASE_NAME,
            "reason": str(exc),
            "eligible_participants": 0,
            "eligible_postbaseline_records": 0,
            "metrics": None,
            "checks": None,
            "thresholds": asdict(thresholds),
            "prediction_interval_status": UQ_STATUS,
        }
    if len(eligible) < thresholds.minimum_participants:
        return {
            "decision": "MODEL_INPUT_INSUFFICIENT",
            "pass": False,
            "case_name": NONPASS_CASE_NAME,
            "reason": "eligible participant count is below the frozen minimum",
            "eligible_participants": len(eligible),
            "eligible_postbaseline_records": sum(
                len(rows) for rows in eligible.values()
            ),
            "metrics": None,
            "checks": None,
            "thresholds": asdict(thresholds),
            "prediction_interval_status": UQ_STATUS,
        }

    participant_mae: list[float] = []
    participant_mse: list[float] = []
    participant_bias: list[float] = []
    participant_niae: list[float] = []
    for rows in eligible.values():
        baseline = rows[0].baseline_weight_kg
        errors_percent = np.asarray(
            [
                100.0
                * (row.predicted_weight_kg - row.observed_weight_kg)
                / baseline
                for row in rows
            ]
        )
        participant_mae.append(float(np.mean(np.abs(errors_percent))))
        participant_mse.append(float(np.mean(errors_percent**2)))
        participant_bias.append(float(np.mean(errors_percent)))
        months = np.asarray(
            [row.visit_month for row in rows],
            dtype=float,
        )
        absolute_error_kg = np.asarray(
            [
                abs(row.predicted_weight_kg - row.observed_weight_kg)
                for row in rows
            ]
        )
        duration = months[-1] - months[0]
        if duration <= 0.0:
            return {
                "decision": "QUALIFICATION_INPUT_INVALID",
                "pass": False,
                "case_name": NONPASS_CASE_NAME,
                "reason": "trajectory duration must be positive",
                "eligible_participants": 0,
                "eligible_postbaseline_records": 0,
                "metrics": None,
                "checks": None,
                "thresholds": asdict(thresholds),
                "prediction_interval_status": UQ_STATUS,
            }
        participant_niae.append(
            float(
                100.0
                * _trapezoidal_integral(absolute_error_kg, months)
                / (baseline * duration)
            )
        )

    bias_values = np.asarray(participant_bias)
    bias_ci = _bootstrap_bias_ci(bias_values, thresholds)
    metrics = {
        "mae_percent": float(np.mean(participant_mae)),
        "rmse_percent": float(sqrt(np.mean(participant_mse))),
        "bias_percent": float(np.mean(bias_values)),
        "bias_percent_bootstrap_95_ci": [bias_ci[0], bias_ci[1]],
        "trajectory_niae_percent": float(np.mean(participant_niae)),
    }
    checks = {
        "mae": metrics["mae_percent"] <= thresholds.mae_percent_max,
        "rmse": metrics["rmse_percent"] <= thresholds.rmse_percent_max,
        "absolute_bias": (
            abs(metrics["bias_percent"])
            <= thresholds.absolute_bias_percent_max
        ),
        "bias_ci": (
            bias_ci[0] >= -thresholds.bias_ci_absolute_bound_percent
            and bias_ci[1] <= thresholds.bias_ci_absolute_bound_percent
        ),
        "trajectory": (
            metrics["trajectory_niae_percent"]
            <= thresholds.trajectory_niae_percent_max
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "MODEL_QUALIFICATION_PASSED"
            if passed
            else "MODEL_QUALIFICATION_FAILED"
        ),
        "pass": passed,
        "case_name": PASS_CASE_NAME if passed else NONPASS_CASE_NAME,
        "eligible_participants": len(eligible),
        "eligible_postbaseline_records": sum(
            len(rows) for rows in eligible.values()
        ),
        "metrics": metrics,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "prediction_interval_status": UQ_STATUS,
    }
