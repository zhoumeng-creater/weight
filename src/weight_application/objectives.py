"""L06 registered E0 objectives rewritten from the v1.1 specification.

The legacy scalar fitness and its weights are intentionally not copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WeightObjectiveRegistry:
    names: tuple[str, ...]
    units: tuple[str, ...]
    scalar_fitness_allowed: bool = False


SYNTHETIC_E0_OBJECTIVES = WeightObjectiveRegistry(
    names=("target_mass_error_kg", "intervention_burden_fraction"),
    units=("kg", "fraction"),
)


def evaluate_weight_objectives(
    *,
    predicted_body_mass_kg: float,
    target_mass_kg: float,
    intake_adjustment_kcal_per_day: float,
    activity_expenditure_adjustment_kcal_per_day: float,
) -> tuple[float, float]:
    values = (
        predicted_body_mass_kg,
        target_mass_kg,
        intake_adjustment_kcal_per_day,
        activity_expenditure_adjustment_kcal_per_day,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("weight objective inputs must be finite")
    target_error = abs(predicted_body_mass_kg - target_mass_kg)
    burden = (
        abs(intake_adjustment_kcal_per_day) / 1000.0
        + abs(activity_expenditure_adjustment_kcal_per_day) / 1000.0
    ) / 2.0
    return target_error, burden
