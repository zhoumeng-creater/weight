"""N03 explicit model-internal constraints for the synthetic E0 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WeightConstraintRegistry:
    names: tuple[str, ...]
    units: tuple[str, ...]
    feasibility_rule: str
    role: str
    clinical_safety_claim: bool


SYNTHETIC_E0_CONSTRAINTS = WeightConstraintRegistry(
    names=(
        "minimum_body_mass",
        "maximum_daily_energy_imbalance",
        "nonnegative_fat_mass",
        "nonnegative_lean_mass",
    ),
    units=("kg", "kcal/day", "kg", "kg"),
    feasibility_rule="c_i <= 0",
    role="model_internal_safety_related_constraints",
    clinical_safety_claim=False,
)


def evaluate_weight_constraints(
    *,
    predicted_body_mass_kg: float,
    predicted_fat_mass_kg: float,
    predicted_lean_mass_kg: float,
    minimum_body_mass_kg: float,
    daily_energy_imbalance_kcal: float,
    maximum_daily_energy_imbalance_kcal: float,
) -> tuple[float, float, float, float]:
    values = (
        predicted_body_mass_kg,
        predicted_fat_mass_kg,
        predicted_lean_mass_kg,
        minimum_body_mass_kg,
        daily_energy_imbalance_kcal,
        maximum_daily_energy_imbalance_kcal,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("weight constraint inputs must be finite")
    if minimum_body_mass_kg <= 0.0 or maximum_daily_energy_imbalance_kcal <= 0.0:
        raise ValueError("weight constraint thresholds must be positive")
    return (
        minimum_body_mass_kg - predicted_body_mass_kg,
        abs(daily_energy_imbalance_kcal)
        - maximum_daily_energy_imbalance_kcal,
        -predicted_fat_mass_kg,
        -predicted_lean_mass_kg,
    )
