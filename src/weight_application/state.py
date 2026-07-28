"""Explicit energy-to-mass state dynamics for the R2 synthetic fixture.

This is an E0 software and mathematical correctness model. It is not a
qualified virtual human, participant model, clinical model, or effect model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


class WeightStateError(ValueError):
    """A synthetic state or transition violates its explicit unit contract."""


@dataclass(frozen=True)
class SyntheticWeightState:
    """Synthetic two-compartment state with mass in kilograms."""

    event_id: int
    fat_mass_kg: float
    lean_mass_kg: float
    cumulative_energy_imbalance_kcal: float

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise WeightStateError("event_id must be nonnegative")
        values = (
            self.fat_mass_kg,
            self.lean_mass_kg,
            self.cumulative_energy_imbalance_kcal,
        )
        if not all(isfinite(value) for value in values):
            raise WeightStateError("synthetic state values must be finite")
        if self.fat_mass_kg < 0.0 or self.lean_mass_kg < 0.0:
            raise WeightStateError("mass compartments must be nonnegative")
        if self.body_mass_kg <= 0.0:
            raise WeightStateError("body mass must be positive")

    @property
    def body_mass_kg(self) -> float:
        """Total mass is defined by, rather than independent from, compartments."""

        return self.fat_mass_kg + self.lean_mass_kg

    def to_dict(self) -> dict[str, float | int]:
        return {
            "event_id": self.event_id,
            "fat_mass_kg": self.fat_mass_kg,
            "lean_mass_kg": self.lean_mass_kg,
            "body_mass_kg": self.body_mass_kg,
            "cumulative_energy_imbalance_kcal": (
                self.cumulative_energy_imbalance_kcal
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SyntheticWeightState:
        state = cls(
            event_id=int(value["event_id"]),
            fat_mass_kg=float(value["fat_mass_kg"]),
            lean_mass_kg=float(value["lean_mass_kg"]),
            cumulative_energy_imbalance_kcal=float(
                value["cumulative_energy_imbalance_kcal"]
            ),
        )
        recorded_total = float(value["body_mass_kg"])
        if abs(state.body_mass_kg - recorded_total) > 1e-12:
            raise WeightStateError(
                "recorded body mass differs from compartment mass"
            )
        return state


@dataclass(frozen=True)
class SyntheticWeightProjection:
    """Finite projected compartments before feasibility is classified."""

    event_id: int
    fat_mass_kg: float
    lean_mass_kg: float
    cumulative_energy_imbalance_kcal: float
    energy_imbalance_kcal: float

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise WeightStateError("projected event_id must be nonnegative")
        values = (
            self.fat_mass_kg,
            self.lean_mass_kg,
            self.cumulative_energy_imbalance_kcal,
            self.energy_imbalance_kcal,
        )
        if not all(isfinite(value) for value in values):
            raise WeightStateError("projected state values must be finite")

    @property
    def body_mass_kg(self) -> float:
        return self.fat_mass_kg + self.lean_mass_kg

    def to_state(self) -> SyntheticWeightState:
        return SyntheticWeightState(
            event_id=self.event_id,
            fat_mass_kg=self.fat_mass_kg,
            lean_mass_kg=self.lean_mass_kg,
            cumulative_energy_imbalance_kcal=(
                self.cumulative_energy_imbalance_kcal
            ),
        )


@dataclass(frozen=True)
class SyntheticWeightModel:
    """Auditable linear energy-balance fixture with explicit units."""

    event_days: float
    energy_density_kcal_per_kg: float
    fat_mass_change_fraction: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.event_days)
            or not isfinite(self.energy_density_kcal_per_kg)
            or self.event_days <= 0.0
            or self.energy_density_kcal_per_kg <= 0.0
        ):
            raise WeightStateError(
                "event duration and energy density must be finite and positive"
            )
        if (
            not isfinite(self.fat_mass_change_fraction)
            or not 0.0 <= self.fat_mass_change_fraction <= 1.0
        ):
            raise WeightStateError(
                "fat mass change fraction must lie in [0, 1]"
            )

    def identity(self) -> Mapping[str, Any]:
        return {
            "model_id": "WGT-E0-LINEAR-ENERGY-MASS-FIXTURE",
            "model_version": "1.0.0",
            "qualification_status": "NOT_QUALIFIED_E0_CORRECTNESS_ONLY",
            "event_days": self.event_days,
            "energy_density_kcal_per_kg": (
                self.energy_density_kcal_per_kg
            ),
            "fat_mass_change_fraction": self.fat_mass_change_fraction,
            "action_units": (
                "intake_adjustment_kcal_per_day",
                "activity_expenditure_adjustment_kcal_per_day",
            ),
        }

    def project(
        self,
        state: SyntheticWeightState,
        action: Sequence[float],
    ) -> SyntheticWeightProjection:
        values = tuple(float(value) for value in action)
        if len(values) != 2 or not all(isfinite(value) for value in values):
            raise WeightStateError(
                "weight action must contain two finite energy rates"
            )
        intake_adjustment, activity_expenditure_adjustment = values
        energy_imbalance_kcal = self.event_days * (
            intake_adjustment - activity_expenditure_adjustment
        )
        mass_change_kg = (
            energy_imbalance_kcal / self.energy_density_kcal_per_kg
        )
        fat_change_kg = (
            self.fat_mass_change_fraction * mass_change_kg
        )
        lean_change_kg = mass_change_kg - fat_change_kg
        projection = SyntheticWeightProjection(
            event_id=state.event_id + 1,
            fat_mass_kg=state.fat_mass_kg + fat_change_kg,
            lean_mass_kg=state.lean_mass_kg + lean_change_kg,
            cumulative_energy_imbalance_kcal=(
                state.cumulative_energy_imbalance_kcal
                + energy_imbalance_kcal
            ),
            energy_imbalance_kcal=energy_imbalance_kcal,
        )
        expected_mass = state.body_mass_kg + mass_change_kg
        if abs(projection.body_mass_kg - expected_mass) > 1e-12:
            raise WeightStateError(
                "energy-derived mass change is not mass consistent"
            )
        return projection

    def transition(
        self,
        state: SyntheticWeightState,
        action: Sequence[float],
    ) -> SyntheticWeightState:
        """Commit only a feasible projected state to the outer trajectory."""

        return self.project(state, action).to_state()
