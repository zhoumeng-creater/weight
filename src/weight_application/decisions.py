"""L13 v1.1 weight-decision contract rewritten from the normative specification.

Provenance disposition:
    LEGACY_WEIGHT/solution_generator.py -> L13 REWRITE_FROM_SPEC
    Source SHA-256 eb99de876b467dcf357d315104c7222ad2bb5389a56835af691f89f8d2987168

No legacy silent fallback or mixed clinical-style decision vector is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class DecisionContractError(ValueError):
    """An E0 weight action violates its explicit dimension or bounds."""


@dataclass(frozen=True)
class WeightDecisionContract:
    names: tuple[str, ...]
    units: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        dimension = len(self.names)
        if (
            dimension == 0
            or len(self.units) != dimension
            or len(self.lower_bounds) != dimension
            or len(self.upper_bounds) != dimension
        ):
            raise DecisionContractError("decision fields must be nonempty and aligned")
        if any(
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower > upper
            for lower, upper in zip(
                self.lower_bounds,
                self.upper_bounds,
                strict=True,
            )
        ):
            raise DecisionContractError("decision bounds must be finite and ordered")

    @property
    def dimension(self) -> int:
        return len(self.names)

    def _array(self, vector: Sequence[float]) -> np.ndarray:
        try:
            values = np.asarray(vector, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DecisionContractError(
                "weight action must contain finite numeric values"
            ) from exc
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise DecisionContractError(
                "weight action must contain two finite components"
            )
        return values

    def validate(self, vector: Sequence[float]) -> tuple[float, ...]:
        values = self._array(vector)
        for value, lower, upper in zip(
            values,
            self.lower_bounds,
            self.upper_bounds,
            strict=True,
        ):
            if value < lower or value > upper:
                raise DecisionContractError("weight action is outside its bounds")
        return tuple(float(value) for value in values)

    def repair(self, vector: Sequence[float]) -> np.ndarray:
        values = self._array(vector)
        return np.clip(values, self.lower_bounds, self.upper_bounds)

    def neutral_action(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=float)


SYNTHETIC_E0_DECISIONS = WeightDecisionContract(
    names=(
        "intake_adjustment_kcal_per_day",
        "activity_expenditure_adjustment_kcal_per_day",
    ),
    units=("kcal/day", "kcal/day"),
    lower_bounds=(-1000.0, 0.0),
    upper_bounds=(1000.0, 1000.0),
)
