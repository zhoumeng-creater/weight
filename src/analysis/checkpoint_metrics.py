"""Frozen E1/E2 nHV and checkpoint endpoint calculations.

This module implements only result-stage pure functions.  It neither discovers
reference fronts from observed method outputs nor reads formal result roots.
Static/CDF normalization therefore requires independently derived exact
extrema.  The historical R5 10,000-point sampling target is provenance, not a
scale-validity condition, under the versioned R8C reference amendment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


# Historical R5 nominal sampling target.  Do not use this as a validity gate.
ANALYTIC_REFERENCE_FRONT_POINTS = 10_000
FULL_CHECKPOINT_FRACTIONS = tuple(index / 20 for index in range(21))
TRANSFER_CHECKPOINT_FRACTIONS = FULL_CHECKPOINT_FRACTIONS[:5]
NEGATIVE_TRANSFER_DIFFERENCE_THRESHOLD = -0.01
_STATIC_CDF_REFERENCE_COORDINATE = 1.1
_MIN_NORMALIZATION_RANGE = 1e-12


class FormalMetricError(ValueError):
    """An input violates the frozen R5 endpoint contract."""


def _finite_matrix(
    values: Sequence[Sequence[float]],
    *,
    dimension: int,
    field: str,
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FormalMetricError(f"{field} must be a numeric matrix") from error
    if matrix.size == 0:
        return np.empty((0, dimension), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise FormalMetricError(
            f"{field} must have shape (n, {dimension})"
        )
    if not np.all(np.isfinite(matrix)):
        raise FormalMetricError(f"{field} must contain only finite values")
    return matrix


def _reference_vector(reference_point: Sequence[float]) -> np.ndarray:
    try:
        reference = np.asarray(reference_point, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FormalMetricError(
            "reference_point must be a numeric vector"
        ) from error
    if (
        reference.ndim != 1
        or reference.size not in {2, 3}
        or not np.all(np.isfinite(reference))
    ):
        raise FormalMetricError(
            "reference_point must be a finite two- or three-dimensional vector"
        )
    return reference


def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    # Ascending x, then ascending y, makes a single decreasing-y scan both
    # remove dominated/duplicate points and accumulate the exact rectangle
    # union.  This is O(n log n), avoiding a quadratic dominance matrix at
    # every slice of the three-dimensional sweep.
    order = np.lexsort((points[:, 1], points[:, 0]))
    front = points[order]
    previous_y = float(reference[1])
    volume = 0.0
    for x_value, y_value in front:
        y = float(y_value)
        if y >= previous_y:
            continue
        volume += (float(reference[0]) - float(x_value)) * (
            previous_y - y
        )
        previous_y = y
    return float(volume)


def _hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    front = np.unique(points, axis=0)
    if len(front) == 0:
        return 0.0
    z_levels = np.unique(front[:, 2])
    volume = 0.0
    for index, lower_z_value in enumerate(z_levels):
        lower_z = float(lower_z_value)
        upper_z = (
            float(z_levels[index + 1])
            if index + 1 < len(z_levels)
            else float(reference[2])
        )
        if upper_z <= lower_z:
            continue
        active = front[front[:, 2] <= lower_z_value, :2]
        volume += _hypervolume_2d(active, reference[:2]) * (
            upper_z - lower_z
        )
    return float(volume)


def exact_hypervolume(
    front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
) -> float:
    """Return exact minimization hypervolume for a 2D or 3D front.

    Every point must weakly dominate the reference point componentwise.
    Points lying on the reference boundary are accepted but contribute zero.
    Dominated and duplicate points are removed deterministically.
    """

    reference = _reference_vector(reference_point)
    points = _finite_matrix(
        front,
        dimension=int(reference.size),
        field="front",
    )
    if len(points) == 0:
        return 0.0
    if np.any(points > reference):
        raise FormalMetricError(
            "front contains a coordinate beyond the reference point"
        )
    positive_boxes = points[np.all(points < reference, axis=1)]
    if reference.size == 2:
        return _hypervolume_2d(positive_boxes, reference)
    return _hypervolume_3d(positive_boxes, reference)


@dataclass(frozen=True)
class AnalyticReferenceScale:
    """Exact min/max scale derived independently of method outputs."""

    minima: tuple[float, ...]
    maxima: tuple[float, ...]
    point_count: int | None = None

    def __post_init__(self) -> None:
        if self.point_count is not None and (
            type(self.point_count) is not int or self.point_count < 1
        ):
            raise FormalMetricError(
                "reference point_count must be a positive integer when given"
            )
        if len(self.minima) not in {2, 3} or len(self.maxima) != len(
            self.minima
        ):
            raise FormalMetricError(
                "analytic reference scale must be two- or three-dimensional"
            )
        if not all(
            math.isfinite(float(value))
            for value in (*self.minima, *self.maxima)
        ):
            raise FormalMetricError(
                "analytic reference scale must contain only finite values"
            )
        if any(
            float(maximum) < float(minimum)
            for minimum, maximum in zip(self.minima, self.maxima, strict=True)
        ):
            raise FormalMetricError(
                "analytic reference maxima must not be below minima"
            )

    @property
    def objective_dimension(self) -> int:
        return len(self.minima)

    @classmethod
    def from_extrema(
        cls,
        *,
        minima: Sequence[float],
        maxima: Sequence[float],
        point_count: int | None = None,
    ) -> AnalyticReferenceScale:
        """Construct a scale directly from independently proven extrema."""

        return cls(
            minima=tuple(float(value) for value in minima),
            maxima=tuple(float(value) for value in maxima),
            point_count=point_count,
        )

    @classmethod
    def from_reference_front(
        cls,
        reference_front: Sequence[Sequence[float]],
    ) -> AnalyticReferenceScale:
        try:
            matrix = np.asarray(reference_front, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise FormalMetricError(
                "analytic reference front must be a numeric matrix"
            ) from error
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[1] not in {2, 3}
        ):
            raise FormalMetricError(
                "analytic reference front must be a nonempty matrix with "
                "two or three columns"
            )
        if not np.all(np.isfinite(matrix)):
            raise FormalMetricError(
                "analytic reference front must contain only finite values"
            )
        return cls(
            minima=tuple(float(value) for value in np.min(matrix, axis=0)),
            maxima=tuple(float(value) for value in np.max(matrix, axis=0)),
            point_count=int(matrix.shape[0]),
        )


def static_cdf_nhv(
    front: Sequence[Sequence[float]],
    reference_scale: AnalyticReferenceScale,
) -> float:
    """Compute frozen static/CDF nHV from an independent analytic scale."""

    points = _finite_matrix(
        front,
        dimension=reference_scale.objective_dimension,
        field="front",
    )
    if len(points) == 0:
        return 0.0
    minima = np.asarray(reference_scale.minima, dtype=np.float64)
    maxima = np.asarray(reference_scale.maxima, dtype=np.float64)
    ranges = np.maximum(maxima - minima, _MIN_NORMALIZATION_RANGE)
    normalized = np.clip(
        (points - minima) / ranges,
        0.0,
        _STATIC_CDF_REFERENCE_COORDINATE,
    )
    reference = np.full(
        reference_scale.objective_dimension,
        _STATIC_CDF_REFERENCE_COORDINATE,
        dtype=np.float64,
    )
    value = exact_hypervolume(normalized, reference) / (
        _STATIC_CDF_REFERENCE_COORDINATE
        ** reference_scale.objective_dimension
    )
    return float(np.clip(value, 0.0, 1.0))


def rolling_nhv(front: Sequence[Sequence[float]]) -> float:
    """Compute frozen three-objective WGT-RR nHV using f/(1+f)."""

    points = _finite_matrix(front, dimension=3, field="front")
    if len(points) == 0:
        return 0.0
    if np.any(points < 0.0):
        raise FormalMetricError(
            "rolling objectives must be nonnegative before phi transform"
        )
    transformed = points / (1.0 + points)
    value = exact_hypervolume(transformed, (1.0, 1.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _validated_curve(values: Sequence[float]) -> tuple[float, ...]:
    curve = tuple(float(value) for value in values)
    if len(curve) != len(FULL_CHECKPOINT_FRACTIONS):
        raise FormalMetricError("an event nHV curve must contain 21 values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in curve):
        raise FormalMetricError("event nHV values must be finite and in [0,1]")
    return curve


def _normalized_trapezoid(
    values: Sequence[float],
    fractions: Sequence[float],
) -> float:
    area = sum(
        (float(values[index]) + float(values[index + 1]))
        * 0.5
        * (float(fractions[index + 1]) - float(fractions[index]))
        for index in range(len(fractions) - 1)
    )
    width = float(fractions[-1]) - float(fractions[0])
    return float(area / width)


def event_anytime_auc(values: Sequence[float]) -> float:
    """Full-budget normalized trapezoidal AUC on the frozen 21 points."""

    curve = _validated_curve(values)
    return _normalized_trapezoid(curve, FULL_CHECKPOINT_FRACTIONS)


def event_early_auc(values: Sequence[float]) -> float:
    """First-20%-budget normalized trapezoidal AUC on the first five points."""

    curve = _validated_curve(values)
    return _normalized_trapezoid(
        curve[: len(TRANSFER_CHECKPOINT_FRACTIONS)],
        TRANSFER_CHECKPOINT_FRACTIONS,
    )


def _validated_event_curves(
    event_curves: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    curves = tuple(_validated_curve(curve) for curve in event_curves)
    if not curves:
        raise FormalMetricError("at least one event nHV curve is required")
    return curves


def e2_transfer_early_auc(
    event_curves: Sequence[Sequence[float]],
) -> float:
    """Mean early-AUC over post-initial events; event zero is excluded."""

    curves = _validated_event_curves(event_curves)
    if len(curves) < 2:
        raise FormalMetricError(
            "transfer early-AUC requires an initial and a post-change event"
        )
    return float(np.mean([event_early_auc(curve) for curve in curves[1:]]))


def negative_transfer_rate(
    proposed_event_curves: Sequence[Sequence[float]],
    comparator_event_curves: Sequence[Sequence[float]],
) -> float:
    """Fraction of post-initial paired events with early-AUC delta < -0.01."""

    proposed = _validated_event_curves(proposed_event_curves)
    comparator = _validated_event_curves(comparator_event_curves)
    if len(proposed) != len(comparator):
        raise FormalMetricError(
            "negative-transfer inputs must contain the same paired events"
        )
    if len(proposed) < 2:
        raise FormalMetricError(
            "negative-transfer rate requires post-initial events"
        )
    differences = [
        event_early_auc(proposed[index])
        - event_early_auc(comparator[index])
        for index in range(1, len(proposed))
    ]
    return float(
        np.mean(
            np.asarray(differences, dtype=np.float64)
            < NEGATIVE_TRANSFER_DIFFERENCE_THRESHOLD
        )
    )


@dataclass(frozen=True)
class E1E2SequenceEndpoints:
    """Frozen within-sequence aggregation before higher-level statistics."""

    anytime_nhv_auc: float
    final_nhv: float
    transfer_early_auc: float | None


def e1e2_sequence_endpoints(
    event_curves: Sequence[Sequence[float]],
    *,
    include_transfer: bool,
) -> E1E2SequenceEndpoints:
    """Aggregate event curves exactly as frozen for one E1/E2 sequence."""

    curves = _validated_event_curves(event_curves)
    anytime = float(np.mean([event_anytime_auc(curve) for curve in curves]))
    final = float(np.mean([curve[-1] for curve in curves]))
    transfer = e2_transfer_early_auc(curves) if include_transfer else None
    return E1E2SequenceEndpoints(
        anytime_nhv_auc=anytime,
        final_nhv=final,
        transfer_early_auc=transfer,
    )
