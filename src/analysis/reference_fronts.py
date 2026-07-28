"""Result-blind identities and compact exact reference-scale artifacts.

This module stores normalization extrema directly.  It never requires a
continuous Pareto front to be materialized at an arbitrary sample count.
When a true Pareto front is finite, every unique point is stored and the
extrema are derived from that complete finite set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np


REFERENCE_IDENTITY_VERSION = "WGT-V11-REFERENCE-IDENTITY-1.0.0"
FINITE_FRONT_COMPLETENESS_ASSERTION = "ALL_UNIQUE_TRUE_PARETO_POINTS"


class ReferenceArtifactError(ValueError):
    """A reference identity or scale is incomplete or non-canonical."""


def _is_sha256(value: str) -> bool:
    normalized = value.lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _float_tokens(values: Sequence[float]) -> list[str]:
    return [float(value).hex() for value in values]


def _validated_u64(value: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ReferenceArtifactError(
            "master_seed_u64 must be a canonical decimal string"
        )
    if not value.isdecimal() or (len(value) > 1 and value.startswith("0")):
        raise ReferenceArtifactError(
            "master_seed_u64 must be a canonical decimal string"
        )
    parsed = int(value)
    if not 0 <= parsed <= (1 << 64) - 1:
        raise ReferenceArtifactError("master_seed_u64 is outside uint64")
    return value


@dataclass(frozen=True)
class ReferenceIdentity:
    """Identity of an analytic or complete finite reference artifact."""

    suite_id: str
    problem_id: str
    evaluator_binding_sha256: str
    event_id: int = 0
    profile: str | None = None
    master_seed_u64: str | None = None
    time_vector: tuple[float, ...] | None = None
    identity_version: str = REFERENCE_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if not self.suite_id or not self.problem_id:
            raise ReferenceArtifactError(
                "reference suite_id and problem_id must be explicit"
            )
        if not _is_sha256(self.evaluator_binding_sha256):
            raise ReferenceArtifactError(
                "evaluator_binding_sha256 must be a lowercase SHA-256"
            )
        if self.evaluator_binding_sha256 != (
            self.evaluator_binding_sha256.lower()
        ):
            raise ReferenceArtifactError(
                "evaluator_binding_sha256 must be lowercase"
            )
        if type(self.event_id) is not int or self.event_id < 0:
            raise ReferenceArtifactError(
                "reference event_id must be a nonnegative integer"
            )
        if self.identity_version != REFERENCE_IDENTITY_VERSION:
            raise ReferenceArtifactError(
                "unknown reference identity version"
            )
        if self.master_seed_u64 is not None:
            _validated_u64(self.master_seed_u64)
        if self.time_vector is not None and (
            not self.time_vector
            or not all(math.isfinite(value) for value in self.time_vector)
        ):
            raise ReferenceArtifactError(
                "time_vector must contain finite values"
            )
        if self.problem_id == "CDF13":
            if not self.profile:
                raise ReferenceArtifactError(
                    "CDF13 reference identity requires profile"
                )
            if self.master_seed_u64 is None:
                raise ReferenceArtifactError(
                    "CDF13 reference identity requires master_seed_u64"
                )
            if self.time_vector is None or len(self.time_vector) != 5:
                raise ReferenceArtifactError(
                    "CDF13 reference identity requires a five-value "
                    "time_vector"
                )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "identity_version": self.identity_version,
            "suite_id": self.suite_id,
            "problem_id": self.problem_id,
            "event_id": self.event_id,
            "profile": self.profile,
            "master_seed_u64": self.master_seed_u64,
            "time_vector_hex": (
                None
                if self.time_vector is None
                else _float_tokens(self.time_vector)
            ),
            "evaluator_binding_sha256": self.evaluator_binding_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256(_canonical_json(self.canonical_payload())).hexdigest()

    @classmethod
    def for_cdf13(
        cls,
        evaluator: Any,
        *,
        event_id: int,
        master_seed_u64: str,
    ) -> ReferenceIdentity:
        """Bind CDF13 to the exact paired schedule used by its evaluator."""

        seed = _validated_u64(master_seed_u64)
        if getattr(evaluator, "problem_id", None) != "CDF13":
            raise ReferenceArtifactError(
                "CDF13 identity requires a CDF13 evaluator"
            )
        if int(getattr(evaluator, "environment_seed", -1)) != int(seed):
            raise ReferenceArtifactError(
                "CDF13 evaluator environment_seed differs from master seed"
            )
        release = evaluator.release_metadata(event_id)
        time_vector_value = release.get("current_time_vector")
        if not isinstance(time_vector_value, Sequence):
            raise ReferenceArtifactError(
                "CDF13 evaluator did not release its current time vector"
            )
        time_vector = tuple(float(value) for value in time_vector_value)
        return cls(
            suite_id="CDF-1-15",
            problem_id="CDF13",
            profile=str(evaluator.profile),
            event_id=event_id,
            master_seed_u64=seed,
            time_vector=time_vector,
            evaluator_binding_sha256=str(evaluator.binding_sha256),
        )


@dataclass(frozen=True)
class ExactReferenceExtrema:
    """Compact normalization scale derived independently of method outputs."""

    identity: ReferenceIdentity
    minima: tuple[float, ...]
    maxima: tuple[float, ...]
    derivation_id: str
    finite_point_count: int | None = None
    finite_front_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.minima) not in {2, 3} or len(self.maxima) != len(
            self.minima
        ):
            raise ReferenceArtifactError(
                "reference extrema must be two- or three-dimensional"
            )
        if not all(
            math.isfinite(float(value))
            for value in (*self.minima, *self.maxima)
        ):
            raise ReferenceArtifactError(
                "reference extrema must contain finite values"
            )
        if any(
            float(maximum) < float(minimum)
            for minimum, maximum in zip(
                self.minima,
                self.maxima,
                strict=True,
            )
        ):
            raise ReferenceArtifactError(
                "reference maxima must not be below minima"
            )
        if not self.derivation_id:
            raise ReferenceArtifactError(
                "reference extrema require an explicit derivation_id"
            )
        finite_fields = (
            self.finite_point_count is not None,
            self.finite_front_sha256 is not None,
        )
        if finite_fields[0] != finite_fields[1]:
            raise ReferenceArtifactError(
                "finite point count and front hash must appear together"
            )
        if self.finite_point_count is not None:
            if (
                type(self.finite_point_count) is not int
                or self.finite_point_count < 1
                or not _is_sha256(str(self.finite_front_sha256))
            ):
                raise ReferenceArtifactError(
                    "finite front provenance is invalid"
                )

    @property
    def objective_dimension(self) -> int:
        return len(self.minima)

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "reference_identity_sha256": self.identity.identity_sha256,
            "minima_hex": _float_tokens(self.minima),
            "maxima_hex": _float_tokens(self.maxima),
            "derivation_id": self.derivation_id,
            "finite_point_count": self.finite_point_count,
            "finite_front_sha256": self.finite_front_sha256,
        }
        return sha256(_canonical_json(payload)).hexdigest()

    def to_analytic_reference_scale(self) -> Any:
        """Convert to the existing nHV scale without fabricating points."""

        from .checkpoint_metrics import AnalyticReferenceScale

        return AnalyticReferenceScale.from_extrema(
            minima=self.minima,
            maxima=self.maxima,
            point_count=self.finite_point_count,
        )


@dataclass(frozen=True)
class FiniteParetoFront:
    """Complete, canonical storage for a mathematically finite true PF."""

    identity: ReferenceIdentity
    points: tuple[tuple[float, ...], ...]
    derivation_id: str
    completeness_assertion: str = FINITE_FRONT_COMPLETENESS_ASSERTION

    def __post_init__(self) -> None:
        if not self.points:
            raise ReferenceArtifactError(
                "finite Pareto front must contain at least one point"
            )
        dimension = len(self.points[0])
        if dimension not in {2, 3}:
            raise ReferenceArtifactError(
                "finite Pareto front must be two- or three-dimensional"
            )
        if any(len(point) != dimension for point in self.points):
            raise ReferenceArtifactError(
                "finite Pareto front points have inconsistent dimensions"
            )
        if not all(
            math.isfinite(float(value))
            for point in self.points
            for value in point
        ):
            raise ReferenceArtifactError(
                "finite Pareto front must contain finite values"
            )
        canonical = tuple(sorted(set(self.points)))
        if self.points != canonical:
            raise ReferenceArtifactError(
                "finite Pareto front must contain every point once in "
                "lexicographic order"
            )
        if not self.derivation_id:
            raise ReferenceArtifactError(
                "finite Pareto front requires an explicit derivation_id"
            )
        if (
            self.completeness_assertion
            != FINITE_FRONT_COMPLETENESS_ASSERTION
        ):
            raise ReferenceArtifactError(
                "finite Pareto front lacks the completeness assertion"
            )

    @classmethod
    def from_points(
        cls,
        *,
        identity: ReferenceIdentity,
        points: Sequence[Sequence[float]],
        derivation_id: str,
    ) -> FiniteParetoFront:
        """Deduplicate and sort the supplied complete finite true PF."""

        try:
            matrix = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ReferenceArtifactError(
                "finite Pareto front must be a numeric matrix"
            ) from error
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[1] not in {2, 3}
            or not np.all(np.isfinite(matrix))
        ):
            raise ReferenceArtifactError(
                "finite Pareto front must be a nonempty finite matrix"
            )
        unique_points = tuple(
            sorted(
                {
                    tuple(float(value) for value in row)
                    for row in matrix.tolist()
                }
            )
        )
        return cls(
            identity=identity,
            points=unique_points,
            derivation_id=derivation_id,
        )

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "reference_identity_sha256": self.identity.identity_sha256,
            "points_hex": [
                _float_tokens(point)
                for point in self.points
            ],
            "derivation_id": self.derivation_id,
            "completeness_assertion": self.completeness_assertion,
        }
        return sha256(_canonical_json(payload)).hexdigest()

    def extrema(self) -> ExactReferenceExtrema:
        matrix = np.asarray(self.points, dtype=np.float64)
        return ExactReferenceExtrema(
            identity=self.identity,
            minima=tuple(float(value) for value in np.min(matrix, axis=0)),
            maxima=tuple(float(value) for value in np.max(matrix, axis=0)),
            derivation_id=(
                f"{self.derivation_id}/EXACT_EXTREMA_OF_COMPLETE_FINITE_PF"
            ),
            finite_point_count=len(self.points),
            finite_front_sha256=self.artifact_sha256,
        )
