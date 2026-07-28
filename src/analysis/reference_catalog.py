"""Equation-derived, result-blind reference catalog for formal E1/E2.

The formal nHV scale needs only objective-wise extrema.  Continuous Pareto
fronts are therefore represented by their independently derived extrema and
small numerical certificates, never by an arbitrary dense point cloud.  A
front is materialized only when the mathematical front is genuinely finite;
in that case every unique nondominated point is retained.

The derivations in this module are bound to the corrective R8C evaluators:

* LIR-CMOP1--14: the version-of-record Table 8 equations;
* CDF1--15: the authors' CMLSGA commit used as operational authority.

No function in this module reads method outputs or experiment result roots.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import (
    ceil,
    cos,
    floor,
    isfinite,
    pi,
    sin,
    sqrt,
)
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_adapters.cdf_operational import (
    CDFOperationalEvaluator,
    CDF_OPERATIONAL_SUITE_ID,
)
from benchmark_adapters.lircmop_paper import (
    LIRCMOPPaperEvaluator,
    LIRCMOP_PAPER_SUITE_ID,
)

from .reference_fronts import (
    ExactReferenceExtrema,
    FiniteParetoFront,
    ReferenceArtifactError,
    ReferenceIdentity,
)


REFERENCE_CATALOG_ID = "WGT-V11-R8C-E1E2-REFERENCE-CATALOG-01"
REFERENCE_CATALOG_VERSION = "1.0.0"
REFERENCE_CATALOG_EXPECTED_IDENTITIES = 2_294
REFERENCE_ROOT_ABSOLUTE_TOLERANCE = 2.0**-48
REFERENCE_FEASIBILITY_TOLERANCE = 2.0**-44
REFERENCE_GLOBAL_GRID_INTERVALS = 1 << 17
REFERENCE_ROOT_GRID_INTERVALS = 1 << 15
CDF_REFERENCE_SEEDS = (
    "1814705672717120344",
    "11510044127855585889",
    "2013063862857590834",
    "9940308221477475016",
    "10545341458691982268",
)

_CDF_PROFILE_SEVERITY = {"CDF-HARSH": 5, "CDF-MILD": 10}
_LIR_WAVE_CONSTRAINTS = {
    9: (1.4, 1.4, 1.5, 6.0, 2.0),
    10: (1.1, 1.2, 2.0, 4.0, 1.0),
    11: (1.2, 1.2, 1.5, 5.0, 2.1),
    12: (1.6, 1.6, 1.5, 6.0, 2.5),
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _float_hex(value: float) -> str:
    number = float(value)
    if not isfinite(number):
        raise ReferenceArtifactError(
            "reference certificate contains a nonfinite value"
        )
    return number.hex()


def _hex_vector(values: Sequence[float]) -> list[str]:
    return [_float_hex(value) for value in values]


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReferenceDerivation:
    """A compact scale plus its optional complete finite true front."""

    extrema: ExactReferenceExtrema
    finite_front: FiniteParetoFront | None
    certificate: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.finite_front is None:
            if self.extrema.finite_point_count is not None:
                raise ReferenceArtifactError(
                    "finite extrema require their complete finite front"
                )
        else:
            if self.finite_front.identity != self.extrema.identity:
                raise ReferenceArtifactError(
                    "finite front and extrema identities differ"
                )
            if self.finite_front.extrema() != self.extrema:
                raise ReferenceArtifactError(
                    "finite-front extrema were not derived from the front"
                )

    def canonical_record(self) -> dict[str, Any]:
        identity_payload = self.extrema.identity.canonical_payload()
        extrema_payload: dict[str, Any] = {
            "minima_hex": _hex_vector(self.extrema.minima),
            "maxima_hex": _hex_vector(self.extrema.maxima),
            "derivation_id": self.extrema.derivation_id,
            "finite_point_count": self.extrema.finite_point_count,
            "finite_front_sha256": self.extrema.finite_front_sha256,
            "artifact_sha256": self.extrema.artifact_sha256,
        }
        finite_payload: dict[str, Any] | None = None
        if self.finite_front is not None:
            finite_payload = {
                "points_hex": [
                    _hex_vector(point)
                    for point in self.finite_front.points
                ],
                "derivation_id": self.finite_front.derivation_id,
                "completeness_assertion": (
                    self.finite_front.completeness_assertion
                ),
                "artifact_sha256": self.finite_front.artifact_sha256,
            }
        payload: dict[str, Any] = {
            "catalog_version": REFERENCE_CATALOG_VERSION,
            "identity": identity_payload,
            "identity_sha256": self.extrema.identity.identity_sha256,
            "extrema": extrema_payload,
            "finite_front": finite_payload,
            "certificate": dict(self.certificate),
        }
        payload["record_sha256"] = sha256(
            _canonical_json(payload)
        ).hexdigest()
        return payload


def _identity_from_payload(payload: Mapping[str, Any]) -> ReferenceIdentity:
    time_tokens = payload.get("time_vector_hex")
    if time_tokens is None:
        time_vector = None
    elif (
        isinstance(time_tokens, list)
        and time_tokens
        and all(isinstance(value, str) for value in time_tokens)
    ):
        try:
            time_vector = tuple(float.fromhex(value) for value in time_tokens)
        except ValueError as error:
            raise ReferenceArtifactError(
                "reference time vector contains an invalid float token"
            ) from error
    else:
        raise ReferenceArtifactError(
            "reference time_vector_hex is malformed"
        )
    return ReferenceIdentity(
        identity_version=str(payload.get("identity_version", "")),
        suite_id=str(payload.get("suite_id", "")),
        problem_id=str(payload.get("problem_id", "")),
        event_id=payload.get("event_id"),  # type: ignore[arg-type]
        profile=payload.get("profile"),  # type: ignore[arg-type]
        master_seed_u64=payload.get("master_seed_u64"),  # type: ignore[arg-type]
        time_vector=time_vector,
        evaluator_binding_sha256=str(
            payload.get("evaluator_binding_sha256", "")
        ),
    )


def reference_derivation_from_record(
    record: Mapping[str, Any],
) -> ReferenceDerivation:
    """Validate and reconstruct one canonical catalog record."""

    if record.get("catalog_version") != REFERENCE_CATALOG_VERSION:
        raise ReferenceArtifactError("unknown reference catalog version")
    claimed_record_hash = record.get("record_sha256")
    if not isinstance(claimed_record_hash, str):
        raise ReferenceArtifactError("reference record lacks its SHA-256")
    unhashed = dict(record)
    del unhashed["record_sha256"]
    if sha256(_canonical_json(unhashed)).hexdigest() != claimed_record_hash:
        raise ReferenceArtifactError("reference record SHA-256 mismatch")

    identity_payload = record.get("identity")
    extrema_payload = record.get("extrema")
    finite_payload = record.get("finite_front")
    certificate = record.get("certificate")
    if not isinstance(identity_payload, Mapping):
        raise ReferenceArtifactError("reference identity is malformed")
    if not isinstance(extrema_payload, Mapping):
        raise ReferenceArtifactError("reference extrema are malformed")
    if not isinstance(certificate, Mapping):
        raise ReferenceArtifactError("reference certificate is malformed")

    identity = _identity_from_payload(identity_payload)
    if record.get("identity_sha256") != identity.identity_sha256:
        raise ReferenceArtifactError("reference identity SHA-256 mismatch")
    try:
        minima = tuple(
            float.fromhex(value)
            for value in extrema_payload["minima_hex"]
        )
        maxima = tuple(
            float.fromhex(value)
            for value in extrema_payload["maxima_hex"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceArtifactError(
            "reference extrema contain invalid float tokens"
        ) from error

    finite_front: FiniteParetoFront | None = None
    if finite_payload is not None:
        if not isinstance(finite_payload, Mapping):
            raise ReferenceArtifactError(
                "finite reference front is malformed"
            )
        try:
            points = tuple(
                tuple(float.fromhex(value) for value in point)
                for point in finite_payload["points_hex"]
            )
            derivation_id = str(finite_payload["derivation_id"])
            completeness = str(finite_payload["completeness_assertion"])
        except (KeyError, TypeError, ValueError) as error:
            raise ReferenceArtifactError(
                "finite reference front contains invalid tokens"
            ) from error
        finite_front = FiniteParetoFront(
            identity=identity,
            points=points,
            derivation_id=derivation_id,
            completeness_assertion=completeness,
        )
        if (
            finite_payload.get("artifact_sha256")
            != finite_front.artifact_sha256
        ):
            raise ReferenceArtifactError(
                "finite reference front SHA-256 mismatch"
            )

    extrema = ExactReferenceExtrema(
        identity=identity,
        minima=minima,
        maxima=maxima,
        derivation_id=str(extrema_payload.get("derivation_id", "")),
        finite_point_count=extrema_payload.get(  # type: ignore[arg-type]
            "finite_point_count"
        ),
        finite_front_sha256=extrema_payload.get(  # type: ignore[arg-type]
            "finite_front_sha256"
        ),
    )
    if extrema_payload.get("artifact_sha256") != extrema.artifact_sha256:
        raise ReferenceArtifactError("reference extrema SHA-256 mismatch")
    return ReferenceDerivation(
        extrema=extrema,
        finite_front=finite_front,
        certificate=dict(certificate),
    )


def load_reference_catalog(
    catalog_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_lines: int | None = None,
) -> tuple[ReferenceDerivation, ...]:
    """Load a hash-bound JSONL catalog without consulting result artifacts."""

    raw = catalog_path.read_bytes()
    if expected_sha256 is not None and sha256(raw).hexdigest() != (
        expected_sha256
    ):
        raise ReferenceArtifactError("reference catalog file SHA-256 mismatch")
    if not raw.endswith(b"\n"):
        raise ReferenceArtifactError(
            "reference catalog must end with one LF"
        )
    lines = raw.splitlines()
    if expected_lines is not None and len(lines) != expected_lines:
        raise ReferenceArtifactError("reference catalog line count mismatch")
    derivations: list[ReferenceDerivation] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReferenceArtifactError(
                f"invalid JSON on reference catalog line {line_number}"
            ) from error
        if not isinstance(value, Mapping):
            raise ReferenceArtifactError(
                f"reference catalog line {line_number} is not an object"
            )
        derivations.append(reference_derivation_from_record(value))
    identities = [
        derivation.extrema.identity.identity_sha256
        for derivation in derivations
    ]
    if len(identities) != len(set(identities)):
        raise ReferenceArtifactError(
            "reference catalog contains duplicate identities"
        )
    return tuple(derivations)


def _finite_derivation(
    *,
    identity: ReferenceIdentity,
    points: Sequence[Sequence[float]],
    derivation_id: str,
    certificate: Mapping[str, Any],
) -> ReferenceDerivation:
    front = FiniteParetoFront.from_points(
        identity=identity,
        points=points,
        derivation_id=derivation_id,
    )
    return ReferenceDerivation(
        extrema=front.extrema(),
        finite_front=front,
        certificate=certificate,
    )


def _continuous_derivation(
    *,
    identity: ReferenceIdentity,
    minima: Sequence[float],
    maxima: Sequence[float],
    derivation_id: str,
    certificate: Mapping[str, Any],
) -> ReferenceDerivation:
    return ReferenceDerivation(
        extrema=ExactReferenceExtrema(
            identity=identity,
            minima=tuple(float(value) for value in minima),
            maxima=tuple(float(value) for value in maxima),
            derivation_id=derivation_id,
        ),
        finite_front=None,
        certificate=certificate,
    )


def _ellipse_constraint(
    f1: float | np.ndarray,
    f2: float | np.ndarray,
    *,
    p: float,
    q: float,
    a: float,
    b: float,
) -> float | np.ndarray:
    theta = -0.25 * pi
    first = (f1 - p) * cos(theta) - (f2 - q) * sin(theta)
    second = (f1 - p) * sin(theta) + (f2 - q) * cos(theta)
    return first * first / (a * a) + second * second / (b * b) - 0.1


def _refine_boolean_transition(
    predicate: Callable[[float], bool],
    left: float,
    right: float,
    *,
    left_is_feasible: bool,
) -> tuple[float, tuple[float, float]]:
    if not left < right:
        raise ReferenceArtifactError("invalid reference root bracket")
    if predicate(left) != left_is_feasible:
        raise ReferenceArtifactError("left reference bracket state drifted")
    if predicate(right) == left_is_feasible:
        raise ReferenceArtifactError(
            "reference bracket does not contain a state transition"
        )
    lower = float(left)
    upper = float(right)
    while upper - lower > REFERENCE_ROOT_ABSOLUTE_TOLERANCE:
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint == lower or midpoint == upper:
            break
        if predicate(midpoint) == left_is_feasible:
            lower = midpoint
        else:
            upper = midpoint
    value = lower if left_is_feasible else upper
    if not predicate(value):
        raise ReferenceArtifactError(
            "reference root refinement did not return the feasible side"
        )
    return value, (lower, upper)


def _feasible_x_extrema(
    constraint: Callable[[np.ndarray], np.ndarray],
    scalar_constraint: Callable[[float], float],
) -> tuple[float, float, dict[str, Any]]:
    grid = np.linspace(
        0.0,
        1.0,
        REFERENCE_ROOT_GRID_INTERVALS + 1,
        dtype=np.float64,
    )
    values = np.asarray(constraint(grid), dtype=np.float64)
    if values.shape != grid.shape or not np.all(np.isfinite(values)):
        raise ReferenceArtifactError(
            "reference feasibility scan returned invalid values"
        )
    feasible = values >= -REFERENCE_FEASIBILITY_TOLERANCE
    indices = np.flatnonzero(feasible)
    if not len(indices):
        raise ReferenceArtifactError(
            "equation-derived reference has no feasible ideal point"
        )

    def predicate(value: float) -> bool:
        result = float(scalar_constraint(float(value)))
        return (
            isfinite(result)
            and result >= -REFERENCE_FEASIBILITY_TOLERANCE
        )

    first = int(indices[0])
    last = int(indices[-1])
    first_bracket: tuple[float, float] | None = None
    last_bracket: tuple[float, float] | None = None
    if first == 0:
        minimum = 0.0
    else:
        minimum, first_bracket = _refine_boolean_transition(
            predicate,
            float(grid[first - 1]),
            float(grid[first]),
            left_is_feasible=False,
        )
    if last == REFERENCE_ROOT_GRID_INTERVALS:
        maximum = 1.0
    else:
        maximum, last_bracket = _refine_boolean_transition(
            predicate,
            float(grid[last]),
            float(grid[last + 1]),
            left_is_feasible=True,
        )
    certificate = {
        "method": (
            "ANALYTIC_IDEAL_CURVE_FEASIBILITY_PLUS_DETERMINISTIC_"
            "TRANSITION_BRACKETING"
        ),
        "scan_intervals": REFERENCE_ROOT_GRID_INTERVALS,
        "feasibility_tolerance_hex": _float_hex(
            REFERENCE_FEASIBILITY_TOLERANCE
        ),
        "root_absolute_tolerance_hex": _float_hex(
            REFERENCE_ROOT_ABSOLUTE_TOLERANCE
        ),
        "minimum_x_bracket_hex": (
            None
            if first_bracket is None
            else _hex_vector(first_bracket)
        ),
        "maximum_x_bracket_hex": (
            None
            if last_bracket is None
            else _hex_vector(last_bracket)
        ),
    }
    return minimum, maximum, certificate


def _first_feasible_axis_value(
    constraints: Callable[[np.ndarray], np.ndarray],
    scalar_constraints: Callable[[float], Sequence[float]],
    *,
    lower: float,
    upper: float = 8.0,
) -> tuple[float, dict[str, Any]]:
    grid = np.linspace(
        lower,
        upper,
        REFERENCE_ROOT_GRID_INTERVALS + 1,
        dtype=np.float64,
    )
    values = np.asarray(constraints(grid), dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != grid.size
        or not np.all(np.isfinite(values))
    ):
        raise ReferenceArtifactError(
            "axis feasibility scan returned invalid values"
        )
    feasible = np.all(
        values >= -REFERENCE_FEASIBILITY_TOLERANCE,
        axis=1,
    )
    indices = np.flatnonzero(feasible)
    if not len(indices):
        raise ReferenceArtifactError(
            "no feasible objective-axis endpoint was bracketed"
        )

    def predicate(value: float) -> bool:
        evaluated = tuple(float(item) for item in scalar_constraints(value))
        return (
            all(isfinite(item) for item in evaluated)
            and all(
                item >= -REFERENCE_FEASIBILITY_TOLERANCE
                for item in evaluated
            )
        )

    first = int(indices[0])
    bracket: tuple[float, float] | None = None
    if first == 0:
        result = float(lower)
    else:
        result, bracket = _refine_boolean_transition(
            predicate,
            float(grid[first - 1]),
            float(grid[first]),
            left_is_feasible=False,
        )
    return result, {
        "scan_lower_hex": _float_hex(lower),
        "scan_upper_hex": _float_hex(upper),
        "scan_intervals": REFERENCE_ROOT_GRID_INTERVALS,
        "root_bracket_hex": (
            None if bracket is None else _hex_vector(bracket)
        ),
        "root_absolute_tolerance_hex": _float_hex(
            REFERENCE_ROOT_ABSOLUTE_TOLERANCE
        ),
        "feasibility_tolerance_hex": _float_hex(
            REFERENCE_FEASIBILITY_TOLERANCE
        ),
    }


def derive_lircmop_reference(problem_index: int) -> ReferenceDerivation:
    """Derive one compact reference scale for the paper-faithful LIR-CMOP."""

    evaluator = LIRCMOPPaperEvaluator(problem_index)
    identity = ReferenceIdentity(
        suite_id=LIRCMOP_PAPER_SUITE_ID,
        problem_id=evaluator.problem_id,
        evaluator_binding_sha256=evaluator.binding_sha256,
    )
    base = 0.7057
    common = {
        "authority": "FAN_2019_TABLE_8",
        "decision_bound_reachability_checked": True,
        "observed_method_outputs_used": False,
    }

    if problem_index in {1, 2}:
        return _continuous_derivation(
            identity=identity,
            minima=(0.5, 0.5),
            maxima=(1.5, 1.5),
            derivation_id=f"LIRCMOP{problem_index}_CLOSED_FORM_EXTREMA_V1",
            certificate={
                **common,
                "rule": "G1_EQUALS_G2_EQUALS_0.5_ON_BOUNDARY",
            },
        )
    if problem_index in {3, 4}:
        x_low = 1.0 / 120.0
        x_high = 113.0 / 120.0
        shape = (
            (lambda value: 1.0 - value * value)
            if problem_index == 3
            else (lambda value: 1.0 - sqrt(value))
        )
        return _continuous_derivation(
            identity=identity,
            minima=(0.5 + x_low, 0.5 + shape(x_high)),
            maxima=(0.5 + x_high, 0.5 + shape(x_low)),
            derivation_id=(
                f"LIRCMOP{problem_index}_ANALYTIC_SINE_INTERVAL_EXTREMA_V1"
            ),
            certificate={
                **common,
                "rule": "SIN_20PI_X_GTE_HALF",
                "minimum_x_hex": _float_hex(x_low),
                "maximum_x_hex": _float_hex(x_high),
            },
        )
    if problem_index in {5, 6}:
        return _continuous_derivation(
            identity=identity,
            minima=(base, base),
            maxima=(1.0 + base, 1.0 + base),
            derivation_id=f"LIRCMOP{problem_index}_CLOSED_FORM_EXTREMA_V1",
            certificate={
                **common,
                "rule": "BOTH_IDEAL_AXIS_ENDPOINTS_FEASIBLE",
            },
        )
    if problem_index in {7, 8}:
        p_values = (1.2, 2.25, 3.5)
        q_values = (1.2, 2.25, 3.5)
        a_values = (2.0, 2.5, 2.5)
        b_values = (6.0, 12.0, 10.0)

        def vector_constraints(values: np.ndarray) -> np.ndarray:
            columns = [
                np.asarray(
                    _ellipse_constraint(
                        base,
                        values,
                        p=p,
                        q=q,
                        a=a,
                        b=b,
                    ),
                    dtype=np.float64,
                )
                for p, q, a, b in zip(
                    p_values,
                    q_values,
                    a_values,
                    b_values,
                    strict=True,
                )
            ]
            return np.column_stack(columns)

        def scalar_constraints(value: float) -> tuple[float, ...]:
            return tuple(
                float(
                    _ellipse_constraint(
                        base,
                        value,
                        p=p,
                        q=q,
                        a=a,
                        b=b,
                    )
                )
                for p, q, a, b in zip(
                    p_values,
                    q_values,
                    a_values,
                    b_values,
                    strict=True,
                )
            )

        axis, root_certificate = _first_feasible_axis_value(
            vector_constraints,
            scalar_constraints,
            lower=1.0 + base,
        )
        return _continuous_derivation(
            identity=identity,
            minima=(base, base),
            maxima=(axis, axis),
            derivation_id=(
                f"LIRCMOP{problem_index}_REACHABLE_AXIS_ROOT_EXTREMA_V1"
            ),
            certificate={
                **common,
                "rule": (
                    "FIRST_OBJECTIVE_AXIS_VALUE_OUTSIDE_ALL_THREE_ELLIPSES"
                ),
                "axis_symmetry_used": True,
                **root_certificate,
            },
        )
    if problem_index in {9, 10, 11, 12}:
        p, q, a, b, offset = _LIR_WAVE_CONSTRAINTS[problem_index]
        alpha = 0.25 * pi

        def objective_constraints(
            f1: float | np.ndarray,
            f2: float | np.ndarray,
        ) -> tuple[float | np.ndarray, float | np.ndarray]:
            ellipse = _ellipse_constraint(
                f1,
                f2,
                p=p,
                q=q,
                a=a,
                b=b,
            )
            wave = (
                f1 * sin(alpha)
                + f2 * cos(alpha)
                - np.sin(
                    4.0
                    * pi
                    * (f1 * cos(alpha) - f2 * sin(alpha))
                )
                - offset
            )
            return ellipse, wave

        def y_vector(values: np.ndarray) -> np.ndarray:
            first, second = objective_constraints(0.0, values)
            return np.column_stack((first, second))

        def y_scalar(value: float) -> tuple[float, float]:
            first, second = objective_constraints(0.0, value)
            return float(first), float(second)

        def x_vector(values: np.ndarray) -> np.ndarray:
            first, second = objective_constraints(values, 0.0)
            return np.column_stack((first, second))

        def x_scalar(value: float) -> tuple[float, float]:
            first, second = objective_constraints(value, 0.0)
            return float(first), float(second)

        y_axis, y_certificate = _first_feasible_axis_value(
            y_vector,
            y_scalar,
            lower=1.7057,
        )
        x_axis, x_certificate = _first_feasible_axis_value(
            x_vector,
            x_scalar,
            lower=1.7057,
        )
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(x_axis, y_axis),
            derivation_id=(
                f"LIRCMOP{problem_index}_REACHABLE_AXIS_ROOT_EXTREMA_V1"
            ),
            certificate={
                **common,
                "rule": (
                    "FIRST_REACHABLE_VALUE_ON_EACH_AXIS_SATISFYING_"
                    "ELLIPSE_AND_WAVE"
                ),
                "x_axis": x_certificate,
                "y_axis": y_certificate,
            },
        )
    if problem_index == 13:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0, 0.0),
            maxima=(1.7057, 1.7057, 1.7057),
            derivation_id="LIRCMOP13_MINIMUM_FEASIBLE_SPHERE_EXTREMA_V1",
            certificate={
                **common,
                "rule": "FIRST_OCTANT_SPHERE_RADIUS_1.7057",
            },
        )
    if problem_index == 14:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0, 0.0),
            maxima=(1.75, 1.75, 1.75),
            derivation_id="LIRCMOP14_MINIMUM_FEASIBLE_SPHERE_EXTREMA_V1",
            certificate={
                **common,
                "rule": "FIRST_OCTANT_SPHERE_RADIUS_1.75",
            },
        )
    raise ReferenceArtifactError("LIR-CMOP index must be in 1..14")


def _cdf_identity(
    evaluator: CDFOperationalEvaluator,
    *,
    event_id: int,
    master_seed_u64: str | None,
) -> ReferenceIdentity:
    if evaluator.problem_index == 13:
        if master_seed_u64 is None:
            raise ReferenceArtifactError(
                "CDF13 reference requires its master seed"
            )
        time_vector = evaluator._time_vector(event_id)
    else:
        if master_seed_u64 is not None:
            raise ReferenceArtifactError(
                "only CDF13 reference identities include a seed"
            )
        time_vector = None
    return ReferenceIdentity(
        suite_id=CDF_OPERATIONAL_SUITE_ID,
        problem_id=evaluator.problem_id,
        evaluator_binding_sha256=evaluator.binding_sha256,
        event_id=event_id,
        profile=evaluator.profile,
        master_seed_u64=master_seed_u64,
        time_vector=time_vector,
    )


def _linked_w_minimum(
    x: float | np.ndarray,
    *,
    gt: float,
    decision_bound: float,
    return_arg: bool = False,
) -> float | np.ndarray | tuple[float, float]:
    values = np.asarray(x, dtype=np.float64)
    base = (
        0.8
        * values
        * np.sin(6.0 * pi * values + 0.2 * pi)
        + gt
    )
    lower = np.maximum(-decision_bound - base, 0.5 * values - 0.25)
    upper = decision_bound - base
    threshold = 1.5 * (1.0 - sqrt(2.0) / 2.0)

    first_upper = np.minimum(upper, threshold)
    first_valid = lower <= first_upper
    first_arg = np.minimum(np.maximum(0.0, lower), first_upper)
    first_value = np.where(first_valid, np.abs(first_arg), np.inf)

    second_lower = np.maximum(lower, threshold)
    second_valid = second_lower <= upper
    second_arg = np.minimum(np.maximum(1.0, second_lower), upper)
    second_value = np.where(
        second_valid,
        0.125 + (second_arg - 1.0) ** 2,
        np.inf,
    )
    minimum = np.minimum(first_value, second_value)
    if np.any(~np.isfinite(minimum)):
        raise ReferenceArtifactError(
            "linked CDF constraint is unreachable within decision bounds"
        )
    if values.ndim == 0:
        scalar_minimum = float(minimum)
        if return_arg:
            use_first = float(first_value) <= float(second_value)
            argument = float(first_arg if use_first else second_arg)
            return scalar_minimum, argument
        return scalar_minimum
    if return_arg:
        raise ReferenceArtifactError(
            "vector linked minimum cannot return one argument"
        )
    return minimum


def _golden_minimum(
    function: Callable[[float], float],
    lower: float,
    upper: float,
) -> tuple[float, float, tuple[float, float]]:
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value = function(first)
    second_value = function(second)
    while right - left > REFERENCE_ROOT_ABSOLUTE_TOLERANCE:
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - ratio * (right - left)
            first_value = function(first)
        else:
            left = first
            first = second
            first_value = second_value
            second = left + ratio * (right - left)
            second_value = function(second)
        if first == second or left == right:
            break
    candidates = (
        (left, function(left)),
        (first, first_value),
        (second, second_value),
        (right, function(right)),
    )
    best_x, best_value = min(candidates, key=lambda item: (item[1], item[0]))
    return float(best_x), float(best_value), (left, right)


def _cdf5_global_minimum(gt: float) -> tuple[float, float, dict[str, Any]]:
    shift = abs(gt)
    grid = np.linspace(
        0.0,
        1.0,
        REFERENCE_GLOBAL_GRID_INTERVALS + 1,
        dtype=np.float64,
    )
    values = (
        1.0
        - grid
        + shift
        + np.asarray(
            _linked_w_minimum(
                grid,
                gt=gt,
                decision_bound=2.0,
            )
        )
    )
    local = np.flatnonzero(
        (values <= np.r_[np.inf, values[:-1]])
        & (values <= np.r_[values[1:], np.inf])
    )
    if not len(local):
        raise ReferenceArtifactError("CDF5 lower envelope has no minimum")

    def function(value: float) -> float:
        return float(
            1.0
            - value
            + shift
            + _linked_w_minimum(
                value,
                gt=gt,
                decision_bound=2.0,
            )
        )

    candidates: list[tuple[float, float, tuple[float, float]]] = []
    spacing = 1.0 / REFERENCE_GLOBAL_GRID_INTERVALS
    for index in local:
        lower = max(0.0, float(grid[index]) - spacing)
        upper = min(1.0, float(grid[index]) + spacing)
        candidates.append(_golden_minimum(function, lower, upper))
    candidates.extend(
        (
            (0.0, function(0.0), (0.0, 0.0)),
            (0.5, function(0.5), (0.5, 0.5)),
            (1.0, function(1.0), (1.0, 1.0)),
        )
    )
    best_value = min(value for _x, value, _bracket in candidates)
    tied = [
        item
        for item in candidates
        if item[1] <= best_value + REFERENCE_FEASIBILITY_TOLERANCE
    ]
    best_x, evaluated, bracket = min(tied, key=lambda item: item[0])
    return best_x, evaluated, {
        "method": (
            "DECISION_BOUND_REACHABLE_PIECEWISE_ANALYTIC_LOWER_ENVELOPE_"
            "GLOBAL_GRID_PLUS_LOCAL_GOLDEN_REFINEMENT"
        ),
        "global_grid_intervals": REFERENCE_GLOBAL_GRID_INTERVALS,
        "grid_spacing_hex": _float_hex(spacing),
        "local_root_absolute_tolerance_hex": _float_hex(
            REFERENCE_ROOT_ABSOLUTE_TOLERANCE
        ),
        "winning_x_bracket_hex": _hex_vector(bracket),
        "candidate_local_minima": len(local),
        "earliest_equal_minimizer_rule": True,
    }


def _nondominated_points(
    points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    unique = sorted(
        {
            (float(point[0]), float(point[1]))
            for point in points
        }
    )
    selected: list[tuple[float, float]] = []
    for index, point in enumerate(unique):
        dominated = False
        for other_index, other in enumerate(unique):
            if index == other_index:
                continue
            if (
                other[0] <= point[0]
                and other[1] <= point[1]
                and (other[0] < point[0] or other[1] < point[1])
            ):
                dominated = True
                break
        if not dominated:
            selected.append(point)
    return tuple(selected)


def _cdf11_finite_front(gt: float) -> tuple[tuple[float, float], ...]:
    candidates: set[float] = {0.0, 1.0}
    for integer in range(floor(gt) - 1, ceil(gt) + 22):
        value = (integer - gt) / 20.0
        if 0.0 <= value <= 1.0:
            candidates.add(float(value))
    points: list[tuple[float, float]] = []
    for x_value in sorted(candidates):
        phase = 20.0 * x_value + gt
        nearest = round(phase)
        ripple = (
            0.0
            if abs(phase - nearest) <= REFERENCE_FEASIBILITY_TOLERANCE
            else 0.15 * abs(sin(pi * phase))
        )
        w_value = float(
            _linked_w_minimum(
                x_value,
                gt=0.0,
                decision_bound=1.0,
            )
        )
        points.append(
            (
                x_value + ripple,
                1.0 - x_value + w_value + ripple,
            )
        )
    return _nondominated_points(points)


def _cdf15_feasible_intervals(gt: float) -> tuple[tuple[float, float], ...]:
    minimum_z = gt
    maximum_z = gt + 2.0
    intervals: list[tuple[float, float]] = []
    for integer in range(floor(minimum_z) - 2, ceil(maximum_z) + 2):
        lower_z = max(minimum_z, integer + 0.5)
        upper_z = min(maximum_z, integer + 1.0)
        if lower_z <= upper_z:
            lower_x = sqrt(max(0.0, (lower_z - gt) / 2.0))
            upper_x = sqrt(max(0.0, (upper_z - gt) / 2.0))
            intervals.append((lower_x, upper_x))
    if not intervals:
        raise ReferenceArtifactError("CDF15 has no feasible ideal interval")
    return tuple(intervals)


def derive_cdf_reference(
    problem_index: int,
    *,
    profile: str,
    event_id: int,
    master_seed_u64: str | None = None,
) -> ReferenceDerivation:
    """Derive one event-specific corrective CDF reference artifact."""

    if profile not in _CDF_PROFILE_SEVERITY:
        raise ReferenceArtifactError("unknown CDF reference profile")
    if not 0 <= event_id < 60:
        raise ReferenceArtifactError("CDF reference event must be in 0..59")
    if problem_index == 13:
        if master_seed_u64 is None:
            raise ReferenceArtifactError("CDF13 reference requires a seed")
        environment_seed = int(master_seed_u64)
    else:
        if master_seed_u64 is not None:
            raise ReferenceArtifactError(
                "only CDF13 reference identities include a seed"
            )
        environment_seed = 0
    evaluator = CDFOperationalEvaluator(
        problem_index,
        profile,
        environment_seed,
    )
    identity = _cdf_identity(
        evaluator,
        event_id=event_id,
        master_seed_u64=master_seed_u64,
    )
    severity = _CDF_PROFILE_SEVERITY[profile]
    time_value = event_id / severity
    gt = sin(0.5 * pi * time_value)
    shift = abs(gt)
    common = {
        "authority": "CMLSGA_COMMIT_1926A5A1",
        "profile": profile,
        "event_id": event_id,
        "decision_bound_reachability_checked": True,
        "observed_method_outputs_used": False,
    }

    if problem_index == 1:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(1.0, 1.0),
            derivation_id="CDF1_OPERATIONAL_FEASIBLE_SUBCURVE_EXTREMA_V1",
            certificate={
                **common,
                "rule": "ORACLE_CONSTRAINTS_ON_IDEAL_UDF2_CURVE",
                "both_axis_endpoints_feasible": True,
            },
        )
    if problem_index == 2:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.125),
            maxima=(1.0, 1.0),
            derivation_id="CDF2_REACHABLE_PIECEWISE_FRONT_EXTREMA_V1",
            certificate={
                **common,
                "rule": "CF4_THREE_BRANCH_FRONT_WITH_REACHABLE_R_EQUALS_1",
            },
        )
    if problem_index == 3:
        points = tuple(
            (index / 20.0, 1.0 - index / 20.0)
            for index in range(21)
        )
        return _finite_derivation(
            identity=identity,
            points=points,
            derivation_id="CDF3_COMPLETE_21_POINT_FRONT_V1",
            certificate={
                **common,
                "rule": "RIPPLE_ZEROS_X_EQUALS_I_OVER_20",
            },
        )
    if problem_index == 4:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(1.0, 1.0),
            derivation_id="CDF4_CONSTRAINED_IDEAL_CURVE_EXTREMA_V1",
            certificate={
                **common,
                "rule": "OBJECTIVE_CONSTRAINT_SUBSET_WITH_BOTH_ENDPOINTS",
            },
        )
    if problem_index == 5:
        x_at_minimum, minimum_f2, search_certificate = (
            _cdf5_global_minimum(gt)
        )
        return _continuous_derivation(
            identity=identity,
            minima=(shift, minimum_f2),
            maxima=(shift + x_at_minimum, 1.0 + shift),
            derivation_id="CDF5_BOUND_REACHABLE_LOWER_ENVELOPE_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "LOWER_ENVELOPE_OVER_CONSTRAINT_AND_X2_BOUND_"
                    "REACHABLE_RESIDUALS"
                ),
                "x_at_minimum_f2_hex": _float_hex(x_at_minimum),
                **search_certificate,
            },
        )
    if problem_index == 6:
        return _continuous_derivation(
            identity=identity,
            minima=(shift, shift),
            maxima=(1.0 + shift, 1.0 + shift),
            derivation_id="CDF6_OPERATIONAL_CF6_BRANCH_EXTREMA_V1",
            certificate={
                **common,
                "rule": "ORACLE_NO_G_SHIFT_CONSTRAINTS_WITH_SHIFTED_OBJECTIVES",
            },
        )
    if problem_index == 7:
        points = tuple(
            (
                shift + index / 20.0,
                shift + 1.0 - index / 20.0,
            )
            for index in range(21)
        )
        return _finite_derivation(
            identity=identity,
            points=points,
            derivation_id="CDF7_COMPLETE_SHIFTED_21_POINT_FRONT_V1",
            certificate={
                **common,
                "rule": "SHIFTED_RIPPLE_ZEROS_X_EQUALS_I_OVER_20",
            },
        )
    if problem_index == 8:
        multiplier = 0.5 + shift

        def vector_constraint(values: np.ndarray) -> np.ndarray:
            return (
                np.sqrt(values)
                - multiplier * values**multiplier
                - np.sin(
                    2.0
                    * pi
                    * (
                        np.sqrt(values)
                        + multiplier * values**multiplier
                    )
                )
            )

        def scalar_constraint(value: float) -> float:
            return float(vector_constraint(np.asarray(value)))

        minimum_x, maximum_x, root_certificate = _feasible_x_extrema(
            vector_constraint,
            scalar_constraint,
        )
        return _continuous_derivation(
            identity=identity,
            minima=(
                minimum_x,
                1.0 - multiplier * maximum_x**multiplier,
            ),
            maxima=(
                maximum_x,
                1.0 - multiplier * minimum_x**multiplier,
            ),
            derivation_id="CDF8_REACHABLE_CONSTRAINED_IDEAL_CURVE_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "LARGEST_AND_SMALLEST_X_SATISFYING_CF3_"
                    "OBJECTIVE_CONSTRAINT"
                ),
                **root_certificate,
            },
        )
    if problem_index == 9:
        multiplier = 0.5 + shift
        maximum_x = min(1.0, 1.0 / multiplier)
        q_end = max(
            0.0,
            1.0 - (multiplier * maximum_x) ** multiplier,
        )
        if q_end >= 0.5:
            transformed = q_end * q_end
        elif q_end >= 0.25:
            transformed = 0.5 * q_end
        else:
            transformed = 0.25 * sqrt(q_end)
        return _continuous_derivation(
            identity=identity,
            minima=(shift, shift + transformed),
            maxima=(shift + maximum_x, shift + 1.0),
            derivation_id="CDF9_REAL_DOMAIN_REACHABLE_CF6_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "REAL_DOMAIN_X_LE_MIN_1_1_OVER_M_AND_REACHABLE_"
                    "THREE_BRANCH_CF6_FRONT"
                ),
                "real_domain_maximum_x_hex": _float_hex(maximum_x),
                "q_at_maximum_x_hex": _float_hex(q_end),
                "q_below_zero_policy": (
                    "CHARGED_TYPED_CDFDomainUndefinedError_NO_EXTENSION"
                ),
            },
        )
    if problem_index == 10:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(1.0, 1.0),
            derivation_id="CDF10_OPERATIONAL_TWO_CONSTRAINT_FRONT_EXTREMA_V1",
            certificate={
                **common,
                "rule": "ORACLE_W_NOT_W_SQUARED_AND_TWO_CF6_CONSTRAINTS",
            },
        )
    if problem_index == 11:
        points = _cdf11_finite_front(gt)
        return _finite_derivation(
            identity=identity,
            points=points,
            derivation_id="CDF11_COMPLETE_BOUND_REACHABLE_RIPPLE_ZERO_FRONT_V1",
            certificate={
                **common,
                "rule": (
                    "RIPPLE_ZEROS_PLUS_ENDPOINTS_THEN_EXACT_"
                    "NONDOMINATED_FILTER_WITH_X2_BOUND_REACHABILITY"
                ),
                "candidate_count_before_dominance": (
                    len(
                        {
                            0.0,
                            1.0,
                            *(
                                (integer - gt) / 20.0
                                for integer in range(
                                    floor(gt) - 1,
                                    ceil(gt) + 22,
                                )
                                if 0.0
                                <= (integer - gt) / 20.0
                                <= 1.0
                            ),
                        }
                    )
                ),
            },
        )
    if problem_index == 12:
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(1.0, 1.0),
            derivation_id="CDF12_OPERATIONAL_SIN_COS_FRONT_EXTREMA_V1",
            certificate={
                **common,
                "rule": "ORACLE_ODD_SIN_EVEN_COS_WITH_BOTH_ENDPOINTS",
            },
        )
    if problem_index == 13:
        time_vector = evaluator._time_vector(event_id)
        g = tuple(sin(0.5 * pi * value) for value in time_vector)
        random_shift = abs(g[2])
        multiplier = 0.5 + abs(g[3])
        exponent = 0.5 + abs(g[4])

        def vector_constraint(values: np.ndarray) -> np.ndarray:
            offset = (
                random_shift
                + multiplier
                * (
                    (values + random_shift) ** exponent
                    - values**exponent
                )
            )
            phase = (
                multiplier
                * (
                    (values + random_shift) ** exponent
                    + values**exponent
                )
                - random_shift
            )
            return offset - np.sin(2.0 * pi * phase)

        def scalar_constraint(value: float) -> float:
            return float(vector_constraint(np.asarray(value)))

        minimum_x, maximum_x, root_certificate = _feasible_x_extrema(
            vector_constraint,
            scalar_constraint,
        )
        return _continuous_derivation(
            identity=identity,
            minima=(
                random_shift + minimum_x,
                1.0
                - multiplier * maximum_x**exponent
                + random_shift,
            ),
            maxima=(
                random_shift + maximum_x,
                1.0
                - multiplier * minimum_x**exponent
                + random_shift,
            ),
            derivation_id="CDF13_SEED_TIME_REACHABLE_IDEAL_CURVE_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "FULL_FIVE_COMPONENT_TIME_VECTOR_AND_SEED_BOUND_"
                    "CF4_FEASIBILITY"
                ),
                "time_vector_hex": _hex_vector(time_vector),
                "g_vector_hex": _hex_vector(g),
                "k_t1": ceil(10.0 * g[0]),
                **root_certificate,
            },
        )
    if problem_index == 14:
        if event_id % (2 * severity) == 0:
            points = tuple(
                (index / 20.0, 1.0 - index / 20.0)
                for index in range(21)
            )
            return _finite_derivation(
                identity=identity,
                points=points,
                derivation_id="CDF14_COMPLETE_ZERO_SHIFT_21_POINT_FRONT_V1",
                certificate={
                    **common,
                    "rule": (
                        "EXACT_RATIONAL_EVENT_CLASS_G_EQUALS_ZERO_"
                        "RIPPLE_ZEROS"
                    ),
                },
            )
        return _continuous_derivation(
            identity=identity,
            minima=(0.0, 0.0),
            maxima=(1.0, 1.0),
            derivation_id="CDF14_NONZERO_SHIFT_FEASIBLE_INTERVAL_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "ABS_SIN_20PI_X_LE_ABS_G_WITH_BOTH_ENDPOINTS"
                ),
            },
        )
    if problem_index == 15:
        intervals = _cdf15_feasible_intervals(gt)
        minimum_x = min(interval[0] for interval in intervals)
        maximum_x = max(interval[1] for interval in intervals)
        return _continuous_derivation(
            identity=identity,
            minima=(minimum_x, 1.0 - maximum_x * maximum_x),
            maxima=(maximum_x, 1.0 - minimum_x * minimum_x),
            derivation_id="CDF15_OPERATIONAL_ANALYTIC_INTERVAL_EXTREMA_V1",
            certificate={
                **common,
                "rule": (
                    "SIN_2PI_TIMES_2X_SQUARED_PLUS_G_LE_ZERO_"
                    "ANALYTIC_INTERVALS"
                ),
                "feasible_intervals_hex": [
                    _hex_vector(interval) for interval in intervals
                ],
            },
        )
    raise ReferenceArtifactError("CDF index must be in 1..15")


def iter_formal_reference_derivations(
    seeds: Sequence[str] = CDF_REFERENCE_SEEDS,
) -> Iterator[ReferenceDerivation]:
    """Yield all 2,294 formal reference identities in canonical order."""

    if tuple(seeds) != CDF_REFERENCE_SEEDS:
        raise ReferenceArtifactError(
            "formal CDF13 seed set differs from the frozen five seeds"
        )
    for problem_index in range(1, 15):
        yield derive_lircmop_reference(problem_index)
    for profile in ("CDF-HARSH", "CDF-MILD"):
        for problem_index in range(1, 16):
            for event_id in range(60):
                if problem_index == 13:
                    for seed in seeds:
                        yield derive_cdf_reference(
                            problem_index,
                            profile=profile,
                            event_id=event_id,
                            master_seed_u64=seed,
                        )
                else:
                    yield derive_cdf_reference(
                        problem_index,
                        profile=profile,
                        event_id=event_id,
                    )


def materialize_reference_catalog(
    path: Path,
) -> tuple[str, int, int, dict[str, int]]:
    """Write the canonical compact JSONL artifact and return its identity."""

    records = [
        derivation.canonical_record()
        for derivation in iter_formal_reference_derivations()
    ]
    records.sort(
        key=lambda record: (
            str(record["identity"]["suite_id"]),
            str(record["identity"]["problem_id"]),
            str(record["identity"]["profile"]),
            int(record["identity"]["event_id"]),
            str(record["identity"]["master_seed_u64"]),
        )
    )
    if len(records) != REFERENCE_CATALOG_EXPECTED_IDENTITIES:
        raise ReferenceArtifactError(
            "formal reference catalog identity count drifted"
        )
    encoded = b"".join(_canonical_json(record) + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    breakdown = {
        "lircmop_static": 14,
        "cdf_non_cdf13": 14 * 2 * 60,
        "cdf13_seed_time": 5 * 2 * 60,
        "finite_front_records": sum(
            record["finite_front"] is not None for record in records
        ),
        "continuous_front_records": sum(
            record["finite_front"] is None for record in records
        ),
    }
    return sha256(encoded).hexdigest(), len(encoded), len(records), breakdown


def bound_file(path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Return a stable repository-relative file identity for a manifest."""

    resolved = path.resolve()
    relative = resolved.relative_to(repository_root.resolve()).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
