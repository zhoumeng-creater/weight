"""Exact public evaluator bindings used by the R4 bridge.

The static binding delegates to the MIT-licensed jMetalPy 1.7.0 LIR-CMOP
implementation.  The dynamic binding is an independent Python transcription
of the equations in Grudniewski and Sobey's CC-BY-4.0 CDF1--15 paper.  The
authors' GPL-3.0-or-later C++ implementation is retained only as a provenance
oracle; it is not imported or vendored by this package.

Every returned constraint uses the project convention ``c <= 0``.  Both
upstream implementations use non-negative values for feasibility, so the
boundary performs one explicit sign reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
from math import ceil, cos, exp, isfinite, pi, sin, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np


JMETALPY_DISTRIBUTION = "jmetalpy"
JMETALPY_VERSION = "1.7.0"
JMETALPY_TAG_COMMIT = "3d5d0072876f84db32d1336e38e06b90319d01ee"
JMETALPY_WHEEL_SHA256 = (
    "57a86bb939695459b0f7deb4723abbbdf4c28fb83bf8036cfdaf20b17b6778e9"
)
CDF_PAPER_DOI = "10.1007/s11047-020-09799-y"
CDF_ORACLE_REPOSITORY = "https://bitbucket.org/Pag1c18/cmlsga"
CDF_ORACLE_COMMIT = "1926a5a1c89adf0a5e5e70449adbec62750a108a"
CDF_ORACLE_CPP_SHA256 = (
    "48b2c256f4bdec6ed4f81f8edd82a03753bc51550776e1ae84b2d6fcbc18fa7a"
)

_CDF_BOUNDS = {
    1: (-1.0, 2.0),
    2: (-2.0, 2.0),
    3: (-1.0, 1.0),
    4: (-2.0, 2.0),
    5: (-2.0, 2.0),
    6: (-2.0, 2.0),
    7: (-2.0, 2.0),
    8: (-1.0, 2.0),
    9: (-2.0, 2.0),
    10: (-2.0, 2.0),
    11: (-1.0, 1.0),
    12: (-1.0, 1.0),
    13: (-2.0, 2.0),
    14: (0.0, 1.0),
    15: (-2.0, 2.0),
}
_CDF_CONSTRAINT_COUNTS = {
    1: 2,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 1,
    8: 1,
    9: 2,
    10: 2,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
}
_CDF_PROFILE_SEVERITY = {"CDF-HARSH": 5, "CDF-MILD": 10}
_LIRCMOP_CONSTRAINT_COUNT_PATCH = {7: 3, 8: 3}
_LIRCMOP_OBJECTIVE_COUNT_PATCH = {13: 3, 14: 3}


class R4EvaluatorBindingError(RuntimeError):
    """An exact R4 evaluator dependency or equation contract is unavailable."""


def _module_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _binding_sha256(*parts: str) -> str:
    return sha256("|".join(parts).encode("utf-8")).hexdigest()


def _require_jmetalpy() -> None:
    try:
        version = importlib.metadata.version(JMETALPY_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise R4EvaluatorBindingError(
            "jmetalpy==1.7.0 is required by the R4 benchmark binding"
        ) from exc
    if version != JMETALPY_VERSION:
        raise R4EvaluatorBindingError(
            f"jmetalpy version {version!r} differs from frozen 1.7.0"
        )


@dataclass(frozen=True)
class LIRCMOPEvaluator:
    """Source-bound jMetalPy LIR-CMOP evaluator with canonical constraints."""

    problem_index: int

    def __post_init__(self) -> None:
        if not 1 <= self.problem_index <= 14:
            raise R4EvaluatorBindingError("LIR-CMOP index must be in 1..14")
        _require_jmetalpy()
        module = __import__(
            "jmetal.problem.multiobjective.lircmop",
            fromlist=[f"LIRCMOP{self.problem_index}"],
        )
        problem_type = getattr(module, f"LIRCMOP{self.problem_index}")
        if self.problem_index in _LIRCMOP_CONSTRAINT_COUNT_PATCH:
            constraint_count = _LIRCMOP_CONSTRAINT_COUNT_PATCH[
                self.problem_index
            ]

            class ConstraintCountPatchedProblem(problem_type):
                """Correct upstream metadata to match its three equations."""

                def number_of_constraints(self) -> int:
                    return constraint_count

            problem_type = ConstraintCountPatchedProblem
        problem = problem_type()
        if self.problem_index in _LIRCMOP_OBJECTIVE_COUNT_PATCH:
            problem.obj_directions = [problem.MINIMIZE] * 3
            problem.obj_labels = ["f1", "f2", "f3"]
        object.__setattr__(self, "_problem", problem)

    @property
    def problem_id(self) -> str:
        return f"LIRCMOP{self.problem_index}"

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._problem.lower_bound)

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._problem.upper_bound)

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(
            f"f{index}"
            for index in range(1, self._problem.number_of_objectives() + 1)
        )

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(
            f"c{index}"
            for index in range(1, self._problem.number_of_constraints() + 1)
        )

    @property
    def binding_sha256(self) -> str:
        return _binding_sha256(
            JMETALPY_WHEEL_SHA256,
            JMETALPY_TAG_COMMIT,
            _module_sha256(),
            self.problem_id,
        )

    def __call__(
        self, vector: Sequence[float], event_id: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if event_id != 0:
            raise R4EvaluatorBindingError("LIR-CMOP is a TS1 benchmark")
        values = tuple(float(value) for value in vector)
        if len(values) != self._problem.number_of_variables():
            raise R4EvaluatorBindingError("LIR-CMOP vector has wrong dimension")
        from jmetal.core.solution import FloatSolution

        solution = FloatSolution(
            list(self._problem.lower_bound),
            list(self._problem.upper_bound),
            self._problem.number_of_objectives(),
            self._problem.number_of_constraints(),
        )
        solution.variables = list(values)
        evaluated = self._problem.evaluate(solution)
        objectives = tuple(float(value) for value in evaluated.objectives)
        constraints = tuple(-float(value) for value in evaluated.constraints)
        return objectives, constraints

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
    ) -> tuple[tuple[Sequence[float], Sequence[float]], ...]:
        """Evaluate rows in input order through the unchanged scalar oracle.

        LIR-CMOP is bound to the upstream jMetalPy implementation, so this
        deliberately avoids an independent vectorized transcription.  The
        ordered scalar loop gives the shared generation-batch path an exact
        kernel without changing any per-candidate floating-point operation.
        """

        if event_id != 0:
            raise R4EvaluatorBindingError("LIR-CMOP is a TS1 benchmark")
        return tuple(self(vector, event_id) for vector in vectors)


def _sgn(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _groups(
    vector: np.ndarray,
    residual,
    transform=lambda value, _index: value * value,
) -> tuple[float, float, float, float]:
    count_odd = count_even = 0.0
    sum_odd = sum_even = 0.0
    for index in range(2, 11):
        value = float(transform(float(residual(index)), index))
        if index % 2 == 1:
            count_odd += 1.0
            sum_odd += value
        else:
            count_even += 1.0
            sum_even += value
    return count_odd, count_even, sum_odd, sum_even


@dataclass(frozen=True)
class CDFEvaluator:
    """CDF1--15 evaluator with a result-blind, paired environment schedule."""

    problem_index: int
    profile: str
    environment_seed: int = 0

    def __post_init__(self) -> None:
        if self.problem_index not in _CDF_BOUNDS:
            raise R4EvaluatorBindingError("CDF index must be in 1..15")
        if self.profile not in _CDF_PROFILE_SEVERITY:
            raise R4EvaluatorBindingError("unknown CDF profile")
        if self.environment_seed < 0:
            raise R4EvaluatorBindingError(
                "CDF environment seed must be nonnegative"
            )

    @property
    def problem_id(self) -> str:
        return f"CDF{self.problem_index}"

    @property
    def severity_ns(self) -> int:
        return _CDF_PROFILE_SEVERITY[self.profile]

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        lower, _ = _CDF_BOUNDS[self.problem_index]
        return (0.0,) + (lower,) * 9

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        _, upper = _CDF_BOUNDS[self.problem_index]
        return (1.0,) + (upper,) * 9

    @property
    def objective_names(self) -> tuple[str, str]:
        return ("f1", "f2")

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(
            f"c{index}"
            for index in range(
                1, _CDF_CONSTRAINT_COUNTS[self.problem_index] + 1
            )
        )

    @property
    def binding_sha256(self) -> str:
        return _binding_sha256(
            CDF_PAPER_DOI,
            CDF_ORACLE_COMMIT,
            CDF_ORACLE_CPP_SHA256,
            _module_sha256(),
            self.problem_id,
            self.profile,
        )

    @property
    def environment_schedule_commitment(self) -> str:
        schedule = ",".join(
            f"{value:.17g}"
            for event_id in range(60)
            for value in self._time_vector(event_id)
        )
        return sha256(schedule.encode("ascii")).hexdigest()

    def _time_vector(self, event_id: int) -> tuple[float, ...]:
        if not 0 <= event_id < 60:
            raise R4EvaluatorBindingError("CDF event must be in 0..59")
        counts = np.zeros(5, dtype=float)
        if event_id:
            rng = np.random.Generator(np.random.PCG64(self.environment_seed))
            selected = rng.integers(0, 5, size=event_id)
            for index in selected:
                counts[int(index)] += 1.0
        return tuple(float(value / self.severity_ns) for value in counts)

    def release_metadata(self, event_id: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile": self.profile,
            "severity_ns": self.severity_ns,
            "time_value": event_id / self.severity_ns,
            "schedule_commitment_sha256": (
                self.environment_schedule_commitment
            ),
        }
        if self.problem_index == 13:
            payload["current_time_vector"] = self._time_vector(event_id)
        return payload

    def __call__(
        self, vector: Sequence[float], event_id: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        x = np.asarray(vector, dtype=float)
        if x.shape != (10,) or not np.all(np.isfinite(x)):
            raise R4EvaluatorBindingError(
                "CDF vector must contain ten finite values"
            )
        if np.any(x < self.lower_bounds) or np.any(x > self.upper_bounds):
            raise R4EvaluatorBindingError("CDF vector is outside its bounds")
        time_value = event_id / self.severity_ns
        objectives, upstream_constraints = self._evaluate_equations(
            x, time_value, self._time_vector(event_id)
        )
        values = (*objectives, *upstream_constraints)
        if not all(isfinite(value) for value in values):
            raise FloatingPointError("CDF evaluator returned a nonfinite value")
        if len(upstream_constraints) != len(self.constraint_names):
            raise R4EvaluatorBindingError(
                "CDF constraint count differs from its registry"
            )
        return objectives, tuple(-value for value in upstream_constraints)

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
    ) -> tuple[tuple[Sequence[float], Sequence[float]], ...]:
        """Evaluate an ordered candidate matrix with a scalar reference fallback."""

        matrix = np.asarray(vectors, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1:] != (10,)
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < self.lower_bounds)
            or np.any(matrix > self.upper_bounds)
        ):
            raise R4EvaluatorBindingError(
                "CDF candidate batch is nonfinite, outside bounds, "
                "or wrong shape"
            )
        if not 0 <= event_id < 60:
            raise R4EvaluatorBindingError("CDF event must be in 0..59")
        time_value = event_id / self.severity_ns
        time_vector = self._time_vector(event_id)
        rows: list[tuple[Sequence[float], Sequence[float]]] = []
        for row in matrix:
            objectives, upstream_constraints = self._evaluate_equations(
                row, time_value, time_vector
            )
            values = (*objectives, *upstream_constraints)
            if not all(isfinite(value) for value in values):
                raise FloatingPointError(
                    "CDF evaluator returned a nonfinite value"
                )
            if len(upstream_constraints) != len(self.constraint_names):
                raise R4EvaluatorBindingError(
                    "CDF constraint count differs from its registry"
                )
            rows.append(
                (
                    objectives,
                    tuple(-value for value in upstream_constraints),
                )
            )
        return tuple(rows)

    def _evaluate_equations(
        self,
        x: np.ndarray,
        time_value: float,
        time_vector: tuple[float, ...],
    ) -> tuple[tuple[float, float], tuple[float, ...]]:
        index = self.problem_index
        gt = sin(0.5 * pi * time_value)

        if index == 1:
            def residual(i):
                exponent = 0.5 * (2.0 + 3.0 * (i - 2) / 8.0) + abs(gt)
                return x[i - 1] - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, (1.0 - x[0]) ** 2 + 2.0 * s2 / n2)
            temp = 0.5 * (1.0 - x[0]) - (1.0 - x[0]) ** 2
            temp2 = 0.25 * sqrt(1.0 - x[0]) - 0.5 * (1.0 - x[0])
            c = (
                x[1] - x[0] ** (1.0 + abs(gt)) - _sgn(temp) * sqrt(abs(temp)),
                x[3]
                - x[0] ** (1.375 + abs(gt))
                - _sgn(temp2) * sqrt(abs(temp2)),
            )
            return (float(f[0]), float(f[1])), tuple(float(v) for v in c)

        if index == 2:
            def residual(i):
                return x[i - 1] - sin(6.0 * pi * x[0] + i * pi / 10.0)

            def transform(y, i):
                if i == 2:
                    threshold = 1.5 * (1.0 - sqrt(2.0) / 2.0)
                    return abs(y) if y < threshold else 0.125 + (y - 1.0) ** 2
                return (y - gt) ** 2

            _, _, s1, s2 = _groups(x, residual, transform)
            f = (x[0] + s1, 1.0 - x[0] + s2)
            temp = (
                x[1]
                - sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - 0.5 * x[0]
                + 0.25
            )
            return (float(f[0]), float(f[1])), (
                float(temp / (1.0 + exp(4.0 * abs(temp)))),
            )

        if index == 3:
            def residual(i):
                exponent = 0.5 * (2.0 + 3.0 * (i - 2) / 8.0) + abs(gt)
                return x[i - 1] - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            ripple = 0.15 * abs(sin(20.0 * pi * x[0]))
            f = (
                x[0] + 2.0 * s1 / n1 + ripple,
                1.0 - x[0] + 2.0 * s2 / n2 + ripple,
            )
            return (float(f[0]), float(f[1])), (
                float(x[1] - x[0] ** (1.0 + abs(gt))),
            )

        if index == 4:
            def residual(i):
                exponent = 0.5 * (1.0 + 3.0 * (i - 2) / 8.0) + abs(gt)
                return x[i - 1] - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, 1.0 - x[0] ** 2 + 2.0 * s2 / n2)
            c = f[0] + f[1] - abs(sin(10.0 * pi * (f[0] - f[1] + 1.0))) - 1.0
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 5:
            def residual(i):
                wave = (
                    cos(6.0 * pi * x[0] + i * pi / 10.0)
                    if i % 2 == 1
                    else sin(6.0 * pi * x[0] + i * pi / 10.0)
                )
                return x[i - 1] - 0.8 * x[0] * wave - gt

            def transform(y, i):
                if i == 2:
                    threshold = 1.5 * (1.0 - sqrt(2.0) / 2.0)
                    return abs(y) if y < threshold else 0.125 + (y - 1.0) ** 2
                return 2.0 * y * y - cos(4.0 * pi * y) + 1.0

            _, _, s1, s2 = _groups(x, residual, transform)
            f = (x[0] + s1 + abs(gt), 1.0 - x[0] + s2 + abs(gt))
            c = (
                x[1]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - 0.5 * x[0]
                + 0.25
                - gt
            )
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 6:
            def residual(i):
                if i % 2 == 1:
                    return (
                        x[i - 1]
                        - 0.8 * x[0] * cos(6.0 * pi * x[0] + i * pi / 10.0)
                        - abs(gt)
                    )
                value = x[i - 1] - 0.8 * x[0] * sin(
                    6.0 * pi * x[0] + i * pi / 10.0
                )
                return value if i in {2, 4} else value - abs(gt)

            _, _, s1, s2 = _groups(x, residual)
            f = (x[0] + s1 + abs(gt), (1.0 - x[0]) ** 2 + s2 + abs(gt))
            temp = 0.5 * (1.0 - x[0]) - (1.0 - x[0]) ** 2
            temp2 = 0.25 * sqrt(1.0 - x[0]) - 0.5 * (1.0 - x[0])
            c = (
                x[1]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - _sgn(temp) * sqrt(abs(temp)),
                x[3]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 4.0 * pi / 10.0)
                - _sgn(temp2) * sqrt(abs(temp2)),
            )
            return (float(f[0]), float(f[1])), tuple(float(v) for v in c)

        if index == 7:
            def residual(i):
                exponent = 0.5 * (1.0 + 3.0 * (i - 2) / 8.0)
                return x[i - 1] - gt - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            f = (
                x[0] + abs(gt) + 2.0 * s1 / n1,
                1.0 - x[0] + abs(gt) + 2.0 * s2 / n2,
            )
            c = (
                f[0]
                + f[1]
                - 2.0 * abs(gt)
                - abs(sin(10.0 * pi * (f[0] - f[1] + 1.0)))
                - 1.0
            )
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 8:
            mt = 0.5 + abs(gt)

            def residual(i):
                exponent = 0.5 * (2.0 + 3.0 * (i - 2) / 8.0)
                return x[i - 1] - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, 1.0 - mt * x[0] ** mt + 2.0 * s2 / n2)
            c = f[1] + sqrt(f[0]) - sin(2.0 * pi * (sqrt(f[0]) - f[1] + 1.0)) - 1.0
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 9:
            mt = ht = 0.5 + abs(gt)

            def residual(i):
                wave = (
                    cos(6.0 * pi * x[0] + i * pi / 10.0)
                    if i % 2 == 1
                    else sin(6.0 * pi * x[0] + i * pi / 10.0)
                )
                return x[i - 1] - 0.8 * x[0] * wave

            _, _, s1, s2 = _groups(x, residual)
            shaped = 1.0 - (mt * x[0]) ** ht
            f = (x[0] + s1 + abs(gt), shaped**2 + s2 + abs(gt))
            if shaped < 0.0:
                raise FloatingPointError(
                    "CDF9 signed square-root constraint is outside its real domain"
                )
            temp = 0.5 * shaped - shaped**2
            temp2 = 0.25 * sqrt(shaped) - 0.5 * shaped
            c = (
                x[1]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - _sgn(temp) * sqrt(abs(temp)),
                x[3]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 4.0 * pi / 10.0)
                - _sgn(temp2) * sqrt(abs(temp2)),
            )
            return (float(f[0]), float(f[1])), tuple(float(v) for v in c)

        if index == 10:
            ht = 0.5 + abs(gt)

            def residual(i):
                wave = (
                    cos(6.0 * pi * x[0] + i * pi / 10.0)
                    if i % 2 == 1
                    else sin(6.0 * pi * x[0] + i * pi / 10.0)
                )
                return x[i - 1] - wave

            def transform(y, i):
                return y * y if i in {2, 4} else 2.0 * y * y - cos(4.0 * pi * y) + 1.0

            _, _, s1, s2 = _groups(x, residual, transform)
            f = (x[0] + s1, (1.0 - x[0]) ** ht + s2)
            temp = 0.5 * (1.0 - x[0]) - (1.0 - x[0]) ** 2
            temp2 = 0.25 * sqrt(1.0 - x[0]) - 0.5 * (1.0 - x[0])
            c = (
                x[1]
                - sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - _sgn(temp) * sqrt(abs(temp)),
                x[3]
                - sin(6.0 * pi * x[0] + 4.0 * pi / 10.0)
                - _sgn(temp2) * sqrt(abs(temp2)),
            )
            return (float(f[0]), float(f[1])), tuple(float(v) for v in c)

        if index == 11:
            def residual(i):
                wave = (
                    cos(6.0 * pi * x[0] + i * pi / 10.0)
                    if i % 2 == 1
                    else sin(6.0 * pi * x[0] + i * pi / 10.0)
                )
                return x[i - 1] - 0.8 * x[0] * wave

            def transform(y, i):
                if i == 2:
                    threshold = 1.5 * (1.0 - sqrt(2.0) / 2.0)
                    return abs(y) if y < threshold else 0.125 + (y - 1.0) ** 2
                return y * y - cos(4.0 * pi * y) + 1.0

            _, _, s1, s2 = _groups(x, residual, transform)
            ripple = 0.15 * abs(sin((20.0 * x[0] + gt) * pi))
            f = (x[0] + s1 + ripple, 1.0 - x[0] + s2 + ripple)
            c = (
                x[1]
                - 0.8 * x[0] * sin(6.0 * pi * x[0] + 2.0 * pi / 10.0)
                - 0.5 * x[0]
                + 0.25
            )
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 12:
            ht = 0.5 + abs(gt)

            def residual(i):
                wave = (
                    sin(6.0 * pi * x[0] + i * pi / 10.0)
                    if i % 2 == 1
                    else cos(6.0 * pi * x[0] + i * pi / 10.0)
                )
                return x[i - 1] - wave

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, 1.0 - x[0] ** ht + 2.0 * s2 / n2)
            temp = f[1] + sqrt(f[0]) - sin(2.0 * pi * (sqrt(f[0]) - f[1] + 1.0)) - 1.0
            c = temp / (1.0 + exp(4.0 * abs(temp)))
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 13:
            g = tuple(sin(0.5 * pi * value) for value in time_vector)
            kt1 = ceil(10.0 * g[0])
            ht4 = 0.5 + abs(g[3])
            ht5 = 0.5 + abs(g[4])

            def residual(i):
                return (
                    x[i - 1]
                    - sin(6.0 * pi * x[0] + (i + kt1) * pi / 10.0)
                    - g[1]
                )

            n1, n2, s1, s2 = _groups(x, residual)
            f = (
                x[0] + abs(g[2]) + 2.0 * s1 / n1,
                1.0 - ht4 * x[0] ** ht5 + abs(g[2]) + 2.0 * s2 / n2,
            )
            c = f[1] + ht4 * f[0] ** ht5 - sin(
                2.0 * pi * (ht4 * f[0] ** ht5 - f[1] + 1.0)
            ) - 1.0
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 14:
            def residual(i):
                exponent = 0.5 * (1.0 + 3.0 * (i - 2) / 8.0)
                return x[i - 1] - x[0] ** exponent

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, 1.0 - x[0] + 2.0 * s2 / n2)
            c = f[0] + f[1] - abs(sin(10.0 * pi * (f[0] - f[1] + 1.0))) - 1.0 + abs(gt)
            return (float(f[0]), float(f[1])), (float(c),)

        if index == 15:
            def residual(i):
                return x[i - 1] - sin(6.0 * pi * x[0] + i * pi / 10.0)

            n1, n2, s1, s2 = _groups(x, residual)
            f = (x[0] + 2.0 * s1 / n1, 1.0 - x[0] ** 2 + 2.0 * s2 / n2)
            c = f[1] + f[0] ** 2 - sin(
                2.0 * pi * (f[0] ** 2 - f[1] + 1.0 + gt)
            ) - 1.0
            return (float(f[0]), float(f[1])), (float(c),)

        raise AssertionError("unreachable CDF index")
