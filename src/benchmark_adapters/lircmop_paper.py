"""Paper-faithful LIR-CMOP1--14 evaluator for the R8C amendment.

The equations are transcribed from Table 8 of the version of record:

    Fan et al., Soft Computing 23, 12491--12510 (2019)
    https://doi.org/10.1007/s00500-019-03794-x
    https://link.springer.com/article/10.1007/s00500-019-03794-x/tables/8

The paper writes feasible constraints as ``c >= 0``.  This boundary returns
their exact negatives because the project-wide convention is ``c <= 0``.
The historical R4 jMetalPy evaluator remains unchanged in
``benchmark_adapters.r4_evaluators``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import cos, isfinite, pi, sin, sqrt
from pathlib import Path

import numpy as np


LIRCMOP_PAPER_DOI = "10.1007/s00500-019-03794-x"
LIRCMOP_PAPER_TABLE = (
    "https://link.springer.com/article/"
    "10.1007/s00500-019-03794-x/tables/8"
)
LIRCMOP_PAPER_EVALUATOR_VERSION = "1.0.0"
LIRCMOP_PAPER_SUITE_ID = "LIR-CMOP-PAPER-2019-TABLE-8"

_CONSTRAINT_COUNTS = {
    1: 2,
    2: 2,
    3: 3,
    4: 3,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 3,
}
_ELLIPSES = {
    5: ((1.6, 2.5), (1.6, 2.5), (2.0, 2.0), (4.0, 8.0)),
    6: ((1.8, 2.8), (1.8, 2.8), (2.0, 2.0), (8.0, 8.0)),
    7: (
        (1.2, 2.25, 3.5),
        (1.2, 2.25, 3.5),
        (2.0, 2.5, 2.5),
        (6.0, 12.0, 10.0),
    ),
    8: (
        (1.2, 2.25, 3.5),
        (1.2, 2.25, 3.5),
        (2.0, 2.5, 2.5),
        (6.0, 12.0, 10.0),
    ),
}
_WAVE_CONSTRAINTS = {
    9: (1.4, 1.4, 1.5, 6.0, 2.0),
    10: (1.1, 1.2, 2.0, 4.0, 1.0),
    11: (1.2, 1.2, 1.5, 5.0, 2.1),
    12: (1.6, 1.6, 1.5, 6.0, 2.5),
}


class LIRCMOPPaperBindingError(ValueError):
    """A candidate or problem index violates the paper evaluator contract."""


def _module_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _binding_sha256(problem_id: str) -> str:
    payload = "|".join(
        (
            LIRCMOP_PAPER_DOI,
            LIRCMOP_PAPER_TABLE,
            LIRCMOP_PAPER_EVALUATOR_VERSION,
            _module_sha256(),
            problem_id,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _linkage_sums(
    x: np.ndarray,
    *,
    indexed_angles: bool,
) -> tuple[float, float]:
    x1 = float(x[0])
    odd_sum = 0.0
    even_sum = 0.0
    for index in range(2, 29, 2):
        one_based = index + 1
        angle = (
            0.5 * one_based * pi * x1 / 30.0
            if indexed_angles
            else 0.5 * pi * x1
        )
        odd_sum += (float(x[index]) - sin(angle)) ** 2
    for index in range(1, 30, 2):
        one_based = index + 1
        angle = (
            0.5 * one_based * pi * x1 / 30.0
            if indexed_angles
            else 0.5 * pi * x1
        )
        even_sum += (float(x[index]) - cos(angle)) ** 2
    return odd_sum, even_sum


def _paper_ellipse_constraint(
    f1: float,
    f2: float,
    *,
    p: float,
    q: float,
    a: float,
    b: float,
) -> float:
    theta = -0.25 * pi
    first = (f1 - p) * cos(theta) - (f2 - q) * sin(theta)
    second = (f1 - p) * sin(theta) + (f2 - q) * cos(theta)
    return first * first / (a * a) + second * second / (b * b) - 0.1


@dataclass(frozen=True)
class LIRCMOPPaperEvaluator:
    """Direct implementation of the fourteen version-of-record equations."""

    problem_index: int

    def __post_init__(self) -> None:
        if self.problem_index not in _CONSTRAINT_COUNTS:
            raise LIRCMOPPaperBindingError(
                "LIR-CMOP paper index must be in 1..14"
            )

    @property
    def problem_id(self) -> str:
        return f"LIRCMOP{self.problem_index}"

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return (0.0,) * 30

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return (1.0,) * 30

    @property
    def objective_names(self) -> tuple[str, ...]:
        count = 3 if self.problem_index in {13, 14} else 2
        return tuple(f"f{index}" for index in range(1, count + 1))

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(
            f"c{index}"
            for index in range(
                1, _CONSTRAINT_COUNTS[self.problem_index] + 1
            )
        )

    @property
    def binding_sha256(self) -> str:
        return _binding_sha256(self.problem_id)

    def __call__(
        self,
        vector: Sequence[float],
        event_id: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if event_id != 0:
            raise LIRCMOPPaperBindingError("LIR-CMOP is a TS1 benchmark")
        x = np.asarray(vector, dtype=float)
        if (
            x.shape != (30,)
            or not np.all(np.isfinite(x))
            or np.any(x < 0.0)
            or np.any(x > 1.0)
        ):
            raise LIRCMOPPaperBindingError(
                "LIR-CMOP vector must contain 30 finite values in [0,1]"
            )

        index = self.problem_index
        if index <= 4:
            objectives, paper_constraints = self._evaluate_1_to_4(x)
        elif index <= 8:
            objectives, paper_constraints = self._evaluate_5_to_8(x)
        elif index <= 12:
            objectives, paper_constraints = self._evaluate_9_to_12(x)
        else:
            objectives, paper_constraints = self._evaluate_13_to_14(x)

        values = (*objectives, *paper_constraints)
        if not all(isfinite(value) for value in values):
            raise FloatingPointError(
                "LIR-CMOP paper evaluator returned a nonfinite value"
            )
        constraints = tuple(-value for value in paper_constraints)
        return objectives, constraints

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
    ) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
        """Preserve scalar operation order for exact batch equivalence."""

        if event_id != 0:
            raise LIRCMOPPaperBindingError("LIR-CMOP is a TS1 benchmark")
        return tuple(self(vector, event_id) for vector in vectors)

    def _evaluate_1_to_4(
        self,
        x: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, ...]]:
        g1, g2 = _linkage_sums(x, indexed_angles=False)
        x1 = float(x[0])
        f1 = x1 + g1
        f2 = (1.0 - x1 * x1 if self.problem_index in {1, 3}
              else 1.0 - sqrt(x1)) + g2
        constraints = [
            (0.51 - g1) * (g1 - 0.5),
            (0.51 - g2) * (g2 - 0.5),
        ]
        if self.problem_index in {3, 4}:
            constraints.append(sin(20.0 * pi * x1) - 0.5)
        return (f1, f2), tuple(constraints)

    def _evaluate_5_to_8(
        self,
        x: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, ...]]:
        g1, g2 = _linkage_sums(x, indexed_angles=True)
        x1 = float(x[0])
        f1 = x1 + 10.0 * g1 + 0.7057
        second_shape = (
            1.0 - sqrt(x1)
            if self.problem_index in {5, 7}
            else 1.0 - x1 * x1
        )
        f2 = second_shape + 10.0 * g2 + 0.7057
        p_values, q_values, a_values, b_values = _ELLIPSES[
            self.problem_index
        ]
        constraints = tuple(
            _paper_ellipse_constraint(
                f1,
                f2,
                p=p,
                q=q,
                a=a,
                b=b,
            )
            for p, q, a, b in zip(
                p_values,
                q_values,
                a_values,
                b_values,
                strict=True,
            )
        )
        return (f1, f2), constraints

    def _evaluate_9_to_12(
        self,
        x: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        g1, g2 = _linkage_sums(x, indexed_angles=True)
        x1 = float(x[0])
        f1 = 1.7057 * x1 * (10.0 * g1 + 1.0)
        second_shape = (
            1.0 - x1 * x1
            if self.problem_index in {9, 12}
            else 1.0 - sqrt(x1)
        )
        f2 = 1.7057 * second_shape * (10.0 * g2 + 1.0)
        p, q, a, b, offset = _WAVE_CONSTRAINTS[self.problem_index]
        ellipse = _paper_ellipse_constraint(
            f1,
            f2,
            p=p,
            q=q,
            a=a,
            b=b,
        )
        alpha = 0.25 * pi
        wave = (
            f1 * sin(alpha)
            + f2 * cos(alpha)
            - sin(
                4.0
                * pi
                * (f1 * cos(alpha) - f2 * sin(alpha))
            )
            - offset
        )
        return (f1, f2), (ellipse, wave)

    def _evaluate_13_to_14(
        self,
        x: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, ...]]:
        g1 = sum(
            10.0 * (float(x[index]) - 0.5) ** 2
            for index in range(2, 30)
        )
        radius = 1.7057 + g1
        x1_angle = 0.5 * pi * float(x[0])
        x2_angle = 0.5 * pi * float(x[1])
        f1 = radius * cos(x1_angle) * cos(x2_angle)
        f2 = radius * cos(x1_angle) * sin(x2_angle)
        f3 = radius * sin(x1_angle)
        radial_square = f1 * f1 + f2 * f2 + f3 * f3
        constraints = [
            (radial_square - 9.0) * (radial_square - 4.0),
            (radial_square - 3.61) * (radial_square - 3.24),
        ]
        if self.problem_index == 14:
            constraints.append(
                (radial_square - 3.0625) * (radial_square - 2.56)
            )
        return (f1, f2, f3), tuple(constraints)
