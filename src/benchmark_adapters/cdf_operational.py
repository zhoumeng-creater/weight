"""Corrective operational binding for the CDF1--15 benchmark suite.

The version-of-record paper contains several equations that conflict with its
own stated Pareto fronts and with the authors' executable implementation.  The
corrective R8C binding therefore uses the authors' fixed CMLSGA commit as its
operational authority and records every paper conflict in a separate,
result-blind amendment.

The historical :class:`benchmark_adapters.r4_evaluators.CDFEvaluator` remains
unchanged.  This module corrects the one Python transcription that differs
from both the paper and the author oracle (CDF1), and gives the pre-existing
CDF9 non-real square-root domain a stable typed error.  It does not invent an
extension outside the source-defined real domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import pi, sin, sqrt
from pathlib import Path

import numpy as np

from .r4_evaluators import (
    CDFEvaluator,
    CDF_ORACLE_COMMIT,
    CDF_ORACLE_CPP_SHA256,
    CDF_PAPER_DOI,
    _groups,
    _sgn,
)


CDF_OPERATIONAL_SUITE_ID = "CDF-1-15-CMLSGA-1926A5A1-OPERATIONAL"
CDF_OPERATIONAL_AUTHORITY_ID = (
    "WGT-V11-R8C-CDF-OPERATIONAL-AUTHORITY-1.0.0"
)
CDF_OPERATIONAL_EVALUATOR_VERSION = "1.0.0"


class CDFDomainUndefinedError(FloatingPointError):
    """A source equation has no real value at an otherwise bounded input."""


def _module_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CDFOperationalEvaluator(CDFEvaluator):
    """Result-blind CDF evaluator bound to the author operational semantics."""

    @property
    def binding_sha256(self) -> str:
        payload = "|".join(
            (
                CDF_OPERATIONAL_AUTHORITY_ID,
                CDF_OPERATIONAL_EVALUATOR_VERSION,
                CDF_PAPER_DOI,
                CDF_ORACLE_COMMIT,
                CDF_ORACLE_CPP_SHA256,
                _module_sha256(),
                super().binding_sha256,
                self.problem_id,
                self.profile,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _evaluate_equations(
        self,
        x: np.ndarray,
        time_value: float,
        time_vector: tuple[float, ...],
    ) -> tuple[tuple[float, float], tuple[float, ...]]:
        if self.problem_index == 1:
            return self._evaluate_cdf1_oracle(x, time_value)
        if self.problem_index == 9:
            gt = sin(0.5 * pi * time_value)
            mt = 0.5 + abs(gt)
            shaped = 1.0 - (mt * float(x[0])) ** mt
            if shaped < 0.0:
                raise CDFDomainUndefinedError(
                    "CDF9 source constraint sqrt(1-(M*x1)^H) "
                    "is outside its real domain"
                )
        return super()._evaluate_equations(x, time_value, time_vector)

    @staticmethod
    def _evaluate_cdf1_oracle(
        x: np.ndarray,
        time_value: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Transcribe CDF1 with both constraint offsets inside the exponents."""

        gt = sin(0.5 * pi * time_value)

        def residual(index: int) -> float:
            exponent = (
                0.5 * (2.0 + 3.0 * (index - 2) / 8.0)
                + abs(gt)
            )
            return float(x[index - 1]) - float(x[0]) ** exponent

        n1, n2, s1, s2 = _groups(x, residual)
        x1 = float(x[0])
        objectives = (
            x1 + 2.0 * s1 / n1,
            (1.0 - x1) ** 2 + 2.0 * s2 / n2,
        )
        first_shape = 0.5 * (1.0 - x1) - (1.0 - x1) ** 2
        second_shape = (
            0.25 * sqrt(1.0 - x1) - 0.5 * (1.0 - x1)
        )
        first_exponent = (
            1.0
            + abs(gt)
            - _sgn(first_shape) * sqrt(abs(first_shape))
        )
        second_exponent = (
            1.375
            + abs(gt)
            - _sgn(second_shape) * sqrt(abs(second_shape))
        )
        constraints = (
            float(x[1]) - x1**first_exponent,
            float(x[3]) - x1**second_exponent,
        )
        return (
            (float(objectives[0]), float(objectives[1])),
            constraints,
        )
