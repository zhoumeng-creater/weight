from __future__ import annotations

import math
import platform

import numpy as np
import pytest

from benchmark_adapters.cdf_operational import CDFOperationalEvaluator


_LINUX_TWO_ULP_KATS = {
    11: (
        (7.004380436663206, 6.437166556824805),
        (0.2131966011250105,),
    ),
    13: (
        (6.4049377857689915, 6.32783472958726),
        (-10.576928457076937,),
    ),
}


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Windows is covered by the original bit-exact CDF known answers",
)
@pytest.mark.parametrize("problem_index", sorted(_LINUX_TWO_ULP_KATS))
def test_linux_cdf_known_answers_remain_within_two_ulps(
    problem_index: int,
) -> None:
    evaluator = CDFOperationalEvaluator(
        problem_index=problem_index,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    lower = np.asarray(evaluator.lower_bounds)
    upper = np.asarray(evaluator.upper_bounds)
    vector = lower + 0.25 * (upper - lower)
    actual = evaluator(vector, 7)
    expected = _LINUX_TWO_ULP_KATS[problem_index]

    for actual_group, expected_group in zip(actual, expected, strict=True):
        for actual_value, expected_value in zip(
            actual_group,
            expected_group,
            strict=True,
        ):
            assert abs(actual_value - expected_value) <= (
                2 * math.ulp(expected_value)
            )
