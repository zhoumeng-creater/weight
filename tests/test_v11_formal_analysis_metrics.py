from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from analysis.checkpoint_metrics import (
    AnalyticReferenceScale,
    FormalMetricError,
    e1e2_sequence_endpoints,
    e2_transfer_early_auc,
    event_anytime_auc,
    event_early_auc,
    exact_hypervolume,
    negative_transfer_rate,
    rolling_nhv,
    static_cdf_nhv,
)


def _curve(first_five: tuple[float, ...], tail: float | None = None) -> tuple[float, ...]:
    if len(first_five) != 5:
        raise AssertionError("fixture requires five early values")
    fill = first_five[-1] if tail is None else tail
    return first_five + (fill,) * 16


def _reference_scale_2d() -> AnalyticReferenceScale:
    axis = np.linspace(0.0, 1.0, 10_000)
    reference = np.column_stack((axis, 1.0 - axis))
    return AnalyticReferenceScale.from_reference_front(reference)


def _inclusion_exclusion_hv(
    front: np.ndarray,
    reference: np.ndarray,
) -> float:
    volume = 0.0
    for size in range(1, len(front) + 1):
        sign = 1.0 if size % 2 else -1.0
        for subset in combinations(front, size):
            lower = np.max(np.asarray(subset), axis=0)
            volume += sign * float(np.prod(reference - lower))
    return volume


def test_exact_hypervolume_2d_known_answers() -> None:
    assert exact_hypervolume([], (1.0, 1.0)) == 0.0
    assert exact_hypervolume([(0.2, 0.3)], (1.0, 1.0)) == pytest.approx(
        0.56
    )
    assert exact_hypervolume(
        [(0.2, 0.8), (0.5, 0.4)],
        (1.0, 1.0),
    ) == pytest.approx(0.36)
    assert exact_hypervolume(
        [(0.2, 0.8), (0.5, 0.4), (0.7, 0.9), (0.2, 0.8)],
        (1.0, 1.0),
    ) == pytest.approx(0.36)


def test_exact_hypervolume_3d_known_answers() -> None:
    assert exact_hypervolume([(0.2, 0.3, 0.4)], (1.0, 1.0, 1.0)) == (
        pytest.approx(0.336)
    )
    assert exact_hypervolume(
        [(0.2, 0.8, 0.2), (0.5, 0.4, 0.5)],
        (1.0, 1.0, 1.0),
    ) == pytest.approx(0.228)
    assert exact_hypervolume(
        [(0.0, 0.0, 0.0)],
        (1.0, 1.0, 1.0),
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_exact_hypervolume_matches_independent_box_union(
    dimension: int,
) -> None:
    rng = np.random.default_rng(8400 + dimension)
    reference = np.ones(dimension)
    for count in range(1, 7):
        for _ in range(10):
            front = rng.uniform(0.0, 0.95, size=(count, dimension))
            expected = _inclusion_exclusion_hv(front, reference)
            assert exact_hypervolume(front, reference) == pytest.approx(
                expected,
                abs=1e-12,
            )


def test_exact_hypervolume_rejects_outside_reference() -> None:
    with pytest.raises(FormalMetricError, match="beyond"):
        exact_hypervolume([(1.01, 0.2)], (1.0, 1.0))


def test_static_cdf_nhv_uses_frozen_normalization_and_clipping() -> None:
    scale = _reference_scale_2d()
    expected = (0.9 * 0.9) / (1.1**2)
    assert static_cdf_nhv([(0.2, 0.2)], scale) == pytest.approx(expected)
    assert static_cdf_nhv([], scale) == 0.0
    assert static_cdf_nhv([(-5.0, -5.0)], scale) == pytest.approx(1.0)
    assert static_cdf_nhv([(50.0, 50.0)], scale) == 0.0


def test_analytic_reference_scale_uses_extrema_not_nominal_sample_count() -> None:
    scale = AnalyticReferenceScale.from_reference_front([(0.0, 1.0)])
    assert scale.minima == (0.0, 1.0)
    assert scale.maxima == (0.0, 1.0)
    assert scale.point_count == 1
    direct = AnalyticReferenceScale.from_extrema(
        minima=(0.0, 0.0),
        maxima=(1.0, 1.0),
    )
    assert direct.point_count is None
    reference = np.zeros((10_000, 2), dtype=float)
    reference[3, 1] = np.nan
    with pytest.raises(FormalMetricError, match="finite"):
        AnalyticReferenceScale.from_reference_front(reference)


def test_rolling_nhv_known_answers_and_nonnegative_guard() -> None:
    assert rolling_nhv([]) == 0.0
    assert rolling_nhv([(0.0, 0.0, 0.0)]) == pytest.approx(1.0)
    assert rolling_nhv([(1.0, 1.0, 1.0)]) == pytest.approx(0.125)
    with pytest.raises(FormalMetricError, match="nonnegative"):
        rolling_nhv([(-0.1, 1.0, 1.0)])


def test_frozen_anytime_and_early_auc_known_answers() -> None:
    assert event_anytime_auc((0.4,) * 21) == pytest.approx(0.4)
    curve = _curve((0.0, 0.25, 0.5, 0.75, 1.0), tail=1.0)
    assert event_early_auc(curve) == pytest.approx(0.5)
    with pytest.raises(FormalMetricError, match="21"):
        event_anytime_auc((0.0, 1.0))


def test_e2_transfer_excludes_event_zero_and_normalizes_each_event() -> None:
    event_zero = _curve((1.0,) * 5)
    event_one = _curve((0.0, 0.25, 0.5, 0.75, 1.0))
    event_two = _curve((0.2,) * 5)
    assert e2_transfer_early_auc(
        (event_zero, event_one, event_two)
    ) == pytest.approx(0.35)


def test_negative_transfer_is_paired_and_strictly_below_minus_point01() -> None:
    initial = _curve((0.0,) * 5)
    exactly_threshold = _curve((0.0,) * 5)
    harmed = _curve((0.0,) * 5)
    improved = _curve((0.60,) * 5)
    rate = negative_transfer_rate(
        (initial, exactly_threshold, harmed, improved),
        (
            initial,
            _curve((0.01,) * 5),
            _curve((0.02,) * 5),
            _curve((0.50,) * 5),
        ),
    )
    assert rate == pytest.approx(1.0 / 3.0)


def test_sequence_endpoints_equal_weight_events() -> None:
    low = _curve((0.2,) * 5, tail=0.2)
    high = _curve((0.8,) * 5, tail=0.8)
    endpoints = e1e2_sequence_endpoints(
        (low, high),
        include_transfer=True,
    )
    assert endpoints.anytime_nhv_auc == pytest.approx(0.5)
    assert endpoints.final_nhv == pytest.approx(0.5)
    assert endpoints.transfer_early_auc == pytest.approx(0.8)
