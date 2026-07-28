from __future__ import annotations

from math import isnan

import numpy as np
import pytest

from dt_ramde_v11.core import (
    Candidate,
    CoreContractViolation,
    dominates,
    environmental_select,
    maintain_nondominated_archive,
    repair_midpoint,
    sample_parameters,
    shade_success_improvement,
    update_mg,
)
from evaluation.contracts import EvaluationResult


def _candidate(
    candidate_id: str,
    objectives: tuple[float, ...],
    constraints: tuple[float, ...] = (-1.0,),
) -> Candidate:
    return Candidate(
        vector=np.asarray([float(candidate_id.removeprefix("c"))]),
        evaluation=EvaluationResult(
            candidate_id=candidate_id,
            objectives=objectives,
            objective_names=tuple(f"f{i}" for i in range(len(objectives))),
            constraints=constraints,
            constraint_names=tuple(f"c{i}" for i in range(len(constraints))),
        ),
        lineage_node_id=f"lineage-{candidate_id}",
    )


def test_pareto_dominance_is_minimization_and_requires_strict_improvement() -> None:
    better = _candidate("c1", (1.0, 2.0))
    worse = _candidate("c2", (2.0, 3.0))
    tradeoff = _candidate("c3", (0.5, 4.0))
    equal = _candidate("c4", (1.0, 2.0))

    assert dominates(better, worse) is True
    assert dominates(worse, better) is False
    assert dominates(better, tradeoff) is False
    assert dominates(better, equal) is False


def test_dominance_rejects_mismatched_objective_identity() -> None:
    left = _candidate("c1", (1.0, 2.0))
    right = Candidate(
        vector=np.asarray([2.0]),
        evaluation=EvaluationResult(
            candidate_id="c2",
            objectives=(2.0, 3.0),
            objective_names=("different", "f1"),
            constraints=(-1.0,),
            constraint_names=("c0",),
        ),
        lineage_node_id="lineage-c2",
    )
    with pytest.raises(CoreContractViolation, match="objective identity"):
        dominates(left, right)


def test_constraint_dominance_prefers_feasible_then_normalized_violation() -> None:
    feasible = _candidate("c1", (10.0,), (-0.1, -2.0))
    infeasible_low = _candidate("c2", (0.0,), (1.5, -1.0))
    infeasible_high = _candidate("c3", (-10.0,), (2.0, -1.0))

    selected = environmental_select(
        [infeasible_high, infeasible_low, feasible],
        population_size=2,
        constraint_scales=(2.0, 1.0),
    )
    assert [candidate.candidate_id for candidate in selected] == ["c1", "c2"]
    assert infeasible_low.normalized_violation((2.0, 1.0)) == 0.75


def test_nondominated_archive_is_feasible_unique_and_deterministic() -> None:
    candidates = [
        _candidate("c1", (1.0, 4.0)),
        _candidate("c2", (2.0, 2.0)),
        _candidate("c3", (4.0, 1.0)),
        _candidate("c4", (3.0, 3.0)),
        _candidate("c5", (0.0, 0.0), (0.1,)),
    ]
    archive = maintain_nondominated_archive(
        candidates,
        capacity=3,
        constraint_scales=(1.0,),
    )
    assert [candidate.candidate_id for candidate in archive] == ["c1", "c3", "c2"]
    assert all(candidate.feasible for candidate in archive)
    assert all(
        not dominates(left, right)
        for left in archive
        for right in archive
        if left is not right
    )

    with pytest.raises(CoreContractViolation, match="duplicate candidate_id"):
        maintain_nondominated_archive(
            [candidates[0], candidates[0]],
            capacity=2,
            constraint_scales=(1.0,),
        )

    with pytest.raises(CoreContractViolation, match="positive"):
        environmental_select(
            candidates[:3],
            population_size=2,
            constraint_scales=(0.0,),
        )


def test_midpoint_repair_and_invalid_shape_are_explicit() -> None:
    repaired, changed = repair_midpoint(
        vector=np.asarray([-2.0, 12.0, 5.0]),
        target=np.asarray([4.0, 6.0, 5.0]),
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
    )
    np.testing.assert_allclose(repaired, [2.0, 8.0, 5.0])
    assert changed is True

    invalid, changed = repair_midpoint(
        vector=np.asarray([1.0, 2.0]),
        target=np.asarray([1.0]),
        lower=np.asarray([0.0]),
        upper=np.asarray([2.0]),
    )
    assert invalid is None
    assert changed is False


def test_parameter_sampling_bounds_and_fallback_are_known_answer() -> None:
    rng = np.random.default_rng(1)
    sample = sample_parameters(
        rng,
        mu_f=0.5,
        mu_cr=0.5,
        f_draws=[-1.0] * 100,
        cr_draw=float("nan"),
    )
    assert sample.f == 0.5
    assert sample.cr == 0.5
    assert sample.fallback_f is True
    assert sample.fallback_cr is True
    assert len(sample.f_draws) == 100
    assert isnan(sample.raw_cr)


def test_weighted_mg_update_matches_frozen_formula() -> None:
    memory_f = [0.5] * 10
    memory_cr = [0.5] * 10
    pointer = update_mg(
        memory_f,
        memory_cr,
        pointer=0,
        successes=((0.5, 0.2, 1.0), (1.0, 0.8, 1.0)),
    )
    assert pointer == 1
    assert memory_f[0] == pytest.approx(5.0 / 6.0)
    assert memory_cr[0] == pytest.approx(0.5)

    unchanged = update_mg(memory_f, memory_cr, pointer, successes=())
    assert unchanged == pointer


def test_shade_success_metric_known_answers_cover_all_three_cases() -> None:
    infeasible_target = _candidate("c1", (100.0, -50.0), (3.0,))
    feasible_trial = _candidate("c2", (200.0, 200.0), (-1.0,))
    feasibility = shade_success_improvement(
        infeasible_target,
        feasible_trial,
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    assert feasibility.success is True
    assert feasibility.reason == "INFEASIBLE_TO_FEASIBLE"
    assert feasibility.delta == pytest.approx(1.75)

    infeasible_trial = _candidate("c3", (-500.0, -500.0), (1.0,))
    cv_reduction = shade_success_improvement(
        infeasible_target,
        infeasible_trial,
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    assert cv_reduction.success is True
    assert cv_reduction.reason == "INFEASIBLE_CV_REDUCTION"
    assert cv_reduction.delta == pytest.approx(0.5)

    feasible_target = _candidate("c4", (2.0, 0.0, -2.0))
    dominating_trial = _candidate("c5", (1.0, 0.0, -4.0))
    pareto = shade_success_improvement(
        feasible_target,
        dominating_trial,
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    assert pareto.success is True
    assert pareto.reason == "FEASIBLE_PARETO_DOMINANCE"
    assert pareto.delta == pytest.approx(2.0 / 9.0)


def test_shade_success_metric_is_scale_invariant_and_rejects_nonimprovements() -> None:
    target = _candidate("c1", (2.0, 0.0, -2.0))
    trial = _candidate("c2", (1.0, 0.0, -4.0))
    baseline = shade_success_improvement(
        target,
        trial,
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    scaled = shade_success_improvement(
        _candidate("c3", (20.0, 0.0, -1.0)),
        _candidate("c4", (10.0, 0.0, -2.0)),
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    assert scaled.delta == pytest.approx(baseline.delta)

    incomparable = shade_success_improvement(
        _candidate("c5", (1.0, 2.0)),
        _candidate("c6", (2.0, 1.0)),
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    )
    assert incomparable.success is False
    assert incomparable.delta == 0.0

    diversity_only = shade_success_improvement(
        target,
        trial,
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=True,
    )
    assert diversity_only.success is False
    assert diversity_only.reason == "PAIRED_TARGET_REMAINS_IN_NEXT_POPULATION"

    discarded = shade_success_improvement(
        target,
        trial,
        (1.0,),
        trial_in_next_population=False,
        target_in_next_population=False,
    )
    assert discarded.success is False
    assert discarded.reason == "TRIAL_NOT_IN_NEXT_POPULATION"


def test_shade_success_weights_drive_existing_history_update_without_repair_penalty() -> None:
    memory_f = [0.5] * 10
    memory_cr = [0.5] * 10
    first_delta = shade_success_improvement(
        _candidate("c1", (2.0,)),
        _candidate("c2", (1.0,)),
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    ).delta
    second_delta = shade_success_improvement(
        _candidate("c3", (4.0,), (3.0,)),
        _candidate("c4", (4.0,), (1.0,)),
        (1.0,),
        trial_in_next_population=True,
        target_in_next_population=False,
    ).delta
    pointer = update_mg(
        memory_f,
        memory_cr,
        0,
        successes=(
            (0.5, 0.2, first_delta),
            (1.0, 0.8, second_delta),
        ),
    )
    normalized_first = first_delta / (first_delta + second_delta)
    normalized_second = 1.0 - normalized_first
    expected_f = (
        normalized_first * 0.5**2 + normalized_second * 1.0**2
    ) / (normalized_first * 0.5 + normalized_second * 1.0)
    expected_cr = normalized_first * 0.2 + normalized_second * 0.8
    assert pointer == 1
    assert memory_f[0] == pytest.approx(expected_f)
    assert memory_cr[0] == pytest.approx(expected_cr)
