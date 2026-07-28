"""v1.1 constrained multi-objective and parameter-memory primitives.

Semantic-port provenance:
    FORMAL_V1/dt_ramde_formal/core.py
    SHA-256 def7b3e8c3c41e088abe1fd50ffc6ab1a2511525151d38742b0eba38ed9f2369

This port uses the shared v1.1 ``EvaluationResult`` contract and removes the
historical hard-coded three-objective terminal selector. Problem-specific
selection remains an adapter-bound pure function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from evaluation.contracts import EvaluationResult


class CoreContractViolation(RuntimeError):
    """A zero-tolerance algorithm-core invariant was violated."""


@dataclass
class Candidate:
    vector: np.ndarray
    evaluation: EvaluationResult
    lineage_node_id: str
    rank: int = 10**9
    crowding: float = 0.0

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)
        if self.vector.ndim != 1 or not np.all(np.isfinite(self.vector)):
            raise CoreContractViolation("candidate vector must be finite and 1-D")
        if not self.lineage_node_id:
            raise CoreContractViolation("lineage_node_id must be nonempty")

    @property
    def candidate_id(self) -> str:
        return self.evaluation.candidate_id

    @property
    def objectives(self) -> tuple[float, ...]:
        return self.evaluation.objectives

    @property
    def constraints(self) -> tuple[float, ...]:
        return self.evaluation.constraints

    @property
    def feasible(self) -> bool:
        return self.evaluation.feasible

    def normalized_violation(self, constraint_scales: Sequence[float]) -> float:
        scales = _validated_scales(self, constraint_scales)
        return sum(
            max(0.0, value) / scale
            for value, scale in zip(self.constraints, scales, strict=True)
        )


@dataclass(frozen=True)
class ShadeSuccess:
    """Result-blind constrained multi-objective SHADE success amount."""

    success: bool
    reason: str
    delta: float


def _validated_scales(
    candidate: Candidate, constraint_scales: Sequence[float]
) -> tuple[float, ...]:
    scales = tuple(float(value) for value in constraint_scales)
    if len(scales) != len(candidate.constraints):
        raise CoreContractViolation("constraint scales must align with constraints")
    if not all(math.isfinite(value) and value > 0.0 for value in scales):
        raise CoreContractViolation("constraint scales must be finite and positive")
    return scales


def _assert_unique_ids(candidates: Sequence[Candidate]) -> None:
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise CoreContractViolation("duplicate candidate_id in candidate collection")


def _assert_common_evaluation_identity(candidates: Sequence[Candidate]) -> None:
    if not candidates:
        return
    objective_names = candidates[0].evaluation.objective_names
    constraint_names = candidates[0].evaluation.constraint_names
    for candidate in candidates[1:]:
        if candidate.evaluation.objective_names != objective_names:
            raise CoreContractViolation(
                "objective identity must match across candidates"
            )
        if candidate.evaluation.constraint_names != constraint_names:
            raise CoreContractViolation(
                "constraint identity must match across candidates"
            )


def dominates(left: Candidate, right: Candidate) -> bool:
    """Return Pareto dominance under the canonical all-minimization contract."""

    if len(left.objectives) != len(right.objectives):
        raise CoreContractViolation("objective dimensions must match")
    if left.evaluation.objective_names != right.evaluation.objective_names:
        raise CoreContractViolation("objective identity must match")
    return all(
        left_value <= right_value
        for left_value, right_value in zip(
            left.objectives, right.objectives, strict=True
        )
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(
            left.objectives, right.objectives, strict=True
        )
    )


def shade_success_improvement(
    target: Candidate,
    trial: Candidate,
    constraint_scales: Sequence[float],
    *,
    trial_in_next_population: bool,
    target_in_next_population: bool,
) -> ShadeSuccess:
    """Return the frozen WGT-SHADE-CMO-SUCCESS-01 pairwise improvement.

    The result includes both the frozen paired population transition and the
    pairwise improvement amount used by SHADE's success history.
    """

    _assert_common_evaluation_identity((target, trial))
    if not trial_in_next_population:
        return ShadeSuccess(
            success=False,
            reason="TRIAL_NOT_IN_NEXT_POPULATION",
            delta=0.0,
        )
    if target_in_next_population:
        return ShadeSuccess(
            success=False,
            reason="PAIRED_TARGET_REMAINS_IN_NEXT_POPULATION",
            delta=0.0,
        )
    target_cv = target.normalized_violation(constraint_scales)
    trial_cv = trial.normalized_violation(constraint_scales)

    if not target.feasible:
        if trial.feasible:
            delta = 1.0 + target_cv / (1.0 + target_cv)
            reason = "INFEASIBLE_TO_FEASIBLE"
        elif trial_cv < target_cv:
            delta = (target_cv - trial_cv) / (1.0 + target_cv)
            reason = "INFEASIBLE_CV_REDUCTION"
        else:
            return ShadeSuccess(
                success=False,
                reason="NO_STRICT_PAIRWISE_IMPROVEMENT",
                delta=0.0,
            )
    elif trial.feasible and dominates(trial, target):
        relative_terms = []
        for target_value, trial_value in zip(
            target.objectives,
            trial.objectives,
            strict=True,
        ):
            denominator = abs(target_value) + abs(trial_value)
            relative_terms.append(
                0.0
                if denominator == 0.0
                else (target_value - trial_value) / denominator
            )
        delta = math.fsum(relative_terms) / len(relative_terms)
        reason = "FEASIBLE_PARETO_DOMINANCE"
    else:
        return ShadeSuccess(
            success=False,
            reason="NO_STRICT_PAIRWISE_IMPROVEMENT",
            delta=0.0,
        )

    if not math.isfinite(delta) or delta <= 0.0:
        raise CoreContractViolation(
            "strict SHADE success must produce a finite positive delta"
        )
    return ShadeSuccess(success=True, reason=reason, delta=delta)


def _assign_crowding(front: Sequence[Candidate]) -> None:
    if not front:
        return
    for candidate in front:
        candidate.crowding = 0.0
    if len(front) <= 2:
        for candidate in front:
            candidate.crowding = math.inf
        return
    for objective_index in range(len(front[0].objectives)):
        ordered = sorted(
            front,
            key=lambda candidate: (
                candidate.objectives[objective_index],
                candidate.candidate_id,
            ),
        )
        ordered[0].crowding = math.inf
        ordered[-1].crowding = math.inf
        span = (
            ordered[-1].objectives[objective_index]
            - ordered[0].objectives[objective_index]
        )
        if span <= 0.0:
            continue
        for position in range(1, len(ordered) - 1):
            candidate = ordered[position]
            if math.isfinite(candidate.crowding):
                candidate.crowding += (
                    ordered[position + 1].objectives[objective_index]
                    - ordered[position - 1].objectives[objective_index]
                ) / span


def _quadratic_nondominated_fronts(
    feasible: Sequence[Candidate],
) -> list[list[Candidate]]:
    """Return canonical fronts from one dominance relation per unordered pair."""

    counts = {candidate.candidate_id: 0 for candidate in feasible}
    dominated: dict[str, list[Candidate]] = {
        candidate.candidate_id: [] for candidate in feasible
    }
    fronts: list[list[Candidate]] = [[]]
    objective_rows = [candidate.objectives for candidate in feasible]
    for left_index, left in enumerate(feasible):
        left_values = objective_rows[left_index]
        for right_index in range(left_index + 1, len(feasible)):
            right = feasible[right_index]
            right_values = objective_rows[right_index]
            left_strictly_better = False
            right_strictly_better = False
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            ):
                if left_value < right_value:
                    left_strictly_better = True
                elif right_value < left_value:
                    right_strictly_better = True
                if left_strictly_better and right_strictly_better:
                    break
            if left_strictly_better and not right_strictly_better:
                dominated[left.candidate_id].append(right)
                counts[right.candidate_id] += 1
            elif right_strictly_better and not left_strictly_better:
                dominated[right.candidate_id].append(left)
                counts[left.candidate_id] += 1
    for left in feasible:
        if counts[left.candidate_id] == 0:
            left.rank = 0
            fronts[0].append(left)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[Candidate] = []
        for left in fronts[front_index]:
            for right in dominated[left.candidate_id]:
                counts[right.candidate_id] -= 1
                if counts[right.candidate_id] == 0:
                    right.rank = front_index + 1
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts


def _two_objective_nondominated_fronts(
    feasible: Sequence[Candidate],
) -> list[list[Candidate]]:
    """Return exact two-objective fronts in O(n log n) time.

    For a two-objective minimization problem, a point's front rank is one plus
    the greatest rank among points that dominate it.  Processing equal first
    objectives together prevents exact objective duplicates from dominating
    one another, while a Fenwick prefix-maximum tree answers the greatest rank
    among earlier first-objective groups whose second objective is no larger.
    """

    if not feasible:
        return [[]]

    ordered = sorted(
        feasible,
        key=lambda candidate: (
            candidate.objectives[0],
            candidate.objectives[1],
            candidate.candidate_id,
        ),
    )
    second_values = sorted({candidate.objectives[1] for candidate in ordered})
    second_indices = {value: index + 1 for index, value in enumerate(second_values)}
    prefix_maximum = [-1] * (len(second_values) + 1)

    def query(index: int) -> int:
        value = -1
        while index > 0:
            value = max(value, prefix_maximum[index])
            index -= index & -index
        return value

    def update(index: int, value: int) -> None:
        while index < len(prefix_maximum):
            prefix_maximum[index] = max(prefix_maximum[index], value)
            index += index & -index

    fronts_by_rank: list[list[Candidate]] = []
    first_start = 0
    while first_start < len(ordered):
        first_value = ordered[first_start].objectives[0]
        first_end = first_start + 1
        while (
            first_end < len(ordered) and ordered[first_end].objectives[0] == first_value
        ):
            first_end += 1

        pending_updates: list[tuple[int, int]] = []
        greatest_same_first_rank = -1
        second_start = first_start
        while second_start < first_end:
            second_value = ordered[second_start].objectives[1]
            second_end = second_start + 1
            while (
                second_end < first_end
                and ordered[second_end].objectives[1] == second_value
            ):
                second_end += 1

            second_index = second_indices[second_value]
            rank = (
                max(
                    query(second_index),
                    greatest_same_first_rank,
                )
                + 1
            )
            if rank == len(fronts_by_rank):
                fronts_by_rank.append([])
            group = ordered[second_start:second_end]
            for candidate in group:
                candidate.rank = rank
                fronts_by_rank[rank].append(candidate)
            greatest_same_first_rank = max(greatest_same_first_rank, rank)
            pending_updates.append((second_index, rank))
            second_start = second_end

        for second_index, rank in pending_updates:
            update(second_index, rank)
        first_start = first_end

    return fronts_by_rank


def assign_rank_and_crowding(candidates: Sequence[Candidate]) -> None:
    """Assign nondominated rank and deterministic crowding to feasible points."""

    _assert_unique_ids(candidates)
    _assert_common_evaluation_identity(candidates)
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        dimension = len(feasible[0].objectives)
        if any(len(candidate.objectives) != dimension for candidate in feasible):
            raise CoreContractViolation("objective dimensions must match")
        fronts = (
            _two_objective_nondominated_fronts(feasible)
            if dimension == 2
            else _quadratic_nondominated_fronts(feasible)
        )
    else:
        fronts = [[]]

    for front in fronts:
        _assign_crowding(front)

    for candidate in candidates:
        if not candidate.feasible:
            candidate.rank = 10**6
            candidate.crowding = 0.0


def constrained_sort_key(
    candidate: Candidate, constraint_scales: Sequence[float]
) -> tuple[Any, ...]:
    if candidate.feasible:
        crowding_key = (
            -candidate.crowding if math.isfinite(candidate.crowding) else -math.inf
        )
        return 0, candidate.rank, crowding_key, candidate.candidate_id
    return (
        1,
        candidate.normalized_violation(constraint_scales),
        candidate.rank,
        candidate.candidate_id,
    )


def environmental_select(
    candidates: Sequence[Candidate],
    *,
    population_size: int,
    constraint_scales: Sequence[float],
) -> list[Candidate]:
    if population_size < 1:
        raise CoreContractViolation("population_size must be positive")
    if len(candidates) < population_size:
        raise CoreContractViolation("candidate pool is smaller than population_size")
    assign_rank_and_crowding(candidates)
    for candidate in candidates:
        _validated_scales(candidate, constraint_scales)
    return sorted(
        candidates,
        key=lambda candidate: constrained_sort_key(candidate, constraint_scales),
    )[:population_size]


def maintain_nondominated_archive(
    candidates: Sequence[Candidate],
    *,
    capacity: int,
    constraint_scales: Sequence[float],
) -> list[Candidate]:
    if capacity < 1:
        raise CoreContractViolation("archive capacity must be positive")
    _assert_unique_ids(candidates)
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return []
    assign_rank_and_crowding(feasible)
    for candidate in feasible:
        _validated_scales(candidate, constraint_scales)
    return sorted(
        (candidate for candidate in feasible if candidate.rank == 0),
        key=lambda candidate: constrained_sort_key(candidate, constraint_scales),
    )[:capacity]


def order_known_nondominated_archive(
    candidates: Sequence[Candidate],
    *,
    capacity: int,
    constraint_scales: Sequence[float],
    fixed_evaluation_schema: bool = False,
) -> list[Candidate]:
    """Order a caller-proven nondominated set without rechecking dominance."""

    if capacity < 1:
        raise CoreContractViolation("archive capacity must be positive")
    _assert_unique_ids(candidates)
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return []
    _assert_common_evaluation_identity(feasible)
    dimension = len(feasible[0].objectives)
    if any(len(candidate.objectives) != dimension for candidate in feasible):
        raise CoreContractViolation("objective dimensions must match")
    if fixed_evaluation_schema:
        # Checkpoint candidates are immutable EvaluationResult instances:
        # construction has already validated aligned lengths and finite values,
        # and the common-identity check above fixes both schemas for this
        # snapshot. Validate the shared scale vector once, not per candidate.
        validated_scales = _validated_scales(
            feasible[0],
            constraint_scales,
        )
        for candidate in feasible:
            candidate.rank = 0
    else:
        validated_scales = tuple(constraint_scales)
        for candidate in feasible:
            candidate.rank = 0
            _validated_scales(candidate, constraint_scales)
    _assign_crowding(feasible)
    return sorted(
        feasible,
        key=lambda candidate: constrained_sort_key(candidate, validated_scales),
    )[:capacity]


def repair_midpoint(
    vector: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    vector = np.asarray(vector, dtype=float)
    target = np.asarray(target, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (
        vector.shape == target.shape == lower.shape == upper.shape
        and vector.ndim == 1
    ):
        return None, False
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(target))
        and np.all(lower <= upper)
    ):
        return None, False
    repaired = vector.copy()
    below = repaired < lower
    above = repaired > upper
    changed = bool(np.any(below) or np.any(above))
    repaired[below] = (lower[below] + target[below]) / 2.0
    repaired[above] = (upper[above] + target[above]) / 2.0
    if not np.all(np.isfinite(repaired)):
        return None, changed
    return repaired, changed


@dataclass(frozen=True)
class ParameterSample:
    f: float
    cr: float
    f_draws: tuple[float, ...]
    raw_cr: float
    fallback_f: bool
    fallback_cr: bool


def sample_parameters(
    rng: np.random.Generator,
    *,
    mu_f: float,
    mu_cr: float,
    f_draws: Sequence[float] | None = None,
    cr_draw: float | None = None,
) -> ParameterSample:
    draws: list[float] = []
    sampled_f: float | None = None
    for attempt in range(100):
        raw = (
            float(f_draws[attempt])
            if f_draws is not None and attempt < len(f_draws)
            else float(mu_f + 0.1 * rng.standard_cauchy())
        )
        draws.append(raw)
        if math.isfinite(raw) and raw > 0.0:
            sampled_f = min(raw, 1.0)
            break
    fallback_f = sampled_f is None
    if fallback_f:
        sampled_f = 0.5

    raw_cr = (
        float(cr_draw) if cr_draw is not None else float(rng.normal(mu_cr, 0.1))
    )
    fallback_cr = not math.isfinite(raw_cr)
    sampled_cr = 0.5 if fallback_cr else float(np.clip(raw_cr, 0.0, 1.0))
    return ParameterSample(
        f=sampled_f,
        cr=sampled_cr,
        f_draws=tuple(draws),
        raw_cr=raw_cr,
        fallback_f=fallback_f,
        fallback_cr=fallback_cr,
    )


def update_mg(
    memory_f: list[float],
    memory_cr: list[float],
    pointer: int,
    successes: Sequence[tuple[float, float, float]],
) -> int:
    if not memory_f or len(memory_f) != len(memory_cr):
        raise CoreContractViolation("M_g memories must be nonempty and aligned")
    if not 0 <= pointer < len(memory_f):
        raise CoreContractViolation("M_g pointer is out of range")
    valid = [
        (float(f_value), float(cr_value), float(weight))
        for f_value, cr_value, weight in successes
        if weight > 0.0
    ]
    if not valid:
        return pointer
    if not all(
        math.isfinite(f_value)
        and math.isfinite(cr_value)
        and math.isfinite(weight)
        and 0.0 < f_value <= 1.0
        and 0.0 <= cr_value <= 1.0
        for f_value, cr_value, weight in valid
    ):
        raise CoreContractViolation("invalid M_g success tuple")

    weights = np.asarray([item[2] for item in valid], dtype=float)
    weights /= weights.sum()
    f_values = np.asarray([item[0] for item in valid], dtype=float)
    cr_values = np.asarray([item[1] for item in valid], dtype=float)
    denominator = float(np.sum(weights * f_values))
    memory_f[pointer] = (
        float(np.sum(weights * f_values * f_values) / denominator)
        if denominator > 0.0
        else 0.5
    )
    memory_cr[pointer] = float(np.sum(weights * cr_values))
    return (pointer + 1) % len(memory_f)
