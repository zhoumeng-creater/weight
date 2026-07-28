"""Shared R4 comparator finalization and contract checks."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from math import isfinite

import numpy as np

from dt_ramde_v11.core import (
    Candidate,
    CoreContractViolation,
    dominates,
    maintain_nondominated_archive,
    order_known_nondominated_archive,
)
from dt_ramde_v11.interfaces import (
    EventProblemAdapter,
    OptimizationResult,
)
from evaluation.contracts import TerminalCode, TerminalOutcome


class ComparatorBindingError(RuntimeError):
    """A comparator cannot satisfy the shared R4 execution contract."""


class ExactNondominatedAccumulator:
    """Incrementally retain the complete feasible global Pareto front."""

    def __init__(self) -> None:
        self._candidate_ids: set[str] = set()
        self._nondominated: list[Candidate] = []
        self._objective_dimension: int | None = None
        self._objective_names: tuple[str, ...] | None = None
        self._front_keys_2d: list[tuple[float, float]] = []
        self._front_groups_2d: dict[
            tuple[float, float], list[Candidate]
        ] = {}
        self._feasible_candidates_3d: list[Candidate] = []

    @property
    def has_candidates(self) -> bool:
        return bool(self._candidate_ids)

    def add(self, candidate: Candidate) -> None:
        candidate_id = candidate.candidate_id
        if candidate_id in self._candidate_ids:
            raise CoreContractViolation(
                "duplicate candidate_id in candidate collection"
            )
        self._candidate_ids.add(candidate_id)
        if not candidate.feasible:
            return
        dimension = len(candidate.objectives)
        objective_names = candidate.evaluation.objective_names
        if self._objective_dimension is None:
            self._objective_dimension = dimension
            self._objective_names = objective_names
        else:
            if dimension != self._objective_dimension:
                raise CoreContractViolation("objective dimensions must match")
            if objective_names != self._objective_names:
                raise CoreContractViolation("objective identity must match")
        if dimension == 2:
            self._add_two_objective(candidate)
            return
        if dimension == 3:
            self._feasible_candidates_3d.append(candidate)
            return
        if any(
            dominates(incumbent, candidate)
            for incumbent in self._nondominated
        ):
            return
        self._nondominated = [
            incumbent
            for incumbent in self._nondominated
            if not dominates(candidate, incumbent)
        ]
        self._nondominated.append(candidate)

    def _add_two_objective(self, candidate: Candidate) -> None:
        key = (
            float(candidate.objectives[0]),
            float(candidate.objectives[1]),
        )
        position = bisect_left(
            self._front_keys_2d,
            (key[0], float("-inf")),
        )
        same_first_objective = (
            position < len(self._front_keys_2d)
            and self._front_keys_2d[position][0] == key[0]
        )
        if same_first_objective:
            incumbent_key = self._front_keys_2d[position]
            if incumbent_key[1] < key[1]:
                return
            if incumbent_key[1] == key[1]:
                self._front_groups_2d[incumbent_key].append(candidate)
                return
        elif (
            position > 0
            and self._front_keys_2d[position - 1][1] <= key[1]
        ):
            return

        while (
            position < len(self._front_keys_2d)
            and self._front_keys_2d[position][1] >= key[1]
        ):
            removed_key = self._front_keys_2d.pop(position)
            del self._front_groups_2d[removed_key]
        self._front_keys_2d.insert(position, key)
        self._front_groups_2d[key] = [candidate]

    def _snapshot_three_objective(self) -> tuple[Candidate, ...]:
        """Return the exact 3-D first front in original insertion order.

        Exact objective duplicates are grouped before the sweep because
        equality alone is not strict Pareto dominance.  In lexicographic
        objective order, a prior distinct group can dominate the current
        group exactly when its second objective is no greater and the minimum
        third objective over that prefix is no greater.  A Fenwick tree stores
        those prefix minima in O(n log n) total time.

        Dominated groups are deliberately inserted into the tree as well:
        their dominators satisfy every later query that they could satisfy,
        by transitivity.
        """

        if not self._feasible_candidates_3d:
            return ()
        groups: dict[tuple[float, float, float], list[Candidate]] = {}
        for candidate in self._feasible_candidates_3d:
            key = (
                float(candidate.objectives[0]),
                float(candidate.objectives[1]),
                float(candidate.objectives[2]),
            )
            groups.setdefault(key, []).append(candidate)

        ordered_keys = sorted(groups)
        second_values = sorted({key[1] for key in ordered_keys})
        second_positions = {
            value: index + 1 for index, value in enumerate(second_values)
        }
        prefix_minimum_third = [float("inf")] * (len(second_values) + 1)
        nondominated_ids: set[str] = set()

        for key in ordered_keys:
            position = second_positions[key[1]]
            minimum_third = float("inf")
            query_position = position
            while query_position:
                minimum_third = min(
                    minimum_third,
                    prefix_minimum_third[query_position],
                )
                query_position -= query_position & -query_position
            if minimum_third > key[2]:
                nondominated_ids.update(
                    candidate.candidate_id for candidate in groups[key]
                )

            update_position = position
            while update_position < len(prefix_minimum_third):
                prefix_minimum_third[update_position] = min(
                    prefix_minimum_third[update_position],
                    key[2],
                )
                update_position += update_position & -update_position

        return tuple(
            candidate
            for candidate in self._feasible_candidates_3d
            if candidate.candidate_id in nondominated_ids
        )

    def snapshot(self) -> tuple[Candidate, ...]:
        if self._objective_dimension == 2:
            return tuple(
                candidate
                for key in self._front_keys_2d
                for candidate in self._front_groups_2d[key]
            )
        if self._objective_dimension == 3:
            return self._snapshot_three_objective()
        return tuple(self._nondominated)


def validate_problem(problem: EventProblemAdapter) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    if (
        problem.decision_dimension < 1
        or lower.shape != (problem.decision_dimension,)
        or upper.shape != lower.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(lower <= upper)
    ):
        raise ComparatorBindingError("problem bounds are invalid")
    scales = tuple(float(value) for value in problem.constraint_scales)
    if not scales or not all(
        isfinite(value) and value > 0.0 for value in scales
    ):
        raise ComparatorBindingError(
            "constraint scales must be finite and positive"
        )
    return lower, upper


def finalize_candidates(
    problem: EventProblemAdapter,
    *,
    event_id: int,
    candidates: Sequence[Candidate],
    archive_capacity: int,
    budget_exhausted: bool = True,
) -> OptimizationResult:
    """Return a typed terminal result from the common feasible archive."""

    if not candidates:
        return OptimizationResult(
            terminal=TerminalOutcome(
                TerminalCode.REJECT_NUMERICAL,
                reason="no finite candidate evaluation was available",
            ),
            archive=(),
        )
    archive = maintain_nondominated_archive(
        candidates,
        capacity=archive_capacity,
        constraint_scales=problem.constraint_scales,
    )
    return _finalize_archive(
        problem,
        event_id=event_id,
        archive=archive,
        budget_exhausted=budget_exhausted,
    )


def finalize_nondominated_candidates(
    problem: EventProblemAdapter,
    *,
    event_id: int,
    candidates: Sequence[Candidate],
    had_finite_candidates: bool,
    archive_capacity: int,
    budget_exhausted: bool = True,
) -> OptimizationResult:
    """Finalize a caller-proven complete nondominated feasible set."""

    if not candidates and not had_finite_candidates:
        return OptimizationResult(
            terminal=TerminalOutcome(
                TerminalCode.REJECT_NUMERICAL,
                reason="no finite candidate evaluation was available",
            ),
            archive=(),
        )
    archive = order_known_nondominated_archive(
        candidates,
        capacity=archive_capacity,
        constraint_scales=problem.constraint_scales,
    )
    return _finalize_archive(
        problem,
        event_id=event_id,
        archive=archive,
        budget_exhausted=budget_exhausted,
    )


def _finalize_archive(
    problem: EventProblemAdapter,
    *,
    event_id: int,
    archive: Sequence[Candidate],
    budget_exhausted: bool,
) -> OptimizationResult:
    evaluations = tuple(candidate.evaluation for candidate in archive)
    if not archive:
        code = (
            TerminalCode.REJECT_BUDGET_NO_FEASIBLE
            if budget_exhausted
            else TerminalCode.REJECT_NO_FEASIBLE
        )
        return OptimizationResult(
            terminal=TerminalOutcome(
                code,
                reason="shared constrained archive contains no feasible point",
            ),
            archive=(),
        )
    safe = sorted(
        (
            candidate
            for candidate in archive
            if problem.safety_filter(candidate.evaluation, event_id)
        ),
        key=lambda candidate: candidate.candidate_id,
    )
    if not safe:
        return OptimizationResult(
            terminal=TerminalOutcome(
                TerminalCode.REJECT_SAFETY_FILTER,
                reason="feasible archive contains no safety-filtered point",
            ),
            archive=evaluations,
        )
    selector = getattr(problem, "select_candidate", None)
    selected = selector(tuple(safe)) if callable(selector) else safe[0]
    safe_ids = {candidate.candidate_id for candidate in safe}
    if selected.candidate_id not in safe_ids:
        raise ComparatorBindingError(
            "problem selector returned a candidate outside the safe archive"
        )
    return OptimizationResult(
        terminal=TerminalOutcome(
            TerminalCode.ACCEPTED,
            candidate_id=selected.candidate_id,
        ),
        archive=evaluations,
        selected_vector=tuple(float(value) for value in selected.vector),
    )
