"""Executable E3 domain baselines bound at R4, without effect authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dt_ramde_v11.core import Candidate
from dt_ramde_v11.interfaces import (
    EventProblemAdapter,
    OptimizationResult,
)
from evaluation.contracts import EvaluationLedgerPort

from .common import (
    ComparatorBindingError,
    finalize_candidates,
    validate_problem,
)
from .matched_de import MatchedParetoDE


@dataclass(frozen=True)
class FixedEnergyDeficitBaseline:
    """A fixed -500 kcal/day intake adjustment with no activity increase."""

    intake_adjustment_kcal_per_day: float = -500.0
    activity_adjustment_kcal_per_day: float = 0.0
    archive_capacity: int = 1

    method_id = "FIXED_ENERGY_DEFICIT_POLICY"
    method_version = "1.0.0-r4"

    def identity(self) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "role": "E3_domain_policy_baseline",
            "policy": {
                "intake_adjustment_kcal_per_day": (
                    self.intake_adjustment_kcal_per_day
                ),
                "activity_adjustment_kcal_per_day": (
                    self.activity_adjustment_kcal_per_day
                ),
            },
            "search": False,
            "cross_event_credit": False,
            "effect_execution_allowed": False,
        }

    def optimize(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        budget: int,
        seed: int,
        ledger: EvaluationLedgerPort,
        initialization_vectors: list[list[float]] | None = None,
    ) -> OptimizationResult:
        del seed
        lower, upper = validate_problem(problem)
        if problem.decision_dimension != 2:
            raise ComparatorBindingError(
                "fixed energy deficit requires the two-field weight action"
            )
        if budget < 1:
            raise ComparatorBindingError(
                "fixed energy deficit requires one shared evaluation"
            )
        vector = np.asarray(
            [
                self.intake_adjustment_kcal_per_day,
                self.activity_adjustment_kcal_per_day,
            ],
            dtype=float,
        )
        if np.any(vector < lower) or np.any(vector > upper):
            raise ComparatorBindingError(
                "fixed energy deficit lies outside the weight action bounds"
            )
        candidate_id = f"{self.method_id}:event:{event_id}:candidate:00000001"
        evaluation = problem.evaluate(
            vector, event_id, ledger, candidate_id
        )
        candidate = Candidate(vector, evaluation, candidate_id)
        return finalize_candidates(
            problem,
            event_id=event_id,
            candidates=[candidate],
            archive_capacity=self.archive_capacity,
            budget_exhausted=False,
        )


@dataclass
class ConventionalRollingPlannerBaseline:
    """Fresh per-event SHADE planner with no cross-event state or credit."""

    population_size: int = 20
    archive_capacity: int = 100

    method_id = "CONVENTIONAL_ROLLING_PLANNER_NO_CROSS_EVENT_CREDIT"
    method_version = "1.0.0-r4"

    def identity(self) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "role": "E3_domain_rolling_planner_baseline",
            "inner_search": "matched_SHADE_Pareto_chassis",
            "new_search_state_each_event": True,
            "warm_start": False,
            "execution_credit": False,
            "rejection_credit": False,
            "lineage_credit": False,
            "effect_execution_allowed": False,
        }

    def optimize(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        budget: int,
        seed: int,
        ledger: EvaluationLedgerPort,
        initialization_vectors: list[list[float]] | None = None,
    ) -> OptimizationResult:
        planner = MatchedParetoDE(
            mode="shade",
            population_size=self.population_size,
            archive_capacity=self.archive_capacity,
            method_id_override=self.method_id,
        )
        return planner.optimize(
            problem,
            event_id=event_id,
            budget=budget,
            seed=seed,
            ledger=ledger,
            initialization_vectors=initialization_vectors,
        )
