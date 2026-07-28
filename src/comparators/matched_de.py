"""Matched Pareto-DE chassis for the R4 fixed, jDE, and SHADE baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from dt_ramde_v11.core import (
    Candidate,
    environmental_select,
    repair_midpoint,
    sample_parameters,
    shade_success_improvement,
    update_mg,
)
from dt_ramde_v11.interfaces import (
    EventProblemAdapter,
    OptimizationResult,
)
from evaluation.contracts import (
    EvaluationLedgerPort,
    EvaluationResult,
    NumericalEvaluationError,
)
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry

from .common import (
    ComparatorBindingError,
    ExactNondominatedAccumulator,
    finalize_nondominated_candidates,
    validate_problem,
)


MatchedMode = Literal["fixed", "jde", "shade"]


@dataclass
class MatchedParetoDE:
    """One common constrained Pareto-DE chassis with three parameter modes."""

    mode: MatchedMode
    population_size: int = 20
    archive_capacity: int = 100
    fixed_f: float = 0.5
    fixed_cr: float = 0.9
    jde_tau_f: float = 0.1
    jde_tau_cr: float = 0.1
    shade_memory_size: int = 5
    method_id_override: str | None = None

    method_version = "1.0.0-r4"

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "jde", "shade"}:
            raise ComparatorBindingError("unknown matched DE mode")
        if self.population_size < 4 or self.archive_capacity < 1:
            raise ComparatorBindingError(
                "matched DE requires population >= 4 and a positive archive"
            )
        if not 0.0 < self.fixed_f <= 1.0 or not 0.0 <= self.fixed_cr <= 1.0:
            raise ComparatorBindingError("fixed F/CR values are invalid")
        if not 0.0 <= self.jde_tau_f <= 1.0 or not 0.0 <= self.jde_tau_cr <= 1.0:
            raise ComparatorBindingError("jDE adaptation probabilities are invalid")
        if self.shade_memory_size < 1:
            raise ComparatorBindingError("SHADE memory must be nonempty")

    @property
    def method_id(self) -> str:
        if self.method_id_override is not None:
            if not self.method_id_override:
                raise ComparatorBindingError(
                    "method_id_override must be nonempty when provided"
                )
            return self.method_id_override
        return {
            "fixed": "MATCHED_FIXED_DE_PARETO",
            "jde": "MATCHED_JDE_PARETO",
            "shade": "MATCHED_SHADE_PARETO",
        }[self.mode]

    def identity(self) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "family": "matched_project_pareto_de_chassis",
            "parameter_mode": self.mode,
            "population_size": self.population_size,
            "archive_capacity": self.archive_capacity,
            "mutation": "DE/rand/1/bin",
            "repair": "shared_midpoint_then_joint_reevaluation",
            "constraint_handling": (
                "shared_feasibility_normalized_violation_rank_crowding"
            ),
            "cross_event_state": False,
            "fixed": {"F": self.fixed_f, "CR": self.fixed_cr},
            "jde": {
                "tau_F": self.jde_tau_f,
                "tau_CR": self.jde_tau_cr,
                "F_support": [0.1, 1.0],
                "CR_support": [0.0, 1.0],
            },
            "shade": {
                "memory_size": self.shade_memory_size,
                "success_rule": "WGT-SHADE-CMO-SUCCESS-01",
            },
            "effect_execution_allowed": False,
        }

    def _candidate_id(
        self, *, event_id: int, seed: int, sequence: int
    ) -> str:
        return (
            f"{self.method_id}:event:{event_id}:seed:{seed}:"
            f"candidate:{sequence:08d}"
        )

    @staticmethod
    def _evaluate_ordered(
        problem: EventProblemAdapter,
        *,
        vectors: Sequence[Sequence[float]],
        candidate_ids: Sequence[str],
        event_id: int,
        ledger: EvaluationLedgerPort,
    ) -> tuple[EvaluationResult | None, ...]:
        batch_evaluator = getattr(problem, "evaluate_batch", None)
        if callable(batch_evaluator):
            try:
                results = tuple(
                    batch_evaluator(
                        vectors,
                        event_id,
                        ledger,
                        candidate_ids,
                    )
                )
            except BatchEvaluationUnavailableBeforeEntry:
                pass
            else:
                if len(results) != len(vectors):
                    raise ComparatorBindingError(
                        "batch evaluator returned the wrong result count"
                    )
                return results
        outcomes: list[EvaluationResult | None] = []
        for vector, candidate_id in zip(
            vectors, candidate_ids, strict=True
        ):
            try:
                result = problem.evaluate(
                    vector, event_id, ledger, candidate_id
                )
            except NumericalEvaluationError:
                outcomes.append(None)
            else:
                outcomes.append(result)
        return tuple(outcomes)

    def optimize(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        budget: int,
        seed: int,
        ledger: EvaluationLedgerPort,
        initialization_vectors: Sequence[Sequence[float]] | None = None,
    ) -> OptimizationResult:
        lower, upper = validate_problem(problem)
        if budget < self.population_size:
            raise ComparatorBindingError(
                "budget must cover matched-DE initialization"
            )
        if budget % self.population_size != 0:
            raise ComparatorBindingError(
                "matched-DE bridge requires a whole generational budget"
            )
        rng = np.random.Generator(np.random.PCG64(seed))
        sequence = 0
        history = ExactNondominatedAccumulator()
        population: list[Candidate] = []
        parameter_by_id: dict[str, tuple[float, float]] = {}
        memory_f = [0.5] * self.shade_memory_size
        memory_cr = [0.5] * self.shade_memory_size
        memory_pointer = 0

        queued = (
            []
            if initialization_vectors is None
            else [
                np.asarray(vector, dtype=float).copy()
                for vector in initialization_vectors
            ]
        )
        if len(queued) > self.population_size:
            raise ComparatorBindingError(
                "too many shared initialization vectors"
            )
        for vector in queued:
            if (
                vector.shape != lower.shape
                or not np.all(np.isfinite(vector))
                or np.any(vector < lower)
                or np.any(vector > upper)
            ):
                raise ComparatorBindingError(
                    "shared initialization vector violates problem bounds"
                )
        if len(queued) == self.population_size:
            batch_evaluator = getattr(problem, "evaluate_batch", None)
            if callable(batch_evaluator):
                initial_ids = [
                    self._candidate_id(
                        event_id=event_id,
                        seed=seed,
                        sequence=index + 1,
                    )
                    for index in range(self.population_size)
                ]
                try:
                    initial_results = tuple(
                        batch_evaluator(
                            queued,
                            event_id,
                            ledger,
                            initial_ids,
                        )
                    )
                except BatchEvaluationUnavailableBeforeEntry:
                    pass
                else:
                    if len(initial_results) != self.population_size:
                        raise ComparatorBindingError(
                            "matched-DE initialization batch has wrong size"
                        )
                    sequence = self.population_size
                    for vector, candidate_id, evaluation in zip(
                        queued,
                        initial_ids,
                        initial_results,
                        strict=True,
                    ):
                        if evaluation.candidate_id != candidate_id:
                            raise ComparatorBindingError(
                                "matched-DE initialization changed "
                                "candidate order"
                            )
                        candidate = Candidate(
                            vector, evaluation, candidate_id
                        )
                        population.append(candidate)
                        history.add(candidate)
                        parameter_by_id[candidate_id] = (
                            self.fixed_f,
                            self.fixed_cr,
                        )
        while len(population) < self.population_size and sequence < budget:
            vector = (
                queued[len(population)]
                if len(population) < len(queued)
                else rng.uniform(lower, upper)
            )
            sequence += 1
            candidate_id = self._candidate_id(
                event_id=event_id, seed=seed, sequence=sequence
            )
            try:
                evaluation = problem.evaluate(
                    vector, event_id, ledger, candidate_id
                )
            except NumericalEvaluationError:
                continue
            candidate = Candidate(vector, evaluation, candidate_id)
            population.append(candidate)
            history.add(candidate)
            parameter_by_id[candidate_id] = (self.fixed_f, self.fixed_cr)

        if len(population) < self.population_size:
            return finalize_nondominated_candidates(
                problem,
                event_id=event_id,
                candidates=history.snapshot(),
                had_finite_candidates=history.has_candidates,
                archive_capacity=self.archive_capacity,
                budget_exhausted=True,
            )

        while sequence < budget:
            trials: list[Candidate] = []
            trial_parameters: dict[str, tuple[float, float]] = {}
            paired: list[tuple[Candidate, Candidate, float, float]] = []
            plans: list[
                tuple[Candidate, np.ndarray | None, str, float, float]
            ] = []
            for target_index, target in enumerate(population):
                donor_pool = [
                    index
                    for index in range(self.population_size)
                    if index != target_index
                ]
                donor_indices = rng.choice(
                    donor_pool, size=3, replace=False
                )
                if self.mode == "fixed":
                    f_value, cr_value = self.fixed_f, self.fixed_cr
                elif self.mode == "jde":
                    f_value, cr_value = parameter_by_id[target.candidate_id]
                    if rng.random() < self.jde_tau_f:
                        f_value = 0.1 + 0.9 * rng.random()
                    if rng.random() < self.jde_tau_cr:
                        cr_value = rng.random()
                else:
                    memory_index = int(rng.integers(0, self.shade_memory_size))
                    sampled = sample_parameters(
                        rng,
                        mu_f=memory_f[memory_index],
                        mu_cr=memory_cr[memory_index],
                    )
                    f_value, cr_value = sampled.f, sampled.cr
                base, left, right = (
                    population[int(index)].vector
                    for index in donor_indices
                )
                mutant = base + f_value * (left - right)
                mask = rng.random(problem.decision_dimension) < cr_value
                mask[int(rng.integers(0, problem.decision_dimension))] = True
                raw_trial = np.where(mask, mutant, target.vector)
                repaired, _ = repair_midpoint(
                    raw_trial, target.vector, lower, upper
                )
                sequence += 1
                candidate_id = self._candidate_id(
                    event_id=event_id, seed=seed, sequence=sequence
                )
                plans.append(
                    (
                        target,
                        repaired,
                        candidate_id,
                        f_value,
                        cr_value,
                    )
                )

            if all(repaired is not None for _, repaired, *_ in plans):
                evaluations = self._evaluate_ordered(
                    problem,
                    vectors=[
                        repaired
                        for _, repaired, *_ in plans
                        if repaired is not None
                    ],
                    candidate_ids=[
                        candidate_id
                        for _, _, candidate_id, _, _ in plans
                    ],
                    event_id=event_id,
                    ledger=ledger,
                )
            else:
                scalar_evaluations: list[EvaluationResult | None] = []
                for _, repaired, candidate_id, _, _ in plans:
                    if repaired is None:
                        if hasattr(ledger, "record_repair_failure"):
                            ledger.record_repair_failure(
                                candidate_id=candidate_id,
                                event_id=event_id,
                                reason="shared midpoint repair failed",
                            )
                        scalar_evaluations.append(None)
                        continue
                    try:
                        evaluation = problem.evaluate(
                            repaired, event_id, ledger, candidate_id
                        )
                    except NumericalEvaluationError:
                        scalar_evaluations.append(None)
                    else:
                        scalar_evaluations.append(evaluation)
                evaluations = tuple(scalar_evaluations)
            if len(evaluations) != len(plans):
                raise ComparatorBindingError(
                    "matched-DE evaluator returned a partial generation"
                )
            for plan, evaluation in zip(
                plans,
                evaluations,
                strict=True,
            ):
                target, repaired, candidate_id, f_value, cr_value = plan
                if evaluation is None:
                    continue
                if evaluation.candidate_id != candidate_id:
                    raise ComparatorBindingError(
                        "matched-DE batch evaluator changed candidate order"
                    )
                assert repaired is not None
                trial = Candidate(
                    repaired, evaluation, candidate_id
                )
                trials.append(trial)
                history.add(trial)
                trial_parameters[candidate_id] = (f_value, cr_value)
                paired.append((target, trial, f_value, cr_value))

            next_population = environmental_select(
                [*population, *trials],
                population_size=self.population_size,
                constraint_scales=problem.constraint_scales,
            )
            survivor_ids = {
                candidate.candidate_id for candidate in next_population
            }
            next_parameters: dict[str, tuple[float, float]] = {}
            for candidate in next_population:
                next_parameters[candidate.candidate_id] = (
                    trial_parameters.get(
                        candidate.candidate_id,
                        parameter_by_id.get(
                            candidate.candidate_id,
                            (self.fixed_f, self.fixed_cr),
                        ),
                    )
                )
            if self.mode == "shade":
                successes: list[tuple[float, float, float]] = []
                for target, trial, f_value, cr_value in paired:
                    success = shade_success_improvement(
                        target,
                        trial,
                        problem.constraint_scales,
                        trial_in_next_population=(
                            trial.candidate_id in survivor_ids
                        ),
                        target_in_next_population=(
                            target.candidate_id in survivor_ids
                        ),
                    )
                    if success.success:
                        successes.append(
                            (f_value, cr_value, success.delta)
                        )
                memory_pointer = update_mg(
                    memory_f,
                    memory_cr,
                    memory_pointer,
                    successes,
                )
            population = next_population
            parameter_by_id = next_parameters

        return finalize_nondominated_candidates(
            problem,
            event_id=event_id,
            candidates=history.snapshot(),
            had_finite_candidates=history.has_candidates,
            archive_capacity=self.archive_capacity,
            budget_exhausted=True,
        )
