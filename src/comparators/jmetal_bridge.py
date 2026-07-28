"""Budget-bounded jMetalPy comparators using the project evaluator and ledger."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
import importlib.metadata
import random
from typing import Any, Literal, Mapping

import numpy as np

from benchmark_adapters.r4_evaluators import (
    JMETALPY_TAG_COMMIT,
    JMETALPY_VERSION,
    JMETALPY_WHEEL_SHA256,
)
from dt_ramde_v11.core import Candidate
from dt_ramde_v11.interfaces import (
    EventProblemAdapter,
    OptimizationResult,
)
from evaluation.contracts import EvaluationLedgerPort, NumericalEvaluationError
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry

from .common import (
    ComparatorBindingError,
    finalize_candidates,
    validate_problem,
)


JMetalMode = Literal["gde3", "nsgaii_static", "nsgaii_dynamic_restart"]


def _require_jmetalpy() -> None:
    try:
        version = importlib.metadata.version("jmetalpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ComparatorBindingError(
            "jmetalpy==1.7.0 is required by the R4 external comparators"
        ) from exc
    if version != JMETALPY_VERSION:
        raise ComparatorBindingError(
            f"jmetalpy version {version!r} differs from frozen 1.7.0"
        )


@dataclass
class JMetalComparator:
    """GDE3, static NSGA-II, or budget-bounded dynamic NSGA-II restart."""

    mode: JMetalMode
    population_size: int = 20
    archive_capacity: int = 100
    _previous_vectors: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False
    )

    method_version = "jmetalpy-1.7.0+r4-bridge.2"

    def __post_init__(self) -> None:
        if self.mode not in {
            "gde3",
            "nsgaii_static",
            "nsgaii_dynamic_restart",
        }:
            raise ComparatorBindingError("unknown jMetal comparator mode")
        if self.population_size < 4 or self.archive_capacity < 1:
            raise ComparatorBindingError(
                "jMetal bridge requires population >= 4 and positive archive"
            )

    @property
    def method_id(self) -> str:
        return {
            "gde3": "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
            "nsgaii_static": "JMETALPY_1_7_NSGAII_STATIC_CMOEA",
            "nsgaii_dynamic_restart": (
                "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE"
            ),
        }[self.mode]

    def identity(self) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "upstream_distribution": "jmetalpy",
            "upstream_release": JMETALPY_VERSION,
            "upstream_tag_commit": JMETALPY_TAG_COMMIT,
            "upstream_wheel_sha256": JMETALPY_WHEEL_SHA256,
            "upstream_license": "MIT",
            "native_search": True,
            "native_evaluator": False,
            "native_budget_counter": False,
            "shared_project_evaluator_and_ledger": True,
            "constraint_convention_bridge": "project_c_le_0_to_jmetal_g_ge_0",
            "numerical_failure_rule": (
                "charged_candidate_marked_infeasible_and_excluded"
            ),
            "boundary_rule": (
                "shared_midpoint_for_GDE3; bounded_native_SBX_polynomial_"
                "operators_for_NSGAII"
            ),
            "parameters": (
                {"F": 0.5, "CR": 0.9, "K": 0.5}
                if self.mode == "gde3"
                else {
                    "SBX_probability": 0.9,
                    "SBX_distribution_index": 20.0,
                    "mutation_probability": "1/D",
                    "mutation_distribution_index": 20.0,
                }
            ),
            "population_size": self.population_size,
            "archive_capacity": self.archive_capacity,
            "dynamic_response": (
                "none"
                if self.mode != "nsgaii_dynamic_restart"
                else "retain_shifted_half_population_then_random_refill"
            ),
            "upstream_dynamic_class_used_unchanged": False,
            "effect_execution_allowed": False,
        }

    def _make_problem(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        seed: int,
        ledger: EvaluationLedgerPort,
    ):
        from jmetal.core.problem import FloatProblem
        from jmetal.core.solution import FloatSolution

        method_id = self.method_id

        class ProjectProblem(FloatProblem):
            def __init__(self) -> None:
                super().__init__()
                self.lower_bound = [
                    float(value) for value in problem.lower_bounds
                ]
                self.upper_bound = [
                    float(value) for value in problem.upper_bounds
                ]
                self.directions = [
                    self.MINIMIZE
                    for _ in range(len(getattr(problem, "objective_names", ())))
                ]
                self.labels = list(
                    getattr(problem, "objective_names", ())
                )
                self.counter = 0

            def number_of_variables(self) -> int:
                return problem.decision_dimension

            def number_of_objectives(self) -> int:
                names = tuple(getattr(problem, "objective_names", ()))
                if not names:
                    raise ComparatorBindingError(
                        "problem must expose objective_names for jMetal"
                    )
                return len(names)

            def number_of_constraints(self) -> int:
                return len(problem.constraint_scales)

            def create_solution(self) -> FloatSolution:
                solution = super().create_solution()
                return solution

            def _candidate_id(self, sequence: int) -> str:
                return (
                    f"{method_id}:event:{event_id}:seed:{seed}:"
                    f"candidate:{sequence:08d}"
                )

            def _store_result(
                self,
                solution: FloatSolution,
                result,
            ) -> FloatSolution:
                solution.objectives = list(result.objectives)
                solution.constraints = [
                    -float(value) for value in result.constraints
                ]
                solution.attributes["project_evaluation"] = result
                return solution

            def evaluate(self, solution: FloatSolution) -> FloatSolution:
                self.counter += 1
                candidate_id = self._candidate_id(self.counter)
                try:
                    result = problem.evaluate(
                        solution.variables,
                        event_id,
                        ledger,
                        candidate_id,
                    )
                except NumericalEvaluationError:
                    solution.objectives = [
                        float("inf")
                        for _ in range(self.number_of_objectives())
                    ]
                    solution.constraints = [
                        float("-inf")
                        for _ in range(self.number_of_constraints())
                    ]
                    solution.attributes["project_evaluation_failed"] = True
                    return solution
                return self._store_result(solution, result)

            def evaluate_solutions(self, solutions):
                values = list(solutions)
                if not values:
                    return values
                batch_evaluator = getattr(problem, "evaluate_batch", None)
                if not callable(batch_evaluator):
                    return [self.evaluate(solution) for solution in values]
                candidate_ids = [
                    self._candidate_id(self.counter + index + 1)
                    for index in range(len(values))
                ]
                try:
                    results = tuple(
                        batch_evaluator(
                            [solution.variables for solution in values],
                            event_id,
                            ledger,
                            candidate_ids,
                        )
                    )
                except BatchEvaluationUnavailableBeforeEntry:
                    return [self.evaluate(solution) for solution in values]
                if len(results) != len(values):
                    raise ComparatorBindingError(
                        "jMetal batch evaluator returned the wrong result count"
                    )
                self.counter += len(values)
                for solution, candidate_id, result in zip(
                    values, candidate_ids, results, strict=True
                ):
                    if result.candidate_id != candidate_id:
                        raise ComparatorBindingError(
                            "jMetal batch evaluator changed candidate order"
                        )
                    self._store_result(solution, result)
                return values

            def name(self) -> str:
                return f"ProjectLedgerBridge({problem.adapter_id})"

        return ProjectProblem()

    def _initial_vectors(
        self, problem: EventProblemAdapter, event_id: int
    ) -> list[np.ndarray]:
        if self.mode != "nsgaii_dynamic_restart" or event_id == 0:
            return []
        retained = self._previous_vectors[: self.population_size // 2]
        return [
            np.asarray(problem.shift_solution(vector), dtype=float)
            for vector in retained
        ]

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
        _require_jmetalpy()
        validate_problem(problem)
        if budget < self.population_size:
            raise ComparatorBindingError(
                "budget must cover jMetal population initialization"
            )
        if budget % self.population_size != 0:
            raise ComparatorBindingError(
                "jMetal bridge requires a whole generational budget"
            )
        if self.mode == "nsgaii_static" and event_id != 0:
            raise ComparatorBindingError(
                "static NSGA-II binding accepts event zero only"
            )

        from jmetal.algorithm.multiobjective.gde3 import GDE3
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.operator import PolynomialMutation, SBXCrossover
        from jmetal.operator.crossover import DifferentialEvolutionCrossover
        from jmetal.util.comparator import (
            DominanceWithConstraintsComparator,
        )
        from jmetal.util.evaluator import Evaluator
        from jmetal.util.termination_criterion import StoppingByEvaluations

        bridge_problem = self._make_problem(
            problem,
            event_id=event_id,
            seed=seed,
            ledger=ledger,
        )
        retained = [
            np.asarray(vector, dtype=float).copy()
            for vector in self._initial_vectors(problem, event_id)
        ]
        shared = (
            []
            if initialization_vectors is None
            else [
                np.asarray(vector, dtype=float).copy()
                for vector in initialization_vectors
            ]
        )
        required_shared = self.population_size - len(retained)
        if len(shared) not in {0, required_shared, self.population_size}:
            raise ComparatorBindingError(
                "shared initialization vector count differs from the "
                "jMetal population requirement"
            )
        queued = retained + shared[:required_shared]
        lower = np.asarray(problem.lower_bounds, dtype=float)
        upper = np.asarray(problem.upper_bounds, dtype=float)
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

        class RestartGenerator:
            def new(self, current_problem):
                solution = current_problem.create_solution()
                if queued:
                    vector = queued.pop(0)
                    if vector.shape != (problem.decision_dimension,):
                        raise ComparatorBindingError(
                            "shifted dynamic vector has wrong shape"
                        )
                    solution.variables = vector.tolist()
                return solution

        comparator = DominanceWithConstraintsComparator()
        termination = StoppingByEvaluations(max_evaluations=budget)

        class ProjectBatchEvaluator(Evaluator):
            def evaluate(self, solution_list, current_problem):
                return current_problem.evaluate_solutions(solution_list)

        population_evaluator = ProjectBatchEvaluator()

        class SharedMidpointDECrossover(DifferentialEvolutionCrossover):
            """Keep GDE3 search native except for the frozen boundary rule."""

            def execute(self, parents):
                if len(parents) != self.get_number_of_parents():
                    raise ComparatorBindingError(
                        "GDE3 received the wrong number of parents"
                    )
                child = copy.deepcopy(self.current_individual)
                forced = random.randint(0, len(child.variables) - 1)
                for index in range(len(child.variables)):
                    if random.random() < self.CR or index == forced:
                        raw = (
                            parents[2].variables[index]
                            + self.F
                            * (
                                parents[0].variables[index]
                                - parents[1].variables[index]
                            )
                        )
                        lower = child.lower_bound[index]
                        upper = child.upper_bound[index]
                        if raw < lower:
                            raw = 0.5 * (
                                lower
                                + self.current_individual.variables[index]
                            )
                        elif raw > upper:
                            raw = 0.5 * (
                                upper
                                + self.current_individual.variables[index]
                            )
                        child.variables[index] = raw
                return [child]

        if self.mode == "gde3":
            algorithm = GDE3(
                problem=bridge_problem,
                population_size=self.population_size,
                cr=0.9,
                f=0.5,
                k=0.5,
                termination_criterion=termination,
                population_generator=RestartGenerator(),
                population_evaluator=population_evaluator,
                dominance_comparator=comparator,
            )
            algorithm.crossover_operator = SharedMidpointDECrossover(
                CR=0.9,
                F=0.5,
                K=0.5,
            )
        else:
            algorithm = NSGAII(
                problem=bridge_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=PolynomialMutation(
                    probability=1.0 / problem.decision_dimension,
                    distribution_index=20.0,
                ),
                crossover=SBXCrossover(
                    probability=0.9,
                    distribution_index=20.0,
                ),
                termination_criterion=termination,
                population_generator=RestartGenerator(),
                population_evaluator=population_evaluator,
                dominance_comparator=comparator,
            )

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cfe_before = int(ledger.snapshot()["cfe"])
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            algorithm.run()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        cfe_after = int(ledger.snapshot()["cfe"])
        if cfe_after - cfe_before != budget:
            raise ComparatorBindingError(
                "jMetal search did not consume the exact shared CFE budget"
            )

        solutions = list(algorithm.result())
        candidates: list[Candidate] = []
        candidates_by_evaluation_id: dict[str, Candidate] = {}
        for solution in solutions:
            if solution.attributes.get("project_evaluation_failed"):
                continue
            evaluation = solution.attributes.get("project_evaluation")
            if evaluation is None:
                raise ComparatorBindingError(
                    "jMetal solution bypassed the project evaluator"
                )
            candidate = Candidate(
                np.asarray(solution.variables, dtype=float),
                evaluation,
                evaluation.candidate_id,
            )
            existing = candidates_by_evaluation_id.get(candidate.candidate_id)
            if existing is None:
                candidates_by_evaluation_id[candidate.candidate_id] = candidate
                candidates.append(candidate)
                continue
            if (
                existing.evaluation != candidate.evaluation
                or not np.array_equal(existing.vector, candidate.vector)
            ):
                raise ComparatorBindingError(
                    "jMetal reused one project candidate_id for "
                    "inconsistent final solutions"
                )
        if self.mode == "nsgaii_dynamic_restart":
            self._previous_vectors = [
                candidate.vector.copy() for candidate in candidates
            ]
        return finalize_candidates(
            problem,
            event_id=event_id,
            candidates=candidates,
            archive_capacity=self.archive_capacity,
            budget_exhausted=True,
        )
