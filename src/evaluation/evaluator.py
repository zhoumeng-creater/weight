"""Single joint objective/constraint evaluator for every v1.1 method."""

from __future__ import annotations

from typing import Callable, Hashable, Sequence

from .contracts import (
    EvaluationContractError,
    EvaluationResult,
    NumericalEvaluationError,
)
from .firewall import InformationSnapshot
from .ledger import EvaluationLedger


class RepairFailed(RuntimeError):
    """A candidate failed validation before entering the evaluator."""


class BatchEvaluationUnavailableBeforeEntry(RuntimeError):
    """A natural batch could not be proven valid before any ledger entry."""


class ExecutionTimeoutBeforeEntry(RuntimeError):
    """A supervisor requested a typed task timeout before the next CFE."""


JointEvaluator = Callable[
    [Sequence[float], InformationSnapshot],
    tuple[tuple[float, ...], tuple[float, ...]],
]
JointBatchEvaluator = Callable[
    [Sequence[Sequence[float]], InformationSnapshot],
    Sequence[tuple[Sequence[float], Sequence[float]]],
]
RepairFunction = Callable[
    [Sequence[float], Sequence[float]], Sequence[float] | None
]


class SharedEvaluator:
    """Charge one CFE before every joint objective/constraint evaluator entry."""

    def __init__(
        self,
        *,
        objective_names: tuple[str, ...],
        constraint_names: tuple[str, ...],
        evaluate_joint: JointEvaluator,
        evaluate_joint_batch: JointBatchEvaluator | None = None,
    ) -> None:
        if not objective_names:
            raise ValueError("objective_names must be nonempty")
        self.objective_names = objective_names
        self.constraint_names = constraint_names
        self._evaluate_joint = evaluate_joint
        self._evaluate_joint_batch = evaluate_joint_batch
        self._cache: dict[
            Hashable, tuple[tuple[float, ...], tuple[float, ...]]
        ] = {}

    def evaluate(
        self,
        *,
        vector: Sequence[float],
        event_id: int,
        candidate_id: str,
        information: InformationSnapshot,
        ledger: EvaluationLedger,
        atomic_steps: int,
        cache_key: Hashable | None = None,
        origin: str = "trial",
    ) -> EvaluationResult:
        cached = cache_key is not None and cache_key in self._cache
        ledger.charge_candidate(
            candidate_id=candidate_id,
            event_id=event_id,
            atomic_steps=atomic_steps,
            origin=origin,
            cached=cached,
            metadata={"information_hash": information.information_hash},
        )
        if cached:
            objectives, constraints = self._cache[cache_key]
        else:
            try:
                objectives, constraints = self._evaluate_joint(vector, information)
                objectives = tuple(float(value) for value in objectives)
                constraints = tuple(float(value) for value in constraints)
            except Exception as error:
                ledger.record_evaluation_failure(
                    candidate_id=candidate_id,
                    event_id=event_id,
                    error=error,
                )
                raise NumericalEvaluationError(
                    "joint evaluator raised a nonrecoverable numerical error"
                ) from error
        try:
            result = EvaluationResult(
                candidate_id=candidate_id,
                objectives=objectives,
                objective_names=self.objective_names,
                constraints=constraints,
                constraint_names=self.constraint_names,
            )
        except EvaluationContractError as error:
            ledger.record_evaluation_failure(
                candidate_id=candidate_id,
                event_id=event_id,
                error=error,
            )
            if "must be finite" in str(error):
                raise NumericalEvaluationError(
                    "joint evaluator returned a non-finite value"
                ) from error
            raise
        if not cached and cache_key is not None:
            self._cache[cache_key] = (objectives, constraints)
        return result

    def evaluate_batch(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        candidate_ids: Sequence[str],
        information: InformationSnapshot,
        ledger: EvaluationLedger,
        atomic_steps: int,
        origin: str = "trial",
    ) -> tuple[EvaluationResult, ...]:
        """Prevalidate a pure batch, then atomically commit its ordered CFEs."""

        vector_values = tuple(vectors)
        id_values = tuple(str(value) for value in candidate_ids)
        if len(vector_values) != len(id_values):
            raise ValueError("batch vectors and candidate_ids must align")
        if not vector_values:
            return ()
        if self._evaluate_joint_batch is None:
            raise BatchEvaluationUnavailableBeforeEntry(
                "adapter has no ordered batch kernel"
            )
        try:
            rows = tuple(
                self._evaluate_joint_batch(vector_values, information)
            )
            if len(rows) != len(vector_values):
                raise ValueError(
                    "batch evaluator returned the wrong row count"
                )
            results = tuple(
                EvaluationResult(
                    candidate_id=candidate_id,
                    objectives=tuple(float(value) for value in objectives),
                    objective_names=self.objective_names,
                    constraints=tuple(float(value) for value in constraints),
                    constraint_names=self.constraint_names,
                )
                for candidate_id, (objectives, constraints) in zip(
                    id_values, rows, strict=True
                )
            )
        except Exception as error:
            raise BatchEvaluationUnavailableBeforeEntry(
                "batch evaluator could not prevalidate every row"
            ) from error

        charge_batch = getattr(ledger, "charge_candidate_batch", None)
        if not callable(charge_batch):
            raise BatchEvaluationUnavailableBeforeEntry(
                "ledger has no atomic batch charge"
            )
        try:
            charge_batch(
                candidate_ids=id_values,
                event_id=event_id,
                atomic_steps=atomic_steps,
                origin=origin,
                metadata={"information_hash": information.information_hash},
            )
        except Exception as error:
            raise BatchEvaluationUnavailableBeforeEntry(
                "ordered batch could not be charged atomically"
            ) from error
        return results

    def evaluate_after_repair(
        self,
        *,
        raw_vector: Sequence[float],
        target_vector: Sequence[float],
        repair: RepairFunction,
        event_id: int,
        candidate_id: str,
        information: InformationSnapshot,
        ledger: EvaluationLedger,
        atomic_steps: int,
        cache_key: Hashable | None = None,
        origin: str = "trial",
    ) -> EvaluationResult:
        try:
            repaired = repair(raw_vector, target_vector)
        except Exception as error:
            reason = f"repair failed before joint evaluator entry: {error}"
            ledger.record_repair_failure(
                candidate_id=candidate_id,
                event_id=event_id,
                reason=reason,
            )
            raise RepairFailed(reason) from error
        if repaired is None:
            reason = "repair failed before joint evaluator entry"
            ledger.record_repair_failure(
                candidate_id=candidate_id,
                event_id=event_id,
                reason=reason,
            )
            raise RepairFailed(reason)
        return self.evaluate(
            vector=repaired,
            event_id=event_id,
            candidate_id=candidate_id,
            information=information,
            ledger=ledger,
            atomic_steps=atomic_steps,
            cache_key=cache_key,
            origin=origin,
        )
