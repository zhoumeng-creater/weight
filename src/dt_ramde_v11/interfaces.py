"""Distinct problem-side and algorithm-side v1.1 integration contracts.

Port provenance:
    FORMAL_V1/dt_ramde_formal/interfaces.py
    SHA-256 1c5477bf6d725355ee7c1636925a77fd152e17e7ac64d2eeb11b70d67e7dfe6a

The historical seven-method problem adapter is retained under its v1.1 name.
The algorithm-side protocol is new and prevents comparators from masquerading
as problem adapters or bypassing the shared ledger argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from evaluation.contracts import (
    EvaluationContractError,
    EvaluationLedgerPort,
    EvaluationResult,
    TerminalCode,
    TerminalOutcome,
)
from evaluation.firewall import InformationSnapshot


@dataclass(frozen=True)
class OptimizationResult:
    """Typed optimizer output preserving the feasible archive and terminal state."""

    terminal: TerminalOutcome
    archive: tuple[EvaluationResult, ...]
    selected_vector: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        archive_ids = {candidate.candidate_id for candidate in self.archive}
        if len(archive_ids) != len(self.archive):
            raise EvaluationContractError("archive candidate_id values must be unique")
        if self.terminal.code is TerminalCode.ACCEPTED:
            if self.terminal.candidate_id not in archive_ids:
                raise EvaluationContractError(
                    "ACCEPTED candidate_id must be present in the archive"
                )
            if (
                self.selected_vector is None
                or not self.selected_vector
                or not all(isfinite(value) for value in self.selected_vector)
            ):
                raise EvaluationContractError(
                    "ACCEPTED optimizer result requires a finite selected vector"
                )
        elif self.selected_vector is not None:
            raise EvaluationContractError(
                "non-ACCEPTED optimizer result cannot carry a selected vector"
            )


@runtime_checkable
class EventProblemAdapter(Protocol):
    """Problem/domain boundary shared by benchmarks and the weight case."""

    adapter_id: str
    adapter_version: str
    decision_dimension: int
    atomic_steps_per_evaluation: int
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    constraint_scales: Sequence[float]

    def identity(self) -> Mapping[str, Any]: ...

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot: ...

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedgerPort,
        candidate_id: str,
    ) -> EvaluationResult: ...

    def safety_filter(self, result: EvaluationResult, event_id: int) -> bool: ...

    def shift_solution(self, vector: Sequence[float]) -> Sequence[float]: ...

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedgerPort,
    ) -> Mapping[str, Any]: ...

    def first_action(self, vector: Sequence[float]) -> Sequence[float]: ...

    def fallback_action(self, event_id: int) -> Sequence[float]: ...


@runtime_checkable
class OrderedBatchEventProblemAdapter(Protocol):
    """Optional generation-sized evaluator preserving candidate order."""

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        ledger: EvaluationLedgerPort,
        candidate_ids: Sequence[str],
    ) -> tuple[EvaluationResult, ...]: ...


@runtime_checkable
class OptimizerOrComparator(Protocol):
    """Algorithm boundary; all evaluation flows through the supplied problem."""

    method_id: str
    method_version: str

    def identity(self) -> Mapping[str, Any]: ...

    def optimize(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        budget: int,
        seed: int,
        ledger: EvaluationLedgerPort,
    ) -> OptimizationResult: ...


@runtime_checkable
class TerminalSelector(Protocol):
    """Problem-bound pure selector applied identically across all methods."""

    selector_id: str
    selector_version: str

    def identity(self) -> Mapping[str, Any]: ...

    def select(self, archive: Sequence[EvaluationResult]) -> str | None: ...
