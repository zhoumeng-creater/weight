"""Shared typed evaluation and terminal-outcome contracts for v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Protocol


class EvaluationContractError(ValueError):
    """An evaluator result violates the shared v1.1 representation."""


class NumericalEvaluationError(RuntimeError):
    """A charged evaluator entry failed numerically and cannot be recovered."""


class EvaluationLedgerPort(Protocol):
    """Minimum shared ledger surface supplied to every method and adapter."""

    def charge_candidate(
        self,
        *,
        candidate_id: str,
        event_id: int,
        atomic_steps: int,
        origin: str = "trial",
        cached: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def charge_candidate_batch(
        self,
        *,
        candidate_ids: tuple[str, ...],
        event_id: int,
        atomic_steps: int,
        origin: str = "trial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def snapshot(self) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class EvaluationResult:
    """A finite vector objective with constraints expressed as ``c <= 0``."""

    candidate_id: str
    objectives: tuple[float, ...]
    objective_names: tuple[str, ...]
    constraints: tuple[float, ...]
    constraint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise EvaluationContractError("candidate_id must be nonempty")
        if not self.objectives:
            raise EvaluationContractError("at least one objective is required")
        if len(self.objectives) != len(self.objective_names):
            raise EvaluationContractError("objective names and values must align")
        if len(self.constraints) != len(self.constraint_names):
            raise EvaluationContractError("constraint names and values must align")
        if len(set(self.objective_names)) != len(self.objective_names):
            raise EvaluationContractError("objective names must be unique")
        if len(set(self.constraint_names)) != len(self.constraint_names):
            raise EvaluationContractError("constraint names must be unique")
        if not all(isfinite(value) for value in self.objectives):
            raise EvaluationContractError("objective values must be finite")
        if not all(isfinite(value) for value in self.constraints):
            raise EvaluationContractError("constraint values must be finite")

    @property
    def feasible(self) -> bool:
        return all(value <= 0.0 for value in self.constraints)

    @property
    def total_violation(self) -> float:
        return sum(max(0.0, value) for value in self.constraints)


class TerminalCode(str, Enum):
    """The mutually exclusive F22 terminal classifications."""

    ACCEPTED = "ACCEPTED"
    REJECT_SAFETY_FILTER = "REJECT_SAFETY_FILTER"
    REJECT_NO_FEASIBLE = "REJECT_NO_FEASIBLE"
    REJECT_BUDGET_NO_FEASIBLE = "REJECT_BUDGET_NO_FEASIBLE"
    REJECT_TIMEOUT = "REJECT_TIMEOUT"
    REJECT_NUMERICAL = "REJECT_NUMERICAL"
    INVALID_STATE_INTEGRITY = "INVALID_STATE_INTEGRITY"


@dataclass(frozen=True)
class TerminalOutcome:
    """A typed terminal result; rejection is never represented as missing data."""

    code: TerminalCode
    candidate_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.code is TerminalCode.ACCEPTED and not self.candidate_id:
            raise EvaluationContractError(
                "ACCEPTED terminal outcome requires candidate_id"
            )
