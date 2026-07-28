"""Append-only joint evaluator and resource ledger for v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class LedgerIntegrityError(RuntimeError):
    """The append-only budget record is internally inconsistent."""


class BudgetExceeded(RuntimeError):
    """A new evaluator call would exceed the logical CFE ceiling."""


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_metadata(item)
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_freeze_metadata(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class EvaluationCharge:
    sequence: int
    candidate_id: str
    event_id: int
    atomic_steps: int
    origin: str
    cached: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationFailure:
    sequence: int
    candidate_id: str
    event_id: int
    failure_type: str
    reason: str


@dataclass(frozen=True)
class EvaluationFailure:
    sequence: int
    candidate_id: str
    event_id: int
    failure_type: str
    reason: str


class EvaluationLedger:
    """Charge every joint objective/constraint evaluator entry exactly once."""

    def __init__(self, *, max_cfe: int) -> None:
        if max_cfe < 1:
            raise ValueError("max_cfe must be positive")
        self.max_cfe = int(max_cfe)
        self._evaluations: list[EvaluationCharge] = []
        self._validation_failures: list[ValidationFailure] = []
        self._evaluation_failures: list[EvaluationFailure] = []
        self._candidate_ids: set[str] = set()
        self._execution_transition_count = 0
        self._atomic_steps = 0
        self._sequence = 0

    @property
    def cfe(self) -> int:
        return len(self._evaluations)

    @property
    def atomic_steps(self) -> int:
        return self._atomic_steps

    @property
    def repair_failure_count(self) -> int:
        return len(self._validation_failures)

    @property
    def evaluation_failure_count(self) -> int:
        return len(self._evaluation_failures)

    @property
    def execution_transition_count(self) -> int:
        return self._execution_transition_count

    @property
    def evaluations(self) -> tuple[EvaluationCharge, ...]:
        return tuple(self._evaluations)

    @property
    def validation_failures(self) -> tuple[ValidationFailure, ...]:
        return tuple(self._validation_failures)

    @property
    def evaluation_failures(self) -> tuple[EvaluationFailure, ...]:
        return tuple(self._evaluation_failures)

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def charge_candidate(
        self,
        *,
        candidate_id: str,
        event_id: int,
        atomic_steps: int,
        origin: str = "trial",
        cached: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if len(self._evaluations) >= self.max_cfe:
            raise BudgetExceeded("CFE budget exhausted before evaluator entry")
        if not candidate_id:
            raise LedgerIntegrityError("candidate_id must be nonempty")
        if candidate_id in self._candidate_ids:
            raise LedgerIntegrityError("candidate_id was charged more than once")
        if event_id < 0:
            raise LedgerIntegrityError("event_id must be nonnegative")
        if atomic_steps < 1:
            raise LedgerIntegrityError("atomic_steps must be positive")
        if not origin:
            raise LedgerIntegrityError("evaluation origin must be explicit")

        self._candidate_ids.add(candidate_id)
        self._evaluations.append(
            EvaluationCharge(
                sequence=self._next_sequence(),
                candidate_id=candidate_id,
                event_id=int(event_id),
                atomic_steps=int(atomic_steps),
                origin=origin,
                cached=bool(cached),
                metadata=_freeze_metadata(metadata or {}),
            )
        )
        self._atomic_steps += int(atomic_steps)

    def charge_candidate_batch(
        self,
        *,
        candidate_ids: tuple[str, ...],
        event_id: int,
        atomic_steps: int,
        origin: str = "trial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically append an ordered all-success evaluation batch."""

        ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
        if not ids:
            return
        if len(self._evaluations) + len(ids) > self.max_cfe:
            raise BudgetExceeded(
                "CFE budget exhausted before batch evaluator entry"
            )
        if any(not candidate_id for candidate_id in ids):
            raise LedgerIntegrityError("candidate_id must be nonempty")
        if len(set(ids)) != len(ids):
            raise LedgerIntegrityError(
                "candidate_id was duplicated within one batch"
            )
        if any(candidate_id in self._candidate_ids for candidate_id in ids):
            raise LedgerIntegrityError(
                "candidate_id was charged more than once"
            )
        if event_id < 0:
            raise LedgerIntegrityError("event_id must be nonnegative")
        if atomic_steps < 1:
            raise LedgerIntegrityError("atomic_steps must be positive")
        if not origin:
            raise LedgerIntegrityError("evaluation origin must be explicit")

        frozen_metadata = _freeze_metadata(metadata or {})
        first_sequence = self._sequence + 1
        charges = [
            EvaluationCharge(
                sequence=first_sequence + index,
                candidate_id=candidate_id,
                event_id=int(event_id),
                atomic_steps=int(atomic_steps),
                origin=origin,
                cached=False,
                metadata=frozen_metadata,
            )
            for index, candidate_id in enumerate(ids)
        ]
        self._evaluations.extend(charges)
        self._candidate_ids.update(ids)
        self._atomic_steps += int(atomic_steps) * len(ids)
        self._sequence += len(ids)

    def record_repair_failure(
        self, *, candidate_id: str, event_id: int, reason: str
    ) -> None:
        self._validation_failures.append(
            ValidationFailure(
                sequence=self._next_sequence(),
                candidate_id=candidate_id,
                event_id=int(event_id),
                failure_type="REPAIR_FAILED",
                reason=reason,
            )
        )

    def record_evaluation_failure(
        self,
        *,
        candidate_id: str,
        event_id: int,
        error: Exception,
    ) -> None:
        if candidate_id not in self._candidate_ids:
            raise LedgerIntegrityError(
                "evaluation failure cannot be recorded before CFE charge"
            )
        self._evaluation_failures.append(
            EvaluationFailure(
                sequence=self._next_sequence(),
                candidate_id=candidate_id,
                event_id=int(event_id),
                failure_type=type(error).__name__,
                reason=str(error),
            )
        )

    def record_execution(self) -> None:
        self._execution_transition_count += 1

    def snapshot(self) -> dict[str, int]:
        cfe = self.cfe
        return {
            "cfe": cfe,
            "objective_calls": cfe,
            "constraint_calls": cfe,
            "scenario_evaluations": cfe,
            "atomic_model_steps": self.atomic_steps,
            "execution_transition_count": self.execution_transition_count,
            "repair_failed": self.repair_failure_count,
            "evaluation_failures": self.evaluation_failure_count,
        }

    def assert_joint_contract(
        self, *, atomic_steps_per_evaluation: int | None = None
    ) -> None:
        snapshot = self.snapshot()
        if not (
            snapshot["cfe"]
            == snapshot["objective_calls"]
            == snapshot["constraint_calls"]
            == snapshot["scenario_evaluations"]
        ):
            raise LedgerIntegrityError("joint evaluator ledgers disagree")
        if (
            atomic_steps_per_evaluation is not None
            and snapshot["atomic_model_steps"]
            != atomic_steps_per_evaluation * snapshot["cfe"]
        ):
            raise LedgerIntegrityError(
                "atomic-step ledger differs from the adapter contract"
            )
