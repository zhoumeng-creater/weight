"""R2 correctness-only DT-RAMDE event engine.

Adaptive-port provenance:
    FORMAL_V1/dt_ramde_formal/engine.py
    SHA-256 a3f4b1bcad5330f91f05147c70879ceae38003c5e486364fc043e7891e2f4659

The v1.0 run permissions, rolling-only evaluator, and hard-coded terminal
scalarization are not ported. This engine accepts only R2 correctness fixtures
and an independently identified terminal selector.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.contracts import (
    EvaluationResult,
    NumericalEvaluationError,
    TerminalCode,
    TerminalOutcome,
)
from evaluation.firewall import (
    InformationBoundaryError,
    InformationSnapshot,
    validate_information_snapshot,
)
from evaluation.evaluator import (
    BatchEvaluationUnavailableBeforeEntry,
    ExecutionTimeoutBeforeEntry,
)
from evaluation.ledger import EvaluationLedger
from evaluation.randomness import RandomStream, derive_rng

from .contracts import (
    AlgorithmConfig,
    COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
    R6ExecutionRequest,
    R8CCorrectiveExecutionRequest,
    R8ExecutionRequest,
)
from .core import (
    Candidate,
    assign_rank_and_crowding,
    constrained_sort_key,
    environmental_select,
    maintain_nondominated_archive,
    repair_midpoint,
    sample_parameters,
    shade_success_improvement,
    update_mg,
)
from .interfaces import EventProblemAdapter, TerminalSelector
from .state import (
    COMPONENTS,
    LineageDAG,
    LineageNode,
    MGState,
    MemoryState,
    PendingCredit,
    StateIntegrityError,
    StateMachine,
    age_prune_bank,
    append_atoms,
    apply_reset_gate,
    close_event_cooldown,
    resolve_pending,
    sample_atom,
)


class NonrecoverableEvaluationError(RuntimeError):
    """An evaluator entry failed after the mandatory joint CFE was charged."""


_NO_RNG_DERIVATION_OPTIONS: Mapping[str, bool] = {}
_COMPACT_RNG_DERIVATION_OPTIONS: Mapping[str, bool] = {
    "include_manifest": False,
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_hash(vector: Sequence[float]) -> str:
    return _canonical_hash(_serializable_vector(vector))


def _serializable_vector(vector: Sequence[float]) -> list[float | str]:
    values: list[float | str] = []
    for raw in np.asarray(vector, dtype=float).reshape(-1):
        value = float(raw)
        values.append(value if math.isfinite(value) else repr(value))
    return values


def _vector_audit_material(
    vector: Sequence[float],
) -> tuple[list[float | str], str]:
    values = _serializable_vector(vector)
    return values, _canonical_hash(values)


def _evaluation_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "objectives": list(result.objectives),
        "objective_names": list(result.objective_names),
        "constraints": list(result.constraints),
        "constraint_names": list(result.constraint_names),
        "feasible": result.feasible,
        "total_violation": result.total_violation,
    }


@dataclass(frozen=True)
class EventRunResult:
    event_id: int
    terminal: TerminalOutcome
    archive: tuple[EvaluationResult, ...]
    information_hash: str
    ledger: Mapping[str, int]
    state_transitions: tuple[Mapping[str, Any], ...]
    credit_resolution_status: str
    resolved_q: float | None
    reset_reason: str | None
    warm_start_seed_count: int
    mg_final: Mapping[str, Any]
    trial_audit: tuple[Mapping[str, Any], ...]
    initialization_audit: Mapping[str, Any]
    lineage_records: tuple[Mapping[str, Any], ...]
    archive_audit: tuple[Mapping[str, Any], ...]
    memory_snapshot: Mapping[str, Any]
    execution_feedback: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "terminal": {
                "code": self.terminal.code.value,
                "candidate_id": self.terminal.candidate_id,
                "reason": self.terminal.reason,
            },
            "archive": [_evaluation_to_dict(item) for item in self.archive],
            "information_hash": self.information_hash,
            "ledger": dict(self.ledger),
            "state_transitions": [
                dict(transition) for transition in self.state_transitions
            ],
            "credit_resolution_status": self.credit_resolution_status,
            "resolved_q": self.resolved_q,
            "reset_reason": self.reset_reason,
            "warm_start_seed_count": self.warm_start_seed_count,
            "mg_final": dict(self.mg_final),
            "trial_audit": [dict(item) for item in self.trial_audit],
            "initialization_audit": dict(self.initialization_audit),
            "lineage_records": [
                dict(item) for item in self.lineage_records
            ],
            "archive_audit": [dict(item) for item in self.archive_audit],
            "memory_snapshot": dict(self.memory_snapshot),
            "execution_feedback": (
                None
                if self.execution_feedback is None
                else dict(self.execution_feedback)
            ),
        }


@dataclass(frozen=True)
class SequenceRunResult:
    config: Mapping[str, Any]
    adapter_identity: Mapping[str, Any]
    selector_identity: Mapping[str, Any]
    events: tuple[EventRunResult, ...]
    persistent_state: Mapping[str, Any]
    effect_estimation_performed: bool = False
    hidden_seed_or_instance_generated: bool = False
    confirmatory_execution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "adapter_identity": dict(self.adapter_identity),
            "selector_identity": dict(self.selector_identity),
            "events": [event.to_dict() for event in self.events],
            "persistent_state": dict(self.persistent_state),
            "effect_estimation_performed": self.effect_estimation_performed,
            "hidden_seed_or_instance_generated": self.hidden_seed_or_instance_generated,
            "confirmatory_execution": self.confirmatory_execution,
        }


@dataclass(frozen=True)
class _GenerationSelectionCache:
    """Generation-static material used by DE parent selection."""

    p_count: int
    population_candidate_ids: tuple[str, ...]
    r1_indices_by_target: tuple[list[int], ...]
    r2_candidates: tuple[Candidate, ...]
    r2_positions_by_candidate_id: Mapping[str, tuple[int, ...]]


def _build_generation_selection_cache(
    population: Sequence[Candidate],
    inferior_archive: Sequence[Candidate],
) -> _GenerationSelectionCache:
    size = len(population)
    p_count = max(
        1,
        math.ceil(min(1.0, max(0.10, 2.0 / size)) * size),
    )
    population_candidate_ids = tuple(
        candidate.candidate_id for candidate in population
    )
    r1_indices_by_target = tuple(
        [index for index in range(size) if index != target_index]
        for target_index in range(size)
    )
    r2_candidates = tuple(population) + tuple(inferior_archive)
    positions: dict[str, list[int]] = {}
    for position, candidate in enumerate(r2_candidates):
        positions.setdefault(candidate.candidate_id, []).append(position)
    return _GenerationSelectionCache(
        p_count=p_count,
        population_candidate_ids=population_candidate_ids,
        r1_indices_by_target=r1_indices_by_target,
        r2_candidates=r2_candidates,
        r2_positions_by_candidate_id={
            candidate_id: tuple(candidate_positions)
            for candidate_id, candidate_positions in positions.items()
        },
    )


class DTRAMDE:
    """Contract-bound engine for FULL and its F22 mechanism ablations."""

    method_version = "0.1.0.dev0"

    def __init__(
        self,
        config: AlgorithmConfig,
        memory: MemoryState | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        self.config = config
        self.memory = memory or MemoryState()
        self._clock = clock
        self._uses_default_monotonic_clock = clock is time.monotonic
        self._batch_deadline_guard_seconds = 5.0
        self._compact_checkpoint_audit = (
            config.audit_materialization
            == COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION
        )
        self.method_id = config.method_label
        if isinstance(
            config.execution_request,
            (R8ExecutionRequest, R8CCorrectiveExecutionRequest),
        ):
            self._execution_role = (
                "R8C_corrective_formal_public_execution"
                if isinstance(
                    config.execution_request,
                    R8CCorrectiveExecutionRequest,
                )
                else "R8_frozen_formal_public_execution"
            )
        elif isinstance(config.execution_request, R6ExecutionRequest):
            self._execution_role = "R6_result_blind_engineering_pilot_only"
        else:
            self._execution_role = "R2_correctness_only"
        self.run_id = _canonical_hash(
            {"config": config.to_dict(), "purpose": self._execution_role}
        )[:24]
        if self.memory.checkpoint_checksum is not None:
            self.memory.verify_checkpoint()
        if self.memory.run_binding_id is None:
            if self.memory.event_index >= 0:
                raise StateIntegrityError(
                    "persisted state is missing its run binding"
                )
            self.memory.run_binding_id = self.run_id
        elif self.memory.run_binding_id != self.run_id:
            raise StateIntegrityError(
                "persisted state run binding differs from configuration"
            )
        self.memory.validate()
        if self.memory.invalidated:
            raise StateIntegrityError(
                "persisted run state is permanently invalidated"
            )
        self.memory.validate_timing(config.timing_mode)
        self.memory.seal_checkpoint()

    def _resource_exhausted(self, deadline: float) -> bool:
        return float(self._clock()) >= deadline

    def _batch_allowed(self, deadline: float) -> bool:
        if not self._uses_default_monotonic_clock:
            return False
        return (
            deadline - float(self._clock())
            >= self._batch_deadline_guard_seconds
        )

    def identity(self) -> Mapping[str, Any]:
        identity = {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "variant": self.config.variant,
            "role": self._execution_role,
        }
        if self._compact_checkpoint_audit:
            identity["audit_materialization"] = (
                COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION
            )
        return identity

    def _validate_bindings(
        self, problem: EventProblemAdapter, selector: TerminalSelector
    ) -> tuple[np.ndarray, np.ndarray]:
        if problem.adapter_id != self.config.adapter_id:
            raise StateIntegrityError(
                "adapter identity differs from algorithm configuration"
            )
        if problem.adapter_version != self.config.adapter_version:
            raise StateIntegrityError(
                "adapter version differs from algorithm configuration"
            )
        if (
            problem.atomic_steps_per_evaluation
            != self.config.atomic_steps_per_evaluation
        ):
            raise StateIntegrityError(
                "adapter atomic-step multiplier differs from configuration"
            )
        if problem.decision_dimension < 1:
            raise StateIntegrityError("decision dimension must be positive")
        lower = np.asarray(problem.lower_bounds, dtype=float)
        upper = np.asarray(problem.upper_bounds, dtype=float)
        if (
            lower.shape != (problem.decision_dimension,)
            or upper.shape != (problem.decision_dimension,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or not np.all(lower <= upper)
        ):
            raise StateIntegrityError("adapter bounds are invalid")
        scales = tuple(float(value) for value in problem.constraint_scales)
        if not scales or not all(
            math.isfinite(value) and value > 0.0 for value in scales
        ):
            raise StateIntegrityError("constraint scales must be finite and positive")
        if (
            selector.selector_id != self.config.selector_id
            or selector.selector_version != self.config.selector_version
        ):
            raise StateIntegrityError(
                "terminal selector identity differs from configuration"
            )
        if (
            self.memory.pending_credit is not None
            and self.memory.pending_credit.adapter_version
            != problem.adapter_version
        ):
            raise StateIntegrityError(
                "pending credit adapter version differs from current adapter"
            )
        return lower, upper

    def _evaluate(
        self,
        problem: EventProblemAdapter,
        *,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> Candidate:
        cfe_before = ledger.cfe
        failures_before = ledger.evaluation_failure_count
        try:
            result = problem.evaluate(vector, event_id, ledger, candidate_id)
        except Exception as error:
            cfe_after = ledger.cfe
            if (
                isinstance(error, ExecutionTimeoutBeforeEntry)
                and cfe_after == cfe_before
            ):
                raise
            if cfe_after != cfe_before + 1:
                raise StateIntegrityError(
                    "adapter must charge exactly one joint CFE even when evaluation fails"
                ) from error
            failures_after = ledger.evaluation_failure_count
            if failures_after == failures_before:
                ledger.record_evaluation_failure(
                    candidate_id=candidate_id,
                    event_id=event_id,
                    error=error,
                )
            elif failures_after != failures_before + 1:
                raise StateIntegrityError(
                    "adapter must log exactly one evaluation failure"
                ) from error
            if isinstance(error, StateIntegrityError):
                raise
            if isinstance(error, NumericalEvaluationError):
                raise NonrecoverableEvaluationError(
                    "joint evaluator raised a nonrecoverable error"
                ) from error
            raise StateIntegrityError(
                "adapter/evaluator contract failed after CFE entry"
            ) from error
        cfe_after = ledger.cfe
        if cfe_after != cfe_before + 1:
            raise StateIntegrityError(
                "adapter must charge exactly one joint CFE per evaluate call"
            )
        if not isinstance(result, EvaluationResult):
            raise StateIntegrityError(
                "adapter must return the shared EvaluationResult type"
            )
        if result.candidate_id != candidate_id:
            raise StateIntegrityError("adapter changed candidate identity")
        if len(result.constraints) != len(problem.constraint_scales):
            raise StateIntegrityError(
                "adapter constraint scales do not align with results"
            )
        return Candidate(
            vector=vector,
            evaluation=result,
            lineage_node_id=candidate_id,
        )

    def _evaluate_batch(
        self,
        problem: EventProblemAdapter,
        *,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_ids: Sequence[str],
    ) -> tuple[Candidate, ...]:
        batch_evaluator = getattr(problem, "evaluate_batch", None)
        if not callable(batch_evaluator):
            raise BatchEvaluationUnavailableBeforeEntry(
                "problem adapter has no ordered batch method"
            )
        cfe_before = ledger.cfe
        failures_before = ledger.evaluation_failure_count
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
            if (
                ledger.cfe != cfe_before
                or ledger.evaluation_failure_count != failures_before
            ):
                raise StateIntegrityError(
                    "batch fallback was requested after ledger mutation"
                )
            raise
        except ExecutionTimeoutBeforeEntry:
            if (
                ledger.cfe != cfe_before
                or ledger.evaluation_failure_count != failures_before
            ):
                raise StateIntegrityError(
                    "task timeout was requested after batch ledger mutation"
                )
            raise
        except Exception as error:
            if (
                ledger.cfe != cfe_before
                or ledger.evaluation_failure_count != failures_before
            ):
                raise StateIntegrityError(
                    "batch evaluator failed after ledger mutation"
                ) from error
            raise StateIntegrityError(
                "batch evaluator violated its pre-entry contract"
            ) from error
        if len(results) != len(vectors):
            raise StateIntegrityError(
                "batch evaluator returned the wrong result count"
            )
        if ledger.cfe != cfe_before + len(results):
            raise StateIntegrityError(
                "batch evaluator did not charge one CFE per result"
            )
        if ledger.evaluation_failure_count != failures_before:
            raise StateIntegrityError(
                "all-success batch recorded an evaluation failure"
            )
        candidates: list[Candidate] = []
        for vector, candidate_id, result in zip(
            vectors, candidate_ids, results, strict=True
        ):
            if not isinstance(result, EvaluationResult):
                raise StateIntegrityError(
                    "batch adapter must return EvaluationResult values"
                )
            if result.candidate_id != candidate_id:
                raise StateIntegrityError(
                    "batch adapter changed candidate identity"
                )
            if len(result.constraints) != len(problem.constraint_scales):
                raise StateIntegrityError(
                    "batch adapter constraint scales do not align"
                )
            candidates.append(
                Candidate(
                    vector=vector,
                    evaluation=result,
                    lineage_node_id=candidate_id,
                )
            )
        return tuple(candidates)

    def _initialize_population(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        ledger: EvaluationLedger,
        lineage: LineageDAG,
        deadline: float,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[
        list[Candidate],
        int,
        str | None,
        bool,
        Mapping[str, Any],
    ]:
        population_size = self.config.population_size
        vectors: list[np.ndarray] = []
        vector_sources: list[dict[str, Any]] = []
        warm_start_attempts: list[dict[str, Any]] = []
        components = COMPONENTS[self.config.variant]
        if components.warm_start and self.memory.transfer_allowed:
            for source_index, previous in enumerate(self.memory.solution_memory):
                shifted = np.asarray(problem.shift_solution(previous), dtype=float)
                if (
                    shifted.shape != (problem.decision_dimension,)
                    or not np.all(np.isfinite(shifted))
                    or np.any(shifted < lower_bounds)
                    or np.any(shifted > upper_bounds)
                ):
                    raise StateIntegrityError("shifted warm-start vector is invalid")
                duplicate = any(
                    np.array_equal(shifted, vector) for vector in vectors
                )
                if not self._compact_checkpoint_audit:
                    input_vector, input_hash = _vector_audit_material(previous)
                    shifted_vector, shifted_hash = _vector_audit_material(shifted)
                    attempt = {
                        "source_index": source_index,
                        "input_vector": input_vector,
                        "input_hash": input_hash,
                        "shifted_vector": shifted_vector,
                        "shifted_hash": shifted_hash,
                        "duplicate": duplicate,
                    }
                    warm_start_attempts.append(attempt)
                if not duplicate:
                    vectors.append(shifted)
                    if not self._compact_checkpoint_audit:
                        vector_sources.append(
                            {
                                "source": "warm_start",
                                "source_index": source_index,
                                "input_hash": input_hash,
                                "shifted_hash": shifted_hash,
                            }
                        )
                if len(vectors) >= math.ceil(0.25 * population_size):
                    break
        warm_count = len(vectors)
        rng, initialization_manifest = derive_rng(
            self.config.algorithm_seed,
            stream=RandomStream.INITIALIZATION,
            experiment_id=self.config.configuration_evidence_id,
            unit_id=problem.adapter_id,
            event_id=event_id,
            substream="initialization",
            **(
                _COMPACT_RNG_DERIVATION_OPTIONS
                if self._compact_checkpoint_audit
                else _NO_RNG_DERIVATION_OPTIONS
            ),
        )
        while len(vectors) < population_size:
            vectors.append(rng.uniform(lower_bounds, upper_bounds))
            if not self._compact_checkpoint_audit:
                vector_sources.append(
                    {
                        "source": "random",
                        "rng_draw_index": len(vectors) - warm_count - 1,
                    }
                )

        population: list[Candidate] = []
        root_audit: list[dict[str, Any]] = []
        numerical_candidate: str | None = None
        timed_out = False
        if self._batch_allowed(deadline):
            root_ids = [
                f"e{event_id}:root:{index:06d}"
                for index in range(len(vectors))
            ]
            try:
                batch_population = self._evaluate_batch(
                    problem,
                    vectors=vectors,
                    event_id=event_id,
                    ledger=ledger,
                    candidate_ids=root_ids,
                )
            except BatchEvaluationUnavailableBeforeEntry:
                batch_population = ()
            else:
                for index, candidate in enumerate(batch_population):
                    if self._compact_checkpoint_audit:
                        lineage_node = LineageNode(
                            node_id=root_ids[index],
                            event_id=event_id,
                            generation=-1,
                            target_predecessor=None,
                            f=None,
                            cr=None,
                            survival=True,
                        )
                    else:
                        vector_values, vector_hash = _vector_audit_material(
                            vectors[index]
                        )
                        root_audit.append(
                            {
                                "candidate_id": root_ids[index],
                                **vector_sources[index],
                                "vector": vector_values,
                                "vector_hash": vector_hash,
                                "evaluation_status": "completed",
                            }
                        )
                        lineage_node = LineageNode(
                            node_id=root_ids[index],
                            event_id=event_id,
                            generation=-1,
                            target_predecessor=None,
                            f=None,
                            cr=None,
                            survival=True,
                            pre_repair_hash=vector_hash,
                            post_repair_hash=vector_hash,
                            objectives=candidate.evaluation.objectives,
                            constraints=candidate.evaluation.constraints,
                            feasible=candidate.feasible,
                            normalized_cv=candidate.normalized_violation(
                                problem.constraint_scales
                            ),
                        )
                    lineage.add(lineage_node)
                    population.append(candidate)
                return (
                    population,
                    warm_count,
                    None,
                    self._resource_exhausted(deadline),
                    (
                        {}
                        if self._compact_checkpoint_audit
                        else {
                            "rng": initialization_manifest,
                            "warm_start_attempts": warm_start_attempts,
                            "roots": root_audit,
                        }
                    ),
                )
        for index, vector in enumerate(vectors):
            candidate_id = f"e{event_id}:root:{index:06d}"
            root_record: dict[str, Any] | None = None
            vector_hash = ""
            if not self._compact_checkpoint_audit:
                vector_values, vector_hash = _vector_audit_material(vector)
                root_record = {
                    "candidate_id": candidate_id,
                    **vector_sources[index],
                    "vector": vector_values,
                    "vector_hash": vector_hash,
                    "evaluation_status": "pending",
                }
            if self._resource_exhausted(deadline):
                if root_record is not None:
                    root_record["evaluation_status"] = "not_started_timeout"
                    root_audit.append(root_record)
                timed_out = True
                break
            try:
                candidate = self._evaluate(
                    problem,
                    vector=vector,
                    event_id=event_id,
                    ledger=ledger,
                    candidate_id=candidate_id,
                )
            except NonrecoverableEvaluationError:
                if root_record is not None:
                    root_record["evaluation_status"] = "numerical_failure"
                    root_audit.append(root_record)
                lineage.add(
                    LineageNode(
                        node_id=candidate_id,
                        event_id=event_id,
                        generation=-1,
                        target_predecessor=None,
                        f=None,
                        cr=None,
                        survival=False,
                        pre_repair_hash=vector_hash,
                        post_repair_hash=vector_hash,
                        evaluation_status="numerical_failure",
                    )
                )
                numerical_candidate = candidate_id
                break
            if root_record is not None:
                root_record["evaluation_status"] = "completed"
                root_audit.append(root_record)
            if self._compact_checkpoint_audit:
                lineage_node = LineageNode(
                    node_id=candidate_id,
                    event_id=event_id,
                    generation=-1,
                    target_predecessor=None,
                    f=None,
                    cr=None,
                    survival=True,
                )
            else:
                lineage_node = LineageNode(
                    node_id=candidate_id,
                    event_id=event_id,
                    generation=-1,
                    target_predecessor=None,
                    f=None,
                    cr=None,
                    survival=True,
                    pre_repair_hash=vector_hash,
                    post_repair_hash=vector_hash,
                    objectives=candidate.evaluation.objectives,
                    constraints=candidate.evaluation.constraints,
                    feasible=candidate.feasible,
                    normalized_cv=candidate.normalized_violation(
                        problem.constraint_scales
                    ),
                )
            lineage.add(lineage_node)
            population.append(candidate)
            if self._resource_exhausted(deadline):
                timed_out = True
                break
        return (
            population,
            warm_count,
            numerical_candidate,
            timed_out,
            (
                {}
                if self._compact_checkpoint_audit
                else {
                    "rng": initialization_manifest,
                    "warm_start_attempts": warm_start_attempts,
                    "roots": root_audit,
                }
            ),
        )

    def _parameter_source(
        self,
        *,
        problem: EventProblemAdapter,
        event_id: int,
        generation: int,
        target_index: int,
        mg: MGState,
    ) -> tuple[float, float, str, Mapping[str, Any] | None]:
        rng, manifest = derive_rng(
            self.config.algorithm_seed,
            stream=RandomStream.ALGORITHM,
            experiment_id=self.config.configuration_evidence_id,
            unit_id=problem.adapter_id,
            method_id=self.method_id,
            event_id=event_id,
            generation=generation,
            target_index=target_index,
            substream="parameter",
            **(
                _COMPACT_RNG_DERIVATION_OPTIONS
                if self._compact_checkpoint_audit
                else _NO_RNG_DERIVATION_OPTIONS
            ),
        )
        component = COMPONENTS[self.config.variant]
        atom = None
        has_eligible_atom = any(
            item.signed_credit > 0.0 and item.age <= 5
            for item in self.memory.bank
        )
        if (
            component.parameter_memory
            and self.memory.transfer_allowed
            and has_eligible_atom
            and float(rng.random()) < 0.5 * self.memory.tau
        ):
            atom = sample_atom(self.memory, rng)
            if atom is None:
                raise StateIntegrityError(
                    "eligible parameter bank could not produce an atom"
                )
        if atom is None:
            slot = int(rng.integers(0, len(mg.memory_f)))
            mu_f = mg.memory_f[slot]
            mu_cr = mg.memory_cr[slot]
            source = f"M_g:{slot}"
        else:
            mu_f, mu_cr = atom.f, atom.cr
            source = f"M_k:{atom.source_event}:{atom.lineage_node_id}"
        sample = sample_parameters(rng, mu_f=mu_f, mu_cr=mu_cr)
        audit = (
            None
            if self._compact_checkpoint_audit
            else {
                "rng": manifest,
                "fallback_f": sample.fallback_f,
                "fallback_cr": sample.fallback_cr,
                "f_draws": list(sample.f_draws),
                "raw_cr": sample.raw_cr,
                "mu_f": mu_f,
                "mu_cr": mu_cr,
            }
        )
        return sample.f, sample.cr, source, audit

    def _trial_vector(
        self,
        population: list[Candidate],
        inferior_archive: list[Candidate],
        ranked_population: Sequence[Candidate],
        *,
        problem: EventProblemAdapter,
        event_id: int,
        generation: int,
        target_index: int,
        f_value: float,
        cr_value: float,
        selection_cache: _GenerationSelectionCache | None = None,
    ) -> tuple[np.ndarray, Mapping[str, Any] | None]:
        if selection_cache is None:
            selection_cache = _build_generation_selection_cache(
                population,
                inferior_archive,
            )
        rng, operator_manifest = derive_rng(
            self.config.algorithm_seed,
            stream=RandomStream.ALGORITHM,
            experiment_id=self.config.configuration_evidence_id,
            unit_id=problem.adapter_id,
            method_id=self.method_id,
            event_id=event_id,
            generation=generation,
            target_index=target_index,
            substream="operator",
            **(
                _COMPACT_RNG_DERIVATION_OPTIONS
                if self._compact_checkpoint_audit
                else _NO_RNG_DERIVATION_OPTIONS
            ),
        )
        pbest = ranked_population[
            int(rng.integers(0, selection_cache.p_count))
        ]
        r1_position = int(
            rng.choice(selection_cache.r1_indices_by_target[target_index])
        )
        r1 = population[r1_position]
        target = population[target_index]
        excluded_ids = {
            selection_cache.population_candidate_ids[target_index],
            selection_cache.population_candidate_ids[r1_position],
        }
        excluded_positions = sorted(
            position
            for candidate_id in excluded_ids
            for position in selection_cache.r2_positions_by_candidate_id[
                candidate_id
            ]
        )
        r2_count = len(selection_cache.r2_candidates) - len(
            excluded_positions
        )
        if r2_count <= 0:
            raise StateIntegrityError("no legal r2 candidate")
        r2_position = int(rng.integers(0, r2_count))
        for excluded_position in excluded_positions:
            if excluded_position > r2_position:
                break
            r2_position += 1
        r2 = selection_cache.r2_candidates[r2_position]
        mutant = (
            target.vector
            + f_value * (pbest.vector - target.vector)
            + f_value * (r1.vector - r2.vector)
        )
        j_rand_rng, j_rand_manifest = derive_rng(
            self.config.algorithm_seed,
            stream=RandomStream.ALGORITHM,
            experiment_id=self.config.configuration_evidence_id,
            unit_id=problem.adapter_id,
            method_id=self.method_id,
            event_id=event_id,
            generation=generation,
            target_index=target_index,
            substream="j_rand",
            **(
                _COMPACT_RNG_DERIVATION_OPTIONS
                if self._compact_checkpoint_audit
                else _NO_RNG_DERIVATION_OPTIONS
            ),
        )
        j_rand = int(j_rand_rng.integers(0, len(mutant)))
        mask = rng.random(len(mutant)) < cr_value
        mask[j_rand] = True
        audit = (
            None
            if self._compact_checkpoint_audit
            else {
                "pbest": pbest,
                "r1": r1,
                "r2": r2,
                "j_rand": j_rand,
                "rng": {
                    "operator": operator_manifest,
                    "j_rand": j_rand_manifest,
                },
            }
        )
        return np.where(mask, mutant, target.vector), audit

    def _try_batched_generation(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        generation: int,
        population: list[Candidate],
        ranked_population: Sequence[Candidate],
        inferior_archive: list[Candidate],
        selection_cache: _GenerationSelectionCache,
        mg: MGState,
        ledger: EvaluationLedger,
        lineage: LineageDAG,
        trial_audit: list[dict[str, Any]],
        deadline: float,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[
        list[Candidate],
        dict[str, dict[str, Any]],
        bool,
    ]:
        """Attempt one all-success natural batch with zero-side-effect fallback."""

        if not self._batch_allowed(deadline):
            raise BatchEvaluationUnavailableBeforeEntry(
                "deadline guard requires scalar evaluation"
            )
        plans: list[dict[str, Any]] = []
        for target_index, target in enumerate(population):
            if self._resource_exhausted(deadline):
                raise BatchEvaluationUnavailableBeforeEntry(
                    "deadline reached while constructing generation"
                )
            f_value, cr_value, source, parameter_audit = (
                self._parameter_source(
                    problem=problem,
                    event_id=event_id,
                    generation=generation,
                    target_index=target_index,
                    mg=mg,
                )
            )
            raw, operator = self._trial_vector(
                population,
                inferior_archive,
                ranked_population,
                problem=problem,
                event_id=event_id,
                generation=generation,
                target_index=target_index,
                f_value=f_value,
                cr_value=cr_value,
                selection_cache=selection_cache,
            )
            repaired, changed = repair_midpoint(
                raw,
                target.vector,
                lower_bounds,
                upper_bounds,
            )
            if repaired is None:
                raise BatchEvaluationUnavailableBeforeEntry(
                    "repair failure requires scalar ledger ordering"
                )
            node_id = (
                f"e{event_id}:g{generation:06d}:t{target_index:06d}"
            )
            pre_repair_hash = ""
            post_repair_hash = ""
            base_audit: dict[str, Any] | None = None
            if not self._compact_checkpoint_audit:
                assert operator is not None
                assert parameter_audit is not None
                raw_vector, pre_repair_hash = _vector_audit_material(raw)
                repaired_vector, post_repair_hash = _vector_audit_material(
                    repaired
                )
                operator_audit = {
                    "pbest_id": operator["pbest"].candidate_id,
                    "r1_id": operator["r1"].candidate_id,
                    "r2_id": operator["r2"].candidate_id,
                    "j_rand": operator["j_rand"],
                    "rng": operator["rng"],
                }
                changed_coordinates = [
                    int(index)
                    for index in np.flatnonzero(raw != repaired)
                ]
                base_audit = {
                    "node_id": node_id,
                    "f": f_value,
                    "cr": cr_value,
                    "source": source,
                    "parameter_audit": parameter_audit,
                    "operator_audit": operator_audit,
                    "raw_vector": raw_vector,
                    "repaired_vector": repaired_vector,
                    "pre_repair_hash": pre_repair_hash,
                    "post_repair_hash": post_repair_hash,
                    "repaired": changed,
                    "repair_changed_coordinates": changed_coordinates,
                    "repair_reason": (
                        "midpoint_bound_repair" if changed else None
                    ),
                    "selection": "pending",
                    "archive_admission": False,
                    "inferior_parent_archive_admission": False,
                    "paired_target_id": target.candidate_id,
                    "mg_mode": COMPONENTS[self.config.variant].mg_mode,
                    "mg_success": False,
                    "mg_success_reason": "PENDING",
                    "mg_success_delta": 0.0,
                    "mg_success_weight": 0.0,
                }
            plans.append(
                {
                    "target": target,
                    "target_index": target_index,
                    "repaired": repaired,
                    "changed": changed,
                    "node_id": node_id,
                    "f": f_value,
                    "cr": cr_value,
                    "source": source,
                    "operator": operator,
                    "pre_repair_hash": pre_repair_hash,
                    "post_repair_hash": post_repair_hash,
                    "base_audit": base_audit,
                }
            )

        candidates = self._evaluate_batch(
            problem,
            vectors=[plan["repaired"] for plan in plans],
            event_id=event_id,
            ledger=ledger,
            candidate_ids=[plan["node_id"] for plan in plans],
        )
        trials: list[Candidate] = []
        metadata: dict[str, dict[str, Any]] = {}
        for plan, trial in zip(plans, candidates, strict=True):
            node_id = plan["node_id"]
            trial.lineage_node_id = node_id
            operator = plan["operator"]
            target = plan["target"]
            if self._compact_checkpoint_audit:
                lineage_node = LineageNode(
                    node_id=node_id,
                    event_id=event_id,
                    generation=generation,
                    target_predecessor=target.lineage_node_id,
                    f=plan["f"],
                    cr=plan["cr"],
                    survival=False,
                )
            else:
                assert operator is not None
                lineage_node = LineageNode(
                    node_id=node_id,
                    event_id=event_id,
                    generation=generation,
                    target_predecessor=target.lineage_node_id,
                    f=plan["f"],
                    cr=plan["cr"],
                    survival=False,
                    target_id=target.candidate_id,
                    pbest_id=operator["pbest"].candidate_id,
                    r1_id=operator["r1"].candidate_id,
                    r2_id=operator["r2"].candidate_id,
                    parameter_source=plan["source"],
                    j_rand=operator["j_rand"],
                    pre_repair_hash=plan["pre_repair_hash"],
                    post_repair_hash=plan["post_repair_hash"],
                    repaired=plan["changed"],
                    objectives=trial.evaluation.objectives,
                    constraints=trial.evaluation.constraints,
                    feasible=trial.feasible,
                    normalized_cv=trial.normalized_violation(
                        problem.constraint_scales
                    ),
                )
            lineage.add(lineage_node)
            trials.append(trial)
            audit_index = (
                None if self._compact_checkpoint_audit else len(trial_audit)
            )
            metadata[node_id] = {
                "target": target,
                "f": plan["f"],
                "cr": plan["cr"],
                "repaired": plan["changed"],
                "audit_index": audit_index,
            }
            if not self._compact_checkpoint_audit:
                assert plan["base_audit"] is not None
                trial_audit.append(
                    {
                        **plan["base_audit"],
                        "evaluated": True,
                        "repair_failed": False,
                        "evaluation_status": "completed",
                    }
                )
        timed_out = self._resource_exhausted(deadline)
        if timed_out and trials and not self._compact_checkpoint_audit:
            last_audit = trial_audit[
                metadata[trials[-1].lineage_node_id]["audit_index"]
            ]
            last_audit["selection"] = "not_applied_timeout"
            last_audit["mg_success_reason"] = "NOT_APPLIED_TIMEOUT"
        return trials, metadata, timed_out

    def run_event(
        self,
        problem: EventProblemAdapter,
        *,
        selector: TerminalSelector,
        event_id: int,
        prior_feedback: Mapping[str, Any] | None,
    ) -> EventRunResult:
        if self.memory.invalidated:
            raise StateIntegrityError("run state is permanently invalidated")
        try:
            self.memory.verify_checkpoint()
        except StateIntegrityError as error:
            self.memory.invalidated = True
            self.memory.invalidation_reason = (
                f"{type(error).__name__}: {error}"
            )
            self.memory.seal_checkpoint()
            raise
        checkpoint = self.memory.to_dict()
        try:
            return self._run_event_impl(
                problem,
                selector=selector,
                event_id=event_id,
                prior_feedback=prior_feedback,
            )
        except Exception as error:
            restored = MemoryState.from_dict(checkpoint)
            for field_name in restored.__dataclass_fields__:
                setattr(
                    self.memory,
                    field_name,
                    getattr(restored, field_name),
                )
            self.memory.invalidated = True
            self.memory.invalidation_reason = (
                f"{type(error).__name__}: {error}"
            )
            self.memory.seal_checkpoint()
            if isinstance(
                error,
                (StateIntegrityError, ExecutionTimeoutBeforeEntry),
            ):
                raise
            raise StateIntegrityError(
                "unexpected event failure invalidated the run"
            ) from error

    def _run_event_impl(
        self,
        problem: EventProblemAdapter,
        *,
        selector: TerminalSelector,
        event_id: int,
        prior_feedback: Mapping[str, Any] | None,
    ) -> EventRunResult:
        lower_bounds, upper_bounds = self._validate_bindings(problem, selector)
        if event_id != self.memory.event_index + 1:
            raise StateIntegrityError("event index must be strictly monotone")
        if self.config.timing_mode == "TS1_single_event" and (
            event_id != 0 or prior_feedback is not None
        ):
            raise StateIntegrityError("TS1 has no prior or future event")

        deadline = (
            float(self._clock()) + self.config.event_time_limit_seconds
        )
        ledger = EvaluationLedger(max_cfe=self.config.cfe_per_event)
        information = problem.freeze_information(event_id, prior_feedback)
        if not isinstance(information, InformationSnapshot):
            raise StateIntegrityError(
                "adapter must return an InformationSnapshot from the shared firewall"
            )
        if information.decision_time != event_id:
            raise StateIntegrityError(
                "information snapshot decision time must equal event_id"
            )
        try:
            information = validate_information_snapshot(information)
        except InformationBoundaryError as error:
            raise StateIntegrityError(
                "information snapshot integrity validation failed"
            ) from error
        machine = StateMachine(run_id=self.run_id, memory=self.memory)
        machine.event_id = event_id
        machine.transition(
            "EVENT_OPEN",
            information_hash=information.information_hash,
            ledger=ledger,
        )

        prior_terminal = (
            self.memory.pending_credit.terminal_code
            if self.memory.pending_credit is not None
            else None
        )
        atoms, resolved_q, credit_status = resolve_pending(
            self.memory,
            variant=self.config.variant,
            feedback=prior_feedback,
        )
        new_keys = append_atoms(self.memory, atoms)
        machine.transition(
            "PRIOR_CREDIT_RESOLVED",
            information_hash=information.information_hash,
            ledger=ledger,
        )
        hard_reason = (
            "prior_numerical_rejection"
            if prior_terminal is TerminalCode.REJECT_NUMERICAL
            else None
        )
        reset_reason = apply_reset_gate(
            self.memory,
            event_id=event_id,
            variant=self.config.variant,
            hard_reason=hard_reason,
        )
        if reset_reason is not None:
            new_keys = set()
        age_prune_bank(self.memory, newly_added_keys=new_keys)
        machine.transition(
            "RESET_GATE_APPLIED",
            information_hash=information.information_hash,
            ledger=ledger,
        )

        mg = MGState.initialize()
        lineage = LineageDAG()
        (
            population,
            warm_count,
            numerical_candidate,
            timed_out,
            initialization_audit,
        ) = self._initialize_population(
            problem,
            event_id=event_id,
            ledger=ledger,
            lineage=lineage,
            deadline=deadline,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        inferior_archive: list[Candidate] = []
        nondominated_archive = maintain_nondominated_archive(
            population,
            capacity=self.config.population_size,
            constraint_scales=problem.constraint_scales,
        )
        archive_audit: list[dict[str, Any]] = []
        if not self._compact_checkpoint_audit:
            for candidate in nondominated_archive:
                lineage.mark_archive_admission(candidate.lineage_node_id)
            archive_audit.append(
                {
                    "generation": -1,
                    "rng": None,
                    "removed_inferior_ids": [],
                    "nondominated_ids": [
                        candidate.candidate_id
                        for candidate in nondominated_archive
                    ],
                }
            )
        machine.transition(
            "SEARCH_INITIALIZED",
            information_hash=information.information_hash,
            ledger=ledger,
        )
        machine.transition(
            "SEARCHING",
            information_hash=information.information_hash,
            ledger=ledger,
        )

        generation = 0
        trial_audit: list[dict[str, Any]] = []
        while (
            numerical_candidate is None
            and not timed_out
            and ledger.cfe < self.config.cfe_per_event
            and population
        ):
            trials: list[Candidate] = []
            metadata: dict[str, dict[str, Any]] = {}
            ranked_population: list[Candidate] | None = None
            generation_batched = False
            selection_cache = _build_generation_selection_cache(
                population,
                inferior_archive,
            )
            if (
                ledger.cfe + len(population)
                <= self.config.cfe_per_event
                and self._batch_allowed(deadline)
                and callable(getattr(problem, "evaluate_batch", None))
            ):
                assign_rank_and_crowding(population)
                ranked_population = sorted(
                    population,
                    key=lambda candidate: constrained_sort_key(
                        candidate,
                        problem.constraint_scales,
                    ),
                )
                try:
                    (
                        trials,
                        metadata,
                        timed_out,
                    ) = self._try_batched_generation(
                        problem,
                        event_id=event_id,
                        generation=generation,
                        population=population,
                        ranked_population=ranked_population,
                        inferior_archive=inferior_archive,
                        selection_cache=selection_cache,
                        mg=mg,
                        ledger=ledger,
                        lineage=lineage,
                        trial_audit=trial_audit,
                        deadline=deadline,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                    )
                except BatchEvaluationUnavailableBeforeEntry:
                    pass
                else:
                    generation_batched = True
            target_population = () if generation_batched else population
            for target_index, target in enumerate(target_population):
                if self._resource_exhausted(deadline):
                    timed_out = True
                    break
                if ledger.cfe >= self.config.cfe_per_event:
                    break
                if ranked_population is None:
                    assign_rank_and_crowding(population)
                    ranked_population = sorted(
                        population,
                        key=lambda candidate: constrained_sort_key(
                            candidate,
                            problem.constraint_scales,
                        ),
                    )
                f_value, cr_value, source, parameter_audit = self._parameter_source(
                    problem=problem,
                    event_id=event_id,
                    generation=generation,
                    target_index=target_index,
                    mg=mg,
                )
                raw, operator = self._trial_vector(
                    population,
                    inferior_archive,
                    ranked_population,
                    problem=problem,
                    event_id=event_id,
                    generation=generation,
                    target_index=target_index,
                    f_value=f_value,
                    cr_value=cr_value,
                    selection_cache=selection_cache,
                )
                repaired, changed = repair_midpoint(
                    raw,
                    target.vector,
                    lower_bounds,
                    upper_bounds,
                )
                node_id = (
                    f"e{event_id}:g{generation:06d}:t{target_index:06d}"
                )
                pre_repair_hash = ""
                post_repair_hash: str | None = ""
                base_audit: dict[str, Any] | None = None
                if not self._compact_checkpoint_audit:
                    assert operator is not None
                    assert parameter_audit is not None
                    raw_vector, pre_repair_hash = _vector_audit_material(raw)
                    if repaired is None:
                        repaired_vector = None
                        post_repair_hash = None
                    else:
                        repaired_vector, post_repair_hash = (
                            _vector_audit_material(repaired)
                        )
                    operator_audit = {
                        "pbest_id": operator["pbest"].candidate_id,
                        "r1_id": operator["r1"].candidate_id,
                        "r2_id": operator["r2"].candidate_id,
                        "j_rand": operator["j_rand"],
                        "rng": operator["rng"],
                    }
                    changed_coordinates = (
                        []
                        if repaired is None
                        else [
                            int(index)
                            for index in np.flatnonzero(raw != repaired)
                        ]
                    )
                    base_audit = {
                        "node_id": node_id,
                        "f": f_value,
                        "cr": cr_value,
                        "source": source,
                        "parameter_audit": parameter_audit,
                        "operator_audit": operator_audit,
                        "raw_vector": raw_vector,
                        "repaired_vector": repaired_vector,
                        "pre_repair_hash": pre_repair_hash,
                        "post_repair_hash": post_repair_hash,
                        "repaired": changed,
                        "repair_changed_coordinates": changed_coordinates,
                        "repair_reason": (
                            "midpoint_bound_repair" if changed else None
                        ),
                        "selection": "pending",
                        "archive_admission": False,
                        "inferior_parent_archive_admission": False,
                        "paired_target_id": target.candidate_id,
                        "mg_mode": COMPONENTS[self.config.variant].mg_mode,
                        "mg_success": False,
                        "mg_success_reason": "PENDING",
                        "mg_success_delta": 0.0,
                        "mg_success_weight": 0.0,
                    }
                if repaired is None:
                    ledger.record_repair_failure(
                        candidate_id=node_id,
                        event_id=event_id,
                        reason="midpoint repair failed",
                    )
                    if base_audit is not None:
                        trial_audit.append(
                            {
                                **base_audit,
                                "evaluated": False,
                                "repair_failed": True,
                                "repair_reason": "midpoint repair failed",
                                "selection": "not_evaluated",
                                "mg_success_reason": "REPAIR_FAILED",
                            }
                        )
                    continue
                try:
                    trial = self._evaluate(
                        problem,
                        vector=repaired,
                        event_id=event_id,
                        ledger=ledger,
                        candidate_id=node_id,
                    )
                except NonrecoverableEvaluationError:
                    if self._compact_checkpoint_audit:
                        lineage_node = LineageNode(
                            node_id=node_id,
                            event_id=event_id,
                            generation=generation,
                            target_predecessor=target.lineage_node_id,
                            f=f_value,
                            cr=cr_value,
                            survival=False,
                            evaluation_status="numerical_failure",
                        )
                    else:
                        assert operator is not None
                        lineage_node = LineageNode(
                            node_id=node_id,
                            event_id=event_id,
                            generation=generation,
                            target_predecessor=target.lineage_node_id,
                            f=f_value,
                            cr=cr_value,
                            survival=False,
                            target_id=target.candidate_id,
                            pbest_id=operator["pbest"].candidate_id,
                            r1_id=operator["r1"].candidate_id,
                            r2_id=operator["r2"].candidate_id,
                            parameter_source=source,
                            j_rand=operator["j_rand"],
                            pre_repair_hash=pre_repair_hash,
                            post_repair_hash=post_repair_hash or "",
                            repaired=changed,
                            evaluation_status="numerical_failure",
                        )
                    lineage.add(lineage_node)
                    if base_audit is not None:
                        trial_audit.append(
                            {
                                **base_audit,
                                "evaluated": True,
                                "repair_failed": False,
                                "evaluation_status": "numerical_failure",
                                "selection": "not_applied_numerical_stop",
                                "mg_success_reason": "NUMERICAL_FAILURE",
                            }
                        )
                    numerical_candidate = node_id
                    break
                trial.lineage_node_id = node_id
                if self._compact_checkpoint_audit:
                    lineage_node = LineageNode(
                        node_id=node_id,
                        event_id=event_id,
                        generation=generation,
                        target_predecessor=target.lineage_node_id,
                        f=f_value,
                        cr=cr_value,
                        survival=False,
                    )
                else:
                    assert operator is not None
                    lineage_node = LineageNode(
                        node_id=node_id,
                        event_id=event_id,
                        generation=generation,
                        target_predecessor=target.lineage_node_id,
                        f=f_value,
                        cr=cr_value,
                        survival=False,
                        target_id=target.candidate_id,
                        pbest_id=operator["pbest"].candidate_id,
                        r1_id=operator["r1"].candidate_id,
                        r2_id=operator["r2"].candidate_id,
                        parameter_source=source,
                        j_rand=operator["j_rand"],
                        pre_repair_hash=pre_repair_hash,
                        post_repair_hash=post_repair_hash or "",
                        repaired=changed,
                        objectives=trial.evaluation.objectives,
                        constraints=trial.evaluation.constraints,
                        feasible=trial.feasible,
                        normalized_cv=trial.normalized_violation(
                            problem.constraint_scales
                        ),
                    )
                lineage.add(lineage_node)
                trials.append(trial)
                audit_index = (
                    None
                    if self._compact_checkpoint_audit
                    else len(trial_audit)
                )
                metadata[node_id] = {
                    "target": target,
                    "f": f_value,
                    "cr": cr_value,
                    "repaired": changed,
                    "audit_index": audit_index,
                }
                if base_audit is not None:
                    trial_audit.append(
                        {
                            **base_audit,
                            "evaluated": True,
                            "repair_failed": False,
                            "evaluation_status": "completed",
                        }
                    )
                if self._resource_exhausted(deadline):
                    if audit_index is not None:
                        trial_audit[audit_index][
                            "selection"
                        ] = "not_applied_timeout"
                        trial_audit[audit_index][
                            "mg_success_reason"
                        ] = "NOT_APPLIED_TIMEOUT"
                    timed_out = True
                    break
            if numerical_candidate is not None or timed_out or not trials:
                break

            next_population = environmental_select(
                population + trials,
                population_size=self.config.population_size,
                constraint_scales=problem.constraint_scales,
            )
            survivor_ids = {
                candidate.candidate_id for candidate in next_population
            }
            successes: list[tuple[float, float, float]] = []
            success_audit_entries: list[tuple[int, float]] = []
            for trial in trials:
                detail = metadata[trial.lineage_node_id]
                audit_index = detail["audit_index"]
                audit = (
                    None
                    if audit_index is None
                    else trial_audit[audit_index]
                )
                survived = trial.candidate_id in survivor_ids
                if audit is not None:
                    audit["selection"] = (
                        "survived" if survived else "discarded"
                    )
                if not survived:
                    if audit is not None:
                        audit["mg_success_reason"] = (
                            "TRIAL_NOT_IN_NEXT_POPULATION"
                        )
                    continue
                lineage.mark_survival(trial.lineage_node_id, True)
                target = detail["target"]
                target_survived = target.candidate_id in survivor_ids
                archive_target = False
                if self.config.variant == "SHADE_ONLY":
                    shade_success = shade_success_improvement(
                        target,
                        trial,
                        problem.constraint_scales,
                        trial_in_next_population=survived,
                        target_in_next_population=target_survived,
                    )
                    if audit is not None:
                        audit["mg_success_reason"] = shade_success.reason
                    weight = shade_success.delta
                    archive_target = shade_success.success
                else:
                    archive_target = not target_survived
                    weight = (
                        1.0
                        if trial.feasible
                        else (
                            0.25
                            if trial.normalized_violation(
                                problem.constraint_scales
                            )
                            < target.normalized_violation(
                                problem.constraint_scales
                            )
                            else 0.0
                        )
                    )
                    if detail["repaired"]:
                        weight *= 0.5
                    if audit is not None:
                        audit["mg_success_reason"] = (
                            "F22_WEIGHTED_SURVIVOR"
                            if weight > 0.0
                            else "NO_F22_WEIGHTED_SURVIVOR_CREDIT"
                        )
                if archive_target and all(
                    candidate.candidate_id != target.candidate_id
                    for candidate in inferior_archive
                ):
                    inferior_archive.append(target)
                    if audit is not None:
                        audit["inferior_parent_archive_admission"] = True
                if weight > 0.0:
                    successes.append((detail["f"], detail["cr"], weight))
                    if audit is not None:
                        audit["mg_success"] = True
                        audit["mg_success_delta"] = weight
                        success_audit_entries.append((audit_index, weight))

            total_success_delta = math.fsum(
                weight for _, weight in success_audit_entries
            )
            for audit_index, weight in success_audit_entries:
                trial_audit[audit_index]["mg_success_weight"] = (
                    weight / total_success_delta
                )

            archive_rng, archive_manifest = derive_rng(
                self.config.algorithm_seed,
                stream=RandomStream.ALGORITHM,
                experiment_id=self.config.configuration_evidence_id,
                unit_id=problem.adapter_id,
                method_id=self.method_id,
                event_id=event_id,
                generation=generation,
                target_index=self.config.population_size,
                substream="archive",
                **(
                    _COMPACT_RNG_DERIVATION_OPTIONS
                    if self._compact_checkpoint_audit
                    else _NO_RNG_DERIVATION_OPTIONS
                ),
            )
            removed_inferior_ids: list[str] = []
            while len(inferior_archive) > self.config.population_size:
                removed = inferior_archive.pop(
                    int(archive_rng.integers(0, len(inferior_archive)))
                )
                if not self._compact_checkpoint_audit:
                    removed_inferior_ids.append(removed.candidate_id)
            mg.pointer = update_mg(
                mg.memory_f,
                mg.memory_cr,
                mg.pointer,
                successes,
            )
            population = next_population
            unique_archive_candidates = {
                candidate.candidate_id: candidate
                for candidate in nondominated_archive + population
            }
            nondominated_archive = maintain_nondominated_archive(
                list(unique_archive_candidates.values()),
                capacity=self.config.population_size,
                constraint_scales=problem.constraint_scales,
            )
            if not self._compact_checkpoint_audit:
                nondominated_ids = {
                    candidate.candidate_id
                    for candidate in nondominated_archive
                }
                for candidate in nondominated_archive:
                    lineage.mark_archive_admission(
                        candidate.lineage_node_id
                    )
                for trial in trials:
                    admitted = trial.candidate_id in nondominated_ids
                    audit_index = metadata[
                        trial.lineage_node_id
                    ]["audit_index"]
                    assert audit_index is not None
                    trial_audit[audit_index][
                        "archive_admission"
                    ] = admitted
                archive_audit.append(
                    {
                        "generation": generation,
                        "rng": archive_manifest,
                        "removed_inferior_ids": removed_inferior_ids,
                        "nondominated_ids": sorted(nondominated_ids),
                    }
                )
            generation += 1

        if (
            numerical_candidate is None
            and not timed_out
            and self._resource_exhausted(deadline)
        ):
            timed_out = True
        ledger.assert_joint_contract(
            atomic_steps_per_evaluation=self.config.atomic_steps_per_evaluation
        )
        archive_by_id = {
            candidate.candidate_id: candidate
            for candidate in nondominated_archive
        }
        selected: Candidate | None = None
        if numerical_candidate is not None:
            terminal = TerminalOutcome(
                code=TerminalCode.REJECT_NUMERICAL,
                candidate_id=numerical_candidate,
                reason="joint evaluator returned a numerical failure",
            )
        elif timed_out:
            witness: Candidate | None
            if archive_by_id:
                selected_id = selector.select(
                    tuple(
                        archive_by_id[candidate_id].evaluation
                        for candidate_id in sorted(archive_by_id)
                    )
                )
                if selected_id not in archive_by_id:
                    raise StateIntegrityError(
                        "terminal selector returned an ID outside the archive"
                    )
                witness = archive_by_id[selected_id]
            else:
                witness = (
                    min(
                        population,
                        key=lambda candidate: (
                            candidate.normalized_violation(
                                problem.constraint_scales
                            ),
                            candidate.candidate_id,
                        ),
                    )
                    if population
                    else None
                )
            terminal = TerminalOutcome(
                code=TerminalCode.REJECT_TIMEOUT,
                candidate_id=(
                    witness.candidate_id if witness is not None else None
                ),
                reason="shared event resource limit reached",
            )
        elif archive_by_id:
            selected_id = selector.select(
                tuple(
                    archive_by_id[candidate_id].evaluation
                    for candidate_id in sorted(archive_by_id)
                )
            )
            if selected_id not in archive_by_id:
                raise StateIntegrityError(
                    "terminal selector returned an ID outside the archive"
                )
            selected = archive_by_id[selected_id]
            if problem.safety_filter(selected.evaluation, event_id):
                terminal = TerminalOutcome(
                    code=TerminalCode.ACCEPTED,
                    candidate_id=selected.candidate_id,
                )
            else:
                terminal = TerminalOutcome(
                    code=TerminalCode.REJECT_SAFETY_FILTER,
                    candidate_id=selected.candidate_id,
                    reason="candidate failed the bound safety-related filter",
                )
        else:
            witness = (
                min(
                    population,
                    key=lambda candidate: (
                        candidate.normalized_violation(
                            problem.constraint_scales
                        ),
                        candidate.candidate_id,
                    ),
                )
                if population
                else None
            )
            code = (
                TerminalCode.REJECT_BUDGET_NO_FEASIBLE
                if ledger.cfe == self.config.cfe_per_event
                else TerminalCode.REJECT_NO_FEASIBLE
            )
            terminal = TerminalOutcome(
                code=code,
                candidate_id=(
                    witness.candidate_id if witness is not None else None
                ),
                reason="no feasible candidate remained in the archive",
            )

        accepted = terminal.code is TerminalCode.ACCEPTED
        if accepted:
            machine.transition(
                "TERMINAL_SELECTED",
                information_hash=information.information_hash,
                ledger=ledger,
            )
            machine.transition(
                "ACTION_COMMITTED",
                information_hash=information.information_hash,
                ledger=ledger,
            )
        else:
            machine.transition(
                "TERMINAL_REJECTED",
                information_hash=information.information_hash,
                ledger=ledger,
            )
            machine.transition(
                "NO_ACTION_COMMITTED",
                information_hash=information.information_hash,
                ledger=ledger,
            )

        terminal_candidate = (
            archive_by_id.get(terminal.candidate_id)
            or next(
                (
                    candidate
                    for candidate in population
                    if candidate.candidate_id == terminal.candidate_id
                ),
                None,
            )
        )
        terminal_lineage_id = (
            terminal.candidate_id
            if terminal.candidate_id in lineage.nodes
            else (
                terminal_candidate.lineage_node_id
                if terminal_candidate is not None
                else None
            )
        )
        lineage_weights = (
            lineage.credit_chain(
                terminal_lineage_id,
                mode=COMPONENTS[self.config.variant].lineage_mode,
            )
            if terminal_lineage_id is not None
            else ()
        )
        parameter_values = (
            lineage.parameter_values(lineage_weights)
            if lineage_weights
            else {}
        )
        pending = (
            None
            if self.config.timing_mode == "TS1_single_event"
            else PendingCredit(
                pending_id=f"{self.run_id}:event:{event_id}",
                source_event=event_id,
                terminal_code=terminal.code,
                lineage_weights=lineage_weights,
                parameter_values=parameter_values,
                information_hash=information.information_hash,
                adapter_version=problem.adapter_version,
            )
        )
        self.memory.pending_credit = pending
        machine.transition(
            "PENDING_CREDIT_WRITTEN",
            information_hash=information.information_hash,
            ledger=ledger,
        )
        self.memory.solution_memory = (
            ()
            if self.config.timing_mode == "TS1_single_event"
            else tuple(
                tuple(float(value) for value in candidate.vector)
                for candidate in nondominated_archive
            )
        )
        self.memory.event_index = event_id
        close_event_cooldown(self.memory)
        machine.transition(
            "EVENT_CLOSED",
            information_hash=information.information_hash,
            ledger=ledger,
        )

        action = np.asarray(
            (
                problem.first_action(selected.vector)
                if accepted and selected is not None
                else problem.fallback_action(event_id)
            ),
            dtype=float,
        )
        if (
            action.ndim != 1
            or action.size < 1
            or not np.all(np.isfinite(action))
        ):
            raise StateIntegrityError(
                "adapter returned an invalid committed/fallback action"
            )
        execution_before = ledger.execution_transition_count
        if self.config.variant == "NO_EXECUTION_FEEDBACK":
            problem.execute(action, event_id, accepted, ledger)
            feedback = None
        else:
            feedback = dict(
                problem.execute(action, event_id, accepted, ledger)
            )
        execution_after = ledger.execution_transition_count
        if (
            self.config.timing_mode == "TS2_fixed_periodic_replanning"
            and execution_after != execution_before + 1
        ):
            raise StateIntegrityError(
                "TS2 adapter must record exactly one execution transition"
            )
        ledger.assert_joint_contract(
            atomic_steps_per_evaluation=self.config.atomic_steps_per_evaluation
        )
        self.memory.validate()
        self.memory.seal_checkpoint()
        return EventRunResult(
            event_id=event_id,
            terminal=terminal,
            archive=tuple(
                candidate.evaluation for candidate in nondominated_archive
            ),
            information_hash=information.information_hash,
            ledger=ledger.snapshot(),
            state_transitions=tuple(machine.logs),
            credit_resolution_status=credit_status,
            resolved_q=resolved_q,
            reset_reason=reset_reason,
            warm_start_seed_count=warm_count,
            mg_final={
                "memory_f": list(mg.memory_f),
                "memory_cr": list(mg.memory_cr),
                "pointer": mg.pointer,
            },
            trial_audit=tuple(trial_audit),
            initialization_audit=initialization_audit,
            lineage_records=(
                () if self._compact_checkpoint_audit else lineage.records()
            ),
            archive_audit=tuple(archive_audit),
            memory_snapshot=self.memory.to_dict(),
            execution_feedback=feedback,
        )

    def run_sequence(
        self,
        problem: EventProblemAdapter,
        *,
        selector: TerminalSelector,
        prior_feedback: Mapping[str, Any] | None = None,
    ) -> SequenceRunResult:
        self._validate_bindings(problem, selector)
        if self.memory.event_index < 0 and prior_feedback is not None:
            raise StateIntegrityError(
                "fresh sequence cannot receive prior feedback"
            )
        events: list[EventRunResult] = []
        start_event = self.memory.event_index + 1
        for event_id in range(start_event, self.config.max_events):
            event = self.run_event(
                problem,
                selector=selector,
                event_id=event_id,
                prior_feedback=prior_feedback,
            )
            prior_feedback = event.execution_feedback
            events.append(event)
        return SequenceRunResult(
            config=self.config.to_dict(),
            adapter_identity=dict(problem.identity()),
            selector_identity=dict(selector.identity()),
            events=tuple(events),
            persistent_state=self.memory.to_dict(),
            confirmatory_execution=isinstance(
                self.config.execution_request,
                (R8ExecutionRequest, R8CCorrectiveExecutionRequest),
            ),
        )
