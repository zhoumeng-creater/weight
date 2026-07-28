"""Result-blind task runtime for the frozen R8 public execution."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gzip
from hashlib import sha256
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from comparators.domain_baselines import (
    ConventionalRollingPlannerBaseline,
    FixedEnergyDeficitBaseline,
)
from comparators.jmetal_bridge import JMetalComparator
from comparators.matched_de import MatchedParetoDE
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
    ConfigurationError,
    ExecutionScope,
    FULL_AUDIT_MATERIALIZATION,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
    R8ContractBindings,
    R8ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from dt_ramde_v11.interfaces import OptimizationResult
from e3_inputs.contract import generate_subject_parameters
from evaluation.contracts import (
    EvaluationResult,
    TerminalCode,
    TerminalOutcome,
)
from evaluation.evaluator import (
    BatchEvaluationUnavailableBeforeEntry,
    ExecutionTimeoutBeforeEntry,
)
from evaluation.ledger import EvaluationLedger
from evaluation.randomness import RandomStream, derive_rng
from weight_application.formal_e3_adapter import FormalHallE3Adapter

from .adapters import (
    FormalR8CHallE3Adapter,
    FormalR8WGTRRAdapter,
    make_corrective_cdf_adapter,
    make_corrective_lircmop_adapter,
    make_corrective_wgt_rr_adapter,
    make_formal_cdf_adapter,
    make_formal_lircmop_adapter,
    make_formal_wgt_rr_adapter,
)
from .checkpoint_data import (
    EVENT_SUMMARY_MAX_RECORD_BYTES,
    FORMAT_ID as CHECKPOINT_FORMAT_ID,
    WORKER_CONTROL_REPORT_MAX_BYTES,
    CheckpointMetadata,
    TaskCheckpointWriter,
)
from .schedule import FormalSequenceSpec, canonical_json_bytes


RAW_JSONL_PERSISTENCE = "RAW_EVALUATIONS_JSONL_GZIP_V1"
CHECKPOINT_FRONT_PERSISTENCE = "ENDPOINT_SUFFICIENT_CHECKPOINT_FRONT_V1"
EVENT_SUMMARIES_FILENAME = "event_summaries.jsonl"


def _write_canonical_json_exclusive_fsynced(
    path: Path,
    value: Any,
    *,
    maximum_bytes: int | None = None,
) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if maximum_bytes is not None and len(payload) > maximum_bytes:
        raise FormalRuntimeError(
            "canonical control report exceeds its frozen byte bound"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if os.name == "nt":
        flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class FormalRuntimeSettings:
    population_size: int
    archive_capacity: int
    configuration_evidence_id: str
    corrective: bool = False
    artifact_stage: str = "R8"
    persistence_mode: str = RAW_JSONL_PERSISTENCE

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.archive_capacity < 1:
            raise ValueError("formal runtime population/archive are invalid")
        if not self.configuration_evidence_id:
            raise ValueError(
                "formal runtime configuration evidence ID is required"
            )
        if not self.artifact_stage:
            raise ValueError("formal runtime artifact stage is required")
        if self.persistence_mode not in {
            RAW_JSONL_PERSISTENCE,
            CHECKPOINT_FRONT_PERSISTENCE,
        }:
            raise ValueError("formal runtime persistence mode is invalid")


LEGACY_R8_RUNTIME_SETTINGS = FormalRuntimeSettings(
    population_size=20,
    archive_capacity=20,
    configuration_evidence_id="WGT_V11_R8_PUBLIC_E1_E2_E3_FORMAL",
)
CORRECTIVE_R8C_RUNTIME_SETTINGS = FormalRuntimeSettings(
    population_size=100,
    archive_capacity=100,
    configuration_evidence_id=(
        "WGT_V11_R8C_PUBLIC_E1_E2_THREE_SCENARIO_E3_FORMAL"
    ),
    corrective=True,
    artifact_stage="R8C",
)
CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS = FormalRuntimeSettings(
    population_size=100,
    archive_capacity=100,
    configuration_evidence_id="WGT_V11_R8C_PUBLIC_E1_E2_FORMAL",
    corrective=True,
    artifact_stage="R8C_E1E2",
    persistence_mode=CHECKPOINT_FRONT_PERSISTENCE,
)
FormalExecutionRequest = (
    R8ExecutionRequest | R8CCorrectiveExecutionRequest
)
_FORMAL_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "scope",
        "companion_scope",
        "contracts",
        "frozen_exact_command",
        "author_confirmation_text",
        "author_exact_command_confirmed",
        "formal_effect_execution_requested",
        "participant_data_requested",
        "hidden_generation_requested",
        "results_analysis_requested",
        "results_writing_requested",
        "remote_git_mutation_requested",
        "release_or_distribution_requested",
    }
)


class FormalRuntimeError(RuntimeError):
    """A formal task cannot satisfy its frozen runtime contract."""


RAW_GZIP_COMPRESSLEVEL = 1
SCIENTIFIC_EVENT_DEADLINES_SECONDS = {
    "E1_STATIC": 1800.0,
    "E1_DYNAMIC": 300.0,
    "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE": 300.0,
    "E1_ROLLING": 120.0,
    "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE": 120.0,
}
EVENT_BATCH_DEADLINE_GUARD_SECONDS = 5.0


class ScientificEventTimeoutBeforeEntry(RuntimeError):
    """The common scientific event deadline elapsed before the next CFE."""


def _validated_request_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if set(payload) != _FORMAL_REQUEST_KEYS:
        missing = sorted(_FORMAL_REQUEST_KEYS - set(payload))
        unexpected = sorted(set(payload) - _FORMAL_REQUEST_KEYS)
        raise ConfigurationError(
            "formal request fields differ from the frozen schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not isinstance(payload["contracts"], Mapping):
        raise ConfigurationError("formal request contracts must be an object")
    for key in (
        "request_id",
        "scope",
        "companion_scope",
        "frozen_exact_command",
        "author_confirmation_text",
    ):
        if type(payload[key]) is not str or not payload[key]:
            raise ConfigurationError(
                f"formal request {key} must be a non-empty string"
            )
    for key in _FORMAL_REQUEST_KEYS - {
        "request_id",
        "scope",
        "companion_scope",
        "contracts",
        "frozen_exact_command",
        "author_confirmation_text",
    }:
        if type(payload[key]) is not bool:
            raise ConfigurationError(
                f"formal request {key} must be a JSON boolean"
            )
    return payload


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_r8_request(payload: Mapping[str, Any]) -> R8ExecutionRequest:
    payload = _validated_request_payload(payload)
    contracts = R8ContractBindings(**dict(payload["contracts"]))
    request = R8ExecutionRequest(
        scope=ExecutionScope(payload["scope"]),
        companion_scope=ExecutionScope(payload["companion_scope"]),
        contracts=contracts,
        request_id=payload["request_id"],
        frozen_exact_command=payload["frozen_exact_command"],
        author_confirmation_text=payload["author_confirmation_text"],
        author_exact_command_confirmed=payload[
            "author_exact_command_confirmed"
        ],
        formal_effect_execution_requested=payload[
            "formal_effect_execution_requested"
        ],
        participant_data_requested=payload["participant_data_requested"],
        hidden_generation_requested=payload["hidden_generation_requested"],
        results_analysis_requested=payload["results_analysis_requested"],
        results_writing_requested=payload["results_writing_requested"],
        remote_git_mutation_requested=payload[
            "remote_git_mutation_requested"
        ],
        release_or_distribution_requested=payload[
            "release_or_distribution_requested"
        ],
    )
    request.validate()
    return request


def parse_r8c_request(
    payload: Mapping[str, Any],
) -> R8CCorrectiveExecutionRequest:
    payload = _validated_request_payload(payload)
    contracts = R8CCorrectiveContractBindings(
        **dict(payload["contracts"])
    )
    request = R8CCorrectiveExecutionRequest(
        scope=ExecutionScope(payload["scope"]),
        companion_scope=ExecutionScope(payload["companion_scope"]),
        contracts=contracts,
        request_id=payload["request_id"],
        frozen_exact_command=payload["frozen_exact_command"],
        author_confirmation_text=payload["author_confirmation_text"],
        author_exact_command_confirmed=payload[
            "author_exact_command_confirmed"
        ],
        formal_effect_execution_requested=payload[
            "formal_effect_execution_requested"
        ],
        participant_data_requested=payload["participant_data_requested"],
        hidden_generation_requested=payload["hidden_generation_requested"],
        results_analysis_requested=payload["results_analysis_requested"],
        results_writing_requested=payload["results_writing_requested"],
        remote_git_mutation_requested=payload[
            "remote_git_mutation_requested"
        ],
        release_or_distribution_requested=payload[
            "release_or_distribution_requested"
        ],
    )
    request.validate()
    return request


class RawEvaluationWriter:
    """Stream every evaluator return to one immutable task-local gzip chunk."""

    def __init__(
        self,
        path: Path,
        task_id: str,
        *,
        buffer_size: int = 1 << 20,
        compresslevel: int = RAW_GZIP_COMPRESSLEVEL,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("raw evaluation buffer_size must be positive")
        if not 0 <= compresslevel <= 9:
            raise ValueError("raw gzip compresslevel must be in 0..9")
        self.path = path
        self.task_id = task_id
        self.count = 0
        self._buffer_size = int(buffer_size)
        self.compresslevel = int(compresslevel)
        self._buffer = bytearray()
        self._closed = False
        self._raw = path.open("xb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self._raw,
            compresslevel=self.compresslevel,
            mtime=0,
        )

    def _write_record(self, record: Mapping[str, Any]) -> None:
        self._buffer.extend(canonical_json_bytes(record))
        self._buffer.append(0x0A)
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()
        self.count += 1

    def _flush_buffer(self) -> None:
        if self._buffer:
            self._gzip.write(self._buffer)
            self._buffer.clear()

    def write(
        self,
        *,
        event_id: int,
        vector: Sequence[float],
        result: EvaluationResult,
    ) -> None:
        record = {
            "task_id": self.task_id,
            "event_id": event_id,
            "candidate_id": result.candidate_id,
            "vector": [float(value) for value in vector],
            "objectives": list(result.objectives),
            "objective_names": list(result.objective_names),
            "constraints": list(result.constraints),
            "constraint_names": list(result.constraint_names),
            "feasible": result.feasible,
            "total_violation": result.total_violation,
        }
        self._write_record(record)

    def write_failure(
        self,
        *,
        event_id: int,
        candidate_id: str,
        vector: Sequence[float],
        error: Exception,
        failure_type: str | None = None,
    ) -> None:
        record = {
            "task_id": self.task_id,
            "event_id": event_id,
            "candidate_id": candidate_id,
            "vector": [float(value) for value in vector],
            "evaluation_failure": {
                "type": failure_type or type(error).__name__,
                "reason": str(error),
            },
        }
        self._write_record(record)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._flush_buffer()
            self._gzip.close()
        finally:
            self._raw.close()
            self._closed = True

    def __enter__(self) -> RawEvaluationWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class EndpointCheckpointWriter:
    """Adapt the endpoint-sufficient checkpoint store to the runtime recorder."""

    def __init__(
        self,
        path: Path,
        task_id: str,
        objective_names: Sequence[str],
    ) -> None:
        self.path = path
        self.task_id = task_id
        self.count = 0
        self._writer = TaskCheckpointWriter(
            path,
            CheckpointMetadata(
                task_id=task_id,
                objective_names=tuple(str(name) for name in objective_names),
            ),
        )

    def begin_event(self, *, event_id: int, cfe_budget: int) -> None:
        self._writer.begin_event(
            event_id=event_id,
            cfe_budget=cfe_budget,
        )

    def write(
        self,
        *,
        event_id: int,
        vector: Sequence[float],
        result: EvaluationResult,
    ) -> None:
        self._writer.record_success(
            event_id=event_id,
            vector=vector,
            result=result,
        )
        self.count += 1

    def write_failure(
        self,
        *,
        event_id: int,
        candidate_id: str,
        vector: Sequence[float],
        error: Exception,
        failure_type: str | None = None,
    ) -> None:
        self._writer.record_failure(
            event_id=event_id,
            candidate_id=candidate_id,
            vector=vector,
            error_type=failure_type or type(error).__name__,
            reason=str(error),
        )
        self.count += 1

    def finish_event(self, *, terminal_snapshot: bool = False) -> None:
        self._writer.finish_event(terminal_snapshot=terminal_snapshot)

    def current_front_evaluations(self) -> tuple[EvaluationResult, ...]:
        return self._writer.current_front_evaluations()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> EndpointCheckpointWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


EvaluationWriter = RawEvaluationWriter | EndpointCheckpointWriter


_COMPACT_EVENT_SUMMARY_KEYS = frozenset(
    {
        "event_id",
        "terminal",
        "ledger",
        "evaluation_failure_type_counts",
        "information_hash",
        "execution_feedback",
        "execution_observation",
    }
)
_COMPACT_EVENT_LEDGER_KEYS = frozenset(
    {
        "cfe",
        "objective_calls",
        "constraint_calls",
        "scenario_evaluations",
        "atomic_model_steps",
        "execution_transition_count",
        "repair_failed",
        "evaluation_failures",
    }
)
_COMPACT_EXECUTION_KEYS = frozenset(
    {
        "available",
        "ell_exec",
        "ell_ref",
        "s_exec",
        "hard_constraint_violation",
        "released_at",
    }
)
_COMPACT_PUBLIC_MISSING_EXECUTION_KEYS = (
    _COMPACT_EXECUTION_KEYS | {"reason"}
)
_COMPACT_TERMINAL_CODES = frozenset(code.value for code in TerminalCode)


def _validated_compact_execution_record(
    value: Any,
    *,
    event_id: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FormalRuntimeError(
            "compact execution channel must be an object or null"
        )
    fields = frozenset(value)
    if fields not in {
        _COMPACT_EXECUTION_KEYS,
        _COMPACT_PUBLIC_MISSING_EXECUTION_KEYS,
    }:
        raise FormalRuntimeError(
            "compact execution channel fields differ from the frozen schema"
        )
    if (
        type(value["available"]) is not bool
        or type(value["released_at"]) is not int
        or value["released_at"] != event_id + 1
    ):
        raise FormalRuntimeError(
            "compact execution channel timing is invalid"
        )
    if value["available"]:
        if fields != _COMPACT_EXECUTION_KEYS:
            raise FormalRuntimeError(
                "available compact execution channel has extra fields"
            )
        for field in ("ell_exec", "ell_ref", "s_exec"):
            item = value[field]
            if (
                isinstance(item, bool)
                or not isinstance(item, int | float)
                or not np.isfinite(float(item))
            ):
                raise FormalRuntimeError(
                    "compact execution channel values are invalid"
                )
        if (
            float(value["s_exec"]) <= 0.0
            or type(value["hard_constraint_violation"]) is not bool
        ):
            raise FormalRuntimeError(
                "compact execution channel values are invalid"
            )
    else:
        if any(
            value[field] is not None
            for field in (
                "ell_exec",
                "ell_ref",
                "s_exec",
                "hard_constraint_violation",
            )
        ):
            raise FormalRuntimeError(
                "unavailable compact execution channel is invalid"
            )
        if "reason" in value and value["reason"] != (
            "MISSING_BY_DESIGN_PUBLIC_BENCHMARK"
        ):
            raise FormalRuntimeError(
                "compact execution channel reason is invalid"
            )
    return dict(value)


class DurableEventSummaryWriter:
    """Durably append completed compact event summaries in canonical JSONL."""

    def __init__(
        self,
        path: Path,
        *,
        event_count: int,
        cfe_per_event: int,
        atomic_steps_per_cfe: int,
    ) -> None:
        self.path = path
        if (
            event_count < 1
            or cfe_per_event < 1
            or atomic_steps_per_cfe < 1
        ):
            raise FormalRuntimeError(
                "event-summary schedule bounds are invalid"
            )
        self._event_count = int(event_count)
        self._cfe_per_event = int(cfe_per_event)
        self._atomic_steps_per_cfe = int(atomic_steps_per_cfe)
        self._stream = path.open("xb")
        self._next_event_id = 0
        self._closed = False

    def append(self, summary: Mapping[str, Any]) -> None:
        if self._closed:
            raise FormalRuntimeError("event-summary writer is closed")
        event_id = summary.get("event_id")
        if type(event_id) is not int or event_id != self._next_event_id:
            raise FormalRuntimeError(
                "event summaries must be appended in zero-based order"
            )
        terminal = summary.get("terminal")
        ledger = summary.get("ledger")
        failure_counts = summary.get(
            "evaluation_failure_type_counts"
        )
        information_hash = summary.get("information_hash")
        if (
            set(summary) != _COMPACT_EVENT_SUMMARY_KEYS
            or event_id >= self._event_count
            or not isinstance(terminal, Mapping)
            or set(terminal) != {"candidate_available", "code", "reason"}
            or terminal.get("code") not in _COMPACT_TERMINAL_CODES
            or type(terminal.get("candidate_available")) is not bool
            or (
                terminal.get("reason") is not None
                and not isinstance(terminal.get("reason"), str)
            )
            or not isinstance(ledger, Mapping)
            or set(ledger) != _COMPACT_EVENT_LEDGER_KEYS
            or any(
                type(item) is not int or item < 0
                for item in ledger.values()
            )
        ):
            raise FormalRuntimeError(
                "compact event summary schema or result-blind boundary differs"
            )
        cfe = ledger["cfe"]
        if (
            cfe > self._cfe_per_event
            or ledger["objective_calls"] != cfe
            or ledger["constraint_calls"] != cfe
            or ledger["scenario_evaluations"] != cfe
            or ledger["atomic_model_steps"]
            != cfe * self._atomic_steps_per_cfe
            or ledger["execution_transition_count"] not in {0, 1}
            or ledger["evaluation_failures"] > cfe
            or not isinstance(failure_counts, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count < 1
                for name, count in failure_counts.items()
            )
            or sum(failure_counts.values())
            != ledger["evaluation_failures"]
            or not isinstance(information_hash, str)
            or len(information_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in information_hash
            )
        ):
            raise FormalRuntimeError(
                "compact event summary accounting or hashes are invalid"
            )
        feedback = _validated_compact_execution_record(
            summary.get("execution_feedback"),
            event_id=event_id,
        )
        observation = _validated_compact_execution_record(
            summary.get("execution_observation"),
            event_id=event_id,
        )
        if feedback is not None and feedback != observation:
            raise FormalRuntimeError(
                "compact feedback differs from the execution observation"
            )
        payload = canonical_json_bytes(dict(summary)) + b"\n"
        if len(payload) > EVENT_SUMMARY_MAX_RECORD_BYTES:
            raise FormalRuntimeError(
                "compact event summary exceeds its frozen byte bound"
            )
        written = self._stream.write(payload)
        if written != len(payload):
            raise FormalRuntimeError("event-summary append was incomplete")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._next_event_id += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        finally:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> DurableEventSummaryWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class RecordingAdapter:
    """Transparent problem proxy that persists the frozen evaluation evidence."""

    def __init__(
        self,
        problem: Any,
        writer: EvaluationWriter,
        task_timeout_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._problem = problem
        self._writer = writer
        self._task_timeout_path = task_timeout_path
        self._clock = clock
        self._event_deadlines: dict[int, float] = {}
        self._failure_type_counts: defaultdict[
            int,
            Counter[str],
        ] = defaultdict(Counter)
        self._execution_observations: dict[
            int,
            dict[str, Any] | None,
        ] = {}

    def _raise_if_task_timed_out(self) -> None:
        if (
            self._task_timeout_path is not None
            and self._task_timeout_path.exists()
        ):
            raise ExecutionTimeoutBeforeEntry(
                "formal task timeout requested before evaluator entry"
            )

    def _raise_if_scientific_event_timed_out(self, event_id: int) -> None:
        deadline = self._event_deadlines.get(event_id)
        if deadline is not None and float(self._clock()) >= deadline:
            raise ScientificEventTimeoutBeforeEntry(
                "common scientific event deadline reached before evaluator entry"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def identity(self) -> Mapping[str, Any]:
        return self._problem.identity()

    def freeze_information(self, event_id: int, feedback: Any) -> Any:
        return self._problem.freeze_information(event_id, feedback)

    def begin_event(
        self,
        *,
        event_id: int,
        cfe_budget: int,
        scientific_deadline_seconds: float | None = None,
    ) -> None:
        if event_id in self._failure_type_counts:
            raise FormalRuntimeError(
                "recording adapter event was started more than once"
            )
        self._failure_type_counts[event_id] = Counter()
        begin = getattr(self._writer, "begin_event", None)
        if callable(begin):
            begin(event_id=event_id, cfe_budget=cfe_budget)
        if scientific_deadline_seconds is not None:
            self.start_scientific_deadline(
                event_id=event_id,
                seconds=scientific_deadline_seconds,
            )

    def start_scientific_deadline(
        self,
        *,
        event_id: int,
        seconds: float,
    ) -> None:
        """Start the event clock after checkpoint and pre-algorithm setup."""

        if event_id not in self._failure_type_counts:
            raise FormalRuntimeError(
                "scientific deadline started before recording event start"
            )
        if event_id in self._event_deadlines:
            raise FormalRuntimeError(
                "scientific deadline started more than once"
            )
        duration = float(seconds)
        if not np.isfinite(duration) or duration <= 0.0:
            raise FormalRuntimeError(
                "scientific event deadline must be finite and positive"
            )
        self._event_deadlines[event_id] = (
            float(self._clock()) + duration
        )

    def finish_event(self, *, terminal_snapshot: bool = False) -> None:
        finish = getattr(self._writer, "finish_event", None)
        if callable(finish):
            finish(terminal_snapshot=terminal_snapshot)

    def scientific_event_deadline_reached(self, event_id: int) -> bool:
        deadline = self._event_deadlines.get(event_id)
        return (
            deadline is not None
            and float(self._clock()) >= deadline
        )

    def current_front_evaluations(self) -> tuple[EvaluationResult, ...]:
        if not isinstance(self._writer, EndpointCheckpointWriter):
            raise FormalRuntimeError(
                "scientific event timeout archive requires checkpoint persistence"
            )
        return self._writer.current_front_evaluations()

    def failure_type_counts(self, event_id: int) -> dict[str, int]:
        """Return sorted non-sensitive charged evaluator failure counts."""

        if event_id not in self._failure_type_counts:
            raise FormalRuntimeError(
                "failure counts requested before recording event start"
            )
        return {
            failure_type: int(count)
            for failure_type, count in sorted(
                self._failure_type_counts[event_id].items()
            )
        }

    def execution_observation(
        self,
        event_id: int,
    ) -> dict[str, Any] | None:
        """Return the executed outcome without exposing it to the algorithm."""

        if event_id not in self._execution_observations:
            raise FormalRuntimeError(
                "execution observation requested before event execution"
            )
        observation = self._execution_observations[event_id]
        return None if observation is None else dict(observation)

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        self._raise_if_task_timed_out()
        self._raise_if_scientific_event_timed_out(event_id)
        cfe_before = ledger.cfe
        failures_before = ledger.evaluation_failure_count
        try:
            result = self._problem.evaluate(
                vector,
                event_id,
                ledger,
                candidate_id,
            )
        except Exception as error:
            charged = ledger.cfe - cfe_before
            failures_after = ledger.evaluation_failure_count
            failure_type: str | None = None
            if charged == 1:
                if failures_after == failures_before + 1:
                    failure_type = ledger.evaluation_failures[-1].failure_type
                elif failures_after == failures_before:
                    # The method wrapper will append this charged failure after
                    # the proxy re-raises it.  Count the same outer type now.
                    failure_type = type(error).__name__
                else:
                    raise FormalRuntimeError(
                        "scalar evaluator appended an invalid number of "
                        "failure ledger rows"
                    ) from error
                self._failure_type_counts[event_id][failure_type] += 1
            elif charged != 0:
                raise FormalRuntimeError(
                    "scalar evaluator changed CFE by more than one"
                ) from error
            if isinstance(self._writer, EndpointCheckpointWriter):
                if charged == 1:
                    self._writer.write_failure(
                        event_id=event_id,
                        candidate_id=candidate_id,
                        vector=vector,
                        error=error,
                        failure_type=failure_type,
                    )
            else:
                self._writer.write_failure(
                    event_id=event_id,
                    candidate_id=candidate_id,
                    vector=vector,
                    error=error,
                    failure_type=failure_type,
                )
            raise
        self._writer.write(
            event_id=event_id,
            vector=vector,
            result=result,
        )
        return result

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_ids: Sequence[str],
    ) -> tuple[EvaluationResult, ...]:
        self._raise_if_task_timed_out()
        deadline = self._event_deadlines.get(event_id)
        if deadline is not None:
            remaining = deadline - float(self._clock())
            if remaining <= 0.0:
                raise ScientificEventTimeoutBeforeEntry(
                    "common scientific event deadline reached before batch entry"
                )
            if remaining < EVENT_BATCH_DEADLINE_GUARD_SECONDS:
                raise BatchEvaluationUnavailableBeforeEntry(
                    "scientific deadline guard requires scalar evaluation"
                )
        vector_values = tuple(vectors)
        id_values = tuple(str(value) for value in candidate_ids)
        if len(vector_values) != len(id_values):
            raise ValueError("batch vectors and candidate_ids must align")
        batch_evaluator = getattr(self._problem, "evaluate_batch", None)
        if not callable(batch_evaluator):
            raise BatchEvaluationUnavailableBeforeEntry(
                "problem adapter has no ordered batch method"
            )

        results = tuple(
            batch_evaluator(
                vector_values,
                event_id,
                ledger,
                id_values,
            )
        )
        if len(results) != len(vector_values):
            raise FormalRuntimeError(
                "batch evaluator returned the wrong result count"
            )
        for index, result in enumerate(results):
            if result.candidate_id != id_values[index]:
                raise FormalRuntimeError(
                    "batch evaluator changed candidate order or identity"
                )
            self._writer.write(
                event_id=event_id,
                vector=vector_values[index],
                result=result,
            )
        return results

    def safety_filter(self, result: EvaluationResult, event_id: int) -> bool:
        return self._problem.safety_filter(result, event_id)

    def shift_solution(self, vector: Sequence[float]) -> Sequence[float]:
        return self._problem.shift_solution(vector)

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any] | None:
        if event_id in self._execution_observations:
            raise FormalRuntimeError(
                "recording adapter event was executed more than once"
            )
        observation = self._problem.execute(
            action,
            event_id,
            committed,
            ledger,
        )
        if observation is None:
            self._execution_observations[event_id] = None
        elif isinstance(observation, Mapping):
            self._execution_observations[event_id] = dict(observation)
        else:
            raise FormalRuntimeError(
                "problem execution observation must be a mapping or null"
            )
        return observation

    def first_action(self, vector: Sequence[float]) -> Sequence[float]:
        return self._problem.first_action(vector)

    def fallback_action(self, event_id: int) -> Sequence[float]:
        return self._problem.fallback_action(event_id)


class FormalTerminalSelector:
    selector_id = "WGT-V11-R8-FROZEN-PROBLEM-SELECTOR-01"
    selector_version = "1.0.0"

    def __init__(self, rolling: bool, *, corrective: bool = False) -> None:
        self.rolling = bool(rolling)
        if corrective:
            self.selector_id = (
                "WGT-V11-R8C-CORRECTIVE-PROBLEM-SELECTOR-01"
            )
            self.selector_version = "1.1.0-r8c-corrective"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
            "role": "R8_frozen_public_terminal_selection",
            "rolling_policy": (
                "equal_weight_augmented_Tchebycheff_phi_rho_1e-6"
            ),
            "other_policy": "minimum_first_then_second_objective_candidate_id",
        }

    def select(
        self,
        archive: Sequence[EvaluationResult],
    ) -> str | None:
        values = tuple(archive)
        if not values:
            return None
        if not self.rolling:
            return min(
                values,
                key=lambda result: (
                    result.objectives[0],
                    result.objectives[1:],
                    result.candidate_id,
                ),
            ).candidate_id
        matrix = np.asarray(
            [result.objectives for result in values],
            dtype=float,
        )
        transformed = matrix / (1.0 + matrix)
        ideal = np.min(transformed, axis=0)
        weighted = (transformed - ideal) / matrix.shape[1]
        scores = np.max(weighted, axis=1) + 1e-6 * np.sum(
            weighted,
            axis=1,
        )
        return min(
            zip(scores.tolist(), values, strict=True),
            key=lambda pair: (pair[0], pair[1].candidate_id),
        )[1].candidate_id


def build_problem(
    spec: FormalSequenceSpec,
    *,
    settings: FormalRuntimeSettings = LEGACY_R8_RUNTIME_SETTINGS,
) -> Any:
    master_seed = int(spec.master_seed_u64 or "0")
    corrective = settings.corrective
    if spec.workload_id == "E1_STATIC":
        if spec.problem_index is None:
            raise FormalRuntimeError("static task lacks problem_index")
        factory = (
            make_corrective_lircmop_adapter
            if corrective
            else make_formal_lircmop_adapter
        )
        return factory(spec.problem_index)
    if spec.workload_id in {
        "E1_DYNAMIC",
        "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
    }:
        if spec.problem_index is None or spec.profile is None:
            raise FormalRuntimeError("CDF task lacks problem/profile")
        factory = (
            make_corrective_cdf_adapter
            if corrective
            else make_formal_cdf_adapter
        )
        return factory(
            spec.problem_index,
            profile=spec.profile,
            environment_seed=master_seed,
        )
    if spec.workload_id in {
        "E1_ROLLING",
        "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
    }:
        if spec.rolling_template is None or spec.rolling_index is None:
            raise FormalRuntimeError("rolling task lacks template/index")
        factory = (
            make_corrective_wgt_rr_adapter
            if corrective
            else make_formal_wgt_rr_adapter
        )
        problem = factory(
            spec.rolling_template,
            spec.rolling_index,
        )
        if str(problem.identity()["derived_seed_u64"]) != (
            spec.rolling_seed_u64
        ):
            raise FormalRuntimeError(
                "rolling task seed differs from generated instance"
            )
        return problem
    if spec.workload_id == "E3":
        if (
            spec.subject_id is None
            or spec.subject_seed_u64 is None
            or spec.scenario_id is None
        ):
            raise FormalRuntimeError("E3 task lacks subject/scenario")
        subject = generate_subject_parameters(
            spec.subject_id,
            int(spec.subject_seed_u64),
        )
        adapter_type = (
            FormalR8CHallE3Adapter
            if corrective
            else FormalHallE3Adapter
        )
        return adapter_type(
            subject=subject,
            scenario=spec.scenario_id,
            replicate_index=spec.replicate_index,
            paired_master_seed_u64=master_seed,
        )
    raise FormalRuntimeError(f"unsupported workload {spec.workload_id!r}")


def _shared_initialization(
    problem: Any,
    *,
    settings: FormalRuntimeSettings,
    master_seed: int,
    event_id: int,
) -> list[list[float]]:
    rng, _ = derive_rng(
        master_seed,
        stream=RandomStream.INITIALIZATION,
        experiment_id=settings.configuration_evidence_id,
        unit_id=problem.adapter_id,
        event_id=event_id,
        substream="initialization",
    )
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    return [
        rng.uniform(lower, upper).tolist()
        for _ in range(settings.population_size)
    ]


def _method_seed(
    problem: Any,
    *,
    settings: FormalRuntimeSettings,
    master_seed: int,
    method_id: str,
    event_id: int,
) -> int:
    rng, _ = derive_rng(
        master_seed,
        stream=RandomStream.ALGORITHM,
        experiment_id=settings.configuration_evidence_id,
        unit_id=problem.adapter_id,
        method_id=method_id,
        event_id=event_id,
        substream="comparator_root",
    )
    return int(rng.integers(0, 1 << 63, dtype=np.int64))


def _comparator(
    method_id: str,
    *,
    settings: FormalRuntimeSettings,
) -> Any:
    matched = {
        "MATCHED_FIXED_DE_PARETO": "fixed",
        "MATCHED_JDE_STYLE_PARETO": "jde",
        "MATCHED_SHADE_STYLE_PARETO": "shade",
    }
    if method_id in matched:
        return MatchedParetoDE(
            mode=matched[method_id],
            population_size=settings.population_size,
            archive_capacity=settings.archive_capacity,
            method_id_override=method_id,
        )
    jmetal = {
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE": "gde3",
        "JMETALPY_1_7_NSGAII_STATIC_CMOEA": "nsgaii_static",
        "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE": (
            "nsgaii_dynamic_restart"
        ),
    }
    if method_id in jmetal:
        return JMetalComparator(
            mode=jmetal[method_id],
            population_size=settings.population_size,
            archive_capacity=settings.archive_capacity,
        )
    if method_id == "CONVENTIONAL_ROLLING_PLANNER_NO_CROSS_EVENT_CREDIT":
        return ConventionalRollingPlannerBaseline(
            population_size=settings.population_size,
            archive_capacity=settings.archive_capacity,
        )
    if method_id == "FIXED_ENERGY_DEFICIT_POLICY":
        return FixedEnergyDeficitBaseline(archive_capacity=1)
    raise FormalRuntimeError(f"unknown formal comparator {method_id!r}")


def _variant(spec: FormalSequenceSpec) -> str | None:
    if spec.method_id == "F22_MG_STATIC":
        return "NO_CROSS_EVENT_MEMORY"
    if spec.method_id == "DT-RAMDE_TS2_FULL":
        return "FULL"
    if spec.workload_id.startswith("E2_"):
        return spec.method_id
    return None


def _scientific_event_deadline_seconds(
    spec: FormalSequenceSpec,
    settings: FormalRuntimeSettings,
) -> float:
    if settings.persistence_mode != CHECKPOINT_FRONT_PERSISTENCE:
        return float(spec.timeout_seconds)
    try:
        return SCIENTIFIC_EVENT_DEADLINES_SECONDS[spec.workload_id]
    except KeyError as error:
        raise FormalRuntimeError(
            "compact E1+E2 task lacks a frozen scientific event deadline"
        ) from error


def _scientific_timeout_result(
    problem: RecordingAdapter,
    *,
    event_id: int,
) -> OptimizationResult:
    archive = problem.current_front_evaluations()
    witness_id = (
        min(result.candidate_id for result in archive)
        if archive
        else None
    )
    return OptimizationResult(
        terminal=TerminalOutcome(
            code=TerminalCode.REJECT_TIMEOUT,
            candidate_id=witness_id,
            reason="common scientific event deadline reached",
        ),
        archive=archive,
    )


def _event_summary(
    *,
    event_id: int,
    terminal_code: str,
    terminal_candidate_id: str | None,
    terminal_reason: str | None,
    ledger: Mapping[str, int],
    evaluation_failure_type_counts: Mapping[str, int],
    information_hash: str,
    feedback: Mapping[str, Any] | None,
    execution_observation: Mapping[str, Any] | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    failure_counts = {
        str(failure_type): int(count)
        for failure_type, count in sorted(
            evaluation_failure_type_counts.items()
        )
    }
    if (
        any(
            not failure_type or count < 1
            for failure_type, count in failure_counts.items()
        )
        or sum(failure_counts.values())
        != int(ledger.get("evaluation_failures", 0))
    ):
        raise FormalRuntimeError(
            "evaluation failure type counts differ from the event ledger"
        )
    terminal = {
        "code": terminal_code,
        "reason": terminal_reason,
    }
    if compact:
        terminal["candidate_available"] = terminal_candidate_id is not None
    else:
        terminal["candidate_id"] = terminal_candidate_id
    summary = {
        "event_id": event_id,
        "terminal": terminal,
        "ledger": dict(ledger),
        "evaluation_failure_type_counts": failure_counts,
        "information_hash": information_hash,
        "execution_feedback": (
            None if feedback is None else dict(feedback)
        ),
    }
    if compact:
        summary["execution_observation"] = (
            None
            if execution_observation is None
            else dict(execution_observation)
        )
    return summary


def _run_dt_ramde(
    problem: RecordingAdapter,
    spec: FormalSequenceSpec,
    request: FormalExecutionRequest,
    settings: FormalRuntimeSettings,
    stop_path: Path,
    heartbeat_path: Path,
    event_summary_writer: DurableEventSummaryWriter | None = None,
) -> tuple[list[dict[str, Any]], str, Mapping[str, Any]]:
    variant = _variant(spec)
    if variant is None:
        raise FormalRuntimeError("task is not a DT-RAMDE variant")
    timing = (
        "TS1_single_event"
        if spec.events == 1
        else "TS2_fixed_periodic_replanning"
    )
    method_label = (
        "F22_MG_STATIC"
        if spec.events == 1
        else (
            "DT-RAMDE_TS2_FULL"
            if variant == "FULL"
            else variant
        )
    )
    selector = FormalTerminalSelector(
        isinstance(problem._problem, FormalR8WGTRRAdapter),
        corrective=settings.corrective,
    )
    config = AlgorithmConfig(
        variant=variant,
        population_size=settings.population_size,
        cfe_per_event=spec.cfe_per_event,
        algorithm_seed=int(spec.master_seed_u64 or "0"),
        max_events=spec.events,
        timing_mode=timing,
        method_label=method_label,
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=spec.atomic_steps_per_cfe,
        event_time_limit_seconds=_scientific_event_deadline_seconds(
            spec,
            settings,
        ),
        configuration_evidence_id=settings.configuration_evidence_id,
        execution_request=request,
        audit_materialization=(
            COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION
            if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE
            else FULL_AUDIT_MATERIALIZATION
        ),
    )
    engine = DTRAMDE(config)
    summaries: list[dict[str, Any]] = []
    prior_feedback = None
    status = "COMPLETE"
    for event_id in range(spec.events):
        if stop_path.exists():
            status = "INCOMPLETE_RESOURCE_CEILING"
            break
        problem.begin_event(
            event_id=event_id,
            cfe_budget=spec.cfe_per_event,
        )
        event = engine.run_event(
            problem,
            selector=selector,
            event_id=event_id,
            prior_feedback=prior_feedback,
        )
        prior_feedback = event.execution_feedback
        summary = _event_summary(
            event_id=event_id,
            terminal_code=event.terminal.code.value,
            terminal_candidate_id=event.terminal.candidate_id,
            terminal_reason=event.terminal.reason,
            ledger=event.ledger,
            evaluation_failure_type_counts=(
                problem.failure_type_counts(event_id)
            ),
            information_hash=event.information_hash,
            feedback=event.execution_feedback,
            execution_observation=problem.execution_observation(event_id),
            compact=(
                settings.persistence_mode
                == CHECKPOINT_FRONT_PERSISTENCE
            ),
        )
        if event_summary_writer is not None:
            event_summary_writer.append(summary)
        problem.finish_event(
            terminal_snapshot=(
                int(event.ledger["cfe"]) < spec.cfe_per_event
            )
        )
        summaries.append(summary)
        heartbeat_path.touch()
    return summaries, status, engine.identity()


def _run_comparator(
    problem: RecordingAdapter,
    spec: FormalSequenceSpec,
    settings: FormalRuntimeSettings,
    stop_path: Path,
    heartbeat_path: Path,
    event_summary_writer: DurableEventSummaryWriter | None = None,
) -> tuple[list[dict[str, Any]], str, Mapping[str, Any]]:
    comparator = _comparator(spec.method_id, settings=settings)
    summaries: list[dict[str, Any]] = []
    feedback: Mapping[str, Any] | None = None
    status = "COMPLETE"
    master_seed = int(spec.master_seed_u64 or "0")
    for event_id in range(spec.events):
        if stop_path.exists():
            status = "INCOMPLETE_RESOURCE_CEILING"
            break
        problem.begin_event(
            event_id=event_id,
            cfe_budget=spec.cfe_per_event,
        )
        try:
            initialization = _shared_initialization(
                problem,
                settings=settings,
                master_seed=master_seed,
                event_id=event_id,
            )
            method_seed = _method_seed(
                problem,
                settings=settings,
                master_seed=master_seed,
                method_id=spec.method_id,
                event_id=event_id,
            )
            if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE:
                problem.start_scientific_deadline(
                    event_id=event_id,
                    seconds=_scientific_event_deadline_seconds(
                        spec,
                        settings,
                    ),
                )
            information = problem.freeze_information(event_id, feedback)
            ledger = EvaluationLedger(max_cfe=spec.cfe_per_event)
            result: OptimizationResult = comparator.optimize(
                problem,
                event_id=event_id,
                budget=spec.cfe_per_event,
                seed=method_seed,
                ledger=ledger,
                initialization_vectors=initialization,
            )
            if problem.scientific_event_deadline_reached(event_id):
                result = _scientific_timeout_result(
                    problem,
                    event_id=event_id,
                )
        except ScientificEventTimeoutBeforeEntry:
            result = _scientific_timeout_result(
                problem,
                event_id=event_id,
            )
        accepted = result.terminal.code is TerminalCode.ACCEPTED
        action = (
            problem.first_action(result.selected_vector)
            if accepted and result.selected_vector is not None
            else problem.fallback_action(event_id)
        )
        feedback = dict(
            problem.execute(action, event_id, accepted, ledger)
        )
        ledger.assert_joint_contract(
            atomic_steps_per_evaluation=spec.atomic_steps_per_cfe
        )
        snapshot = ledger.snapshot()
        if (
            snapshot["cfe"] != spec.cfe_per_event
            and result.terminal.code is not TerminalCode.REJECT_TIMEOUT
        ):
            raise FormalRuntimeError(
                "comparator did not consume exact event CFE budget"
            )
        summary = _event_summary(
            event_id=event_id,
            terminal_code=result.terminal.code.value,
            terminal_candidate_id=result.terminal.candidate_id,
            terminal_reason=result.terminal.reason,
            ledger=snapshot,
            evaluation_failure_type_counts=(
                problem.failure_type_counts(event_id)
            ),
            information_hash=information.information_hash,
            feedback=feedback,
            execution_observation=problem.execution_observation(event_id),
            compact=(
                settings.persistence_mode
                == CHECKPOINT_FRONT_PERSISTENCE
            ),
        )
        if event_summary_writer is not None:
            event_summary_writer.append(summary)
        problem.finish_event(
            terminal_snapshot=(
                int(snapshot["cfe"]) < spec.cfe_per_event
            )
        )
        summaries.append(summary)
        heartbeat_path.touch()
    return summaries, status, comparator.identity()


_TYPED_SHORT_CFE_TERMINALS = frozenset(
    {
        TerminalCode.REJECT_NUMERICAL.value,
        TerminalCode.REJECT_TIMEOUT.value,
    }
)


def _validate_task_accounting(
    *,
    spec: FormalSequenceSpec,
    status: str,
    events: Sequence[Mapping[str, Any]],
    recorded_count: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate charged work without turning typed method outcomes into crashes."""

    total_cfe = sum(int(event["ledger"]["cfe"]) for event in events)
    total_atomic = sum(
        int(event["ledger"]["atomic_model_steps"]) for event in events
    )
    if recorded_count != total_cfe:
        raise FormalRuntimeError(
            "persisted charged-evaluation count differs from event ledgers"
        )
    if total_atomic != total_cfe * spec.atomic_steps_per_cfe:
        raise FormalRuntimeError(
            "atomic-model-step accounting differs from charged CFE"
        )
    if len(events) > spec.events:
        raise FormalRuntimeError("task produced more events than scheduled")

    shortfall_events: list[int] = []
    for expected_event_id, event in enumerate(events):
        if int(event["event_id"]) != expected_event_id:
            raise FormalRuntimeError(
                "task event IDs differ from the scheduled zero-based order"
            )
        event_cfe = int(event["ledger"]["cfe"])
        if not 0 <= event_cfe <= spec.cfe_per_event:
            raise FormalRuntimeError("event CFE lies outside its scheduled budget")
        if event_cfe < spec.cfe_per_event:
            terminal = event.get("terminal")
            terminal_code = (
                terminal.get("code")
                if isinstance(terminal, Mapping)
                else None
            )
            if terminal_code not in _TYPED_SHORT_CFE_TERMINALS:
                raise FormalRuntimeError(
                    "short-CFE event lacks a frozen numerical/timeout terminal"
                )
            shortfall_events.append(expected_event_id)

    if status == "COMPLETE" and len(events) != spec.events:
        raise FormalRuntimeError(
            "completed task event count differs from the schedule"
        )
    if status == "COMPLETE" and total_cfe > spec.total_cfe:
        raise FormalRuntimeError("completed task exceeds scheduled CFE")
    if status == "COMPLETE" and total_atomic > spec.total_atomic_steps:
        raise FormalRuntimeError(
            "completed task exceeds scheduled atomic-model steps"
        )
    return total_cfe, total_atomic, tuple(shortfall_events)


def run_task(
    *,
    spec: FormalSequenceSpec,
    request: FormalExecutionRequest,
    task_directory: Path,
    stop_path: Path,
    settings: FormalRuntimeSettings = LEGACY_R8_RUNTIME_SETTINGS,
) -> dict[str, Any]:
    """Execute exactly one scheduled method sequence with no retry."""

    task_directory.mkdir(parents=False, exist_ok=False)
    heartbeat_path = task_directory / "heartbeat"
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    base_problem = build_problem(spec, settings=settings)
    if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE:
        data_path = task_directory / "checkpoint_fronts.cfe"
        objective_names = getattr(base_problem, "objective_names", None)
        if not isinstance(objective_names, Sequence) or isinstance(
            objective_names,
            str | bytes,
        ):
            raise FormalRuntimeError(
                "checkpoint persistence requires fixed objective names"
            )
        writer_context: EvaluationWriter = EndpointCheckpointWriter(
            data_path,
            spec.task_id,
            objective_names,
        )
    else:
        data_path = task_directory / "raw_evaluations.jsonl.gz"
        writer_context = RawEvaluationWriter(data_path, spec.task_id)
    with writer_context as writer:
        event_summary_writer = (
            DurableEventSummaryWriter(
                task_directory / EVENT_SUMMARIES_FILENAME,
                event_count=spec.events,
                cfe_per_event=spec.cfe_per_event,
                atomic_steps_per_cfe=spec.atomic_steps_per_cfe,
            )
            if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE
            else None
        )
        try:
            problem = RecordingAdapter(
                base_problem,
                writer,
                task_directory / "TASK_TIMEOUT_REQUESTED",
            )
            if _variant(spec) is not None:
                events, status, method_identity = _run_dt_ramde(
                    problem,
                    spec,
                    request,
                    settings,
                    stop_path,
                    heartbeat_path,
                    event_summary_writer,
                )
            else:
                events, status, method_identity = _run_comparator(
                    problem,
                    spec,
                    settings,
                    stop_path,
                    heartbeat_path,
                    event_summary_writer,
                )
            recorded_count = writer.count
        finally:
            if event_summary_writer is not None:
                event_summary_writer.close()
    technical_timeout_path = (
        task_directory / "TASK_TIMEOUT_REQUESTED"
    )
    if technical_timeout_path.is_file():
        raise ExecutionTimeoutBeforeEntry(
            "formal task timeout requested before summary publication"
        )
    total_cfe, total_atomic, shortfall_events = _validate_task_accounting(
        spec=spec,
        status=status,
        events=events,
        recorded_count=recorded_count,
    )
    summary: dict[str, Any] = {
        "artifact_role": (
            (
                f"{settings.artifact_stage}_IMMUTABLE_"
                "ENDPOINT_SUFFICIENT_UNANALYZED"
            )
            if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE
            else f"{settings.artifact_stage}_IMMUTABLE_RAW_UNANALYZED"
        ),
        "status": status,
        "task": spec.to_dict(),
        "method_identity": dict(method_identity),
        "adapter_identity": dict(base_problem.identity()),
        "events": events,
        "total_cfe": total_cfe,
        "total_atomic_model_steps": total_atomic,
        "budget_accounting": {
            "scheduled_cfe": spec.total_cfe,
            "charged_cfe": total_cfe,
            "unconsumed_cfe_due_to_typed_terminal": (
                spec.total_cfe - total_cfe
            ),
            "scheduled_atomic_model_steps": spec.total_atomic_steps,
            "charged_atomic_model_steps": total_atomic,
            "typed_short_cfe_event_ids": list(shortfall_events),
            "unused_budget_transferred": False,
        },
        "timeout_semantics": {
            "scientific_event_deadline_seconds": (
                _scientific_event_deadline_seconds(spec, settings)
            ),
            "technical_sequence_hard_ceiling_seconds": (
                int(spec.timeout_seconds)
            ),
            "scientific_event_terminal": (
                TerminalCode.REJECT_TIMEOUT.value
            ),
            "technical_timeout_algorithm_terminal": None,
        },
        "runtime": {
            "attempt": 1,
            "automatic_retries": 0,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
        "permissions": {
            "participant_data_accessed": False,
            "hidden_instance_accessed_or_generated": False,
            "results_analysis_performed": False,
            "results_writing_performed": False,
        },
    }
    if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE:
        summary["charged_evaluation_count"] = recorded_count
        summary["individual_evaluation_rows_persisted"] = 0
        summary["checkpoint_data_format"] = {
            "format_id": CHECKPOINT_FORMAT_ID,
            "encoding": "little-endian IEEE-754 binary64 fixed-shape records",
            "checkpoints_per_event": 21,
            "archive_capacity": settings.archive_capacity,
            "front_valid_count_persisted": True,
            "front_max_constraint_witness_persisted": True,
            "evaluation_stream_sha256_chain_persisted": True,
            "decision_vectors_persisted": False,
            "candidate_ids_persisted": False,
            "terminal_candidate_identity_persisted": False,
            "dominated_evaluations_persisted": False,
            "execution_observation_persisted": True,
            "algorithm_feedback_channel_persisted": True,
            "effect_endpoint_computed": False,
        }
        summary["event_summary_data_format"] = {
            "filename": EVENT_SUMMARIES_FILENAME,
            "encoding": "UTF-8 canonical JSONL with LF records",
            "append_scope": "one durable record per completed event",
            "maximum_record_bytes_including_lf": (
                EVENT_SUMMARY_MAX_RECORD_BYTES
            ),
            "flush_after_each_event": True,
            "fsync_after_each_event": True,
            "candidate_ids_persisted": False,
            "effect_endpoint_computed": False,
        }
    else:
        summary["raw_evaluation_count"] = recorded_count
        summary["raw_evaluation_format"] = {
            "encoding": "UTF-8 canonical JSONL",
            "compression": "gzip",
            "compresslevel": RAW_GZIP_COMPRESSLEVEL,
            "gzip_mtime": 0,
        }
    summary_path = task_directory / "task_summary.json"
    _write_canonical_json_exclusive_fsynced(
        summary_path,
        summary,
        maximum_bytes=(
            WORKER_CONTROL_REPORT_MAX_BYTES
            if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE
            else None
        ),
    )
    if technical_timeout_path.is_file():
        raise ExecutionTimeoutBeforeEntry(
            "formal task timeout requested before manifest publication"
        )
    artifact_paths = [data_path, summary_path]
    if settings.persistence_mode == CHECKPOINT_FRONT_PERSISTENCE:
        artifact_paths.append(
            task_directory / EVENT_SUMMARIES_FILENAME
        )
    artifacts = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }
    manifest = {
        "task_id": spec.task_id,
        "status": status,
        "artifacts": artifacts,
        "task_binding_sha256": sha256(
            canonical_json_bytes(
                {
                    "task": spec.to_dict(),
                    "artifacts": artifacts,
                }
            )
        ).hexdigest(),
    }
    manifest_path = task_directory / "task_manifest.json"
    _write_canonical_json_exclusive_fsynced(manifest_path, manifest)
    heartbeat_path.unlink(missing_ok=True)
    return {
        "task_id": spec.task_id,
        "status": status,
        "total_cfe": total_cfe,
        "total_atomic_model_steps": total_atomic,
        "cpu_seconds": summary["runtime"]["cpu_seconds"],
        "output_bytes": sum(
            path.stat().st_size for path in task_directory.iterdir()
        ),
        "task_manifest_sha256": file_sha256(manifest_path),
    }


def spec_from_dict(payload: Mapping[str, Any]) -> FormalSequenceSpec:
    allowed = set(FormalSequenceSpec.__dataclass_fields__)
    return FormalSequenceSpec(
        **{key: value for key, value in payload.items() if key in allowed}
    )
