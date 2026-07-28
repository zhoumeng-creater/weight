"""Result-blind, fixed-shape checkpoint storage for formal E1/E2 tasks.

The format deliberately stores only a deterministic sequence of Pareto-front
snapshots.  Individual decision vectors, candidate identifiers, constraints,
and failures are committed by a SHA-256 evaluation-stream chain but are not
written as recoverable records.

All numeric payloads use IEEE-754 binary64 in little-endian order.  Each
snapshot has exactly ``archive_capacity`` objective slots, even when the valid
front is smaller, so the file length does not disclose the front size.  Every
unused slot is encoded as all-zero bytes.  One global maximum constraint value
per snapshot witnesses that all valid front points are feasible.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re
import struct
from typing import BinaryIO, Final, Literal

import numpy as np

from comparators.common import ExactNondominatedAccumulator
from dt_ramde_v11.core import Candidate, order_known_nondominated_archive
from evaluation.contracts import EvaluationResult


CHECKPOINTS_PER_EVENT: Final = 21
ARCHIVE_CAPACITY: Final = 100
EVENT_SUMMARY_MAX_RECORD_BYTES: Final = 8 * 1024
WORKER_CONTROL_REPORT_MAX_BYTES: Final = 64 * 1024
FORMAT_VERSION: Final = 1
FORMAT_ID: Final = "WGT_CFE_CHECKPOINT_BINARY_V1_ENDPOINT_SUFFICIENT"

_MAGIC: Final = b"WGTCFE01"
_FILE_PREFIX = struct.Struct("<8sHHI32s")
_RECORD_LENGTH = struct.Struct("<Q")
_RECORD_FIXED = struct.Struct("<4sB3xqH2xQQQQQ64sHHd")
_RECORD_MAGIC: Final = b"CPR1"
_KIND_CHECKPOINT: Final = 0
_KIND_TERMINAL: Final = 1
_TERMINAL_INDEX: Final = 0xFFFF
_METADATA_DOMAIN: Final = b"WGT-CFE-CHECKPOINT-METADATA-v1\x00"
_CHAIN_INITIAL_DOMAIN: Final = b"WGT-CFE-CHAIN-INITIAL-v1\x00"
_CHAIN_EVENT_DOMAIN: Final = b"WGT-CFE-CHAIN-EVENT-v1\x00"
_CHAIN_LINK_DOMAIN: Final = b"WGT-CFE-CHAIN-LINK-v1\x00"
_SUCCESS_DOMAIN: Final = b"WGT-CFE-EVALUATION-SUCCESS-v1\x00"
_FAILURE_DOMAIN: Final = b"WGT-CFE-EVALUATION-FAILURE-v1\x00"
_MAX_FILE_HEADER_BYTES: Final = 4 * 1024
_SHA256_HEX = re.compile(rb"[0-9a-f]{64}")


class CheckpointDataError(ValueError):
    """Checkpoint input or a persisted checkpoint file is invalid."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """The minimal task-local identity needed to interpret front snapshots."""

    task_id: str
    objective_names: tuple[str, ...]
    archive_capacity: int = ARCHIVE_CAPACITY
    checkpoints_per_event: int = CHECKPOINTS_PER_EVENT

    def __post_init__(self) -> None:
        if not self.task_id:
            raise CheckpointDataError("task_id must be nonempty")
        if not self.objective_names:
            raise CheckpointDataError("objective_names must be nonempty")
        if any(not name for name in self.objective_names):
            raise CheckpointDataError("objective names must be nonempty")
        if len(set(self.objective_names)) != len(self.objective_names):
            raise CheckpointDataError("objective names must be unique")
        if len(self.objective_names) not in {2, 3}:
            raise CheckpointDataError("formal E1/E2 objective dimension must be 2 or 3")
        if self.archive_capacity != ARCHIVE_CAPACITY:
            raise CheckpointDataError(f"archive_capacity must be {ARCHIVE_CAPACITY}")
        if self.checkpoints_per_event != CHECKPOINTS_PER_EVENT:
            raise CheckpointDataError(
                f"checkpoints_per_event must be {CHECKPOINTS_PER_EVENT}"
            )
        _encode_text(self.task_id, field="task_id")
        for name in self.objective_names:
            _encode_text(name, field="objective_name")

    @property
    def objective_dimension(self) -> int:
        return len(self.objective_names)


@dataclass(frozen=True)
class CheckpointRecord:
    """One decoded regular or terminal fixed-shape front snapshot."""

    kind: Literal["checkpoint", "terminal"]
    event_id: int
    checkpoint_index: int | None
    cfe: int
    cfe_budget: int
    success_count: int
    failure_count: int
    feasible_count: int
    evaluation_chain_sha256: str
    front_objectives: tuple[tuple[float, ...], ...]
    front_max_constraint: float

    @property
    def valid_count(self) -> int:
        return len(self.front_objectives)


@dataclass(frozen=True)
class CheckpointFile:
    """A small-file convenience representation returned by the strict reader."""

    metadata: CheckpointMetadata
    records: tuple[CheckpointRecord, ...]
    sha256: str


@dataclass(frozen=True)
class E1E2CheckpointStorageEstimate:
    """Worst-case fixed-shape E1/E2 machine-data storage estimate."""

    task_count: int
    event_count: int
    checkpoint_record_count: int
    objective_payload_bytes: int
    max_constraint_payload_bytes: int
    fixed_record_overhead_bytes: int
    file_header_upper_bound_bytes: int
    conservative_total_upper_bound_bytes: int

    @property
    def conservative_total_upper_bound_gib(self) -> float:
        return self.conservative_total_upper_bound_bytes / (1024**3)


def estimate_e1e2_checkpoint_storage() -> E1E2CheckpointStorageEstimate:
    """Return the frozen worst-case fixed-shape E1/E2 storage calculation.

    The schedule comprises 840 one-event static tasks (720 two-objective and
    120 three-objective), 1,950 sixty-event two-objective dynamic tasks, and
    2,240 twenty-event three-objective rolling tasks.
    """

    static_2d_tasks = 720
    static_3d_tasks = 120
    dynamic_2d_tasks = 1950
    dynamic_events = 60
    rolling_3d_tasks = 2240
    rolling_events = 20
    task_count = static_2d_tasks + static_3d_tasks + dynamic_2d_tasks + rolling_3d_tasks
    event_count = (
        static_2d_tasks
        + static_3d_tasks
        + dynamic_2d_tasks * dynamic_events
        + rolling_3d_tasks * rolling_events
    )
    checkpoint_record_count = event_count * CHECKPOINTS_PER_EVENT
    objective_slots = (
        CHECKPOINTS_PER_EVENT
        * ARCHIVE_CAPACITY
        * (
            static_2d_tasks * 2
            + static_3d_tasks * 3
            + dynamic_2d_tasks * dynamic_events * 2
            + rolling_3d_tasks * rolling_events * 3
        )
    )
    objective_payload_bytes = objective_slots * 8
    max_constraint_payload_bytes = checkpoint_record_count * 8
    fixed_record_overhead_bytes = checkpoint_record_count * (
        _RECORD_LENGTH.size + _RECORD_FIXED.size - 8
    )
    file_header_upper_bound_bytes = task_count * _MAX_FILE_HEADER_BYTES
    conservative_total_upper_bound_bytes = (
        objective_payload_bytes
        + max_constraint_payload_bytes
        + fixed_record_overhead_bytes
        + file_header_upper_bound_bytes
    )
    return E1E2CheckpointStorageEstimate(
        task_count=task_count,
        event_count=event_count,
        checkpoint_record_count=checkpoint_record_count,
        objective_payload_bytes=objective_payload_bytes,
        max_constraint_payload_bytes=max_constraint_payload_bytes,
        fixed_record_overhead_bytes=fixed_record_overhead_bytes,
        file_header_upper_bound_bytes=file_header_upper_bound_bytes,
        conservative_total_upper_bound_bytes=(conservative_total_upper_bound_bytes),
    )


def _encode_text(value: str, *, field: str) -> bytes:
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise CheckpointDataError(f"{field} must be valid UTF-8") from error
    if len(encoded) > 0xFFFFFFFF:
        raise CheckpointDataError(f"{field} is too long")
    return struct.pack("<I", len(encoded)) + encoded


def _encode_float_sequence(values: Sequence[float], *, field: str) -> bytes:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise CheckpointDataError(f"{field} must contain only finite values")
    return struct.pack("<I", len(converted)) + struct.pack(
        f"<{len(converted)}d",
        *converted,
    )


def _encode_text_sequence(values: Sequence[str], *, field: str) -> bytes:
    if len(values) > 0xFFFFFFFF:
        raise CheckpointDataError(f"{field} has too many entries")
    return struct.pack("<I", len(values)) + b"".join(
        _encode_text(value, field=field) for value in values
    )


def _metadata_payload(metadata: CheckpointMetadata) -> bytes:
    payload = bytearray(_METADATA_DOMAIN)
    payload.extend(_encode_text(metadata.task_id, field="task_id"))
    payload.extend(
        struct.pack(
            "<HHH",
            metadata.objective_dimension,
            metadata.archive_capacity,
            metadata.checkpoints_per_event,
        )
    )
    payload.extend(
        _encode_text_sequence(
            metadata.objective_names,
            field="objective_name",
        )
    )
    if _FILE_PREFIX.size + len(payload) > _MAX_FILE_HEADER_BYTES:
        raise CheckpointDataError(
            "encoded metadata exceeds the fixed file-header upper bound"
        )
    return bytes(payload)


def _success_encoding(
    *,
    event_id: int,
    vector: Sequence[float],
    result: EvaluationResult,
) -> bytes:
    payload = bytearray(_SUCCESS_DOMAIN)
    payload.extend(struct.pack("<q", event_id))
    payload.extend(_encode_text(result.candidate_id, field="candidate_id"))
    payload.extend(_encode_float_sequence(vector, field="vector"))
    payload.extend(_encode_float_sequence(result.objectives, field="objectives"))
    payload.extend(
        _encode_text_sequence(
            result.objective_names,
            field="objective_name",
        )
    )
    payload.extend(_encode_float_sequence(result.constraints, field="constraints"))
    payload.extend(
        _encode_text_sequence(
            result.constraint_names,
            field="constraint_name",
        )
    )
    payload.extend(struct.pack("<B", int(result.feasible)))
    payload.extend(struct.pack("<d", float(result.total_violation)))
    return bytes(payload)


def _failure_encoding(
    *,
    event_id: int,
    candidate_id: str,
    vector: Sequence[float],
    error_type: str,
    reason: str,
) -> bytes:
    payload = bytearray(_FAILURE_DOMAIN)
    payload.extend(struct.pack("<q", event_id))
    payload.extend(_encode_text(candidate_id, field="candidate_id"))
    payload.extend(_encode_float_sequence(vector, field="vector"))
    payload.extend(_encode_text(error_type, field="error_type"))
    payload.extend(_encode_text(reason, field="reason"))
    return bytes(payload)


def front_max_constraint(results: Sequence[EvaluationResult]) -> float:
    """Return one witness for feasibility of every supplied front point.

    With finite constraint values, the result is nonpositive if and only if
    every constraint of every result is nonpositive.  An empty front, or a
    front whose points have no constraints, uses ``0.0`` (vacuous feasibility).
    """

    return max(
        (float(constraint) for result in results for constraint in result.constraints),
        default=0.0,
    )


class TaskCheckpointWriter:
    """Write one exclusive, deterministic task-local checkpoint file.

    The caller must submit exactly one ``record_success`` or ``record_failure``
    call for every charged and completed CFE, in evaluator return order.
    """

    def __init__(self, path: Path, metadata: CheckpointMetadata) -> None:
        self.path = Path(path)
        self.metadata = metadata
        payload = _metadata_payload(metadata)
        prefix = _FILE_PREFIX.pack(
            _MAGIC,
            FORMAT_VERSION,
            0,
            len(payload),
            sha256(payload).digest(),
        )
        self._stream = self.path.open("xb")
        try:
            self._stream.write(prefix)
            self._stream.write(payload)
        except BaseException:
            self._stream.close()
            raise
        self._closed = False
        self._event_id: int | None = None
        self._event_budget = 0
        self._event_cfe = 0
        self._success_count = 0
        self._failure_count = 0
        self._feasible_count = 0
        self._next_checkpoint_index = 0
        self._last_event_id: int | None = None
        self._seen_candidate_ids: set[str] = set()
        self._constraint_names: tuple[str, ...] | None = None
        self._accumulator = ExactNondominatedAccumulator()
        self._chain = sha256(_CHAIN_INITIAL_DOMAIN + payload).digest()

    def _require_open(self) -> None:
        if self._closed:
            raise CheckpointDataError("checkpoint writer is closed")

    def begin_event(self, *, event_id: int, cfe_budget: int) -> None:
        """Begin a strictly ordered event and write its zero-CFE checkpoint."""

        self._require_open()
        if self._event_id is not None:
            raise CheckpointDataError(
                "the active event must be finished before another begins"
            )
        if event_id < 0:
            raise CheckpointDataError("event_id must be nonnegative")
        if self._last_event_id is not None and event_id <= self._last_event_id:
            raise CheckpointDataError("event_id values must be strictly increasing")
        intervals = CHECKPOINTS_PER_EVENT - 1
        if cfe_budget <= 0 or cfe_budget % intervals != 0:
            raise CheckpointDataError(
                f"cfe_budget must be positive and divisible by {intervals}"
            )
        self._event_id = int(event_id)
        self._event_budget = int(cfe_budget)
        self._event_cfe = 0
        self._success_count = 0
        self._failure_count = 0
        self._feasible_count = 0
        self._next_checkpoint_index = 0
        self._seen_candidate_ids = set()
        self._constraint_names = None
        self._accumulator = ExactNondominatedAccumulator()
        event_encoding = _CHAIN_EVENT_DOMAIN + struct.pack(
            "<qQ", self._event_id, self._event_budget
        )
        self._chain = sha256(event_encoding + self._chain).digest()
        self._write_snapshot(kind="checkpoint", checkpoint_index=0)
        self._next_checkpoint_index = 1

    def _validate_candidate_id(self, candidate_id: str) -> None:
        if not candidate_id:
            raise CheckpointDataError("candidate_id must be nonempty")
        _encode_text(candidate_id, field="candidate_id")
        if candidate_id in self._seen_candidate_ids:
            raise CheckpointDataError("candidate_id must be unique within an event")
        self._seen_candidate_ids.add(candidate_id)

    def record_success(
        self,
        *,
        event_id: int,
        vector: Sequence[float],
        result: EvaluationResult,
    ) -> None:
        """Commit one successful charged evaluation and update the exact front."""

        self._require_active_event(event_id)
        if result.objective_names != self.metadata.objective_names:
            raise CheckpointDataError(
                "evaluation objective names differ from checkpoint metadata"
            )
        if len(result.objectives) != self.metadata.objective_dimension:
            raise CheckpointDataError(
                "evaluation objective dimension differs from metadata"
            )
        if (
            self._constraint_names is not None
            and result.constraint_names != self._constraint_names
        ):
            raise CheckpointDataError(
                "constraint identity must remain fixed within an event"
            )
        self._validate_candidate_id(result.candidate_id)
        encoded = _success_encoding(
            event_id=event_id,
            vector=vector,
            result=result,
        )
        if self._constraint_names is None:
            self._constraint_names = result.constraint_names
        self._chain = sha256(_CHAIN_LINK_DOMAIN + self._chain + encoded).digest()
        self._success_count += 1
        if result.feasible:
            self._feasible_count += 1
        candidate = Candidate(
            vector=np.empty(0, dtype=float),
            evaluation=result,
            lineage_node_id=f"checkpoint:{event_id}:{result.candidate_id}",
        )
        self._accumulator.add(candidate)
        self._advance_cfe()

    def record_failure(
        self,
        *,
        event_id: int,
        candidate_id: str,
        vector: Sequence[float],
        error_type: str,
        reason: str,
    ) -> None:
        """Commit one failed charged evaluation without persisting its details."""

        self._require_active_event(event_id)
        if not error_type:
            raise CheckpointDataError("error_type must be nonempty")
        self._validate_candidate_id(candidate_id)
        encoded = _failure_encoding(
            event_id=event_id,
            candidate_id=candidate_id,
            vector=vector,
            error_type=error_type,
            reason=reason,
        )
        self._chain = sha256(_CHAIN_LINK_DOMAIN + self._chain + encoded).digest()
        self._failure_count += 1
        self._advance_cfe()

    def _require_active_event(self, event_id: int) -> None:
        self._require_open()
        if self._event_id is None:
            raise CheckpointDataError("no checkpoint event is active")
        if event_id != self._event_id:
            raise CheckpointDataError("record event_id differs from active event")
        if self._event_cfe >= self._event_budget:
            raise CheckpointDataError("event CFE budget is already exhausted")

    def _advance_cfe(self) -> None:
        self._event_cfe += 1
        interval = self._event_budget // (CHECKPOINTS_PER_EVENT - 1)
        if (
            self._next_checkpoint_index < CHECKPOINTS_PER_EVENT
            and self._event_cfe == self._next_checkpoint_index * interval
        ):
            self._write_snapshot(
                kind="checkpoint",
                checkpoint_index=self._next_checkpoint_index,
            )
            self._next_checkpoint_index += 1

    def _front_candidates(self) -> tuple[Candidate, ...]:
        values = self._accumulator.snapshot()
        if not values:
            return ()
        constraint_count = len(values[0].constraints)
        ordered = order_known_nondominated_archive(
            values,
            capacity=self.metadata.archive_capacity,
            constraint_scales=(1.0,) * constraint_count,
            fixed_evaluation_schema=True,
        )
        return tuple(ordered)

    def current_front_objectives(self) -> tuple[tuple[float, ...], ...]:
        """Return the current capacity-truncated deterministic front ordering."""

        self._require_open()
        if self._event_id is None:
            raise CheckpointDataError("no checkpoint event is active")
        return tuple(
            tuple(float(value) for value in candidate.objectives)
            for candidate in self._front_candidates()
        )

    def current_front_evaluations(self) -> tuple[EvaluationResult, ...]:
        """Return the current deterministic feasible front for runtime control."""

        self._require_open()
        if self._event_id is None:
            raise CheckpointDataError("no checkpoint event is active")
        return tuple(
            candidate.evaluation for candidate in self._front_candidates()
        )

    def _write_snapshot(
        self,
        *,
        kind: Literal["checkpoint", "terminal"],
        checkpoint_index: int | None,
    ) -> None:
        if self._event_id is None:
            raise CheckpointDataError("no checkpoint event is active")
        front = self._front_candidates()
        valid_count = len(front)
        capacity = self.metadata.archive_capacity
        dimension = self.metadata.objective_dimension
        if valid_count > capacity:
            raise CheckpointDataError("front exceeds checkpoint capacity")
        payload = bytearray(capacity * dimension * 8)
        offset = 0
        for candidate in front:
            objectives = candidate.objectives
            if len(objectives) != dimension:
                raise CheckpointDataError("front objective dimension changed")
            struct.pack_into(
                f"<{dimension}d",
                payload,
                offset,
                *objectives,
            )
            offset += dimension * 8
        max_constraint = front_max_constraint(
            tuple(candidate.evaluation for candidate in front)
        )
        if not math.isfinite(max_constraint) or max_constraint > 0.0:
            raise CheckpointDataError(
                "front maximum constraint is not a feasibility witness"
            )
        kind_code = _KIND_CHECKPOINT if kind == "checkpoint" else _KIND_TERMINAL
        stored_index = (
            int(checkpoint_index) if checkpoint_index is not None else _TERMINAL_INDEX
        )
        fixed = _RECORD_FIXED.pack(
            _RECORD_MAGIC,
            kind_code,
            self._event_id,
            stored_index,
            self._event_cfe,
            self._event_budget,
            self._success_count,
            self._failure_count,
            self._feasible_count,
            self._chain.hex().encode("ascii"),
            valid_count,
            dimension,
            max_constraint,
        )
        body_length = len(fixed) + len(payload)
        self._stream.write(_RECORD_LENGTH.pack(body_length))
        self._stream.write(fixed)
        self._stream.write(payload)

    def finish_event(self, *, terminal_snapshot: bool = False) -> None:
        """Finish the active event, optionally persisting a terminal snapshot.

        A partial event must use ``terminal_snapshot=True``.  A full event
        already has checkpoint 20 and must not append a duplicate terminal
        record.  This keeps every complete or partial event at no more than
        the frozen 21 fixed-shape records.
        """

        self._require_open()
        if self._event_id is None:
            raise CheckpointDataError("no checkpoint event is active")
        if self._event_cfe < self._event_budget and not terminal_snapshot:
            raise CheckpointDataError("a partial event requires a terminal snapshot")
        if self._event_cfe == self._event_budget and terminal_snapshot:
            raise CheckpointDataError(
                "a full event must use checkpoint 20 as its terminal snapshot"
            )
        if terminal_snapshot:
            self._write_snapshot(kind="terminal", checkpoint_index=None)
        self._last_event_id = self._event_id
        self._event_id = None

    def close(self) -> None:
        """Close the file, preserving an active event as a terminal snapshot."""

        if self._closed:
            return
        try:
            if self._event_id is not None:
                self.finish_event(
                    terminal_snapshot=self._event_cfe < self._event_budget
                )
            self._stream.flush()
        finally:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> TaskCheckpointWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class _PayloadDecoder:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, length: int) -> bytes:
        if length < 0 or self.offset + length > len(self.payload):
            raise CheckpointDataError("metadata is truncated")
        value = self.payload[self.offset : self.offset + length]
        self.offset += length
        return value

    def unpack(self, format_string: str) -> tuple[object, ...]:
        layout = struct.Struct(format_string)
        return layout.unpack(self.take(layout.size))

    def text(self, *, field: str) -> str:
        (length,) = self.unpack("<I")
        try:
            return self.take(int(length)).decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise CheckpointDataError(f"metadata {field} is not valid UTF-8") from error


def _decode_metadata(payload: bytes) -> CheckpointMetadata:
    decoder = _PayloadDecoder(payload)
    if decoder.take(len(_METADATA_DOMAIN)) != _METADATA_DOMAIN:
        raise CheckpointDataError("metadata domain is invalid")
    task_id = decoder.text(field="task_id")
    dimension, capacity, checkpoints = decoder.unpack("<HHH")
    (name_count,) = decoder.unpack("<I")
    names = tuple(decoder.text(field="objective_name") for _ in range(int(name_count)))
    if decoder.offset != len(payload):
        raise CheckpointDataError("metadata contains trailing bytes")
    if int(dimension) != len(names):
        raise CheckpointDataError("metadata objective dimension and names differ")
    return CheckpointMetadata(
        task_id=task_id,
        objective_names=names,
        archive_capacity=int(capacity),
        checkpoints_per_event=int(checkpoints),
    )


def _read_exact(stream: BinaryIO, length: int, *, context: str) -> bytes:
    payload = stream.read(length)
    if len(payload) != length:
        raise CheckpointDataError(f"{context} is truncated")
    return payload


def _read_metadata(stream: BinaryIO) -> CheckpointMetadata:
    prefix = _read_exact(
        stream,
        _FILE_PREFIX.size,
        context="checkpoint file header",
    )
    magic, version, flags, payload_length, expected_hash = _FILE_PREFIX.unpack(prefix)
    if magic != _MAGIC:
        raise CheckpointDataError("checkpoint file magic is invalid")
    if version != FORMAT_VERSION:
        raise CheckpointDataError("checkpoint file version is unsupported")
    if flags != 0:
        raise CheckpointDataError("checkpoint file flags must be zero")
    if _FILE_PREFIX.size + payload_length > _MAX_FILE_HEADER_BYTES:
        raise CheckpointDataError("checkpoint file header is too large")
    payload = _read_exact(
        stream,
        payload_length,
        context="checkpoint metadata",
    )
    if sha256(payload).digest() != expected_hash:
        raise CheckpointDataError("checkpoint metadata hash differs")
    return _decode_metadata(payload)


class CheckpointReader:
    """Streaming strict reader for task-local checkpoint files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream = self.path.open("rb")
        try:
            self.metadata = _read_metadata(self._stream)
        except BaseException:
            self._stream.close()
            raise
        self._consumed = False
        self._closed = False

    def records(self) -> Iterator[CheckpointRecord]:
        if self._closed:
            raise CheckpointDataError("checkpoint reader is closed")
        if self._consumed:
            raise CheckpointDataError("checkpoint records can be iterated once")
        self._consumed = True
        previous_event_id: int | None = None
        expected_checkpoint_index = 0
        previous_cfe = 0
        previous_success = 0
        previous_failure = 0
        previous_feasible = 0
        event_finished = True
        regular_complete = False
        while True:
            length_bytes = self._stream.read(_RECORD_LENGTH.size)
            if not length_bytes:
                break
            if len(length_bytes) != _RECORD_LENGTH.size:
                raise CheckpointDataError("record length is truncated")
            (body_length,) = _RECORD_LENGTH.unpack(length_bytes)
            expected_body_length = _RECORD_FIXED.size + (
                self.metadata.archive_capacity * self.metadata.objective_dimension * 8
            )
            if body_length != expected_body_length:
                raise CheckpointDataError(
                    "record length differs from the fixed-shape format"
                )
            body = _read_exact(
                self._stream,
                body_length,
                context="checkpoint record",
            )
            fixed = body[: _RECORD_FIXED.size]
            payload = body[_RECORD_FIXED.size :]
            (
                record_magic,
                kind_code,
                event_id,
                stored_index,
                cfe,
                cfe_budget,
                success_count,
                failure_count,
                feasible_count,
                chain_hex,
                valid_count,
                dimension,
                max_constraint,
            ) = _RECORD_FIXED.unpack(fixed)
            if record_magic != _RECORD_MAGIC:
                raise CheckpointDataError("record magic is invalid")
            if kind_code not in {_KIND_CHECKPOINT, _KIND_TERMINAL}:
                raise CheckpointDataError("record kind is invalid")
            if event_id < 0:
                raise CheckpointDataError("record event_id is negative")
            if dimension != self.metadata.objective_dimension:
                raise CheckpointDataError("record objective dimension differs")
            if valid_count > self.metadata.archive_capacity:
                raise CheckpointDataError("record valid_count exceeds capacity")
            if not math.isfinite(max_constraint) or max_constraint > 0.0:
                raise CheckpointDataError(
                    "front maximum constraint is not a feasibility witness"
                )
            if valid_count == 0 and max_constraint != 0.0:
                raise CheckpointDataError("empty front maximum constraint must be zero")
            if _SHA256_HEX.fullmatch(chain_hex) is None:
                raise CheckpointDataError("evaluation chain SHA-256 format is invalid")
            intervals = self.metadata.checkpoints_per_event - 1
            if cfe_budget <= 0 or cfe_budget % intervals != 0:
                raise CheckpointDataError("record CFE budget is invalid")
            if success_count + failure_count != cfe:
                raise CheckpointDataError(
                    "record success/failure counts do not equal CFE"
                )
            if feasible_count > success_count:
                raise CheckpointDataError(
                    "record feasible count exceeds successful evaluations"
                )
            if valid_count > feasible_count:
                raise CheckpointDataError(
                    "record front size exceeds feasible evaluations"
                )

            is_new_event = event_id != previous_event_id
            if is_new_event:
                if previous_event_id is not None and (
                    event_id <= previous_event_id
                    or not (event_finished or regular_complete)
                ):
                    raise CheckpointDataError(
                        "record event ordering or termination is invalid"
                    )
                expected_checkpoint_index = 0
                previous_cfe = 0
                previous_success = 0
                previous_failure = 0
                previous_feasible = 0
                event_finished = False
                regular_complete = False
            elif event_finished:
                raise CheckpointDataError(
                    "a terminal/full event has additional records"
                )

            if (
                cfe < previous_cfe
                or success_count < previous_success
                or failure_count < previous_failure
                or feasible_count < previous_feasible
            ):
                raise CheckpointDataError("record counts are not monotonic")

            if kind_code == _KIND_CHECKPOINT:
                if stored_index >= self.metadata.checkpoints_per_event:
                    raise CheckpointDataError(
                        "checkpoint index exceeds the frozen range"
                    )
                if stored_index != expected_checkpoint_index:
                    raise CheckpointDataError("checkpoint indexes are not consecutive")
                expected_cfe = stored_index * cfe_budget // intervals
                if cfe != expected_cfe:
                    raise CheckpointDataError(
                        "checkpoint CFE differs from the frozen fraction"
                    )
                checkpoint_index: int | None = int(stored_index)
                kind: Literal["checkpoint", "terminal"] = "checkpoint"
                expected_checkpoint_index += 1
                if stored_index == intervals:
                    regular_complete = True
            else:
                if stored_index != _TERMINAL_INDEX:
                    raise CheckpointDataError("terminal record index marker is invalid")
                if regular_complete:
                    raise CheckpointDataError(
                        "full event cannot append a terminal record"
                    )
                if cfe > cfe_budget:
                    raise CheckpointDataError("terminal record exceeds the CFE budget")
                next_checkpoint_cfe = (
                    expected_checkpoint_index * cfe_budget // intervals
                )
                if not regular_complete and cfe >= next_checkpoint_cfe:
                    raise CheckpointDataError(
                        "terminal record skips a required checkpoint"
                    )
                checkpoint_index = None
                kind = "terminal"
                event_finished = True

            slot_width = dimension * 8
            valid_bytes = int(valid_count) * slot_width
            if any(payload[valid_bytes:]):
                raise CheckpointDataError(
                    "fixed-shape padding slots must contain all-zero bytes"
                )
            values = struct.unpack(
                f"<{self.metadata.archive_capacity * dimension}d",
                payload,
            )
            objectives: list[tuple[float, ...]] = []
            for index in range(int(valid_count)):
                start = index * dimension
                row = tuple(float(value) for value in values[start : start + dimension])
                if not all(math.isfinite(value) for value in row):
                    raise CheckpointDataError("valid front slot is nonfinite")
                objectives.append(row)

            record = CheckpointRecord(
                kind=kind,
                event_id=int(event_id),
                checkpoint_index=checkpoint_index,
                cfe=int(cfe),
                cfe_budget=int(cfe_budget),
                success_count=int(success_count),
                failure_count=int(failure_count),
                feasible_count=int(feasible_count),
                evaluation_chain_sha256=chain_hex.decode("ascii"),
                front_objectives=tuple(objectives),
                front_max_constraint=float(max_constraint),
            )
            previous_event_id = int(event_id)
            previous_cfe = int(cfe)
            previous_success = int(success_count)
            previous_failure = int(failure_count)
            previous_feasible = int(feasible_count)
            yield record
        if (
            previous_event_id is not None
            and not event_finished
            and not regular_complete
        ):
            raise CheckpointDataError(
                "final event lacks checkpoint 20 or a terminal snapshot"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._closed = True

    def __enter__(self) -> CheckpointReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a checkpoint file without loading it in memory."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_file(path: Path) -> CheckpointFile:
    """Strictly decode a checkpoint file into memory (for small files/tests)."""

    with CheckpointReader(path) as reader:
        metadata = reader.metadata
        records = tuple(reader.records())
    return CheckpointFile(
        metadata=metadata,
        records=records,
        sha256=file_sha256(path),
    )
