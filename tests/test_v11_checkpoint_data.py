from __future__ import annotations

from hashlib import sha256
import struct
from pathlib import Path

import numpy as np
import pytest

from dt_ramde_v11 import core as core_module
from dt_ramde_v11.core import Candidate, maintain_nondominated_archive
from evaluation.contracts import EvaluationResult
from formal_execution import checkpoint_data
from formal_execution.checkpoint_data import (
    ARCHIVE_CAPACITY,
    CHECKPOINTS_PER_EVENT,
    CheckpointDataError,
    CheckpointMetadata,
    TaskCheckpointWriter,
    estimate_e1e2_checkpoint_storage,
    front_max_constraint,
    read_checkpoint_file,
)


def _result(
    candidate_id: str,
    objectives: tuple[float, ...],
    *,
    constraint: float = -1.0,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        objectives=objectives,
        objective_names=tuple(f"objective_{index}" for index in range(len(objectives))),
        constraints=(constraint,),
        constraint_names=("constraint",),
    )


def _candidate(result: EvaluationResult) -> Candidate:
    return Candidate(
        vector=np.empty(0, dtype=float),
        evaluation=result,
        lineage_node_id=f"reference:{result.candidate_id}",
    )


def _reference_front(
    results: list[EvaluationResult],
) -> tuple[tuple[float, ...], ...]:
    archive = maintain_nondominated_archive(
        [_candidate(result) for result in results],
        capacity=ARCHIVE_CAPACITY,
        constraint_scales=(1.0,),
    )
    return tuple(candidate.objectives for candidate in archive)


def _random_results(
    *,
    dimension: int,
    count: int,
    seed: int,
) -> list[EvaluationResult]:
    rng = np.random.default_rng(seed)
    rows: list[EvaluationResult] = []
    for index in range(count):
        objectives = tuple(
            float(value) for value in rng.uniform(-2.0, 5.0, size=dimension)
        )
        constraint = -float(rng.uniform(0.0, 2.0))
        if index % 17 == 0:
            constraint = float(rng.uniform(0.01, 1.0))
        rows.append(
            _result(
                f"candidate-{index:04d}",
                objectives,
                constraint=constraint,
            )
        )
    return rows


@pytest.mark.parametrize("dimension", [2, 3])
def test_every_checkpoint_matches_full_offline_reference(
    tmp_path: Path,
    dimension: int,
) -> None:
    budget = 200
    results = _random_results(
        dimension=dimension,
        count=budget,
        seed=400 + dimension,
    )
    path = tmp_path / f"random-{dimension}d.cfe"
    metadata = CheckpointMetadata(
        task_id=f"random-{dimension}d",
        objective_names=tuple(f"objective_{index}" for index in range(dimension)),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=3, cfe_budget=budget)
        for index, result in enumerate(results):
            writer.record_success(
                event_id=3,
                vector=(float(index), -float(index)),
                result=result,
            )
        writer.finish_event()

    decoded = read_checkpoint_file(path)
    assert decoded.metadata == metadata
    assert len(decoded.records) == CHECKPOINTS_PER_EVENT
    interval = budget // (CHECKPOINTS_PER_EVENT - 1)
    for record in decoded.records:
        assert record.kind == "checkpoint"
        assert record.checkpoint_index is not None
        stop = record.checkpoint_index * interval
        expected = _reference_front(results[:stop])
        assert record.cfe == stop
        assert record.front_objectives == expected
        assert record.front_max_constraint <= 0.0


def _write_deterministic_fixture(path: Path) -> None:
    metadata = CheckpointMetadata(
        task_id="deterministic-task",
        objective_names=("objective_0", "objective_1"),
    )
    rows = _random_results(dimension=2, count=40, seed=91)
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=40)
        for index, result in enumerate(rows):
            writer.record_success(
                event_id=0,
                vector=(index + 0.25, index - 0.75),
                result=result,
            )
        writer.finish_event()


def test_identical_inputs_produce_identical_bytes_and_hash(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.cfe"
    right = tmp_path / "right.cfe"
    _write_deterministic_fixture(left)
    _write_deterministic_fixture(right)

    assert left.read_bytes() == right.read_bytes()
    expected_hash = sha256(left.read_bytes()).hexdigest()
    assert read_checkpoint_file(left).sha256 == expected_hash
    assert read_checkpoint_file(right).sha256 == expected_hash
    assert struct.pack("<d", 0.0) in left.read_bytes()


def test_checkpoint_validates_shared_constraint_scales_once_per_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = core_module._validated_scales

    def tracked(
        candidate: Candidate,
        constraint_scales: tuple[float, ...],
    ) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return original(candidate, constraint_scales)

    monkeypatch.setattr(core_module, "_validated_scales", tracked)
    path = tmp_path / "single-schema-validation.cfe"
    metadata = CheckpointMetadata(
        task_id="single-schema-validation",
        objective_names=("objective_0", "objective_1"),
    )
    results = [
        _result(
            f"tradeoff-{index:03d}",
            (float(index), float(19 - index)),
        )
        for index in range(20)
    ]
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index, result in enumerate(results):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=result,
            )
        writer.finish_event()

    assert calls == 20


def test_checkpoint_uses_capacity_100_deterministic_crowding_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capacity.cfe"
    metadata = CheckpointMetadata(
        task_id="capacity",
        objective_names=("objective_0", "objective_1"),
    )
    results = [
        _result(
            f"tradeoff-{index:03d}",
            (float(index), float(149 - index)),
        )
        for index in range(150)
    ]
    results.extend(
        _result(
            f"dominated-{index:03d}",
            (1000.0 + index, 1000.0 + index),
        )
        for index in range(50)
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=200)
        for index, result in enumerate(results):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=result,
            )
        writer.finish_event()

    final = read_checkpoint_file(path).records[-1]
    assert final.valid_count == ARCHIVE_CAPACITY
    assert final.front_objectives == _reference_front(results)


def test_dominated_point_removal_does_not_change_current_front(
    tmp_path: Path,
) -> None:
    metadata = CheckpointMetadata(
        task_id="dominated-point",
        objective_names=("objective_0", "objective_1"),
    )
    dominating = _result("dominant", (1.0, 1.0))
    dominated = _result("dominated", (2.0, 3.0))

    with TaskCheckpointWriter(tmp_path / "with.cfe", metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.record_success(
            event_id=0,
            vector=(0.0,),
            result=dominating,
        )
        writer.record_success(
            event_id=0,
            vector=(1.0,),
            result=dominated,
        )
        with_dominated = writer.current_front_objectives()
        writer.finish_event(terminal_snapshot=True)

    with TaskCheckpointWriter(tmp_path / "without.cfe", metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.record_success(
            event_id=0,
            vector=(0.0,),
            result=dominating,
        )
        without_dominated = writer.current_front_objectives()
        writer.finish_event(terminal_snapshot=True)

    assert with_dominated == without_dominated == ((1.0, 1.0),)


def test_failure_partial_terminal_and_no_per_evaluation_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cfe"
    metadata = CheckpointMetadata(
        task_id="partial-task",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=8, cfe_budget=40)
        writer.record_success(
            event_id=8,
            vector=(12345.125, -98765.5),
            result=_result("private-candidate-id", (1.0, 2.0)),
        )
        writer.record_failure(
            event_id=8,
            candidate_id="private-failure-id",
            vector=(77777.25,),
            error_type="PrivateNumericalFailure",
            reason="private failure detail",
        )
        for index in range(5):
            writer.record_success(
                event_id=8,
                vector=(float(index),),
                result=_result(
                    f"partial-{index}",
                    (2.0 + index, 1.0 + index),
                ),
            )
        writer.finish_event(terminal_snapshot=True)

    decoded = read_checkpoint_file(path)
    assert [record.cfe for record in decoded.records] == [0, 2, 4, 6, 7]
    terminal = decoded.records[-1]
    assert terminal.kind == "terminal"
    assert terminal.checkpoint_index is None
    assert terminal.success_count == 6
    assert terminal.failure_count == 1
    assert terminal.feasible_count == 6
    assert len(terminal.evaluation_chain_sha256) == 64
    raw = path.read_bytes()
    for private_value in (
        b"private-candidate-id",
        b"private-failure-id",
        b"PrivateNumericalFailure",
        b"private failure detail",
        struct.pack("<2d", 12345.125, -98765.5),
    ):
        assert private_value not in raw


def test_context_exit_seals_an_unfinished_event_as_readable_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "automatic-terminal.cfe"
    metadata = CheckpointMetadata(
        task_id="automatic-terminal",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.record_success(
            event_id=0,
            vector=(0.5,),
            result=_result("candidate-0", (1.0, 2.0)),
        )

    decoded = read_checkpoint_file(path)
    assert [(record.kind, record.cfe) for record in decoded.records] == [
        ("checkpoint", 0),
        ("checkpoint", 1),
        ("terminal", 1),
    ]
    assert decoded.records[-1].evaluation_chain_sha256 == (
        decoded.records[-2].evaluation_chain_sha256
    )


def test_reader_rejects_terminal_appended_after_full_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-full-terminal.cfe"
    metadata = CheckpointMetadata(
        task_id="duplicate-full-terminal",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(20):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(
                    f"candidate-{index}",
                    (float(index), float(20 - index)),
                ),
            )
        writer.finish_event()

    raw = path.read_bytes()
    record_size = (
        checkpoint_data._RECORD_LENGTH.size
        + checkpoint_data._RECORD_FIXED.size
        + metadata.archive_capacity
        * metadata.objective_dimension
        * 8
    )
    last = raw[-record_size:]
    length_prefix = last[: checkpoint_data._RECORD_LENGTH.size]
    fixed_start = checkpoint_data._RECORD_LENGTH.size
    fixed_end = fixed_start + checkpoint_data._RECORD_FIXED.size
    values = list(
        checkpoint_data._RECORD_FIXED.unpack(
            last[fixed_start:fixed_end]
        )
    )
    values[1] = checkpoint_data._KIND_TERMINAL
    values[3] = checkpoint_data._TERMINAL_INDEX
    duplicate_terminal = (
        length_prefix
        + checkpoint_data._RECORD_FIXED.pack(*values)
        + last[fixed_end:]
    )
    path.write_bytes(raw + duplicate_terminal)
    with pytest.raises(
        CheckpointDataError,
        match="full event cannot append",
    ):
        read_checkpoint_file(path)


def test_chain_commits_nonpersisted_vector_and_failure_details(
    tmp_path: Path,
) -> None:
    def write(
        path: Path,
        *,
        vector: tuple[float, ...],
        reason: str,
    ) -> str:
        metadata = CheckpointMetadata(
            task_id="chain-commitment",
            objective_names=("objective_0", "objective_1"),
        )
        with TaskCheckpointWriter(path, metadata) as writer:
            writer.begin_event(event_id=0, cfe_budget=20)
            writer.record_success(
                event_id=0,
                vector=vector,
                result=_result("same-success", (1.0, 2.0)),
            )
            writer.record_failure(
                event_id=0,
                candidate_id="same-failure",
                vector=(3.0,),
                error_type="Failure",
                reason=reason,
            )
            writer.finish_event(terminal_snapshot=True)
        return read_checkpoint_file(path).records[-1].evaluation_chain_sha256

    first = write(
        tmp_path / "first.cfe",
        vector=(1.0,),
        reason="first",
    )
    second = write(
        tmp_path / "second.cfe",
        vector=(2.0,),
        reason="second",
    )
    assert first != second


def test_fixed_shape_padding_and_corruption_are_strictly_validated(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean.cfe"
    metadata = CheckpointMetadata(
        task_id="padding",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(clean, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.record_success(
            event_id=0,
            vector=(0.0,),
            result=_result("only-point", (4.0, 5.0)),
        )
        writer.finish_event(terminal_snapshot=True)
    decoded = read_checkpoint_file(clean)
    assert decoded.records[-1].valid_count == 1

    corrupt_padding = tmp_path / "corrupt-padding.cfe"
    payload = bytearray(clean.read_bytes())
    payload[-1] = 1
    corrupt_padding.write_bytes(payload)
    with pytest.raises(CheckpointDataError, match="padding"):
        read_checkpoint_file(corrupt_padding)

    corrupt_magic = tmp_path / "corrupt-magic.cfe"
    payload = bytearray(clean.read_bytes())
    payload[0] ^= 0xFF
    corrupt_magic.write_bytes(payload)
    with pytest.raises(CheckpointDataError, match="magic"):
        read_checkpoint_file(corrupt_magic)

    truncated = tmp_path / "truncated.cfe"
    truncated.write_bytes(clean.read_bytes()[:-1])
    with pytest.raises(CheckpointDataError, match="truncated"):
        read_checkpoint_file(truncated)


def test_full_event_rejects_duplicate_terminal_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full-terminal.cfe"
    metadata = CheckpointMetadata(
        task_id="full-terminal",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(20):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(
                    f"candidate-{index}",
                    (float(index), float(20 - index)),
                ),
            )
        with pytest.raises(
            CheckpointDataError,
            match="checkpoint 20",
        ):
            writer.finish_event(terminal_snapshot=True)
        writer.finish_event()
    decoded = read_checkpoint_file(path)
    assert len(decoded.records) == CHECKPOINTS_PER_EVENT
    assert decoded.records[-1].checkpoint_index == 20
    assert decoded.records[-1].kind == "checkpoint"
    assert decoded.records[-1].cfe == 20


def test_storage_estimate_is_exact_and_below_seven_gib() -> None:
    estimate = estimate_e1e2_checkpoint_storage()
    assert estimate.task_count == 5030
    assert estimate.event_count == 162640
    assert estimate.checkpoint_record_count == 3_415_440
    assert estimate.objective_payload_bytes == 6_219_360_000
    assert estimate.max_constraint_payload_bytes == 27_323_520
    assert estimate.conservative_total_upper_bound_bytes < 7 * 1024**3
    assert estimate.conservative_total_upper_bound_gib < 7.0


def test_global_constraint_witness_is_nonpositive_iff_front_is_feasible() -> None:
    feasible = (
        _result("feasible-a", (1.0, 2.0), constraint=-0.25),
        _result("feasible-b", (2.0, 1.0), constraint=0.0),
    )
    mixed_constraints = EvaluationResult(
        candidate_id="infeasible",
        objectives=(1.5, 1.5),
        objective_names=("objective_0", "objective_1"),
        constraints=(-5.0, 0.125, -0.5),
        constraint_names=("constraint_0", "constraint_1", "constraint_2"),
    )
    infeasible = feasible + (mixed_constraints,)

    assert front_max_constraint(()) == 0.0
    assert (front_max_constraint(feasible) <= 0.0) is all(
        result.feasible for result in feasible
    )
    assert (front_max_constraint(infeasible) <= 0.0) is all(
        result.feasible for result in infeasible
    )


def test_format_has_no_paper_endpoint_fields(tmp_path: Path) -> None:
    path = tmp_path / "blind.cfe"
    metadata = CheckpointMetadata(
        task_id="blind",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.finish_event(terminal_snapshot=True)

    record_fields = set(read_checkpoint_file(path).records[0].__dataclass_fields__)
    forbidden = {
        "nhv",
        "auc",
        "hypervolume",
        "p_value",
        "confidence_interval",
    }
    assert record_fields.isdisjoint(forbidden)
    lower_bytes = path.read_bytes().lower()
    assert all(field.encode("ascii") not in lower_bytes for field in forbidden)
