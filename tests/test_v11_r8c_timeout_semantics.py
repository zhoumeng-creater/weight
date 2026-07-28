from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.evaluator import (
    BatchEvaluationUnavailableBeforeEntry,
    ExecutionTimeoutBeforeEntry,
)
from evaluation.ledger import EvaluationLedger
from formal_execution import runtime
from formal_execution.checkpoint_data import read_checkpoint_file
from formal_execution.runtime import (
    CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    EndpointCheckpointWriter,
    FormalRuntimeError,
    RecordingAdapter,
)
from formal_execution.schedule import FormalSequenceSpec


def _spec(
    *,
    workload_id: str = "E1_STATIC",
    method_id: str = "MATCHED_FIXED_DE_PARETO",
) -> FormalSequenceSpec:
    return FormalSequenceSpec(
        schedule_index=0,
        workload_id=workload_id,
        unit_id="LIRCMOP1",
        method_id=method_id,
        replicate_index=0,
        master_seed_u64="20260726",
        events=1,
        cfe_per_event=200,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=1,
        problem_id="LIRCMOP1",
        task_namespace="r8c",
    )


@pytest.mark.parametrize(
    ("workload_id", "expected_seconds"),
    [
        ("E1_STATIC", 1800.0),
        ("E1_DYNAMIC", 300.0),
        ("E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE", 300.0),
        ("E1_ROLLING", 120.0),
        ("E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE", 120.0),
    ],
)
def test_f23_event_deadlines_are_not_r5_sequence_hard_caps(
    workload_id: str,
    expected_seconds: float,
) -> None:
    spec = SimpleNamespace(
        workload_id=workload_id,
        timeout_seconds=21_600,
    )
    assert runtime._scientific_event_deadline_seconds(
        spec,
        CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    ) == expected_seconds
    assert spec.timeout_seconds != expected_seconds


def test_task_timeout_has_priority_and_is_not_scientific_reject(
    tmp_path: Path,
) -> None:
    spec = _spec()
    base_problem = runtime.build_problem(
        spec,
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    checkpoint_path = tmp_path / "partial.cfe"
    timeout_marker = tmp_path / "TASK_TIMEOUT_REQUESTED"
    now = [0.0]
    with EndpointCheckpointWriter(
        checkpoint_path,
        spec.task_id,
        base_problem.objective_names,
    ) as writer:
        problem = RecordingAdapter(
            base_problem,
            writer,
            timeout_marker,
            clock=lambda: now[0],
        )
        problem.begin_event(
            event_id=0,
            cfe_budget=spec.cfe_per_event,
            scientific_deadline_seconds=1.0,
        )
        now[0] = 1.0
        timeout_marker.write_text("TASK_TIMEOUT\n", encoding="utf-8")
        with pytest.raises(ExecutionTimeoutBeforeEntry):
            problem.evaluate(
                [0.5] * base_problem.decision_dimension,
                0,
                EvaluationLedger(max_cfe=spec.cfe_per_event),
                "technical-timeout",
            )

    decoded = read_checkpoint_file(checkpoint_path)
    assert decoded.records[-1].kind == "terminal"
    assert decoded.records[-1].cfe == 0


def test_batch_guard_falls_back_before_scientific_deadline(
    tmp_path: Path,
) -> None:
    spec = _spec()
    base_problem = runtime.build_problem(
        spec,
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    now = [0.0]
    with EndpointCheckpointWriter(
        tmp_path / "batch.cfe",
        spec.task_id,
        base_problem.objective_names,
    ) as writer:
        problem = RecordingAdapter(
            base_problem,
            writer,
            clock=lambda: now[0],
        )
        problem.begin_event(
            event_id=0,
            cfe_budget=spec.cfe_per_event,
            scientific_deadline_seconds=10.0,
        )
        now[0] = 6.0
        with pytest.raises(
            BatchEvaluationUnavailableBeforeEntry,
            match="deadline guard",
        ):
            problem.evaluate_batch(
                [[0.5] * base_problem.decision_dimension],
                0,
                EvaluationLedger(max_cfe=spec.cfe_per_event),
                ["batch-guard"],
            )


def test_checkpoint_begin_does_not_consume_scientific_deadline() -> None:
    now = [0.0]

    class SlowCheckpointWriter:
        def begin_event(self, *, event_id: int, cfe_budget: int) -> None:
            del event_id, cfe_budget
            now[0] = 40.0

    problem = RecordingAdapter(
        SimpleNamespace(),
        SlowCheckpointWriter(),
        clock=lambda: now[0],
    )
    problem.begin_event(
        event_id=0,
        cfe_budget=200,
        scientific_deadline_seconds=10.0,
    )
    now[0] = 49.999
    assert problem.scientific_event_deadline_reached(0) is False
    now[0] = 50.0
    assert problem.scientific_event_deadline_reached(0) is True
    with pytest.raises(FormalRuntimeError, match="more than once"):
        problem.start_scientific_deadline(event_id=0, seconds=10.0)


def test_common_comparator_event_timeout_keeps_partial_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    base_problem = runtime.build_problem(
        spec,
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    now = [0.0]
    checkpoint_path = tmp_path / "comparator-timeout.cfe"

    class TimeoutComparator:
        def optimize(
            self,
            problem,
            *,
            event_id,
            budget,
            seed,
            ledger,
            initialization_vectors,
        ):
            assert now[0] == 25.0
            assert problem.scientific_event_deadline_reached(event_id) is False
            problem.evaluate(
                initialization_vectors[0],
                event_id,
                ledger,
                "first-valid",
            )
            now[0] = 1825.0
            problem.evaluate(
                initialization_vectors[1],
                event_id,
                ledger,
                "must-not-enter",
            )
            raise AssertionError("deadline must stop before the second CFE")

        def identity(self):
            return {"method_id": "TIMEOUT_FIXTURE"}

    monkeypatch.setattr(
        runtime,
        "_comparator",
        lambda method_id, settings: TimeoutComparator(),
    )
    original_initialization = runtime._shared_initialization

    def delayed_initialization(*args, **kwargs):
        values = original_initialization(*args, **kwargs)
        now[0] = 25.0
        return values

    monkeypatch.setattr(
        runtime,
        "_shared_initialization",
        delayed_initialization,
    )
    with EndpointCheckpointWriter(
        checkpoint_path,
        spec.task_id,
        base_problem.objective_names,
    ) as writer:
        problem = RecordingAdapter(
            base_problem,
            writer,
            clock=lambda: now[0],
        )
        events, status, _ = runtime._run_comparator(
            problem,
            spec,
            CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
            tmp_path / "STOP",
            tmp_path / "heartbeat",
        )
        assert writer.count == 1

    assert status == "COMPLETE"
    assert events[0]["terminal"]["code"] == "REJECT_TIMEOUT"
    assert events[0]["ledger"]["cfe"] == 1
    assert isinstance(events[0]["terminal"]["candidate_available"], bool)
    decoded = read_checkpoint_file(checkpoint_path)
    assert decoded.records[-1].kind == "terminal"
    assert decoded.records[-1].cfe == 1


def test_compact_task_summary_binds_both_timeout_layers(
    tmp_path: Path,
) -> None:
    spec = _spec()
    task_directory = tmp_path / "task"
    request = SimpleNamespace()
    monkey_request = request
    # The DT-RAMDE request is validated when the engine is constructed, so use
    # the comparator path for this summary-only binding test.
    runtime.run_task(
        spec=spec,
        request=monkey_request,
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    summary = json.loads(
        (task_directory / "task_summary.json").read_text(encoding="utf-8")
    )
    assert summary["timeout_semantics"] == {
        "scientific_event_deadline_seconds": 1800.0,
        "technical_sequence_hard_ceiling_seconds": 3600,
        "scientific_event_terminal": "REJECT_TIMEOUT",
        "technical_timeout_algorithm_terminal": None,
    }
