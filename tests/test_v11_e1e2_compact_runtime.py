from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analysis import (
    AnalyticReferenceScale,
    CheckpointAnalysisError,
    NumericalContinuousEndpointExcluded,
    read_manifest_bound_complete_task_nhv,
)
from dt_ramde_v11.contracts import (
    ExecutionScope,
    R8CCorrectiveContractBindings,
    R8CCorrectiveExecutionRequest,
)
from evaluation.evaluator import (
    BatchEvaluationUnavailableBeforeEntry,
    ExecutionTimeoutBeforeEntry,
)
from evaluation.ledger import EvaluationLedger
from formal_execution import runtime as runtime_module
from formal_execution.adapters import FormalR8CStaticAdapter
from formal_execution.checkpoint_data import read_checkpoint_file
from formal_execution.public_rolling import generate_public_instance
from formal_execution.runtime import (
    CHECKPOINT_FRONT_PERSISTENCE,
    CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    FormalRuntimeError,
    RecordingAdapter,
    _event_summary,
    _validate_task_accounting,
    _write_canonical_json_exclusive_fsynced,
    run_task,
)
from formal_execution.schedule import FormalSequenceSpec


def _request() -> R8CCorrectiveExecutionRequest:
    command = "test-only-e1e2-command"
    return R8CCorrectiveExecutionRequest(
        scope=ExecutionScope.BENCHMARK_EFFECT,
        companion_scope=ExecutionScope.BENCHMARK_EFFECT,
        contracts=R8CCorrectiveContractBindings(
            protocol_id="WGT-JOURNAL-2026-01",
            r5_contract_id=(
                "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
            ),
            r5_contract_sha256=(
                "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
            ),
            r5a_contract_id="WGT-V11-R5A-E3-INPUT-CONTRACT-01",
            r5a_contract_sha256=(
                "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
            ),
            corrective_protocol_id=(
                "WGT-V11-R8C-RESULT-BLIND-CORRECTIVE-PROTOCOL-01"
            ),
            corrective_protocol_sha256=(
                "dfe74d041f36b12fd13cb86e1fa2bba5483bbd871a7749b2c98e09160ee39b43"
            ),
            r8c_formal_contract_id=(
                "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
                "FORMAL-EXECUTION-CONTRACT-01"
            ),
            r8c_formal_contract_sha256="a" * 64,
            formal_schedule_id="WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01",
            formal_schedule_sha256=(
                "db468253fb1430749d9f816d19532e428ca1054a86f399f80b12575a5c45282d"
            ),
            source_git_commit="b" * 40,
            source_git_tree="c" * 40,
        ),
        request_id=(
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "EXECUTION-REQUEST-20260726-01"
        ),
        frozen_exact_command=command,
        author_confirmation_text=command,
        author_exact_command_confirmed=True,
    )


def _spec() -> FormalSequenceSpec:
    return FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="MATCHED_FIXED_DE_PARETO",
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


def _cdf9_spec(method_id: str) -> FormalSequenceSpec:
    return FormalSequenceSpec(
        schedule_index=9,
        workload_id="E1_DYNAMIC",
        unit_id="CDF9/CDF-HARSH",
        method_id=method_id,
        replicate_index=0,
        master_seed_u64="20260726",
        events=6,
        cfe_per_event=100,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=9,
        problem_id="CDF9",
        profile="CDF-HARSH",
        task_namespace="r8c",
    )


def _rolling_no_feedback_spec() -> FormalSequenceSpec:
    instance = generate_public_instance("RR-SMOOTH", 0)
    return FormalSequenceSpec(
        schedule_index=10,
        workload_id="E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
        unit_id="RR-SMOOTH/0",
        method_id="NO_EXECUTION_FEEDBACK",
        replicate_index=0,
        master_seed_u64="20260726",
        events=2,
        cfe_per_event=200,
        atomic_steps_per_cfe=6,
        timeout_seconds=3600,
        rolling_template="RR-SMOOTH",
        rolling_index=0,
        rolling_seed_u64=str(instance["derived_seed_u64"]),
        reused_full_workload_id="E1_ROLLING",
        task_namespace="r8c",
    )


@pytest.mark.parametrize(
    "filename",
    ["task_summary.json", "task_manifest.json"],
)
def test_control_json_is_exclusive_canonical_and_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    fsynced: list[int] = []
    monkeypatch.setattr(
        runtime_module.os,
        "fsync",
        lambda descriptor: fsynced.append(descriptor),
    )
    path = tmp_path / filename
    value = {"z": 2, "a": 1}

    _write_canonical_json_exclusive_fsynced(path, value)

    assert path.read_bytes() == b'{"a":1,"z":2}\n'
    assert len(fsynced) == 1
    with pytest.raises(FileExistsError):
        _write_canonical_json_exclusive_fsynced(path, {"replacement": True})
    assert path.read_bytes() == b'{"a":1,"z":2}\n'

    oversize_path = tmp_path / f"oversize-{filename}"
    with pytest.raises(FormalRuntimeError, match="frozen byte bound"):
        _write_canonical_json_exclusive_fsynced(
            oversize_path,
            value,
            maximum_bytes=4,
        )
    assert not oversize_path.exists()


def test_e1e2_runtime_persists_only_endpoint_sufficient_checkpoints(
    tmp_path: Path,
) -> None:
    task_directory = tmp_path / "compact"
    result = run_task(
        spec=_spec(),
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    assert result["status"] == "COMPLETE"
    assert (
        CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS.persistence_mode
        == CHECKPOINT_FRONT_PERSISTENCE
    )
    assert not (task_directory / "raw_evaluations.jsonl.gz").exists()
    checkpoint_path = task_directory / "checkpoint_fronts.cfe"
    decoded = read_checkpoint_file(checkpoint_path)
    assert decoded.metadata.task_id == _spec().task_id
    assert len(decoded.records) == 21
    assert [record.cfe for record in decoded.records] == list(
        range(0, 201, 10)
    )
    assert decoded.records[-1].success_count == 200
    assert decoded.records[-1].failure_count == 0
    assert all(
        record.front_max_constraint <= 0.0
        for record in decoded.records
    )

    summary_path = task_directory / "task_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    assert summary_bytes == (
        runtime_module.canonical_json_bytes(summary) + b"\n"
    )
    event_summary_path = task_directory / "event_summaries.jsonl"
    event_summary_bytes = event_summary_path.read_bytes()
    assert b"\r\n" not in event_summary_bytes
    assert event_summary_bytes.endswith(b"\n")
    persisted_events = [
        json.loads(line)
        for line in event_summary_bytes.decode("utf-8").splitlines()
    ]
    assert persisted_events == summary["events"]
    assert event_summary_bytes == b"".join(
        runtime_module.canonical_json_bytes(event) + b"\n"
        for event in persisted_events
    )
    assert b'"candidate_id"' not in event_summary_bytes
    assert b'"candidate_ids"' not in event_summary_bytes
    assert b'"nhv"' not in event_summary_bytes.lower()
    assert b'"auc"' not in event_summary_bytes.lower()
    assert summary["charged_evaluation_count"] == 200
    assert summary["individual_evaluation_rows_persisted"] == 0
    assert summary["budget_accounting"] == {
        "scheduled_cfe": 200,
        "charged_cfe": 200,
        "unconsumed_cfe_due_to_typed_terminal": 0,
        "scheduled_atomic_model_steps": 200,
        "charged_atomic_model_steps": 200,
        "typed_short_cfe_event_ids": [],
        "unused_budget_transferred": False,
    }
    assert "reason" in summary["events"][0]["terminal"]
    assert summary["events"][0]["terminal"].keys() == {
        "candidate_available",
        "code",
        "reason",
    }
    assert "candidate_id" not in summary["events"][0]["terminal"]
    assert "raw_evaluation_count" not in summary
    assert (
        summary["checkpoint_data_format"]["effect_endpoint_computed"]
        is False
    )
    assert (
        summary["checkpoint_data_format"][
            "terminal_candidate_identity_persisted"
        ]
        is False
    )
    assert (
        summary["checkpoint_data_format"]["execution_observation_persisted"]
        is True
    )
    assert summary["event_summary_data_format"] == {
        "filename": "event_summaries.jsonl",
        "encoding": "UTF-8 canonical JSONL with LF records",
        "append_scope": "one durable record per completed event",
        "maximum_record_bytes_including_lf": 8192,
        "flush_after_each_event": True,
        "fsync_after_each_event": True,
        "candidate_ids_persisted": False,
        "effect_endpoint_computed": False,
    }
    manifest_path = task_directory / "task_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == (
        runtime_module.canonical_json_bytes(manifest) + b"\n"
    )
    assert set(manifest["artifacts"]) == {
        "checkpoint_fronts.cfe",
        "event_summaries.jsonl",
        "task_summary.json",
    }

    axis = np.linspace(0.0, 1.0, 10_000)
    bound = read_manifest_bound_complete_task_nhv(
        task_directory,
        expected_task=_spec().to_dict(),
        expected_task_manifest_sha256=sha256(
            (task_directory / "task_manifest.json").read_bytes()
        ).hexdigest(),
        mode="STATIC_CDF",
        analytic_reference_scales={
            0: AnalyticReferenceScale.from_reference_front(
                np.column_stack((axis, 1.0 - axis))
            )
        },
    )
    assert bound.task_id == _spec().task_id
    assert bound.task_manifest_sha256 is not None
    assert bound.task_summary_sha256 is not None


def test_sixty_event_success_report_stays_below_worker_control_limit(
    tmp_path: Path,
) -> None:
    spec = replace(
        _cdf9_spec("DT-RAMDE_TS2_FULL"),
        events=60,
        cfe_per_event=100,
    )
    task_directory = tmp_path / "sixty-events"

    run_task(
        spec=spec,
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    summary_path = task_directory / "task_summary.json"
    assert 0 < summary_path.stat().st_size <= 64 * 1024


def test_preexisting_stop_publishes_incomplete_task_summary_and_manifest(
    tmp_path: Path,
) -> None:
    stop_path = tmp_path / "STOP_DISPATCH"
    stop_path.write_text("RESOURCE_CEILING\n", encoding="utf-8")
    task_directory = tmp_path / "resource-stopped-task"

    result = run_task(
        spec=_spec(),
        request=_request(),
        task_directory=task_directory,
        stop_path=stop_path,
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    summary = json.loads(
        (task_directory / "task_summary.json").read_bytes()
    )
    manifest = json.loads(
        (task_directory / "task_manifest.json").read_bytes()
    )
    assert result["status"] == "INCOMPLETE_RESOURCE_CEILING"
    assert summary["status"] == "INCOMPLETE_RESOURCE_CEILING"
    assert summary["events"] == []
    assert manifest["status"] == "INCOMPLETE_RESOURCE_CEILING"


def test_no_feedback_ablation_persists_independent_execution_observation(
    tmp_path: Path,
) -> None:
    spec = _rolling_no_feedback_spec()
    task_directory = tmp_path / "no-feedback"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    summary = json.loads(
        (task_directory / "task_summary.json").read_text(encoding="utf-8")
    )
    assert [
        event["execution_feedback"] for event in summary["events"]
    ] == [None, None]
    observations = [
        event["execution_observation"] for event in summary["events"]
    ]
    assert any(observation["available"] for observation in observations)
    available_observations = [
        observation
        for observation in observations
        if observation["available"]
    ]
    assert all(
        isinstance(observation["ell_exec"], float)
        and isinstance(observation["hard_constraint_violation"], bool)
        for observation in available_observations
    )

    bound = read_manifest_bound_complete_task_nhv(
        task_directory,
        expected_task=spec.to_dict(),
        expected_task_manifest_sha256=sha256(
            (task_directory / "task_manifest.json").read_bytes()
        ).hexdigest(),
        mode="ROLLING",
    )
    assert bound.execution_observations == tuple(observations)


def test_durable_event_summary_writer_rejects_nested_schema_drift(
    tmp_path: Path,
) -> None:
    observation = {
        "available": True,
        "ell_exec": 1.0,
        "ell_ref": 1.0,
        "s_exec": 1.0,
        "hard_constraint_violation": False,
        "released_at": 1,
    }
    valid = _event_summary(
        event_id=0,
        terminal_code="ACCEPTED",
        terminal_candidate_id="candidate-0",
        terminal_reason="fixture",
        ledger={
            "cfe": 20,
            "objective_calls": 20,
            "constraint_calls": 20,
            "scenario_evaluations": 20,
            "atomic_model_steps": 20,
            "execution_transition_count": 1,
            "repair_failed": 0,
            "evaluation_failures": 0,
        },
        evaluation_failure_type_counts={},
        information_hash="a" * 64,
        feedback=observation,
        execution_observation=observation,
        compact=True,
    )
    invalid_values: dict[str, dict[str, object]] = {}

    terminal_extra = deepcopy(valid)
    terminal_extra["terminal"]["candidate_id"] = "forbidden"
    invalid_values["terminal-extra"] = terminal_extra

    ledger_extra = deepcopy(valid)
    ledger_extra["ledger"]["objective_front"] = []
    invalid_values["ledger-extra"] = ledger_extra

    failure_mismatch = deepcopy(valid)
    failure_mismatch["evaluation_failure_type_counts"] = {
        "SyntheticFailure": 1
    }
    invalid_values["failure-mismatch"] = failure_mismatch

    feedback_vector = deepcopy(valid)
    feedback_vector["execution_feedback"]["decision_vector"] = [0.1]
    invalid_values["feedback-vector"] = feedback_vector

    observation_effect = deepcopy(valid)
    observation_effect["execution_observation"]["effect_size"] = 0.5
    invalid_values["observation-effect"] = observation_effect

    for name, invalid in invalid_values.items():
        path = tmp_path / f"{name}.jsonl"
        with runtime_module.DurableEventSummaryWriter(
            path,
            event_count=1,
            cfe_per_event=20,
            atomic_steps_per_cfe=1,
        ) as writer:
            with pytest.raises(FormalRuntimeError, match="compact"):
                writer.append(invalid)
        assert path.read_bytes() == b""


def test_durable_event_summary_writer_rejects_oversize_before_write(
    tmp_path: Path,
) -> None:
    valid = _event_summary(
        event_id=0,
        terminal_code="ACCEPTED",
        terminal_candidate_id="candidate-0",
        terminal_reason=(
            "X" * runtime_module.EVENT_SUMMARY_MAX_RECORD_BYTES
        ),
        ledger={
            "cfe": 20,
            "objective_calls": 20,
            "constraint_calls": 20,
            "scenario_evaluations": 20,
            "atomic_model_steps": 20,
            "execution_transition_count": 0,
            "repair_failed": 0,
            "evaluation_failures": 0,
        },
        evaluation_failure_type_counts={},
        information_hash="a" * 64,
        feedback=None,
        execution_observation=None,
        compact=True,
    )
    path = tmp_path / "oversize.jsonl"
    with runtime_module.DurableEventSummaryWriter(
        path,
        event_count=1,
        cfe_per_event=20,
        atomic_steps_per_cfe=1,
    ) as writer:
        with pytest.raises(FormalRuntimeError, match="frozen byte bound"):
            writer.append(valid)
    assert path.read_bytes() == b""


def test_compact_checkpoint_file_is_batch_scalar_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: 123.0)
    monkeypatch.setattr(runtime_module.time, "process_time", lambda: 45.0)
    batch_dir = tmp_path / "batch"
    scalar_dir = tmp_path / "scalar"
    run_task(
        spec=_spec(),
        request=_request(),
        task_directory=batch_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    def unavailable(self, vectors, event_id, ledger, candidate_ids):
        del self, vectors, event_id, ledger, candidate_ids
        raise BatchEvaluationUnavailableBeforeEntry(
            "forced scalar reference"
        )

    monkeypatch.setattr(
        FormalR8CStaticAdapter,
        "evaluate_batch",
        unavailable,
    )
    run_task(
        spec=_spec(),
        request=_request(),
        task_directory=scalar_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    for name in (
        "checkpoint_fronts.cfe",
        "event_summaries.jsonl",
        "task_summary.json",
        "task_manifest.json",
    ):
        assert (batch_dir / name).read_bytes() == (
            scalar_dir / name
        ).read_bytes()


@pytest.mark.parametrize(
    "spec",
    [
        replace(_spec(), method_id="F22_MG_STATIC"),
        replace(
            _cdf9_spec("DT-RAMDE_TS2_FULL"),
            events=2,
            cfe_per_event=200,
        ),
        replace(
            _rolling_no_feedback_spec(),
            workload_id="E1_ROLLING",
            method_id="DT-RAMDE_TS2_FULL",
            reused_full_workload_id=None,
        ),
    ],
    ids=["static", "cdf", "rolling"],
)
def test_compact_engine_is_scientifically_exact_to_full_audit_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: FormalSequenceSpec,
) -> None:
    compact_events: list[Any] = []
    reference_events: list[Any] = []
    original_run_event = runtime_module.DTRAMDE.run_event
    active_capture = compact_events

    def capture_run_event(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        event = original_run_event(self, *args, **kwargs)
        active_capture.append(event)
        return event

    monkeypatch.setattr(
        runtime_module.DTRAMDE,
        "run_event",
        capture_run_event,
    )
    compact_dir = tmp_path / "compact-engine"
    reference_dir = tmp_path / "full-audit-reference"
    run_task(
        spec=spec,
        request=_request(),
        task_directory=compact_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    active_capture = reference_events
    monkeypatch.setattr(
        runtime_module,
        "COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION",
        "full",
    )
    run_task(
        spec=spec,
        request=_request(),
        task_directory=reference_dir,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    assert (compact_dir / "checkpoint_fronts.cfe").read_bytes() == (
        reference_dir / "checkpoint_fronts.cfe"
    ).read_bytes()
    assert (compact_dir / "event_summaries.jsonl").read_bytes() == (
        reference_dir / "event_summaries.jsonl"
    ).read_bytes()
    assert len(compact_events) == len(reference_events) == spec.events
    for compact, reference in zip(
        compact_events,
        reference_events,
        strict=True,
    ):
        assert compact.terminal == reference.terminal
        assert compact.ledger == reference.ledger
        assert compact.archive == reference.archive
        assert compact.execution_feedback == reference.execution_feedback
        assert compact.mg_final == reference.mg_final
        assert (
            compact.warm_start_seed_count
            == reference.warm_start_seed_count
        )
        assert compact.trial_audit == ()
        assert compact.initialization_audit == {}
        assert compact.lineage_records == ()
        assert compact.archive_audit == ()
        assert reference.trial_audit
        assert reference.initialization_audit
        assert reference.lineage_records
        assert reference.archive_audit


def test_completed_event_summary_survives_abnormal_task_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_finish_event = RecordingAdapter.finish_event
    original_fsync = runtime_module.os.fsync
    fsync_calls: list[int] = []

    def tracked_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        original_fsync(file_descriptor)

    def fail_after_checkpoint_finalization(
        self: RecordingAdapter,
        *,
        terminal_snapshot: bool = False,
    ) -> None:
        original_finish_event(
            self,
            terminal_snapshot=terminal_snapshot,
        )
        raise RuntimeError("synthetic interruption after completed event")

    monkeypatch.setattr(runtime_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(
        RecordingAdapter,
        "finish_event",
        fail_after_checkpoint_finalization,
    )
    task_directory = tmp_path / "interrupted"
    with pytest.raises(
        RuntimeError,
        match="synthetic interruption",
    ):
        run_task(
            spec=_spec(),
            request=_request(),
            task_directory=task_directory,
            stop_path=tmp_path / "STOP",
            settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
        )

    event_summary_path = task_directory / "event_summaries.jsonl"
    payload = event_summary_path.read_bytes()
    assert payload.count(b"\n") == 1
    assert payload.endswith(b"\n")
    event = json.loads(payload)
    assert payload == runtime_module.canonical_json_bytes(event) + b"\n"
    assert event["event_id"] == 0
    assert event["ledger"]["cfe"] == _spec().cfe_per_event
    assert "candidate_id" not in event["terminal"]
    assert len(runtime_module.file_sha256(event_summary_path)) == 64
    assert len(fsync_calls) >= 2
    assert not (task_directory / "task_summary.json").exists()
    assert not (task_directory / "task_manifest.json").exists()


@pytest.mark.parametrize(
    "method_id",
    ["DT-RAMDE_TS2_FULL", "MATCHED_FIXED_DE_PARETO"],
)
def test_cdf9_compact_summary_retains_original_failure_type_counts(
    tmp_path: Path,
    method_id: str,
) -> None:
    task_directory = tmp_path / method_id
    spec = _cdf9_spec(method_id)
    run_task(
        spec=spec,
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )

    summary = json.loads(
        (task_directory / "task_summary.json").read_text(encoding="utf-8")
    )
    failure_events = [
        event
        for event in summary["events"]
        if event["evaluation_failure_type_counts"]
    ]
    assert failure_events
    assert any(
        event["evaluation_failure_type_counts"].get(
            "CDFDomainUndefinedError",
            0,
        )
        > 0
        for event in failure_events
    )
    assert all(
        sum(event["evaluation_failure_type_counts"].values())
        == event["ledger"]["evaluation_failures"]
        for event in summary["events"]
    )
    assert all(
        set(event["evaluation_failure_type_counts"])
        <= {"CDFDomainUndefinedError"}
        for event in summary["events"]
    )
    if method_id == "DT-RAMDE_TS2_FULL":
        assert any(
            event["terminal"]["code"] == "REJECT_NUMERICAL"
            for event in failure_events
        )
    axis = np.linspace(0.0, 1.0, 10_000)
    with pytest.raises(
        NumericalContinuousEndpointExcluded,
        match="numerical",
    ):
        read_manifest_bound_complete_task_nhv(
            task_directory,
            expected_task=spec.to_dict(),
            expected_task_manifest_sha256=sha256(
                (task_directory / "task_manifest.json").read_bytes()
            ).hexdigest(),
            mode="STATIC_CDF",
            analytic_reference_scales={
                event_id: AnalyticReferenceScale.from_reference_front(
                    np.column_stack((axis, 1.0 - axis))
                )
                for event_id in range(spec.events)
            },
        )


def test_event_summary_rejects_failure_type_count_ledger_mismatch() -> None:
    with pytest.raises(FormalRuntimeError, match="failure type counts"):
        _event_summary(
            event_id=0,
            terminal_code="REJECT_NUMERICAL",
            terminal_candidate_id=None,
            terminal_reason=None,
            ledger={
                "cfe": 1,
                "atomic_model_steps": 1,
                "evaluation_failures": 1,
            },
            evaluation_failure_type_counts={},
            information_hash="a" * 64,
            feedback=None,
        )


def test_legacy_event_summary_retains_candidate_identity_schema() -> None:
    summary = _event_summary(
        event_id=0,
        terminal_code="ACCEPTED",
        terminal_candidate_id="legacy-candidate",
        terminal_reason=None,
        ledger={
            "cfe": 1,
            "atomic_model_steps": 1,
            "evaluation_failures": 0,
        },
        evaluation_failure_type_counts={},
        information_hash="a" * 64,
        feedback=None,
    )
    assert summary["terminal"] == {
        "candidate_id": "legacy-candidate",
        "code": "ACCEPTED",
        "reason": None,
    }
    assert "execution_observation" not in summary


def test_r9_manifest_join_rejects_schedule_substitution(
    tmp_path: Path,
) -> None:
    task_directory = tmp_path / "bound"
    run_task(
        spec=_spec(),
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    manifest_sha256 = sha256(
        (task_directory / "task_manifest.json").read_bytes()
    ).hexdigest()
    substituted = _spec().to_dict()
    substituted["method_id"] = "MATCHED_JDE_STYLE_PARETO"
    axis = np.linspace(0.0, 1.0, 10_000)
    with pytest.raises(CheckpointAnalysisError, match="identity"):
        read_manifest_bound_complete_task_nhv(
            task_directory,
            expected_task=substituted,
            expected_task_manifest_sha256=manifest_sha256,
            mode="STATIC_CDF",
            analytic_reference_scales={
                0: AnalyticReferenceScale.from_reference_front(
                    np.column_stack((axis, 1.0 - axis))
                )
            },
        )


def test_r9_manifest_join_rejects_failure_type_ledger_mismatch(
    tmp_path: Path,
) -> None:
    task_directory = tmp_path / "failure-count-mismatch"
    spec = _spec()
    run_task(
        spec=spec,
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    summary_path = task_directory / "task_summary.json"
    manifest_path = task_directory / "task_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["events"][0]["evaluation_failure_type_counts"] = {
        "SyntheticNumericalFailure": 1
    }
    summary_path.write_bytes(
        runtime_module.canonical_json_bytes(summary) + b"\n"
    )
    event_summary_path = task_directory / "event_summaries.jsonl"
    event_summary_path.write_bytes(
        b"".join(
            runtime_module.canonical_json_bytes(event) + b"\n"
            for event in summary["events"]
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["task_summary.json"] = {
        "bytes": summary_path.stat().st_size,
        "sha256": sha256(summary_path.read_bytes()).hexdigest(),
    }
    manifest["artifacts"]["event_summaries.jsonl"] = {
        "bytes": event_summary_path.stat().st_size,
        "sha256": sha256(event_summary_path.read_bytes()).hexdigest(),
    }
    manifest["task_binding_sha256"] = sha256(
        runtime_module.canonical_json_bytes(
            {
                "task": spec.to_dict(),
                "artifacts": manifest["artifacts"],
            }
        )
    ).hexdigest()
    manifest_path.write_bytes(
        runtime_module.canonical_json_bytes(manifest) + b"\n"
    )

    axis = np.linspace(0.0, 1.0, 10_000)
    with pytest.raises(
        CheckpointAnalysisError,
        match="failure-type counts differ",
    ):
        read_manifest_bound_complete_task_nhv(
            task_directory,
            expected_task=spec.to_dict(),
            expected_task_manifest_sha256=sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            mode="STATIC_CDF",
            analytic_reference_scales={
                0: AnalyticReferenceScale.from_reference_front(
                    np.column_stack((axis, 1.0 - axis))
                )
            },
        )


def test_r9_manifest_join_rejects_checkpoint_budget_substitution(
    tmp_path: Path,
) -> None:
    task_directory = tmp_path / "checkpoint-budget-substitution"
    spec = _spec()
    run_task(
        spec=spec,
        request=_request(),
        task_directory=task_directory,
        stop_path=tmp_path / "STOP",
        settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
    )
    summary_path = task_directory / "task_summary.json"
    manifest_path = task_directory / "task_manifest.json"
    substituted = spec.to_dict()
    substituted["cfe_per_event"] = 400
    substituted["total_cfe"] = 400
    substituted["total_atomic_steps"] = 400

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["task"] = substituted
    summary_path.write_bytes(
        runtime_module.canonical_json_bytes(summary) + b"\n"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["task_summary.json"] = {
        "bytes": summary_path.stat().st_size,
        "sha256": sha256(summary_path.read_bytes()).hexdigest(),
    }
    manifest["task_binding_sha256"] = sha256(
        runtime_module.canonical_json_bytes(
            {
                "task": substituted,
                "artifacts": manifest["artifacts"],
            }
        )
    ).hexdigest()
    manifest_path.write_bytes(
        runtime_module.canonical_json_bytes(manifest) + b"\n"
    )

    axis = np.linspace(0.0, 1.0, 10_000)
    with pytest.raises(CheckpointAnalysisError, match="CFE budget"):
        read_manifest_bound_complete_task_nhv(
            task_directory,
            expected_task=substituted,
            expected_task_manifest_sha256=sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            mode="STATIC_CDF",
            analytic_reference_scales={
                0: AnalyticReferenceScale.from_reference_front(
                    np.column_stack((axis, 1.0 - axis))
                )
            },
        )


def _accounting_event(
    *,
    event_id: int,
    cfe: int,
    terminal_code: str,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "terminal": {
            "code": terminal_code,
            "candidate_id": None,
            "reason": "fixture",
        },
        "ledger": {
            "cfe": cfe,
            "atomic_model_steps": cfe,
        },
    }


@pytest.mark.parametrize(
    "terminal_code",
    ["REJECT_NUMERICAL", "REJECT_TIMEOUT"],
)
def test_typed_short_cfe_is_a_completed_method_outcome(
    terminal_code: str,
) -> None:
    spec = _spec()
    total_cfe, total_atomic, shortfall_events = _validate_task_accounting(
        spec=spec,
        status="COMPLETE",
        events=[
            _accounting_event(
                event_id=0,
                cfe=73,
                terminal_code=terminal_code,
            )
        ],
        recorded_count=73,
    )
    assert (total_cfe, total_atomic, shortfall_events) == (73, 73, (0,))


def test_untyped_short_cfe_still_fails_closed() -> None:
    with pytest.raises(FormalRuntimeError, match="short-CFE"):
        _validate_task_accounting(
            spec=_spec(),
            status="COMPLETE",
            events=[
                _accounting_event(
                    event_id=0,
                    cfe=73,
                    terminal_code="ACCEPTED",
                )
            ],
            recorded_count=73,
        )


def test_task_timeout_marker_stops_before_scalar_or_batch_cfe(
    tmp_path: Path,
) -> None:
    timeout_path = tmp_path / "TASK_TIMEOUT_REQUESTED"
    timeout_path.write_text("TASK_TIMEOUT\n", encoding="utf-8")

    class NeverCalledProblem:
        def evaluate(self, *args, **kwargs):
            raise AssertionError("scalar evaluator must not be entered")

        def evaluate_batch(self, *args, **kwargs):
            raise AssertionError("batch evaluator must not be entered")

    adapter = RecordingAdapter(
        NeverCalledProblem(),
        object(),
        timeout_path,
    )
    ledger = EvaluationLedger(max_cfe=2)
    with pytest.raises(ExecutionTimeoutBeforeEntry, match="timeout"):
        adapter.evaluate((0.0,), 0, ledger, "scalar")
    with pytest.raises(ExecutionTimeoutBeforeEntry, match="timeout"):
        adapter.evaluate_batch(((0.0,),), 0, ledger, ("batch",))
    assert ledger.cfe == 0


def test_recording_adapter_accepts_null_execution_observation() -> None:
    class NullExecutionProblem:
        def execute(self, action, event_id, committed, ledger):
            del action, event_id, committed, ledger
            return None

    adapter = RecordingAdapter(NullExecutionProblem(), object())
    adapter.begin_event(event_id=0, cfe_budget=1)
    ledger = EvaluationLedger(max_cfe=1)
    assert adapter.execute((0.0,), 0, False, ledger) is None
    assert adapter.execution_observation(0) is None
