from __future__ import annotations

import csv
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import pytest

from analysis import r8c_batch_outputs
from analysis.r8c_batch_outputs import (
    R8CFreeze,
    R8CIntegrityError,
    R9ExportAuthorizationError,
    export_r9_readable_outputs,
    validate_r8c_e1e2_run,
)
from evaluation.contracts import EvaluationResult
from formal_execution.checkpoint_data import (
    EVENT_SUMMARY_MAX_RECORD_BYTES,
    CheckpointMetadata,
    TaskCheckpointWriter,
    file_sha256,
)
from formal_execution.schedule import (
    FormalSequenceSpec,
    canonical_json_bytes,
)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _commitment(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _gzip_bytes(payload: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(
        fileobj=target,
        mode="wb",
        filename="",
        mtime=0,
    ) as stream:
        stream.write(payload)
    return target.getvalue()


def _result(index: int) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=f"candidate-{index}",
        objectives=(1.0, 1.0, 1.0),
        objective_names=("cost", "risk", "instability"),
        constraints=(-1.0,),
        constraint_names=("feasible",),
    )


def _synthetic_run(tmp_path: Path) -> tuple[Path, R8CFreeze, str]:
    root = tmp_path / "raw"
    root.mkdir()
    tasks_root = root / "tasks"
    tasks_root.mkdir()
    (root / "worker_logs").mkdir()
    spec = FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_ROLLING",
        unit_id="RR-SMOOTH/0",
        method_id="DT-RAMDE_TS2_FULL",
        replicate_index=0,
        master_seed_u64="1",
        events=1,
        cfe_per_event=20,
        atomic_steps_per_cfe=1,
        timeout_seconds=10,
        rolling_template="RR-SMOOTH",
        rolling_index=0,
        rolling_seed_u64="2",
        task_namespace="r8c",
    )
    row = spec.to_dict()
    schedule_raw = canonical_json_bytes(row) + b"\n"
    schedule_path = root / "schedule.jsonl.gz"
    schedule_path.write_bytes(_gzip_bytes(schedule_raw))
    reuse_raw = canonical_json_bytes({"fixture": True}) + b"\n"
    reuse_path = root / "e2_full_reuse_map.jsonl"
    reuse_path.write_bytes(reuse_raw)
    freeze = R8CFreeze(
        schedule_id="SYNTHETIC-R8C-E1E2",
        schedule_sha256=sha256(schedule_raw).hexdigest(),
        reuse_sha256=sha256(reuse_raw).hexdigest(),
        task_count=1,
        total_cfe=20,
        total_atomic_steps=20,
        reuse_rows=1,
    )

    task_directory = tasks_root / spec.task_id
    task_directory.mkdir()
    checkpoint_path = task_directory / "checkpoint_fronts.cfe"
    with TaskCheckpointWriter(
        checkpoint_path,
        CheckpointMetadata(
            task_id=spec.task_id,
            objective_names=("cost", "risk", "instability"),
        ),
    ) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(20):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(index),
            )
        writer.finish_event()
    event_summary = {
        "event_id": 0,
        "ledger": {
            "cfe": 20,
            "objective_calls": 20,
            "constraint_calls": 20,
            "scenario_evaluations": 20,
            "atomic_model_steps": 20,
            "execution_transition_count": 1,
            "repair_failed": 0,
            "evaluation_failures": 0,
        },
        "evaluation_failure_type_counts": {},
        "terminal": {
            "code": "ACCEPTED",
            "reason": "fixture",
            "candidate_available": True,
        },
        "information_hash": "a" * 64,
        "execution_feedback": None,
        "execution_observation": {
            "available": True,
            "ell_exec": 1.0,
            "ell_ref": 1.0,
            "s_exec": 1.0,
            "hard_constraint_violation": False,
            "released_at": 1,
        },
    }
    event_summaries_path = task_directory / "event_summaries.jsonl"
    event_summaries_path.write_bytes(
        canonical_json_bytes(event_summary) + b"\n"
    )
    summary = {
        "artifact_role": (
            "R8C_E1E2_IMMUTABLE_ENDPOINT_SUFFICIENT_UNANALYZED"
        ),
        "status": "COMPLETE",
        "task": row,
        "events": [event_summary],
        "total_cfe": 20,
        "total_atomic_model_steps": 20,
        "charged_evaluation_count": 20,
        "individual_evaluation_rows_persisted": 0,
        "permissions": {"results_analysis_performed": False},
    }
    summary_path = task_directory / "task_summary.json"
    _write_json(summary_path, summary)
    artifacts = {
        "checkpoint_fronts.cfe": _commitment(checkpoint_path),
        "event_summaries.jsonl": _commitment(event_summaries_path),
        "task_summary.json": _commitment(summary_path),
    }
    task_manifest = {
        "task_id": spec.task_id,
        "status": "COMPLETE",
        "artifacts": artifacts,
        "task_binding_sha256": sha256(
            canonical_json_bytes(
                {
                    "task": row,
                    "artifacts": artifacts,
                }
            )
        ).hexdigest(),
    }
    task_manifest_path = task_directory / "task_manifest.json"
    _write_json(task_manifest_path, task_manifest)
    task_manifest_sha = file_sha256(task_manifest_path)

    runtime = {
        "status": "COMPLETE_UNANALYZED",
        "scheduled_task_count": 1,
        "recorded_outcome_count": 1,
        "attempts_per_task": 1,
        "automatic_retries": 0,
        "results_analysis_performed": False,
        "effect_values_read_by_supervisor": False,
        "completed": [
            {
                "task_id": spec.task_id,
                "status": "COMPLETE",
                "total_cfe": 20,
                "total_atomic_model_steps": 20,
                "cpu_seconds": 1.0,
                "wall_seconds": 1.5,
                "peak_rss_bytes": 1024,
                "output_bytes": sum(
                    path.stat().st_size
                    for path in task_directory.iterdir()
                ),
                "automatic_retries": 0,
                "task_manifest_sha256": task_manifest_sha,
            }
        ],
        "failures": [],
    }
    runtime_path = root / "runtime_report.json"
    _write_json(runtime_path, runtime)
    marker_path = tmp_path / "request_consumption_marker.json"
    record_path = root / "request_consumption_record.json"
    launch_path = root / "launch_binding.json"
    _write_json(
        launch_path,
        {
            "paths": {
                "request_consumption_marker": str(marker_path.resolve()),
                "request_consumption_record": (
                    "request_consumption_record.json"
                ),
            }
        },
    )
    _write_json(
        record_path,
        {
            "consumption": "ONE_TIME_FORMAL_SUPERVISOR_START",
            "launch_binding_sha256": file_sha256(launch_path),
        },
    )
    control_paths = (
        schedule_path,
        reuse_path,
        launch_path,
        runtime_path,
        record_path,
    )
    run_manifest = {
        "status": "COMPLETE_UNANALYZED",
        "schedule": {
            "id": freeze.schedule_id,
            "sha256": freeze.schedule_sha256,
            "e2_full_reuse_sha256": freeze.reuse_sha256,
            "method_sequences": 1,
            "completed_sequences": 1,
            "recorded_outcomes": 1,
        },
        "resources": {"automatic_retries": 0},
        "control_artifacts": {
            path.name: _commitment(path) for path in control_paths
        },
        "task_manifest_commitments": {
            spec.task_id: task_manifest_sha
        },
        "permissions": {"effect_analysis": False},
        "analysis_gate": (
            "R9_RAW_LOCK_AND_ANALYSIS_NOT_YET_AUTHORIZED"
        ),
    }
    run_manifest_path = root / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return root, freeze, file_sha256(run_manifest_path)


def _reseal_task_and_run(root: Path) -> str:
    task_directory = next((root / "tasks").iterdir())
    task_manifest_path = task_directory / "task_manifest.json"
    task_manifest = json.loads(
        task_manifest_path.read_text(encoding="utf-8")
    )
    task_manifest["artifacts"] = {
        path.name: _commitment(path)
        for path in task_directory.iterdir()
        if path.is_file() and path.name != "task_manifest.json"
    }
    schedule_row = json.loads(
        gzip.decompress((root / "schedule.jsonl.gz").read_bytes())
    )
    task_manifest["task_binding_sha256"] = sha256(
        canonical_json_bytes(
            {
                "task": schedule_row,
                "artifacts": task_manifest["artifacts"],
            }
        )
    ).hexdigest()
    _write_json(task_manifest_path, task_manifest)
    task_manifest_sha = file_sha256(task_manifest_path)

    runtime_path = root / "runtime_report.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    outcomes = [*runtime["completed"], *runtime["failures"]]
    assert len(outcomes) == 1
    outcomes[0]["task_manifest_sha256"] = task_manifest_sha
    outcomes[0]["output_bytes"] = sum(
        path.stat().st_size for path in task_directory.iterdir()
    )
    _write_json(runtime_path, runtime)

    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    run_manifest["task_manifest_commitments"] = {
        task_directory.name: task_manifest_sha
    }
    run_manifest["control_artifacts"]["runtime_report.json"] = _commitment(
        runtime_path
    )
    _write_json(run_manifest_path, run_manifest)
    return file_sha256(run_manifest_path)


def _convert_to_inexact_hard_kill(
    root: Path,
    *,
    checkpoint_mode: str,
) -> str:
    task_directory = next((root / "tasks").iterdir())
    checkpoint_path = task_directory / "checkpoint_fronts.cfe"
    checkpoint_bytes = checkpoint_path.read_bytes()
    if checkpoint_mode == "zero":
        checkpoint_path.write_bytes(b"")
        (task_directory / "event_summaries.jsonl").write_bytes(b"")
    elif checkpoint_mode == "truncated":
        checkpoint_path.write_bytes(checkpoint_bytes[:17])
        (task_directory / "event_summaries.jsonl").write_bytes(b"")
    elif checkpoint_mode != "complete_prefix":
        raise AssertionError("unknown checkpoint fixture mode")
    (task_directory / "task_summary.json").unlink()

    task_manifest_path = task_directory / "task_manifest.json"
    task_manifest = json.loads(
        task_manifest_path.read_text(encoding="utf-8")
    )
    task_manifest["status"] = "INCOMPLETE_TECHNICAL_TIMEOUT"
    _write_json(task_manifest_path, task_manifest)

    runtime_path = root / "runtime_report.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    completed = runtime["completed"].pop()
    runtime["failures"] = [
        {
            "task_id": completed["task_id"],
            "status": "INCOMPLETE_TECHNICAL_TIMEOUT",
            "outcome_class": "TECHNICAL_SEQUENCE_TIMEOUT",
            "charged_cfe": None,
            "charged_work_exact": False,
            "cpu_seconds": 1.0,
            "wall_seconds": 1.5,
            "peak_rss_bytes": 1024,
            "output_bytes": 0,
            "automatic_retries": 0,
            "task_manifest_sha256": "",
        }
    ]
    _write_json(runtime_path, runtime)

    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    run_manifest["schedule"]["completed_sequences"] = 0
    _write_json(run_manifest_path, run_manifest)
    return _reseal_task_and_run(root)


def test_result_blind_batch_audit_authenticates_control_plane_only(
    tmp_path: Path,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    report = validate_r8c_e1e2_run(
        root,
        expected_run_manifest_sha256=raw_lock,
        _freeze=freeze,
    )
    assert report.authenticated_task_count == 1
    assert report.authenticated_event_count == 1
    assert report.authenticated_checkpoint_record_count == 21
    assert report.authenticated_charged_cfe == 20
    assert report.control_plane_dict()["effect_endpoints_computed"] is False
    assert "anytime_nhv_auc" not in report.control_plane_dict()
    with pytest.raises(R8CIntegrityError, match="raw-lock"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256="0" * 64,
            _freeze=freeze,
        )


def test_batch_audit_fails_closed_on_missing_extra_duplicate_or_hash_drift(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    task_id = next((root / "tasks").iterdir()).name
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_manifest_commitments"]["extra-task"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(R8CIntegrityError, match="missing/extra"):
        validate_r8c_e1e2_run(root, _freeze=freeze)

    manifest["task_manifest_commitments"].pop("extra-task")
    _write_json(manifest_path, manifest)
    summary_path = root / "tasks" / task_id / "task_summary.json"
    summary_path.write_bytes(summary_path.read_bytes() + b" ")
    with pytest.raises(R8CIntegrityError, match="byte count"):
        validate_r8c_e1e2_run(root, _freeze=freeze)


def test_batch_audit_rejects_uncommitted_task_subdirectory(
    tmp_path: Path,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    task_directory = next((root / "tasks").iterdir())
    (task_directory / "uncommitted").mkdir()
    with pytest.raises(R8CIntegrityError, match="non-file"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_batch_audit_remains_valid_after_raw_root_relocation(
    tmp_path: Path,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    relocated = tmp_path / "relocated-raw"
    root.rename(relocated)
    report = validate_r8c_e1e2_run(
        relocated,
        expected_run_manifest_sha256=raw_lock,
        _freeze=freeze,
    )
    assert report.authenticated_task_count == 1


@pytest.mark.parametrize(
    ("checkpoint_mode", "expected_event_count"),
    (("zero", 0), ("truncated", 0), ("complete_prefix", 1)),
)
def test_inexact_hard_kill_checkpoint_is_hash_bound_opaque_partial(
    tmp_path: Path,
    checkpoint_mode: str,
    expected_event_count: int,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    raw_lock = _convert_to_inexact_hard_kill(
        root,
        checkpoint_mode=checkpoint_mode,
    )
    report = validate_r8c_e1e2_run(
        root,
        expected_run_manifest_sha256=raw_lock,
        _freeze=freeze,
    )
    assert report.failed_task_count == 1
    assert report.authenticated_event_count == expected_event_count
    assert report.authenticated_checkpoint_record_count == 0
    assert report.unknown_charged_cfe_task_count == 1


def test_inexact_hard_kill_ignores_only_committed_jsonl_tail_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    _convert_to_inexact_hard_kill(
        root,
        checkpoint_mode="complete_prefix",
    )
    event_path = (
        next((root / "tasks").iterdir()) / "event_summaries.jsonl"
    )
    trailing_fragment = b'{"event_id":1,"execution_observation":'
    event_path.write_bytes(
        event_path.read_bytes() + trailing_fragment
    )
    raw_lock = _reseal_task_and_run(root)

    report = validate_r8c_e1e2_run(
        root,
        expected_run_manifest_sha256=raw_lock,
        _freeze=freeze,
    )
    assert report.authenticated_event_count == 1
    assert report.event_summary_trailing_fragment_task_count == 1
    assert report.event_summary_trailing_fragment_total_bytes == len(
        trailing_fragment
    )
    task = report.tasks[0]
    assert task.event_summary_trailing_fragment_present is True
    assert task.event_summary_trailing_fragment_bytes == len(
        trailing_fragment
    )
    assert task.event_summary_trailing_fragment_sha256 == sha256(
        trailing_fragment
    ).hexdigest()

    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    output_root = tmp_path / "r9-trailing-fragment"
    result = export_r9_readable_outputs(
        root,
        raw_manifest_sha256=raw_lock,
        authorization=(
            "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
        ),
        reference_catalog_path=tmp_path / "unused.jsonl",
        reference_catalog_sha256="0" * 64,
        output_root=output_root,
        include_event_diagnostics=True,
        _freeze=freeze,
    )
    assert result["event_summary_trailing_fragment_task_count"] == 1
    assert result["event_summary_trailing_fragment_total_bytes"] == len(
        trailing_fragment
    )
    with (output_root / "task_endpoints.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        endpoint = next(csv.DictReader(stream))
    assert endpoint["endpoint_status"] == "NOT_COMPUTED_TASK_FAILURE"
    with (output_root / "failure_cost.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        cost = next(csv.DictReader(stream))
    assert cost["event_summary_trailing_fragment_present"] == "True"
    assert cost["event_summary_trailing_fragment_bytes"] == str(
        len(trailing_fragment)
    )
    assert cost["event_summary_trailing_fragment_sha256"] == sha256(
        trailing_fragment
    ).hexdigest()
    with gzip.open(
        output_root / "event_diagnostics.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        diagnostics = [json.loads(line) for line in stream]
    assert len(diagnostics) == 1
    assert diagnostics[0]["event_id"] == 0


@pytest.mark.parametrize("exact_failed_task", (False, True))
def test_jsonl_tail_fragment_remains_strict_for_complete_or_exact_task(
    tmp_path: Path,
    exact_failed_task: bool,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    if exact_failed_task:
        _convert_to_inexact_hard_kill(
            root,
            checkpoint_mode="complete_prefix",
        )
        runtime_path = root / "runtime_report.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        failure = runtime["failures"][0]
        failure["charged_cfe"] = 20
        failure["charged_atomic_model_steps"] = 20
        failure["charged_work_exact"] = True
        _write_json(runtime_path, runtime)
    event_path = (
        next((root / "tasks").iterdir()) / "event_summaries.jsonl"
    )
    event_path.write_bytes(event_path.read_bytes() + b'{"partial":')
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="final LF"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_batch_audit_rejects_oversize_complete_event_record(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    task_directory = next((root / "tasks").iterdir())
    event_path = task_directory / "event_summaries.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["terminal"]["reason"] = "X" * EVENT_SUMMARY_MAX_RECORD_BYTES
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="frozen byte bound"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_batch_audit_rejects_oversize_opaque_tail(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    _convert_to_inexact_hard_kill(
        root,
        checkpoint_mode="complete_prefix",
    )
    event_path = (
        next((root / "tasks").iterdir()) / "event_summaries.jsonl"
    )
    event_path.write_bytes(
        event_path.read_bytes()
        + b"X" * EVENT_SUMMARY_MAX_RECORD_BYTES
    )
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="trailing fragment"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_batch_audit_rejects_arbitrary_nested_event_summary_fields(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    task_directory = next((root / "tasks").iterdir())
    event_path = task_directory / "event_summaries.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["execution_observation"]["decision_vector"] = [0.1, 0.2]
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    summary_path = task_directory / "task_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["events"] = [event]
    _write_json(summary_path, summary)
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="execution channel"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_opaque_partial_exception_does_not_cover_general_task_failure(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    _convert_to_inexact_hard_kill(root, checkpoint_mode="truncated")
    runtime_path = root / "runtime_report.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["failures"][0]["outcome_class"] = "TASK_EXECUTION_FAILURE"
    _write_json(runtime_path, runtime)
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="strict decoding"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_batch_audit_rejects_terminal_code_outside_frozen_enum(
    tmp_path: Path,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    task_directory = next((root / "tasks").iterdir())
    event_path = task_directory / "event_summaries.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["terminal"]["code"] = "MADE_UP_TERMINAL"
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    summary_path = task_directory / "task_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["events"] = [event]
    _write_json(summary_path, summary)
    raw_lock = _reseal_task_and_run(root)
    with pytest.raises(R8CIntegrityError, match="terminal"):
        validate_r8c_e1e2_run(
            root,
            expected_run_manifest_sha256=raw_lock,
            _freeze=freeze,
        )


def test_r9_export_requires_exact_authorization_and_writes_compact_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    with pytest.raises(R9ExportAuthorizationError, match="authorization"):
        export_r9_readable_outputs(
            root,
            raw_manifest_sha256=raw_lock,
            authorization="NO",
            reference_catalog_path=tmp_path / "unused.jsonl",
            reference_catalog_sha256="0" * 64,
            output_root=tmp_path / "unauthorized",
            _freeze=freeze,
        )
    assert not (tmp_path / "unauthorized").exists()

    result = export_r9_readable_outputs(
        root,
        raw_manifest_sha256=raw_lock,
        authorization=(
            "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
        ),
        reference_catalog_path=tmp_path / "unused.jsonl",
        reference_catalog_sha256="0" * 64,
        output_root=tmp_path / "r9",
        include_event_diagnostics=True,
        _freeze=freeze,
    )
    assert result["task_endpoints_rows"] == 1
    assert result["failure_cost_rows"] == 1
    assert result["e2_negative_transfer_rows"] == 0
    assert result["post_execution_hard_violation_rows"] == 1
    assert result[
        "raw_lock_revalidated_immediately_before_publication"
    ] is True
    assert result[
        "all_raw_task_artifacts_reaudited_after_derived_write"
    ] is True
    readme_path = tmp_path / "r9" / "README.md"
    assert readme_path.is_file()
    assert "one row per scheduled sequence" in readme_path.read_text(
        encoding="utf-8"
    )
    with (tmp_path / "r9" / "task_endpoints.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        endpoint_rows = list(csv.DictReader(stream))
    assert len(endpoint_rows) == 1
    assert endpoint_rows[0]["endpoint_status"] == "INCLUDED"
    assert float(endpoint_rows[0]["anytime_nhv_auc"]) == pytest.approx(
        0.121875
    )
    with (tmp_path / "r9" / "failure_cost.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        cost_rows = list(csv.DictReader(stream))
    assert len(cost_rows) == 1
    assert cost_rows[0]["charged_cfe"] == "20"
    assert cost_rows[0]["evaluation_failure_count"] == "0"
    with (tmp_path / "r9" / "e2_negative_transfer.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        negative_transfer_rows = list(csv.DictReader(stream))
    assert negative_transfer_rows == []
    with (
        tmp_path / "r9" / "post_execution_hard_violation.csv"
    ).open(encoding="utf-8", newline="") as stream:
        hard_violation_rows = list(csv.DictReader(stream))
    assert hard_violation_rows[0]["endpoint_status"] == "INCLUDED"
    assert (
        hard_violation_rows[0]["post_execution_hard_violation_rate"]
        == "0.0"
    )
    with gzip.open(
        tmp_path / "r9" / "event_diagnostics.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        diagnostics = [json.loads(line) for line in stream]
    assert len(diagnostics) == 1
    assert diagnostics[0]["final_nhv"] == pytest.approx(0.125)
    export_manifest = json.loads(
        (tmp_path / "r9" / "r9_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert export_manifest["artifacts"]["README.md"] == _commitment(
        readme_path
    )


def test_r9_export_rechecks_raw_lock_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    original = r8c_batch_outputs._gzip_jsonl_bytes

    def mutate_raw_manifest(rows):
        payload = original(rows)
        manifest_path = root / "run_manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        return payload

    monkeypatch.setattr(
        r8c_batch_outputs,
        "_gzip_jsonl_bytes",
        mutate_raw_manifest,
    )
    output_root = tmp_path / "r9-toctou"
    with pytest.raises(R8CIntegrityError, match="manifest changed"):
        export_r9_readable_outputs(
            root,
            raw_manifest_sha256=raw_lock,
            authorization=(
                "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
            ),
            reference_catalog_path=tmp_path / "unused.jsonl",
            reference_catalog_sha256="0" * 64,
            output_root=output_root,
            include_event_diagnostics=True,
            _freeze=freeze,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("source_name", "expected_error"),
    (
        ("runtime_report.json", "runtime report commitment differs"),
        ("task_manifest.json", "source task manifest"),
    ),
)
def test_r9_export_binds_runtime_and_task_manifest_after_initial_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    expected_error: str,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    original_validate = r8c_batch_outputs.validate_r8c_e1e2_run

    def validate_then_mutate(*args, **kwargs):
        report = original_validate(*args, **kwargs)
        if source_name == "runtime_report.json":
            source = root / source_name
        else:
            source = next((root / "tasks").iterdir()) / source_name
        source.write_bytes(source.read_bytes() + b" ")
        return report

    monkeypatch.setattr(
        r8c_batch_outputs,
        "validate_r8c_e1e2_run",
        validate_then_mutate,
    )
    output_root = tmp_path / f"r9-{source_name}"
    with pytest.raises(R8CIntegrityError, match=expected_error):
        export_r9_readable_outputs(
            root,
            raw_manifest_sha256=raw_lock,
            authorization=(
                "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
            ),
            reference_catalog_path=tmp_path / "unused.jsonl",
            reference_catalog_sha256="0" * 64,
            output_root=output_root,
            _freeze=freeze,
        )
    assert not output_root.exists()


def test_r9_export_reaudits_all_task_artifacts_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, freeze, raw_lock = _synthetic_run(tmp_path)
    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    original_write = r8c_batch_outputs._write_exclusive
    mutated = False

    def write_then_mutate(path, payload):
        nonlocal mutated
        original_write(path, payload)
        if not mutated:
            mutated = True
            checkpoint = (
                next((root / "tasks").iterdir())
                / "checkpoint_fronts.cfe"
            )
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")

    monkeypatch.setattr(
        r8c_batch_outputs,
        "_write_exclusive",
        write_then_mutate,
    )
    output_root = tmp_path / "r9-artifact-toctou"
    with pytest.raises(R8CIntegrityError, match="byte count"):
        export_r9_readable_outputs(
            root,
            raw_manifest_sha256=raw_lock,
            authorization=(
                "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
            ),
            reference_catalog_path=tmp_path / "unused.jsonl",
            reference_catalog_sha256="0" * 64,
            output_root=output_root,
            _freeze=freeze,
        )
    assert output_root.is_dir()
    assert not (output_root / "r9_export_manifest.json").exists()


def test_r9_partial_failure_preserves_completed_event_and_missing_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, freeze, _ = _synthetic_run(tmp_path)
    raw_lock = _convert_to_inexact_hard_kill(
        root,
        checkpoint_mode="complete_prefix",
    )
    monkeypatch.setattr(
        r8c_batch_outputs,
        "load_reference_catalog",
        lambda *args, **kwargs: (),
    )
    output_root = tmp_path / "r9-partial"
    result = export_r9_readable_outputs(
        root,
        raw_manifest_sha256=raw_lock,
        authorization=(
            "R9_RAW_MANIFEST_LOCKED_AND_ANALYSIS_AUTHORIZED"
        ),
        reference_catalog_path=tmp_path / "unused.jsonl",
        reference_catalog_sha256="0" * 64,
        output_root=output_root,
        include_event_diagnostics=True,
        _freeze=freeze,
    )
    assert result["task_endpoints_rows"] == 1
    with (output_root / "failure_cost.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        cost = next(csv.DictReader(stream))
    assert cost["charged_cfe"] == ""
    assert cost["charged_work_exact"] == "False"
    assert cost["terminal_code_counts_json"] == '{"ACCEPTED":1}'
    with (output_root / "post_execution_hard_violation.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        hard = next(csv.DictReader(stream))
    assert hard["endpoint_status"] == "NOT_COMPUTED_TASK_FAILURE"
    assert hard["post_execution_hard_violation_rate"] == ""
    assert hard["executed_event_count"] == "1"
    assert hard["hard_violation_event_count"] == "0"
    with gzip.open(
        output_root / "event_diagnostics.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        diagnostics = [json.loads(line) for line in stream]
    assert len(diagnostics) == 1
    assert diagnostics[0]["execution_observation"][
        "hard_constraint_violation"
    ] is False
    assert diagnostics[0]["final_nhv"] is None


def test_hard_violation_summary_does_not_impute_missing_observation() -> None:
    status, rate, count, executed, available, missing, completed = (
        r8c_batch_outputs._hard_violation_summary(
            task_status="COMPLETE",
            events=(
                {
                    "ledger": {"execution_transition_count": 1},
                    "execution_observation": {
                        "available": False,
                        "ell_exec": None,
                        "ell_ref": None,
                        "s_exec": None,
                        "hard_constraint_violation": None,
                        "released_at": 1,
                    }
                },
            ),
            scheduled_event_count=1,
        )
    )
    assert status == "NOT_COMPUTED_MISSING_EXECUTION_OBSERVATION"
    assert rate == ""
    assert count == 0
    assert (executed, available, missing) == (1, 0, 1)
    assert completed == 1


def test_hard_violation_denominator_uses_execution_transitions() -> None:
    status, rate, count, executed, available, missing, completed = (
        r8c_batch_outputs._hard_violation_summary(
            task_status="COMPLETE",
            events=(
                {
                    "ledger": {"execution_transition_count": 0},
                    "execution_observation": None,
                },
            ),
            scheduled_event_count=1,
        )
    )
    assert status == "NOT_COMPUTED_NO_EXECUTED_EVENTS"
    assert rate == ""
    assert count == 0
    assert (executed, available, missing, completed) == (0, 0, 0, 1)


def test_negative_transfer_summary_preserves_pair_status_and_counts() -> None:
    zero = (0.0,) * 21
    low = (0.2,) * 21
    high = (0.5,) * 21
    status, rate, count, denominator = (
        r8c_batch_outputs._negative_transfer_summary(
            full_status="INCLUDED",
            comparator_status="INCLUDED",
            full_curves=(zero, zero, high),
            comparator_curves=(zero, low, low),
            event_count=3,
        )
    )
    assert status == "INCLUDED"
    assert rate == pytest.approx(0.5)
    assert count == 1
    assert denominator == 2

    excluded = r8c_batch_outputs._negative_transfer_summary(
        full_status="NOT_COMPUTED_TASK_FAILURE",
        comparator_status="INCLUDED",
        full_curves=None,
        comparator_curves=(zero, low),
        event_count=2,
    )
    assert excluded == (
        "FULL_NOT_COMPUTED_TASK_FAILURE",
        "",
        "",
        1,
    )


def test_formal_public_freeze_is_not_parameterized_by_cli() -> None:
    assert r8c_batch_outputs.FORMAL_R8C_E1E2_FREEZE.task_count == 5030
    assert (
        r8c_batch_outputs.FORMAL_R8C_E1E2_FREEZE.total_cfe
        == 851_000_000
    )
    assert (
        r8c_batch_outputs.FORMAL_R8C_E1E2_FREEZE.total_atomic_steps
        == 1_971_000_000
    )
    assert (
        r8c_batch_outputs.R8C_E1E2_NEGATIVE_TRANSFER_PAIR_COUNT
        == 2_330
    )
