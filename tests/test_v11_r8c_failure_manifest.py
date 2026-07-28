from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from analysis.r8c_failure_outcomes import (
    FailureOutcomeClosureError,
    validate_failure_outcome_closure,
)
from evaluation.evaluator import ExecutionTimeoutBeforeEntry
from formal_execution.schedule import (
    FormalSequenceSpec,
    canonical_json_bytes,
    schedule_commitment,
)


def _runner():
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        "_test_v11_r8c_failure_runner",
        root / "tools" / "run_v11_r8_formal.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _spec() -> FormalSequenceSpec:
    return FormalSequenceSpec(
        schedule_index=0,
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="MATCHED_FIXED_DE_PARETO",
        replicate_index=0,
        master_seed_u64="20260726",
        events=1,
        cfe_per_event=50_000,
        atomic_steps_per_cfe=1,
        timeout_seconds=3600,
        problem_index=1,
        problem_id="LIRCMOP1",
        task_namespace="r8c",
    )


def _write_failure_run(tmp_path: Path):
    runner = _runner()
    spec = _spec()
    output_root = tmp_path / "run"
    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True)
    schedule_path = output_root / "schedule.jsonl.gz"
    with schedule_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as stream:
            stream.write(canonical_json_bytes(spec.to_dict()) + b"\n")

    task_directory = tasks_root / spec.task_id
    status = "NOT_DISPATCHED_TECHNICAL_STOP_NO_RETRY"
    _, manifest_sha256 = runner._materialize_supervisor_task_outcome(
        profile=runner.CORRECTIVE_E1E2_PROFILE,
        spec=spec,
        task_directory=task_directory,
        status=status,
        failure_class=runner.TECHNICAL_NOT_DISPATCHED,
        reason="GLOBAL_HARD_TIMEOUT",
    )
    failure = {
        "task_id": spec.task_id,
        "schedule_index": spec.schedule_index,
        "status": status,
        "outcome_class": runner.TECHNICAL_NOT_DISPATCHED,
        "return_code": None,
        "hard_timed_out": False,
        "timeout_requested": False,
        "timeout_marker": None,
        "attempt": 1,
        "automatic_retries": 0,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "peak_rss_bytes": 0,
        "output_bytes": runner._directory_bytes(task_directory),
        "scheduled_cfe": spec.total_cfe,
        "charged_cfe": 0,
        "scheduled_atomic_model_steps": spec.total_atomic_steps,
        "charged_atomic_model_steps": 0,
        "charged_work_exact": True,
        "charged_work_source": "NOT_DISPATCHED",
        "task_manifest_sha256": manifest_sha256,
        "algorithm_terminal_code": None,
        "worker_reported_status": None,
        "error_type": None,
        "error": "GLOBAL_HARD_TIMEOUT",
        "logs": {
            "stdout": {"path": "missing.stdout", "missing": True},
            "stderr": {"path": "missing.stderr", "missing": True},
        },
    }
    runtime_report = {
        "status": "INCOMPLETE_RESOURCE_CEILING",
        "scheduled_task_count": 1,
        "dispatched_task_count": 0,
        "recorded_outcome_count": 1,
        "completed_process_count": 0,
        "failed_process_count": 0,
        "failed_outcome_count": 1,
        "not_dispatched_outcome_count": 1,
        "unknown_failure_accounting_count": 0,
        "attempts_per_task": 1,
        "automatic_retries": 0,
        "completed": [],
        "failures": [failure],
        "task_manifest_commitments": {
            spec.task_id: manifest_sha256,
        },
        "effect_values_read_by_supervisor": False,
        "results_analysis_performed": False,
    }
    runtime_path = output_root / "runtime_report.json"
    runtime_path.write_bytes(canonical_json_bytes(runtime_report) + b"\n")
    run_manifest = {
        "status": "INCOMPLETE_RESOURCE_CEILING",
        "resources": {"automatic_retries": 0},
        "schedule": {
            "id": "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01",
            "sha256": schedule_commitment([spec]),
            "method_sequences": 1,
        },
        "control_artifacts": {
            "runtime_report.json": {
                "bytes": runtime_path.stat().st_size,
                "sha256": runner.file_sha256(runtime_path),
            }
        },
        "task_manifest_commitments": {
            spec.task_id: manifest_sha256,
        },
    }
    manifest_path = output_root / "run_manifest.json"
    payload, _ = runner._run_manifest_payload_with_final_output_bytes(
        run_manifest,
        output_bytes_before_manifest=runner._directory_bytes(output_root),
    )
    manifest_path.write_bytes(payload)
    return runner, spec, output_root, runtime_report, run_manifest


def test_run_wide_failure_validator_requires_every_scheduled_commitment(
    tmp_path: Path,
) -> None:
    _, spec, output_root, _, _ = _write_failure_run(tmp_path)
    outcomes = validate_failure_outcome_closure(output_root)
    assert len(outcomes) == 1
    assert outcomes[0].task_id == spec.task_id
    assert outcomes[0].is_technical_failure is True
    assert outcomes[0].charged_cfe == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "algorithm_terminal",
        "missing_commitment",
        "fabricated_inexact_cost",
        "missing_rss",
    ],
)
def test_run_wide_failure_validator_rejects_incomplete_or_false_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    runner, spec, output_root, runtime_report, run_manifest = (
        _write_failure_run(tmp_path)
    )
    if mutation == "algorithm_terminal":
        runtime_report["failures"][0][
            "algorithm_terminal_code"
        ] = "REJECT_TIMEOUT"
    elif mutation == "missing_commitment":
        run_manifest["task_manifest_commitments"] = {}
    elif mutation == "fabricated_inexact_cost":
        runtime_report["failures"][0]["charged_work_exact"] = False
    elif mutation == "missing_rss":
        runtime_report["failures"][0].pop("peak_rss_bytes")
    runtime_path = output_root / "runtime_report.json"
    runtime_path.write_bytes(canonical_json_bytes(runtime_report) + b"\n")
    run_manifest["control_artifacts"]["runtime_report.json"] = {
        "bytes": runtime_path.stat().st_size,
        "sha256": runner.file_sha256(runtime_path),
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.unlink()
    payload, _ = runner._run_manifest_payload_with_final_output_bytes(
        run_manifest,
        output_bytes_before_manifest=runner._directory_bytes(output_root),
    )
    manifest_path.write_bytes(payload)
    with pytest.raises(FailureOutcomeClosureError):
        validate_failure_outcome_closure(output_root)


def test_worker_cooperative_timeout_is_typed_technical_not_algorithmic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    spec = _spec()
    monkeypatch.setattr(
        runner,
        "_load_and_validate",
        lambda contract, request, profile: ({}, object(), [spec]),
    )
    monkeypatch.setattr(
        runner,
        "_validate_worker_launch",
        lambda **kwargs: None,
    )

    def cooperative_timeout(**kwargs):
        task_directory = kwargs["task_directory"]
        task_directory.mkdir()
        (task_directory / runner.TASK_TIMEOUT_MARKER_NAME).write_text(
            "TASK_TIMEOUT\n",
            encoding="utf-8",
        )
        (task_directory / "checkpoint_fronts.cfe").write_bytes(b"partial")
        raise ExecutionTimeoutBeforeEntry("supervisor timeout marker")

    monkeypatch.setattr(runner, "run_task", cooperative_timeout)
    task_directory = tmp_path / "task"
    code = runner._run_worker(
        argparse.Namespace(
            execution_profile="corrective_r8c_e1e2",
            contract=str(tmp_path / "contract.json"),
            request=str(tmp_path / "request.json"),
            schedule_index=0,
            task_id=spec.task_id,
            task_directory=str(task_directory),
            stop_path=str(tmp_path / "STOP"),
        )
    )
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(
        (task_directory / "task_failure.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PARTIAL_TECHNICAL_TIMEOUT_NO_RETRY"
    assert payload["outcome_class"] == "TECHNICAL_SEQUENCE_TIMEOUT"
    assert payload["algorithm_terminal_code"] is None
    assert payload["reason_code"] == payload["outcome_class"]
    assert "error" not in payload
    assert payload["automatic_retries"] == 0
    assert payload["accounting"]["charged_work_exact"] is False
    manifest = json.loads(
        (task_directory / "task_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == payload["status"]
    assert manifest["outcome_class"] == payload["outcome_class"]
