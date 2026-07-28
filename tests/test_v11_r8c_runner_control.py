from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from dt_ramde_v11.contracts import ConfigurationError


def _runner():
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        "_test_v11_r8c_runner_control",
        root / "tools" / "run_v11_r8_formal.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _target_runtime_environment(runner) -> dict:
    evidence = dict(runner._runtime_environment_lock_evidence())
    evidence.update(
        {
            "actual_python_implementation": "CPython",
            "actual_python_version": "3.12.0",
            "actual_system": "Linux",
            "actual_machine": "x86_64",
            "installed_package_versions": dict(
                evidence["locked_package_versions"]
            ),
            "missing_locked_packages": [],
            "version_mismatches": {},
            "interpreter_matches": True,
            "platform_matches": True,
            "all_locked_packages_match": True,
            "target_environment_match": True,
        }
    )
    return evidence


def test_wrapper_rejects_worker_and_profile_override() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "tools" / "run_v11_r8c_e1e2_formal.py"
    for option in (
        "--worker",
        "--execution-profile=legacy_r8",
        "--task-directory=elsewhere",
    ):
        result = __import__("subprocess").run(
            [sys.executable, str(wrapper), option],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "reserved" in result.stderr


def test_direct_worker_without_supervisor_token_fails_before_file_reads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    monkeypatch.delenv(runner.LAUNCH_TOKEN_ENV, raising=False)
    code = runner.main(
        [
            "--execution-profile",
            "corrective_r8c_e1e2",
            "--contract",
            "does-not-exist-contract.json",
            "--request",
            "does-not-exist-request.json",
            "--worker",
            "--schedule-index",
            "0",
            "--task-id",
            "task",
            "--task-directory",
            "task",
            "--stop-path",
            "stop",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "supervisor-internal" in captured.err
    assert "does-not-exist" not in captured.err


def test_atomic_consumption_marker_is_complete_and_single_use(
    tmp_path: Path,
) -> None:
    runner = _runner()
    marker = tmp_path / "request.consumed.json"
    payload = {
        "schema_version": "WGT-R8-REQUEST-CONSUMPTION-1.0",
        "request_id": "request",
        "launch_binding_sha256": "a" * 64,
        "consumption": "ONE_TIME_FORMAL_SUPERVISOR_START",
    }
    assert runner._consume_request_once(
        marker_path=marker,
        payload=payload,
    ) == marker
    assert json.loads(marker.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        runner._consume_request_once(
            marker_path=marker,
            payload={"incomplete": True},
        )
    assert json.loads(marker.read_text(encoding="utf-8")) == payload


def test_worker_log_commitments_are_root_relative_and_relocatable(
    tmp_path: Path,
) -> None:
    runner = _runner()
    from analysis.r8c_batch_outputs import (
        _validate_worker_log_commitments,
    )

    root = tmp_path / "cloud-raw"
    logs = root / "worker_logs"
    logs.mkdir(parents=True)
    task_id = "r8c-0000-portable"
    stdout = logs / f"{task_id}.stdout.log"
    stderr = logs / f"{task_id}.stderr.log"
    stdout.write_bytes(b"out\n")
    stderr.write_bytes(b"err\n")
    bindings = {
        task_id: {
            "stdout": runner._root_relative_file_commitment(
                stdout,
                root=root,
            ),
            "stderr": runner._root_relative_file_commitment(
                stderr,
                root=root,
            ),
        }
    }
    assert bindings[task_id]["stdout"]["path"] == (
        f"worker_logs/{task_id}.stdout.log"
    )
    relocated = tmp_path / "local-raw"
    root.rename(relocated)
    _validate_worker_log_commitments(
        relocated,
        run_manifest={"worker_log_commitments": bindings},
        runtime_report={"worker_log_commitments": bindings},
        task_ids={task_id},
    )


def test_r8c_worker_control_report_is_manifest_bound_and_relocatable(
    tmp_path: Path,
) -> None:
    runner = _runner()
    from analysis.r8c_batch_outputs import (
        _validate_worker_control_report_commitments,
    )

    task_id = "r8c-0000-portable"
    task_payload = {"task_id": task_id, "schedule_index": 0}
    spec = SimpleNamespace(
        task_id=task_id,
        schedule_index=0,
        total_cfe=20,
        total_atomic_steps=20,
        atomic_steps_per_cfe=1,
        to_dict=lambda: task_payload,
    )
    root = tmp_path / "cloud-raw"
    task_directory = root / "tasks" / task_id
    task_directory.mkdir(parents=True)
    report_path = task_directory / "task_supervisor_outcome.json"
    report = {
        "artifact_role": "R8C_E1E2_IMMUTABLE_UNANALYZED",
        "status": "NOT_DISPATCHED_TECHNICAL_STOP_NO_RETRY",
        "outcome_class": runner.TECHNICAL_NOT_DISPATCHED,
        "task": task_payload,
        "reason_code": runner.TECHNICAL_NOT_DISPATCHED,
        "error_type": None,
        "accounting": {
            "scheduled_cfe": 20,
            "scheduled_atomic_model_steps": 20,
            "atomic_steps_per_cfe": 1,
            "charged_cfe": 0,
            "charged_atomic_model_steps": 0,
            "charged_work_exact": True,
            "charged_work_source": "NOT_DISPATCHED",
            "charged_work_recovery_error_type": None,
        },
        "attempt": 1,
        "automatic_retries": 0,
        "algorithm_terminal_code": None,
        "results_analysis_performed": False,
    }
    runner._write_json_exclusive(report_path, report)
    artifact_binding = {
        "bytes": report_path.stat().st_size,
        "sha256": runner.file_sha256(report_path),
    }
    runner._write_json_exclusive(
        task_directory / "task_manifest.json",
        {
            "task_id": task_id,
            "status": report["status"],
            "artifacts": {report_path.name: artifact_binding},
        },
    )
    runner._write_json_exclusive(
        root / "launch_binding.json",
        {
            "paths": {
                "worker_control_reports": (
                    "TASK_MANIFEST_COMMITTED_TASK_ARTIFACTS"
                )
            }
        },
    )
    selected = runner._r8c_e1e2_worker_report(
        task_directory,
        spec=spec,
    )
    assert selected is not None
    binding = runner._worker_report_commitment(
        selected,
        output_root=root,
    )
    assert binding["path"] == (
        f"tasks/{task_id}/task_supervisor_outcome.json"
    )
    commitments = {task_id: binding}
    control = {
        "worker_control_report_commitments": commitments,
        "raw_worker_stdout_persisted": False,
        "raw_worker_stderr_persisted": False,
    }
    relocated = tmp_path / "local-raw"
    root.rename(relocated)
    _validate_worker_control_report_commitments(
        relocated,
        run_manifest=control,
        runtime_report=control,
        specs={task_id: spec},
    )
    assert not (relocated / "worker_logs").exists()


def test_r8c_worker_control_report_rejects_exception_message(
    tmp_path: Path,
) -> None:
    runner = _runner()
    task_id = "r8c-no-error-message"
    task_payload = {"task_id": task_id, "schedule_index": 0}
    spec = SimpleNamespace(task_id=task_id, to_dict=lambda: task_payload)
    task_directory = tmp_path / task_id
    task_directory.mkdir()
    report = {
        "artifact_role": "R8C_E1E2_IMMUTABLE_UNANALYZED",
        "status": "TASK_FAILED_NO_RETRY",
        "outcome_class": runner.TASK_EXECUTION_FAILURE,
        "task": task_payload,
        "reason_code": runner.TASK_EXECUTION_FAILURE,
        "error_type": "RuntimeError",
        "error": "objective=[0.1, 0.2]",
        "accounting": {},
        "attempt": 1,
        "automatic_retries": 0,
        "algorithm_terminal_code": None,
        "results_analysis_performed": False,
    }
    runner._write_json_exclusive(
        task_directory / "task_supervisor_outcome.json",
        report,
    )
    with pytest.raises(
        ConfigurationError,
        match="blind schema|fields differ",
    ):
        runner._r8c_e1e2_worker_report(task_directory, spec=spec)


def test_r8c_worker_control_report_rejects_oversized_payload(
    tmp_path: Path,
) -> None:
    runner = _runner()
    task_id = "r8c-oversized-report"
    task_payload = {"task_id": task_id, "schedule_index": 0}
    spec = SimpleNamespace(
        task_id=task_id,
        total_cfe=20,
        total_atomic_steps=20,
        atomic_steps_per_cfe=1,
        to_dict=lambda: task_payload,
    )
    task_directory = tmp_path / task_id
    task_directory.mkdir()
    oversized_code = "X" * runner.R8C_E1E2_WORKER_REPORT_MAX_BYTES
    runner._write_json_exclusive(
        task_directory / "task_supervisor_outcome.json",
        {
            "artifact_role": "R8C_E1E2_IMMUTABLE_UNANALYZED",
            "status": "TASK_FAILED_NO_RETRY",
            "outcome_class": oversized_code,
            "task": task_payload,
            "reason_code": oversized_code,
            "error_type": None,
            "accounting": {
                "scheduled_cfe": 20,
                "scheduled_atomic_model_steps": 20,
                "atomic_steps_per_cfe": 1,
                "charged_cfe": 0,
                "charged_atomic_model_steps": 0,
                "charged_work_exact": True,
                "charged_work_source": "NOT_STARTED",
                "charged_work_recovery_error_type": None,
            },
            "attempt": 1,
            "automatic_retries": 0,
            "algorithm_terminal_code": None,
            "results_analysis_performed": False,
        },
    )

    with pytest.raises(ConfigurationError, match="strict byte bound"):
        runner._r8c_e1e2_worker_report(task_directory, spec=spec)


def test_r8c_worker_control_report_rejects_oversize_before_write(
    tmp_path: Path,
) -> None:
    runner = _runner()
    path = tmp_path / "task_supervisor_outcome.json"
    with pytest.raises(ConfigurationError, match="strict byte bound"):
        runner._write_json_exclusive(
            path,
            {"payload": "X" * 100},
            maximum_bytes=16,
        )
    assert not path.exists()


def test_worker_failure_arbitration_never_labels_timeout_as_complete() -> None:
    runner = _runner()
    item = {
        "timeout_requested": True,
        "forced_termination_reason": None,
    }
    payload = {"status": "COMPLETE"}
    failure_class = runner._worker_failure_class(item, payload)
    assert failure_class == runner.TECHNICAL_SEQUENCE_TIMEOUT
    assert runner._worker_failure_status(
        failure_class=failure_class,
        payload=payload,
    ) == "PARTIAL_TECHNICAL_TIMEOUT_NO_RETRY"


def test_graceful_resource_stop_is_a_technical_resource_outcome() -> None:
    runner = _runner()
    payload = {
        "status": "INCOMPLETE_RESOURCE_CEILING",
        "total_cfe": 40,
        "total_atomic_model_steps": 40,
    }
    failure_class = runner._worker_failure_class({}, payload)
    assert failure_class == runner.TECHNICAL_RESOURCE_TERMINATION
    assert runner._worker_failure_status(
        failure_class=failure_class,
        payload=payload,
    ) == "INCOMPLETE_RESOURCE_CEILING"


def test_incomplete_task_summary_is_valid_control_report_but_not_success(
    tmp_path: Path,
) -> None:
    runner = _runner()
    task_id = "r8c-resource-stop-summary"
    task_payload = {"task_id": task_id, "schedule_index": 0}
    spec = SimpleNamespace(task_id=task_id, to_dict=lambda: task_payload)
    task_directory = tmp_path / task_id
    task_directory.mkdir()
    runner._write_json_exclusive(
        task_directory / "task_summary.json",
        {
            "artifact_role": (
                "R8C_E1E2_IMMUTABLE_ENDPOINT_SUFFICIENT_UNANALYZED"
            ),
            "status": "INCOMPLETE_RESOURCE_CEILING",
            "task": task_payload,
            "method_identity": {},
            "adapter_identity": {},
            "events": [],
            "total_cfe": 0,
            "total_atomic_model_steps": 0,
            "budget_accounting": {},
            "timeout_semantics": {},
            "runtime": {"cpu_seconds": 0.25},
            "permissions": {"results_analysis_performed": False},
            "charged_evaluation_count": 0,
            "individual_evaluation_rows_persisted": 0,
            "checkpoint_data_format": {},
            "event_summary_data_format": {},
        },
    )

    report = runner._r8c_e1e2_worker_report(
        task_directory,
        spec=spec,
    )
    payload = runner._normalized_worker_payload_from_report(
        report,
        task_directory=task_directory,
    )

    assert payload is not None
    assert payload["status"] == "INCOMPLETE_RESOURCE_CEILING"
    assert payload["status"] != "COMPLETE"
    failure_class = runner._worker_failure_class({}, payload)
    assert failure_class == runner.TECHNICAL_RESOURCE_TERMINATION
    assert runner._worker_failure_status(
        failure_class=failure_class,
        payload=payload,
    ) == "INCOMPLETE_RESOURCE_CEILING"


def test_directory_size_tolerates_file_disappearing_between_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    transient = tmp_path / "transient"
    transient.write_bytes(b"1234")
    original_stat = Path.stat
    target_calls = 0

    def flaky_stat(path: Path, *args, **kwargs):
        nonlocal target_calls
        if path == transient:
            target_calls += 1
            if target_calls >= 2:
                raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert runner._directory_bytes(tmp_path) == 0


def test_incremental_output_accounting_scans_root_once_and_tracks_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    tasks_root = output_root / "tasks"
    logs_root = output_root / "worker_logs"
    task_directory = tasks_root / "task-1"
    tasks_root.mkdir(parents=True)
    logs_root.mkdir()
    (output_root / "launch-binding.json").write_bytes(b"base")
    calls: list[Path] = []
    original_directory_bytes = runner._directory_bytes

    def counted_directory_bytes(path: Path) -> int:
        calls.append(Path(path).resolve())
        return original_directory_bytes(path)

    monkeypatch.setattr(
        runner,
        "_directory_bytes",
        counted_directory_bytes,
    )
    accounting = runner._IncrementalOutputAccounting(output_root)
    stdout_path = logs_root / "task-1.stdout.log"
    stderr_path = logs_root / "task-1.stderr.log"
    accounting.begin_worker(
        "task-1",
        task_directory=task_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    task_directory.mkdir()
    (task_directory / "checkpoint.cfe").write_bytes(b"12345")
    stdout_path.write_bytes(b"12")
    stderr_path.write_bytes(b"")

    assert accounting.current_bytes() == 11
    (task_directory / "checkpoint.cfe").write_bytes(b"123456789")
    stdout_path.write_bytes(b"123")
    assert accounting.current_bytes() == 16
    assert calls.count(output_root.resolve()) == 1
    assert calls.count(task_directory.resolve()) == 2


def test_incremental_output_accounting_completion_transfer_does_not_double_count(
    tmp_path: Path,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    task_directory = output_root / "tasks" / "task-1"
    logs_root = output_root / "worker_logs"
    task_directory.mkdir(parents=True)
    logs_root.mkdir()
    (output_root / "fixed-control").write_bytes(b"1234")
    accounting = runner._IncrementalOutputAccounting(output_root)
    stdout_path = logs_root / "task-1.stdout.log"
    stderr_path = logs_root / "task-1.stderr.log"
    accounting.begin_worker(
        "task-1",
        task_directory=task_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    (task_directory / "checkpoint.cfe").write_bytes(b"12345")
    stdout_path.write_bytes(b"12")
    stderr_path.write_bytes(b"1")

    assert accounting.current_bytes() == 12
    assert accounting.finish_worker("task-1") == 8
    assert accounting.current_bytes() == 12
    assert accounting.committed_worker_bytes == 8
    assert accounting.reserve_scope_count == 0
    with pytest.raises(ConfigurationError, match="inactive formal task"):
        accounting.finish_worker("task-1")


def test_incremental_output_accounting_reserve_covers_active_scan_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    task_directory = output_root / "tasks" / "task-1"
    logs_root = output_root / "worker_logs"
    task_directory.mkdir(parents=True)
    logs_root.mkdir()
    accounting = runner._IncrementalOutputAccounting(output_root)
    stdout_path = logs_root / "task-1.stdout.log"
    stderr_path = logs_root / "task-1.stderr.log"
    accounting.begin_worker(
        "task-1",
        task_directory=task_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    (task_directory / "created-after-scan").write_bytes(b"1234")
    monkeypatch.setattr(
        runner,
        "_worker_output_bytes",
        lambda scope: 0,
    )

    observed = accounting.current_bytes()
    reserve = accounting.reserve_bytes(
        control_plane_reserve=8,
        inflight_write_reserve_per_worker=4,
    )
    assert observed == accounting.startup_baseline_bytes
    assert observed + reserve >= runner._directory_bytes(output_root) + 8


def test_incremental_output_accounting_launch_failure_is_committed_and_reserved(
    tmp_path: Path,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    task_directory = output_root / "tasks" / "failed-task"
    logs_root = output_root / "worker_logs"
    (output_root / "tasks").mkdir(parents=True)
    logs_root.mkdir()
    accounting = runner._IncrementalOutputAccounting(output_root)
    stdout_path = logs_root / "failed-task.stdout.log"
    stderr_path = logs_root / "failed-task.stderr.log"
    accounting.begin_worker(
        "failed-task",
        task_directory=task_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    stdout_path.write_bytes(b"launch")
    stderr_path.write_bytes(b"failed")

    assert accounting.finish_worker(
        "failed-task",
        retain_inflight_reserve=True,
    ) == 12
    assert accounting.committed_worker_bytes == 12
    assert accounting.current_bytes() == 12
    assert accounting.reserve_scope_count == 1
    assert accounting.reserve_bytes(
        control_plane_reserve=10,
        inflight_write_reserve_per_worker=20,
    ) == 30


def test_run_manifest_final_output_bytes_include_the_manifest_itself(
    tmp_path: Path,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    (output_root / "existing.bin").write_bytes(b"x" * 257)
    manifest = {
        "status": "TEST",
        "resources": {"automatic_retries": 0},
    }
    before = runner._directory_bytes(output_root)
    payload, expected_total = (
        runner._run_manifest_payload_with_final_output_bytes(
            manifest,
            output_bytes_before_manifest=before,
        )
    )
    (output_root / "run_manifest.json").write_bytes(payload)
    assert runner._directory_bytes(output_root) == expected_total
    decoded = json.loads(payload)
    assert decoded["resources"]["total_output_bytes"] == expected_total
    assert decoded["resources"]["total_output_bytes_scope"] == (
        "ENTIRE_OUTPUT_ROOT_INCLUDING_THIS_RUN_MANIFEST"
    )


def test_worker_timeout_requests_graceful_checkpoint_before_hard_kill(
    tmp_path: Path,
) -> None:
    runner = _runner()

    class Process:
        def __init__(self) -> None:
            self.killed = False

        def poll(self):
            return None

        def kill(self) -> None:
            self.killed = True

    process = Process()
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    item = {
        "process": process,
        "task_directory": task_directory,
    }

    action = runner._advance_worker_timeout(
        item,
        now=100.0,
        runtime_seconds=10.0,
        timeout_seconds=10.0,
    )
    marker = task_directory / runner.TASK_TIMEOUT_MARKER_NAME
    assert action == "SOFT_TIMEOUT_REQUESTED"
    assert marker.read_bytes() == b"FORMAL_TASK_TIMEOUT_BEFORE_NEXT_CFE\n"
    assert process.killed is False

    assert (
        runner._advance_worker_timeout(
            item,
            now=100.0 + runner.TASK_TIMEOUT_GRACE_SECONDS - 0.01,
            runtime_seconds=20.0,
            timeout_seconds=10.0,
        )
        is None
    )
    assert process.killed is False

    assert runner._advance_worker_timeout(
        item,
        now=100.0 + runner.TASK_TIMEOUT_GRACE_SECONDS,
        runtime_seconds=20.0,
        timeout_seconds=10.0,
    ) == "HARD_TIMEOUT_AFTER_GRACE"
    assert process.killed is True
    assert item["hard_timed_out"] is True


def test_worker_resource_sample_tolerates_poll_proc_exit_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    class Process:
        def __init__(self) -> None:
            self.polls = iter((None, None, 0))

        def poll(self):
            return next(self.polls)

    def unavailable(_process_id: int) -> int:
        raise runner.HostSamplingError("RSS unavailable during exit")

    sleeps: list[float] = []
    monkeypatch.setattr(runner, "_process_rss_bytes", unavailable)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    assert runner._sample_live_worker_resources(Process(), 1234) is None
    assert sleeps == [runner.PROCESS_SAMPLE_EXIT_RACE_GRACE_SECONDS]


def test_worker_resource_sample_recovers_after_transient_proc_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    class Process:
        @staticmethod
        def poll():
            return None

    rss_samples = iter(
        (
            runner.HostSamplingError("transient proc gap"),
            123_456,
        )
    )

    def sample_rss(_process_id: int) -> int:
        value = next(rss_samples)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runner, "_process_rss_bytes", sample_rss)
    monkeypatch.setattr(runner, "_process_cpu_seconds", lambda _pid: 7.25)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    assert runner._sample_live_worker_resources(Process(), 1234) == (
        123_456,
        7.25,
    )


def test_worker_resource_sample_recovers_when_cpu_read_hits_exit_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    class Process:
        @staticmethod
        def poll():
            return None

    cpu_samples = iter(
        (
            runner.HostSamplingError("transient CPU proc gap"),
            3.5,
        )
    )

    def sample_cpu(_process_id: int) -> float:
        value = next(cpu_samples)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runner, "_process_rss_bytes", lambda _pid: 654_321)
    monkeypatch.setattr(runner, "_process_cpu_seconds", sample_cpu)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    assert runner._sample_live_worker_resources(Process(), 1234) == (
        654_321,
        3.5,
    )


def test_worker_resource_sample_final_poll_can_confirm_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    poll_results = iter(
        [None, None] * (runner.PROCESS_SAMPLE_MAX_ATTEMPTS - 1)
        + [None, 0]
    )

    class Process:
        @staticmethod
        def poll():
            return next(poll_results)

    def unavailable(_process_id: int) -> int:
        raise runner.HostSamplingError("exit at final bounded poll")

    monkeypatch.setattr(runner, "_process_rss_bytes", unavailable)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    assert runner._sample_live_worker_resources(Process(), 1234) is None


def test_worker_resource_sample_persistent_live_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    class Process:
        @staticmethod
        def poll():
            return None

    def unavailable(_process_id: int) -> int:
        raise runner.HostSamplingError("persistent sampling failure")

    monkeypatch.setattr(runner, "_process_rss_bytes", unavailable)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(runner.HostSamplingError, match="persistent"):
        runner._sample_live_worker_resources(Process(), 1234)


def test_prelaunch_rejects_candidate_and_qualified_worker_fields_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    marker = tmp_path / "request.consumed"
    gib = 1 << 30
    contract = {
        "launch": {"request_consumption_marker": str(marker)},
        "method_runtime": {
            "logical_threads_per_worker": 1,
            "blas_openmp_threads_per_worker": 1,
            "gpu_allowed_for_this_formal_scope": False,
        },
        "schedule": {"e3_dispatched": False},
        "resources": {
            "candidate_target": {
                "host_fingerprint_sha256": "f" * 64,
            },
            "parallelism": {
                "candidate_workers": 2,
                "max_workers": 2,
                "logical_threads_per_worker": 1,
                "blas_openmp_threads_per_worker": 1,
                "max_worker_peak_rss_bytes": gib,
                "max_pool_peak_rss_bytes": 2 * gib,
                "worker_count_qualified_on_target": True,
            },
            "monitor": {"interval_seconds": 1},
            "timeouts_seconds": {"global_formal_wall": 60},
            "output": {"max_total_bytes": gib},
            "max_total_cpu_seconds": 100,
            "scratch": {
                "stop_dispatch_below_free_bytes": gib,
                "minimum_free_bytes_at_start": 4 * gib,
            },
        },
    }
    args = argparse.Namespace()
    monkeypatch.setattr(
        runner,
        "_validate_source",
        lambda request: {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        runner,
        "_validate_invocation",
        lambda args, contract, request: output_root,
    )
    monkeypatch.setattr(
        runner,
        "host_fingerprint",
        lambda: {
            "effective_logical_processors": 4,
            "memory_bytes": 16 * gib,
        },
    )
    monkeypatch.setattr(
        runner,
        "host_fingerprint_sha256",
        lambda value=None: "f" * 64,
    )
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10 * gib),
    )
    schedule = [SimpleNamespace(workload_id="E1_STATIC")]
    with pytest.raises(
        ConfigurationError,
        match="candidate-only fields",
    ):
        runner._validate_prelaunch(
            args,
            contract,
            object(),
            schedule,
        )


def test_prelaunch_rejects_request_marker_inside_source_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    output_root = tmp_path / "formal-output"
    marker = runner.PROJECT_ROOT / "request-marker-must-not-be-created.json"
    gib = 1 << 30
    contract = {
        "launch": {"request_consumption_marker": str(marker)},
        "method_runtime": {
            "logical_threads_per_worker": 1,
            "blas_openmp_threads_per_worker": 1,
            "gpu_allowed_for_this_formal_scope": False,
        },
        "schedule": {"e3_dispatched": False},
        "resources": {
            "candidate_target": {
                "host_fingerprint_sha256": "f" * 64,
            },
            "parallelism": {
                "max_workers": 2,
                "logical_threads_per_worker": 1,
                "blas_openmp_threads_per_worker": 1,
                "max_worker_peak_rss_bytes": gib,
                "max_pool_peak_rss_bytes": 2 * gib,
                "worker_count_qualified_on_target": True,
            },
            "monitor": {"interval_seconds": 1},
            "timeouts_seconds": {"global_formal_wall": 60},
            "output": {"max_total_bytes": gib},
            "max_total_cpu_seconds": 100,
            "scratch": {
                "stop_dispatch_below_free_bytes": gib,
                "minimum_free_bytes_at_start": 4 * gib,
            },
        },
    }
    monkeypatch.setattr(
        runner,
        "_validate_source",
        lambda request: {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        runner,
        "_validate_invocation",
        lambda args, frozen, request: output_root,
    )
    monkeypatch.setattr(
        runner,
        "host_fingerprint",
        lambda: {
            "effective_logical_processors": 4,
            "memory_bytes": 16 * gib,
        },
    )
    monkeypatch.setattr(
        runner,
        "host_fingerprint_sha256",
        lambda value=None: "f" * 64,
    )
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10 * gib),
    )

    with pytest.raises(ConfigurationError, match="outside the source worktree"):
        runner._validate_prelaunch(
            argparse.Namespace(),
            contract,
            object(),
            [SimpleNamespace(workload_id="E1_STATIC")],
        )


def test_runner_strictly_validates_rng_amendment_and_bound_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    runner._validate_rng_implementation_amendment(upstream)

    wrong_hash = deepcopy(upstream)
    wrong_hash["rng_implementation_amendment"]["sha256"] = "0" * 64
    with pytest.raises(ConfigurationError, match="amendment drifted"):
        runner._validate_rng_implementation_amendment(wrong_hash)

    amendment = json.loads(
        (
            runner.PROJECT_ROOT
            / upstream["rng_implementation_amendment"]["path"]
        ).read_text(encoding="utf-8")
    )
    implementation_path = (
        runner.PROJECT_ROOT
        / amendment["current_equivalent_implementation"]["path"]
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def drift_current_implementation(path: Path) -> str:
        if Path(path).resolve() == implementation_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(
        runner,
        "file_sha256",
        drift_current_implementation,
    )
    with pytest.raises(ConfigurationError, match="implementation drifted"):
        runner._validate_rng_implementation_amendment(upstream)


def test_runner_strictly_validates_timeout_semantics_amendment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    runner._validate_timeout_semantics_amendment(upstream)

    wrong_hash = deepcopy(upstream)
    wrong_hash["timeout_semantics_amendment"]["sha256"] = "0" * 64
    with pytest.raises(
        ConfigurationError,
        match="timeout semantics amendment drifted",
    ):
        runner._validate_timeout_semantics_amendment(wrong_hash)

    binding = upstream["timeout_semantics_amendment"]
    schema_path = (
        runner.PROJECT_ROOT / binding["schema_path"]
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def drift_schema(path: Path) -> str:
        if Path(path).resolve() == schema_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runner, "file_sha256", drift_schema)
    with pytest.raises(
        ConfigurationError,
        match="timeout semantics amendment schema drifted",
    ):
        runner._validate_timeout_semantics_amendment(upstream)


def test_runner_strictly_validates_lircmop_reference_amendment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    runner._validate_lircmop_reference_amendment(upstream)

    wrong_hash = deepcopy(upstream)
    wrong_hash["lircmop_reference_amendment"]["sha256"] = "0" * 64
    with pytest.raises(
        ConfigurationError,
        match="LIR-CMOP/reference amendment drifted",
    ):
        runner._validate_lircmop_reference_amendment(wrong_hash)

    wrong_schema_hash = deepcopy(upstream)
    wrong_schema_hash["lircmop_reference_amendment"][
        "schema_sha256"
    ] = "0" * 64
    with pytest.raises(
        ConfigurationError,
        match="amendment schema drifted",
    ):
        runner._validate_lircmop_reference_amendment(wrong_schema_hash)

    binding = upstream["lircmop_reference_amendment"]
    amendment = json.loads(
        (runner.PROJECT_ROOT / binding["path"]).read_text(encoding="utf-8")
    )
    reference_path = (
        runner.PROJECT_ROOT
        / amendment["implementation_bindings"]["reference_artifacts"]["path"]
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def drift_reference_artifacts(path: Path) -> str:
        if Path(path).resolve() == reference_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(
        runner,
        "file_sha256",
        drift_reference_artifacts,
    )
    with pytest.raises(
        ConfigurationError,
        match="reference_artifacts drifted",
    ):
        runner._validate_lircmop_reference_amendment(upstream)


def test_runner_strictly_validates_reference_catalog_and_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    manifest = runner._validate_reference_catalog_binding(upstream)
    assert manifest["identity_scope"]["actual_total"] == 2_294

    wrong_hash = deepcopy(upstream)
    wrong_hash["reference_catalog"]["artifact_sha256"] = "0" * 64
    with pytest.raises(
        ConfigurationError,
        match="reference catalog artifact drifted",
    ):
        runner._validate_reference_catalog_binding(wrong_hash)

    generator_path = (
        runner.PROJECT_ROOT
        / manifest["source_bindings"]["generator"]["path"]
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def drift_generator(path: Path) -> str:
        if Path(path).resolve() == generator_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runner, "file_sha256", drift_generator)
    with pytest.raises(
        ConfigurationError,
        match="reference catalog generator drifted",
    ):
        runner._validate_reference_catalog_binding(upstream)


def test_runner_strictly_validates_cdf_operational_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    runner._validate_cdf_operational_authority_amendment(upstream)

    wrong_hash = deepcopy(upstream)
    wrong_hash["cdf_operational_authority_amendment"]["sha256"] = "0" * 64
    with pytest.raises(
        ConfigurationError,
        match="CDF operational authority amendment drifted",
    ):
        runner._validate_cdf_operational_authority_amendment(wrong_hash)

    binding = upstream["cdf_operational_authority_amendment"]
    amendment = json.loads(
        (runner.PROJECT_ROOT / binding["path"]).read_text(encoding="utf-8")
    )
    evaluator_path = (
        runner.PROJECT_ROOT
        / amendment["implementation_bindings"]["corrective_evaluator"]["path"]
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def drift_evaluator(path: Path) -> str:
        if Path(path).resolve() == evaluator_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runner, "file_sha256", drift_evaluator)
    with pytest.raises(
        ConfigurationError,
        match="CDF operational corrective_evaluator drifted",
    ):
        runner._validate_cdf_operational_authority_amendment(upstream)


def test_runner_requires_exact_target_linux_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    contract = json.loads(
        (
            runner.PROJECT_ROOT
            / "config"
            / "r8c_e1e2"
            / "r8c_e1e2_formal_execution_contract.json"
        ).read_text(encoding="utf-8")
    )
    upstream = contract["upstream"]
    target_environment = _target_runtime_environment(runner)
    monkeypatch.setattr(
        runner,
        "_runtime_environment_lock_evidence",
        lambda: deepcopy(target_environment),
    )
    runner._validate_linux_runtime_lock(upstream, target_environment)

    report_drift = deepcopy(target_environment)
    report_drift["actual_python_version"] = "3.12.1"
    with pytest.raises(
        ConfigurationError,
        match="qualification environment differs",
    ):
        runner._validate_linux_runtime_lock(upstream, report_drift)

    wrong_count = deepcopy(upstream)
    wrong_count["linux_runtime_lock"]["locked_package_count"] = 36
    with pytest.raises(
        ConfigurationError,
        match="runtime lock target differs",
    ):
        runner._validate_linux_runtime_lock(wrong_count, target_environment)


def test_worker_launch_validation_checks_token_before_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    monkeypatch.delenv(runner.LAUNCH_TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        runner,
        "_validate_source",
        lambda request: pytest.fail("source must not be read without token"),
    )
    with pytest.raises(ConfigurationError, match="launch token"):
        runner._validate_worker_launch(
            args=argparse.Namespace(),
            profile=runner.CORRECTIVE_E1E2_PROFILE,
            contract={},
            request=object(),
            schedule=[],
            spec=object(),
        )


def test_worker_environment_binds_one_thread_and_token() -> None:
    runner = _runner()
    environment = runner._worker_environment("secret-token")
    assert environment[runner.LAUNCH_TOKEN_ENV] == "secret-token"
    assert environment["PYTHONHASHSEED"] == "0"
    assert all(
        environment[name] == "1"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    )
    assert os.pathsep in environment["PYTHONPATH"] or environment["PYTHONPATH"]


def test_failed_worker_manifests_partial_evidence_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    spec = SimpleNamespace(
        task_id="r8-test-partial",
        to_dict=lambda: {
            "task_id": "r8-test-partial",
            "schedule_index": 0,
        },
    )
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

    def fail_after_partial_write(**kwargs):
        task_directory = kwargs["task_directory"]
        task_directory.mkdir()
        (task_directory / "checkpoint_fronts.cfe").write_bytes(b"partial")
        (task_directory / "heartbeat").write_bytes(b"")
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(runner, "run_task", fail_after_partial_write)
    task_directory = tmp_path / "task"
    code = runner._run_worker(
        argparse.Namespace(
            execution_profile="legacy_r8",
            contract=str(tmp_path / "contract.json"),
            request=str(tmp_path / "request.json"),
            schedule_index=0,
            task_id=spec.task_id,
            task_directory=str(task_directory),
            stop_path=str(tmp_path / "STOP"),
        )
    )

    assert code == 3
    assert (task_directory / "checkpoint_fronts.cfe").read_bytes() == (
        b"partial"
    )
    assert not (task_directory / "heartbeat").exists()
    failure = json.loads(
        (task_directory / "task_failure.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (task_directory / "task_manifest.json").read_text(encoding="utf-8")
    )
    assert failure["automatic_retries"] == 0
    assert manifest["status"] == "TASK_FAILED_NO_RETRY"
    assert set(manifest["artifacts"]) == {
        "checkpoint_fronts.cfe",
        "task_failure.json",
    }
    reported = json.loads(capsys.readouterr().out)
    assert reported["task_manifest_sha256"] == runner.file_sha256(
        task_directory / "task_manifest.json"
    )
