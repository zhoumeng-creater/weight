from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.evaluator import ExecutionTimeoutBeforeEntry
from formal_execution import runtime
from formal_execution.runtime import (
    CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
)
from formal_execution.schedule import FormalSequenceSpec


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


def test_timeout_marker_after_evaluations_blocks_summary_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_directory = tmp_path / "task"
    marker = task_directory / "TASK_TIMEOUT_REQUESTED"
    original_exit = runtime.EndpointCheckpointWriter.__exit__

    def exit_and_request_timeout(self, exc_type, exc, traceback):
        original_exit(self, exc_type, exc, traceback)
        marker.write_bytes(b"FORMAL_TASK_TIMEOUT_BEFORE_NEXT_CFE\n")

    monkeypatch.setattr(
        runtime.EndpointCheckpointWriter,
        "__exit__",
        exit_and_request_timeout,
    )
    with pytest.raises(
        ExecutionTimeoutBeforeEntry,
        match="summary publication",
    ):
        runtime.run_task(
            spec=_spec(),
            request=SimpleNamespace(),
            task_directory=task_directory,
            stop_path=tmp_path / "STOP",
            settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
        )
    assert not (task_directory / "task_summary.json").exists()
    assert not (task_directory / "task_manifest.json").exists()


def test_timeout_marker_after_summary_blocks_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_directory = tmp_path / "task"
    summary_path = task_directory / "task_summary.json"
    marker = task_directory / "TASK_TIMEOUT_REQUESTED"
    original_write = runtime._write_canonical_json_exclusive_fsynced

    def write_and_request_timeout(
        path: Path,
        value: object,
        *,
        maximum_bytes: int | None = None,
    ) -> None:
        original_write(
            path,
            value,
            maximum_bytes=maximum_bytes,
        )
        if path == summary_path:
            marker.write_bytes(
                b"FORMAL_TASK_TIMEOUT_BEFORE_NEXT_CFE\n"
            )

    monkeypatch.setattr(
        runtime,
        "_write_canonical_json_exclusive_fsynced",
        write_and_request_timeout,
    )
    with pytest.raises(
        ExecutionTimeoutBeforeEntry,
        match="manifest publication",
    ):
        runtime.run_task(
            spec=_spec(),
            request=SimpleNamespace(),
            task_directory=task_directory,
            stop_path=tmp_path / "STOP",
            settings=CORRECTIVE_R8C_E1E2_RUNTIME_SETTINGS,
        )
    assert summary_path.is_file()
    assert not (task_directory / "task_manifest.json").exists()
