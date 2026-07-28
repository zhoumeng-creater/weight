from __future__ import annotations

import json
from pathlib import Path
from tempfile import gettempdir

import pytest

from resource_pilot.r8c import (
    ARCHIVE_CAPACITY,
    ARTIFACT_ROLE,
    DEFAULT_OUTPUT_ROOT,
    FORBIDDEN_FORMAL_ROOT,
    POPULATION_SIZE,
    ResourcePilotError,
    _run_task,
    assert_control_plane_only,
    build_tasks,
    run_resource_pilot,
    validate_output_root,
)


def test_default_roots_are_portable_system_temporary_paths(
) -> None:
    system_temp = Path(gettempdir()).resolve()
    assert DEFAULT_OUTPUT_ROOT.is_absolute()
    assert FORBIDDEN_FORMAL_ROOT.is_absolute()
    assert DEFAULT_OUTPUT_ROOT.resolve().is_relative_to(system_temp)
    assert FORBIDDEN_FORMAL_ROOT.resolve().is_relative_to(system_temp)
    assert "DT-RAMDE-v11-runs" not in DEFAULT_OUTPUT_ROOT.as_posix()
    assert "DT-RAMDE-v11-runs" not in FORBIDDEN_FORMAL_ROOT.as_posix()


def test_task_matrix_has_all_profiles_and_unique_pilot_ids() -> None:
    tasks = build_tasks(
        worker_count=32,
        repetitions_per_profile=6,
        cfe_per_event=500,
        dynamic_events=2,
    )
    assert len(tasks) == 36
    assert len({task.task_id for task in tasks}) == 36
    assert all(
        task.task_id.startswith("r8c-resource-pilot-w32-")
        for task in tasks
    )
    assert {task.profile.profile_id for task in tasks} == {
        "E1_STATIC",
        "E1_DYNAMIC",
        "E1_ROLLING",
        "E2_DYNAMIC",
        "E2_ROLLING",
        "E3_SUPPORTIVE",
    }


def test_single_task_is_100_100_r6_control_plane_only() -> None:
    task = build_tasks(
        worker_count=1,
        repetitions_per_profile=1,
        cfe_per_event=100,
        dynamic_events=1,
    )[0]
    record = _run_task(task)
    assert record["artifact_role"] == ARTIFACT_ROLE
    assert record["total_cfe"] == 100
    assert record["formal_request_loaded"] is False
    assert record["formal_request_consumed"] is False
    assert record["effect_fields_persisted"] is False
    assert record["effect_analysis_performed"] is False
    assert POPULATION_SIZE == 100
    assert ARCHIVE_CAPACITY == 100
    assert_control_plane_only(record)


def test_effect_bearing_key_is_rejected_recursively() -> None:
    with pytest.raises(ResourcePilotError, match="effect-bearing key"):
        assert_control_plane_only({"nested": [{"objectives": [1.0]}]})


def test_output_root_is_single_use_and_cannot_be_formal(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "r8c-resource-pilot-test"
    validate_output_root(valid)
    valid.mkdir()
    with pytest.raises(ResourcePilotError, match="already exists"):
        validate_output_root(valid)
    with pytest.raises(ResourcePilotError, match="formal root"):
        validate_output_root(FORBIDDEN_FORMAL_ROOT)


def test_smoke_sweep_persists_only_control_plane(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r8c-resource-pilot-pytest-smoke"
    report = run_resource_pilot(
        output_root=output,
        worker_counts=(1,),
        repetitions_per_profile=1,
        cfe_per_event=100,
        dynamic_events=1,
        require_clean_git=False,
    )
    assert report["status"] == "PASS"
    assert report["pilot_design"]["task_count"] == 6
    assert report["target_resource_gate"]["formal_go"] is False
    assert report["target_resource_gate"]["status"] == (
        "NO_GO_TARGET_HOST_UNMEASURED"
    )
    assert (
        report["runtime_contract"]["comparator_family_throughput_measured"]
        is False
    )
    assert (
        report["runtime_contract"][
            "raw_gzip_and_filesystem_contention_measured"
        ]
        is False
    )
    assert_control_plane_only(report)
    persisted = json.loads(
        (output / "resource_qualification_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert_control_plane_only(persisted)
    task_rows = [
        json.loads(line)
        for line in (
            output
            / "workers-01"
            / "task_control_metrics.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(task_rows) == 6
    assert all(row["automatic_retries"] == 0 for row in task_rows)
    assert_control_plane_only(task_rows)
