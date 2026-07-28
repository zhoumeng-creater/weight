from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.contracts import EvaluationResult
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry
from evaluation.ledger import EvaluationLedger
from benchmark_adapters.cdf_operational import (
    CDF_OPERATIONAL_AUTHORITY_ID,
    CDF_OPERATIONAL_SUITE_ID,
)
from formal_execution.checkpoint_data import (
    CHECKPOINTS_PER_EVENT,
    read_checkpoint_file,
)
import resource_pilot.e1e2_fullpath as qualification
from resource_pilot.e1e2_fullpath import (
    ARCHIVE_CAPACITY,
    CHECKPOINT_FILENAME,
    DEFAULT_CFE_PER_EVENT,
    DEFAULT_DYNAMIC_EVENTS,
    DEFAULT_REPETITIONS,
    DEFAULT_WORKERS,
    DYNAMIC_CFE_PER_EVENT,
    ENDPOINT_SUFFICIENT_FORMAT,
    FORMAL_METHODS_BY_WORKLOAD,
    POPULATION_SIZE,
    ROLLING_CFE_PER_EVENT,
    STATIC_CFE_PER_EVENT,
    E1E2QualificationError,
    _run_task,
    _build_problem,
    _synthetic_candidate_id,
    _synthetic_float,
    assert_control_plane_only,
    build_tasks,
    code_identity,
    formal_schedule_weights,
    qualification_profiles,
    run_e1e2_qualification,
    validate_output_root,
)


def test_target_qualification_requires_exact_frozen_sample_design() -> None:
    baseline = {
        "worker_counts": DEFAULT_WORKERS,
        "repetitions": DEFAULT_REPETITIONS,
        "static_cfe_per_event": STATIC_CFE_PER_EVENT,
        "dynamic_cfe_per_event": DYNAMIC_CFE_PER_EVENT,
        "rolling_cfe_per_event": ROLLING_CFE_PER_EVENT,
        "dynamic_events": DEFAULT_DYNAMIC_EVENTS,
        "smoke": False,
        "allow_dirty": False,
        "worktree_clean": True,
        "target_environment_match": True,
        "failed_count": 0,
    }
    assert qualification._is_target_qualifying_design(**baseline)

    for changed in (
        {"worker_counts": (1, 8, 16, 24)},
        {"repetitions": 1},
        {"repetitions": 3},
        {"static_cfe_per_event": 5_000},
        {"static_cfe_per_event": 50_100},
        {"dynamic_cfe_per_event": 100},
        {"dynamic_cfe_per_event": 5_100},
        {"rolling_cfe_per_event": 100},
        {"rolling_cfe_per_event": 5_100},
        {"dynamic_events": 1},
        {"dynamic_events": 2},
        {"dynamic_events": 3},
        {"dynamic_events": 5},
        {"dynamic_events": 7},
        {"target_environment_match": False},
    ):
        candidate = {**baseline, **changed}
        assert not qualification._is_target_qualifying_design(**candidate)


def test_uniform_cfe_override_is_smoke_only(tmp_path: Path) -> None:
    with pytest.raises(E1E2QualificationError, match="only for.*smoke"):
        run_e1e2_qualification(
            output_root=(
                tmp_path / "r8c-e1e2-qualification-uniform-refused"
            ),
            cfe_per_event=100,
            smoke=False,
        )


def test_batch_unavailable_before_entry_leaves_synthetic_stream_untouched() -> None:
    class Problem:
        def evaluate_batch(self, vectors, event_id, ledger, candidate_ids):
            del vectors, event_id, ledger, candidate_ids
            raise BatchEvaluationUnavailableBeforeEntry("forced scalar path")

        def evaluate(self, vector, event_id, ledger, candidate_id):
            del vector
            ledger.charge_candidate(
                candidate_id=candidate_id,
                event_id=event_id,
                atomic_steps=1,
            )
            return EvaluationResult(
                candidate_id=candidate_id,
                objectives=(1.0, 2.0),
                objective_names=("f1", "f2"),
                constraints=(-1.0,),
                constraint_names=("g1",),
            )

    class Writer:
        def __init__(self) -> None:
            self.successes = 0
            self.failures = 0

        def record_success(self, **kwargs) -> None:
            del kwargs
            self.successes += 1

        def record_failure(self, **kwargs) -> None:
            del kwargs
            self.failures += 1

    writer = Writer()
    adapter = qualification._SyntheticRecordingAdapter(Problem(), writer)
    ledger = EvaluationLedger(max_cfe=2)
    with pytest.raises(
        BatchEvaluationUnavailableBeforeEntry,
        match="forced scalar path",
    ):
        adapter.evaluate_batch(
            ((0.1,), (0.2,)),
            0,
            ledger,
            ("batch-0", "batch-1"),
        )

    assert ledger.cfe == 0
    assert adapter.synthetic_record_count == 0
    assert writer.successes == writer.failures == 0

    adapter.evaluate((0.1,), 0, ledger, "scalar-0")
    assert ledger.cfe == 1
    assert adapter.synthetic_record_count == 1
    assert writer.successes + writer.failures == 1


def test_profile_matrix_covers_exact_frozen_schedule_paths() -> None:
    profiles = qualification_profiles()
    profile_keys = {profile.key for profile in profiles}
    case_keys = {profile.case_key for profile in profiles}
    rate_keys = {profile.rate_key for profile in profiles}
    expected = {
        (workload, method)
        for workload, methods in FORMAL_METHODS_BY_WORKLOAD.items()
        for method in methods
    }
    weights = formal_schedule_weights()
    weight_keys = {
        (
            str(row["workload_id"]),
            str(row["method_id"]),
            str(row["projection_rate_class"]),
        )
        for row in weights
    }
    assert len(profiles) == len(case_keys) == 84
    assert len(profile_keys) == 33
    assert profile_keys == expected
    assert rate_keys == weight_keys
    assert len(rate_keys) == 39
    assert {
        profile.representative_case_id for profile in profiles
    } == {
        "LIRCMOP1_2D_HEAD",
        "LIRCMOP12_2D_TAIL",
        "LIRCMOP14_3D",
        "CDF1_HARSH_BASE",
        "CDF9_HARSH_DOMAIN_FALLBACK",
        "CDF13_HARSH_SEED_DEPENDENT",
        "CDF15_MILD_TAIL",
        "WGT_RR_KNOWN_ANSWER",
    }
    assert all(
        profile.events == 6
        for profile in profiles
        if profile.adapter_kind in {"cdf", "rolling"}
    )
    assert len(FORMAL_METHODS_BY_WORKLOAD) == 5
    assert sum(int(row["formal_task_count"]) for row in weights) == 5_030
    assert sum(int(row["formal_cfe"]) for row in weights) == 851_000_000
    assert POPULATION_SIZE == 100
    assert ARCHIVE_CAPACITY == 100
    assert DEFAULT_CFE_PER_EVENT == 5_000


def test_static_qualification_uses_corrective_paper_evaluator() -> None:
    task = next(
        item
        for item in build_tasks(
            output_root=Path("r8c-e1e2-qualification-identity-test"),
            worker_count=1,
            repetitions=1,
            cfe_per_event=100,
            dynamic_events=1,
        )
        if item.profile.representative_case_id == "LIRCMOP14_3D"
        and item.profile.method_id == "F22_MG_STATIC"
    )
    identity = _build_problem(task).identity()
    assert identity["target_suite_id"] == "LIR-CMOP-PAPER-2019-TABLE-8"
    assert identity["target_problem_id"] == "LIRCMOP14"


def test_dynamic_qualification_uses_corrective_cdf_operational_evaluator() -> None:
    task = next(
        item
        for item in build_tasks(
            output_root=Path("r8c-e1e2-qualification-cdf-identity-test"),
            worker_count=1,
            repetitions=1,
            cfe_per_event=100,
            dynamic_events=1,
        )
        if item.profile.representative_case_id
        == "CDF9_HARSH_DOMAIN_FALLBACK"
        and item.profile.method_id == "DT-RAMDE_TS2_FULL"
    )
    identity = _build_problem(task).identity()
    assert identity["target_suite_id"] == CDF_OPERATIONAL_SUITE_ID
    assert identity["target_problem_id"] == "CDF9"
    assert identity["profile"] == "CDF-HARSH"


def test_target_tasks_use_full_workload_specific_event_budgets() -> None:
    tasks = build_tasks(
        output_root=Path("r8c-e1e2-qualification-budget-identity-test"),
        worker_count=1,
        repetitions=1,
        dynamic_events=DEFAULT_DYNAMIC_EVENTS,
    )
    assert {
        task.cfe_per_event
        for task in tasks
        if task.profile.adapter_kind == "lircmop"
    } == {STATIC_CFE_PER_EVENT}
    assert {
        task.cfe_per_event
        for task in tasks
        if task.profile.adapter_kind == "cdf"
    } == {DYNAMIC_CFE_PER_EVENT}
    assert {
        task.cfe_per_event
        for task in tasks
        if task.profile.adapter_kind == "rolling"
    } == {ROLLING_CFE_PER_EVENT}


def test_cdf9_typed_numerical_terminal_remains_a_qualifying_task_outcome(
    tmp_path: Path,
) -> None:
    task = next(
        item
        for item in build_tasks(
            output_root=tmp_path / "r8c-e1e2-qualification-cdf9-typed",
            worker_count=1,
            repetitions=1,
            cfe_per_event=100,
            dynamic_events=DEFAULT_DYNAMIC_EVENTS,
        )
        if item.profile.workload_id == "E1_DYNAMIC"
        and item.profile.method_id == "DT-RAMDE_TS2_FULL"
        and item.profile.representative_case_id
        == "CDF9_HARSH_DOMAIN_FALLBACK"
    )
    record = _run_task(task)
    assert record["status"] == "PASS"
    assert record["cfe_consumed"] < record["scheduled_cfe"]
    assert record["unconsumed_cfe_due_to_typed_terminal"] > 0
    decoded = read_checkpoint_file(
        task.task_directory / CHECKPOINT_FILENAME
    )
    assert any(row.kind == "terminal" for row in decoded.records)


def test_dt_ramde_task_uses_r6_and_writes_deterministic_2d_checkpoints(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r8c-e1e2-qualification-task"
    tasks = build_tasks(
        output_root=output,
        worker_count=1,
        repetitions=1,
        cfe_per_event=100,
        dynamic_events=1,
    )
    task = next(
        item
        for item in tasks
        if item.profile.workload_id == "E1_STATIC"
        and item.profile.method_id == "F22_MG_STATIC"
    )
    record = _run_task(task)
    assert record["status"] == "PASS"
    assert record["population_size"] == 100
    assert record["archive_capacity"] == 100
    assert record["cfe_consumed"] == 100
    assert record["checkpoint_evaluation_count"] == 100
    assert record["checkpoint_synthetic_success_count"] == 98
    assert record["checkpoint_synthetic_failure_count"] == 2
    assert record["checkpoint_record_count"] == CHECKPOINTS_PER_EVENT
    assert record["production_checkpoint_writer_used"] is True
    assert record["endpoint_sufficient_format"] == ENDPOINT_SUFFICIENT_FORMAT
    assert record["r6_engineering_request_used_for_dt_ramde"] is True
    assert record["real_effect_values_persisted"] is False
    assert_control_plane_only(record)

    checkpoint_path = task.task_directory / CHECKPOINT_FILENAME
    decoded = read_checkpoint_file(checkpoint_path)
    assert decoded.metadata.objective_dimension == 2
    assert len(decoded.records) == CHECKPOINTS_PER_EVENT
    assert [row.checkpoint_index for row in decoded.records] == list(range(21))
    assert [row.cfe for row in decoded.records] == list(range(0, 101, 5))
    assert decoded.records[-1].success_count == 98
    assert decoded.records[-1].failure_count == 2
    assert all(
        len(point) == 2
        for row in decoded.records
        for point in row.front_objectives
    )
    assert len(
        {row.evaluation_chain_sha256 for row in decoded.records}
    ) == CHECKPOINTS_PER_EVENT

    duplicate_output = tmp_path / "r8c-e1e2-qualification-task-copy"
    duplicate_task = next(
        item
        for item in build_tasks(
            output_root=duplicate_output,
            worker_count=1,
            repetitions=1,
            cfe_per_event=100,
            dynamic_events=1,
        )
        if item.profile.workload_id == "E1_STATIC"
        and item.profile.method_id == "F22_MG_STATIC"
    )
    duplicate_record = _run_task(duplicate_task)
    assert duplicate_record["status"] == "PASS"
    assert checkpoint_path.read_bytes() == (
        duplicate_task.task_directory / CHECKPOINT_FILENAME
    ).read_bytes()

    assert _synthetic_candidate_id(1) != _synthetic_candidate_id(2)
    assert _synthetic_float(1, 1, 0) != _synthetic_float(2, 1, 0)
    assert _synthetic_float(1, 2, 0) != _synthetic_float(2, 2, 0)


def test_rolling_task_writes_3d_twenty_one_point_checkpoint_semantics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r8c-e1e2-qualification-rolling"
    task = next(
        item
        for item in build_tasks(
            output_root=output,
            worker_count=1,
            repetitions=1,
            cfe_per_event=100,
            dynamic_events=1,
        )
        if item.profile.workload_id == "E1_ROLLING"
        and item.profile.method_id == "DT-RAMDE_TS2_FULL"
    )
    record = _run_task(task)
    assert record["status"] == "PASS"
    assert record["checkpoint_evaluation_count"] == 100
    assert record["checkpoint_synthetic_failure_count"] == 2

    decoded = read_checkpoint_file(
        task.task_directory / CHECKPOINT_FILENAME
    )
    assert decoded.metadata.objective_dimension == 3
    assert len(decoded.records) == CHECKPOINTS_PER_EVENT
    assert [row.checkpoint_index for row in decoded.records] == list(range(21))
    assert [row.cfe for row in decoded.records] == list(range(0, 101, 5))
    assert all(
        len(point) == 3
        for row in decoded.records
        for point in row.front_objectives
    )


def test_matched_comparator_calls_real_optimize_and_exact_budget(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r8c-e1e2-qualification-comparator"
    tasks = build_tasks(
        output_root=output,
        worker_count=1,
        repetitions=1,
        cfe_per_event=100,
        dynamic_events=1,
    )
    task = next(
        item
        for item in tasks
        if item.profile.workload_id == "E1_STATIC"
        and item.profile.method_id == "MATCHED_FIXED_DE_PARETO"
    )
    record = _run_task(task)
    assert record["status"] == "PASS"
    assert record["real_comparator_optimize_called"] is True
    assert record["cfe_consumed"] == 100
    assert record["checkpoint_evaluation_count"] == 100
    assert record["checkpoint_synthetic_success_count"] == 98
    assert record["checkpoint_synthetic_failure_count"] == 2
    assert record["production_checkpoint_writer_used"] is True
    assert record["automatic_retries"] == 0


def test_clean_git_is_default_and_dirty_override_is_nonqualifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(*args: str) -> str:
        if args[0] == "status":
            return "?? synthetic-dirty-file"
        if args[-1] == "HEAD^{tree}":
            return "b" * 40
        return "a" * 40

    monkeypatch.setattr(qualification, "_git", fake_git)
    with pytest.raises(E1E2QualificationError, match="clean committed"):
        code_identity(allow_dirty=False)
    identity = code_identity(allow_dirty=True)
    assert identity["worktree_clean"] is False


def test_output_root_is_independent_and_single_use(tmp_path: Path) -> None:
    with pytest.raises(E1E2QualificationError, match="absolute"):
        validate_output_root(
            Path("r8c-e1e2-qualification-relative")
        )
    wrong = tmp_path / "qualification"
    with pytest.raises(E1E2QualificationError, match="must start"):
        validate_output_root(wrong)
    output = tmp_path / "r8c-e1e2-qualification-once"
    validate_output_root(output)
    output.mkdir()
    with pytest.raises(E1E2QualificationError, match="already exists"):
        validate_output_root(output)
    inside = (
        qualification.PROJECT_ROOT
        / "r8c-e1e2-qualification-inside-worktree"
    )
    assert not inside.exists()
    with pytest.raises(E1E2QualificationError, match="outside"):
        validate_output_root(inside)


def test_smoke_sweep_covers_all_paths_and_json_is_control_plane_only(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jmetal")
    output = tmp_path / "r8c-e1e2-qualification-smoke"
    report = run_e1e2_qualification(
        output_root=output,
        worker_counts=(1,),
        repetitions=1,
        cfe_per_event=100,
        dynamic_events=1,
        allow_dirty=True,
        smoke=True,
    )
    assert report["status"] == "PASS_NONQUALIFYING_DIAGNOSTIC"
    assert report["pilot_design"]["task_count"] == 84
    assert report["failed_task_count"] == 0
    assert report["target_qualification_complete"] is False
    assert report["formal_launch_authorized"] is False
    runtime = report["runtime_contract"]
    assert runtime["production_checkpoint_writer_used"] is True
    assert runtime["endpoint_sufficient_format"] == ENDPOINT_SUFFICIENT_FORMAT
    assert runtime["checkpoint_points_per_event"] == CHECKPOINTS_PER_EVENT
    assert runtime["workload_method_paths_covered"] == 33
    assert runtime["workload_method_case_bindings_covered"] == 84
    assert runtime["unique_representative_benchmark_cases_covered"] == 8
    assert runtime["formal_projection_rate_classes_covered"] == 39
    assert runtime["cdf_operational_suite_id"] == CDF_OPERATIONAL_SUITE_ID
    assert (
        runtime["cdf_operational_authority_id"]
        == CDF_OPERATIONAL_AUTHORITY_ID
    )
    assert runtime["cdf_operational_authority_amendment_id"] == (
        qualification.CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
    )
    assert runtime["dynamic_event_ids_covered"] == [0]
    assert runtime["cdf9_max_domain_stress_event_id"] == 5
    assert runtime["cdf9_max_domain_stress_event_exercised"] is False
    assert (
        runtime["formal_checkpoint_storage_conservative_upper_bound_bytes"]
        == 6_731_786_240
    )
    assert len(
        report["e1_e2_wall_projection"]["projections"][0][
            "method_cfe_weighted_rates"
        ]
    ) == 39
    projection = report["e1_e2_wall_projection"]["projections"][0]
    assert projection["production_checkpoint_writer_used"] is True
    assert projection["projected_wall_hours_with_25_percent_headroom"] == (
        projection["projected_wall_hours"] * 1.25
    )
    memory = projection["memory_qualification"]
    assert memory["rss_safety_factor"] == 1.25
    assert memory["conservative_worker_peak_rss_bytes"] >= (
        memory["measured_max_worker_peak_rss_bytes"] * 1.25
    )
    assert (
        report["worker_recommendation"]["recommended_worker_count"]
        in {None, 1}
    )
    assert (
        projection["formal_checkpoint_storage_conservative_upper_bound_bytes"]
        == 6_731_786_240
    )
    assert_control_plane_only(report)

    checkpoint_paths = tuple(output.rglob("*.cfe"))
    assert len(checkpoint_paths) == 84
    assert not tuple(output.rglob("*.gz"))
    for path in output.rglob("*.json"):
        assert_control_plane_only(
            json.loads(path.read_text(encoding="utf-8"))
        )
    for path in output.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert_control_plane_only(json.loads(line))
