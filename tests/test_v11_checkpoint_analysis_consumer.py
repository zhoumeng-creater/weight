from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.contracts import EvaluationResult
from analysis import (
    AnalyticReferenceScale,
    CheckpointAnalysisError,
    IncompleteCheckpointDataError,
    NumericalContinuousEndpointExcluded,
    read_complete_task_nhv,
)
from analysis import checkpoint_consumer
from formal_execution.checkpoint_data import (
    CheckpointMetadata,
    TaskCheckpointWriter,
    read_checkpoint_file,
)


def _result(
    candidate_id: str,
    objectives: tuple[float, ...],
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        objectives=objectives,
        objective_names=tuple(
            f"objective_{index}" for index in range(len(objectives))
        ),
        constraints=(-1.0,),
        constraint_names=("constraint",),
    )


def _scale_2d() -> AnalyticReferenceScale:
    axis = np.linspace(0.0, 1.0, 10_000)
    return AnalyticReferenceScale.from_reference_front(
        np.column_stack((axis, 1.0 - axis))
    )


def test_consumer_decodes_complete_static_cdf_curves(tmp_path: Path) -> None:
    path = tmp_path / "complete-2d.cfe"
    metadata = CheckpointMetadata(
        task_id="complete-static",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(20):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(f"event-0-{index}", (0.0, 0.0)),
            )
        writer.finish_event()

    decoded = read_complete_task_nhv(
        path,
        mode="STATIC_CDF",
        analytic_reference_scales={0: _scale_2d()},
        expected_event_count=1,
    )
    assert decoded.task_id == "complete-static"
    assert decoded.event_ids == (0,)
    assert decoded.nhv_by_event[0][0] == 0.0
    assert decoded.nhv_by_event[0][1:] == pytest.approx((1.0,) * 20)
    assert len(decoded.checkpoint_file_sha256) == 64


def test_consumer_decodes_complete_rolling_curve(tmp_path: Path) -> None:
    path = tmp_path / "complete-3d.cfe"
    metadata = CheckpointMetadata(
        task_id="complete-rolling",
        objective_names=("objective_0", "objective_1", "objective_2"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(20):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(f"event-0-{index}", (1.0, 1.0, 1.0)),
            )
        writer.finish_event()

    decoded = read_complete_task_nhv(
        path,
        mode="ROLLING",
        expected_event_count=1,
    )
    assert decoded.nhv_by_event[0][0] == 0.0
    assert decoded.nhv_by_event[0][1:] == pytest.approx((0.125,) * 20)


def test_consumer_fails_closed_on_partial_terminal_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cfe"
    metadata = CheckpointMetadata(
        task_id="partial",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        for index in range(3):
            writer.record_success(
                event_id=0,
                vector=(float(index),),
                result=_result(f"partial-{index}", (0.5, 0.5)),
            )
        writer.finish_event(terminal_snapshot=True)

    with pytest.raises(IncompleteCheckpointDataError, match="21"):
        read_complete_task_nhv(
            path,
            mode="STATIC_CDF",
            analytic_reference_scales={0: _scale_2d()},
            expected_event_count=1,
        )

    decoded = read_checkpoint_file(path)
    timeout = checkpoint_consumer._task_nhv_from_decoded(
        decoded,
        mode="STATIC_CDF",
        analytic_reference_scales={0: _scale_2d()},
        expected_event_count=1,
        terminal_codes={0: "REJECT_TIMEOUT"},
    )
    assert timeout.timeout_carried_forward_event_ids == (0,)
    assert len(timeout.nhv_by_event[0]) == 21
    assert timeout.nhv_by_event[0][3:] == pytest.approx(
        (timeout.nhv_by_event[0][3],) * 18
    )
    with pytest.raises(
        NumericalContinuousEndpointExcluded,
        match="numerical",
    ):
        checkpoint_consumer._task_nhv_from_decoded(
            decoded,
            mode="STATIC_CDF",
            analytic_reference_scales={0: _scale_2d()},
            expected_event_count=1,
            terminal_codes={0: "REJECT_NUMERICAL"},
        )


def test_consumer_requires_exact_scale_and_event_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "identity.cfe"
    metadata = CheckpointMetadata(
        task_id="identity",
        objective_names=("objective_0", "objective_1"),
    )
    with TaskCheckpointWriter(path, metadata) as writer:
        writer.begin_event(event_id=4, cfe_budget=20)
        for index in range(20):
            writer.record_failure(
                event_id=4,
                candidate_id=f"failure-{index}",
                vector=(float(index),),
                error_type="NumericalFailure",
                reason="fixture",
            )
        writer.finish_event()

    with pytest.raises(CheckpointAnalysisError, match="zero-based"):
        read_complete_task_nhv(
            path,
            mode="STATIC_CDF",
            analytic_reference_scales={4: _scale_2d()},
            expected_event_count=1,
        )
    with pytest.raises(CheckpointAnalysisError, match="exactly match"):
        read_complete_task_nhv(
            path,
            mode="STATIC_CDF",
            analytic_reference_scales={0: _scale_2d()},
        )
