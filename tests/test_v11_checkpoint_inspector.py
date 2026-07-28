from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from evaluation.contracts import EvaluationResult
from formal_execution.checkpoint_data import (
    CheckpointMetadata,
    TaskCheckpointWriter,
)


def _checkpoint_file(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoint_fronts.bin"
    result = EvaluationResult(
        candidate_id="C-001",
        objectives=(0.25, 0.75),
        objective_names=("f1", "f2"),
        constraints=(-0.5,),
        constraint_names=("c1",),
    )
    with TaskCheckpointWriter(
        path,
        CheckpointMetadata(
            task_id="r8c-test",
            objective_names=("f1", "f2"),
        ),
    ) as writer:
        writer.begin_event(event_id=0, cfe_budget=20)
        writer.record_success(
            event_id=0,
            vector=(0.1,),
            result=result,
        )
        for index in range(2, 21):
            writer.record_failure(
                event_id=0,
                candidate_id=f"C-{index:03d}",
                vector=(float(index),),
                error_type="SyntheticFailure",
                reason="test-only",
            )
        writer.finish_event()
    return path


def _tool() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "inspect_v11_e1e2_checkpoint.py"
    )


def test_integrity_mode_never_prints_effect_values(tmp_path: Path) -> None:
    path = _checkpoint_file(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(_tool()), "--input", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["checkpoint_record_count"] == 21
    assert payload["effect_values_printed"] is False
    assert "front_objectives" not in completed.stdout
    assert "0.25" not in completed.stdout


def test_effect_export_interface_is_disabled_even_with_old_self_auth_flag(
    tmp_path: Path,
) -> None:
    path = _checkpoint_file(tmp_path)
    output = tmp_path / "front.csv"
    denied = subprocess.run(
        [
            sys.executable,
            str(_tool()),
            "--input",
            str(path),
            "--event-id",
            "0",
            "--checkpoint-index",
            "1",
            "--export",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "unrecognized arguments" in denied.stderr
    assert not output.exists()

    self_authorized = subprocess.run(
        [
            sys.executable,
            str(_tool()),
            "--input",
            str(path),
            "--event-id",
            "0",
            "--checkpoint-index",
            "1",
            "--export",
            str(output),
            "--r9-effect-view-authorized",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert self_authorized.returncode == 2
    assert "unrecognized arguments" in self_authorized.stderr
    assert not output.exists()
