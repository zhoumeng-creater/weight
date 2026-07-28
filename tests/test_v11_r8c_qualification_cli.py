from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_test_v11_r8c_qualification_cli",
        root / "tools" / "run_v11_r8c_e1e2_qualification.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exit_code_distinguishes_target_qualification_from_diagnostic() -> None:
    module = _module()
    assert module._completion_exit_code(
        {
            "status": (
                "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
            ),
            "failed_task_count": 0,
        },
        smoke=False,
    ) == 0
    diagnostic = {
        "status": "PASS_NONQUALIFYING_DIAGNOSTIC",
        "failed_task_count": 0,
    }
    assert module._completion_exit_code(
        diagnostic,
        smoke=True,
    ) == 0
    assert module._completion_exit_code(
        diagnostic,
        smoke=False,
    ) == 3
    assert module._completion_exit_code(
        {"status": "FAILED_NO_RETRY", "failed_task_count": 1},
        smoke=False,
    ) == 1
