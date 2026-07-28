"""Run the result-blind R8C E1+E2 full-path target qualification."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import sys


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resource_pilot.e1e2_fullpath import (
    DEFAULT_DYNAMIC_EVENTS,
    DEFAULT_REPETITIONS,
    DEFAULT_WORKERS,
    E1E2QualificationError,
    run_e1e2_qualification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise every staged E1+E2 method/workload path at 100/100 "
            "with deterministic high-entropy synthetic success/failure "
            "streams through the production endpoint checkpoint writer; "
            "no real effect value is persisted."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
        help=(
            "Target sweep is exactly 1 8 16 24 32 48 64; alternate lists "
            "are diagnostic and cannot qualify a host."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument(
        "--cfe-per-event",
        type=int,
        default=None,
        help=(
            "Uniform diagnostic CFE scale; allowed only with --smoke. "
            "The target design fixes static=50000, dynamic=5000 and "
            "rolling=5000."
        ),
    )
    parser.add_argument(
        "--dynamic-events",
        type=int,
        default=DEFAULT_DYNAMIC_EVENTS,
        help=(
            "Target design uses six events (0..5) so CDF9 reaches its "
            "maximum undefined-domain stress; changing this is nonqualifying."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run all 33 paths over 84 workload/method/case bindings once "
            "at 100 CFE with one worker; this cannot qualify a target."
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow an uncommitted tree for diagnostics only; the resulting "
            "report cannot qualify a target."
        ),
    )
    return parser


def _completion_exit_code(
    report: dict[str, object],
    *,
    smoke: bool,
) -> int:
    if report.get("failed_task_count"):
        return 1
    if report.get("status") == (
        "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
    ):
        return 0
    if smoke and report.get("status") == (
        "PASS_NONQUALIFYING_DIAGNOSTIC"
    ):
        return 0
    return 3


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    workers = args.workers
    repetitions = args.repetitions
    cfe_per_event = args.cfe_per_event
    dynamic_events = args.dynamic_events
    if args.smoke:
        workers = [1]
        repetitions = 1
        cfe_per_event = 100
        dynamic_events = 1
    elif cfe_per_event is not None:
        print(
            "E1+E2 qualification refused: --cfe-per-event is a smoke-only "
            "uniform diagnostic override"
        )
        return 2
    try:
        report = run_e1e2_qualification(
            output_root=args.output_root,
            worker_counts=workers,
            repetitions=repetitions,
            cfe_per_event=cfe_per_event,
            dynamic_events=dynamic_events,
            allow_dirty=args.allow_dirty,
            smoke=args.smoke,
        )
    except E1E2QualificationError as error:
        print(f"E1+E2 qualification refused: {error}")
        return 2
    print(
        f"E1+E2 qualification {report['status']}: "
        f"{args.output_root.resolve()}"
    )
    exit_code = _completion_exit_code(report, smoke=args.smoke)
    if exit_code == 3:
        print(
            "E1+E2 qualification completed but did not qualify this "
            "target host"
        )
    return exit_code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
